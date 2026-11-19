from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args_types import Configuration
from comfy.distributed.config import DistributedConfiguration
from comfy.execution_context import context_configuration
from comfy.tensor_parallel import external


class _Coordinator:
    def __init__(self, commands=(), responses=()):
        self.commands = iter(commands)
        self.responses = iter(responses)
        self.broadcasts = []
        self.sent = []

    def broadcast_command(self, command=None):
        if command is not None:
            self.broadcasts.append(command)
            return command
        return next(self.commands)

    def receive_object(self, rank):
        del rank
        return next(self.responses)

    def send_object(self, value, destination):
        self.sent.append((value, destination))


def test_driver_loads_its_shard_and_connects_peer_sizes():
    runtime = external.ExternalTensorParallelRuntime.__new__(external.ExternalTensorParallelRuntime)
    runtime._configuration = DistributedConfiguration(
        rank=0,
        world_size=2,
        tensor_parallel_size=2,
        executor_backend="external_launcher",
        externally_launched=True,
    )
    runtime.logical_devices = (torch.device("cuda", 0), torch.device("cuda", 1))
    runtime.operations = object()
    runtime.coordinator = _Coordinator(responses=({"kind": "ready", "size": 200},))
    runtime.process_groups = SimpleNamespace(control_process_group=object())
    runtime._executors = {}
    runtime._next_model_id = 0
    load_spec = object()
    loaded = []

    root = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=object()),
        model_size=lambda: 100,
    )

    def load_root(operations):
        loaded.append(operations)
        return root

    returned_root, executor = runtime.load_executor(load_spec, load_root)

    assert returned_root is root
    assert loaded == [runtime.operations]
    assert runtime.coordinator.broadcasts == [{"kind": "load_tensor", "model_id": 1, "load_spec": load_spec}]
    assert runtime._executors == {1: executor}
    assert executor.rank_sizes == (100, 200)
    assert executor.close_command == "unload_tensor"
    executor._closed = True


def test_worker_loads_shards_again_after_unload_without_rebuilding_groups(monkeypatch):
    runtime = external.ExternalTensorParallelRuntime.__new__(external.ExternalTensorParallelRuntime)
    runtime._configuration = DistributedConfiguration(
        rank=1,
        world_size=2,
        tensor_parallel_size=2,
        executor_backend="external_launcher",
        externally_launched=True,
    )
    runtime.device = torch.device("cpu")
    runtime.operations = object()
    runtime.coordinator = _Coordinator(
        commands=(
            {"kind": "load_tensor", "load_spec": "first"},
            {"kind": "unload_tensor"},
            {"kind": "load_tensor", "load_spec": "second"},
            {"kind": "close"},
        )
    )
    runtime.process_groups = object()
    runtime._executors = {}
    runtime._closed = False
    loaded = []
    unloaded = []

    def load_rank(load_spec, rank, device, operations):
        loaded.append((load_spec, rank, device, operations))
        return SimpleNamespace(model_size=lambda: 64, model=SimpleNamespace(diffusion_model=object()))

    destroyed = []
    monkeypatch.setattr("comfy.tensor_parallel.loader.load_tensor_parallel_rank", load_rank)
    monkeypatch.setattr("comfy.model_management.soft_empty_cache", lambda: unloaded.append(True))
    monkeypatch.setattr(external, "destroy_independent_process_groups", destroyed.append)

    runtime.run_worker_service()

    assert [item[0] for item in loaded] == ["first", "second"]
    assert all(item[1:] == (1, torch.device("cpu"), runtime.operations) for item in loaded)
    assert [item for item in runtime.coordinator.sent if item[0]["kind"] == "ready"] == [
        ({"kind": "ready", "size": 64}, 0),
        ({"kind": "ready", "size": 64}, 0),
    ]
    assert unloaded == [True]
    assert destroyed == [runtime.process_groups]


def test_external_loader_uses_launcher_devices_without_enumerating_local_devices(monkeypatch):
    from comfy.tensor_parallel import loader

    runtime = SimpleNamespace(
        logical_devices=(torch.device("cuda", 0), torch.device("cuda", 1)),
    )
    result = object()
    seen = []

    class Reader:
        pass

    monkeypatch.setattr(loader, "SafetensorsCheckpointReader", lambda path: Reader())
    monkeypatch.setattr(
        loader,
        "_normalize_detection_state",
        lambda reader: ({}, {}, ""),
    )
    monkeypatch.setattr(
        loader.model_detection,
        "model_config_from_unet",
        lambda *args, **kwargs: SimpleNamespace(
            unet_config={"image_model": "magi2"},
        ),
    )
    monkeypatch.setattr(loader, "is_safetensors_checkpoint", lambda path: True)
    monkeypatch.setattr(
        "comfy.tensor_parallel.external.get_external_tensor_parallel_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        loader.model_management,
        "get_all_torch_devices",
        lambda: pytest.fail("external launch must use launcher ranks"),
    )
    monkeypatch.setattr(
        loader,
        "load_diffusion_model_tensor_parallel",
        lambda path, devices, **kwargs: seen.append((path, devices, kwargs)) or result,
    )
    configuration = Configuration(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
    )

    with context_configuration(configuration):
        assert loader.try_load_diffusion_model_tensor_parallel("model.safetensors") is result

    assert seen == [(
        "model.safetensors",
        runtime.logical_devices,
        {
            "model_options": None,
            "disable_dynamic": False,
            "external_runtime": runtime,
        },
    )]


def test_worker_completes_cached_index_before_opening_shards(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    from comfy import model_downloader as downloader
    from comfy.pipeline_parallel.checkpoint import SafetensorsCheckpointReader

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    index = snapshot / "test.safetensors.index.json"
    names = ("first.safetensors", "second.safetensors")
    index.write_text(json.dumps({"weight_map": {"x": names[0], "y": names[1]}}))
    save_file({"x": torch.ones(1)}, snapshot / names[0])
    with pytest.raises(FileNotFoundError):
        SafetensorsCheckpointReader(index)

    model = downloader.HuggingFile("test/repo", index.name, revision="pinned", companion_files=names)
    monkeypatch.setattr(downloader, "KNOWN_UNET_MODELS", [model])
    downloaded = []

    def fetch(filename, revision, local_files_only=False, **kwargs):
        assert revision == model.revision
        file = snapshot / filename
        if file.is_file():
            return str(file)
        if local_files_only:
            raise downloader.LocalEntryNotFoundError("cache miss")
        downloaded.append(filename)
        save_file({"y": torch.ones(1)}, file)
        return str(file)

    monkeypatch.setattr(downloader, "hf_hub_download", fetch)
    resolved = downloader.resolve_worker_checkpoint(str(index))
    SafetensorsCheckpointReader(resolved)
    assert downloaded == [names[1]]
    assert Path(resolved) == index


def test_worker_drains_unload_and_close_after_device_group_abort(monkeypatch):
    from comfy import model_management

    runtime = external.ExternalTensorParallelRuntime.__new__(external.ExternalTensorParallelRuntime)
    runtime._configuration = DistributedConfiguration(
        rank=1, world_size=2, tensor_parallel_size=2, executor_backend="external_launcher",
        externally_launched=True,
    )
    runtime.device = torch.device("cpu")
    runtime.operations = SimpleNamespace(rank=1, world_size=2, process_group=object())
    runtime.coordinator = _Coordinator(commands=(
        {"kind": "load_tensor", "load_spec": "refiner"},
        {"kind": "forward", "method": "forward", "descriptors": [],
         "structure": ("value", ((), {}))},
        {"kind": "unload_tensor"}, {"kind": "close"},
    ))

    def forward():
        raise RuntimeError("rowwise quantization failed")

    patcher = SimpleNamespace(model_size=lambda: 64, model=SimpleNamespace(
        diffusion_model=SimpleNamespace(forward=forward)))
    closed = []
    monkeypatch.setattr("comfy.tensor_parallel.loader.load_tensor_parallel_rank", lambda *args: patcher)
    monkeypatch.setattr(external, "_broadcast_tensors", lambda *args, **kwargs: {})
    monkeypatch.setattr(external, "_abort_device_group", lambda group: None)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: pytest.fail("synchronized failed device"))
    monkeypatch.setattr(runtime, "close", lambda: closed.append(True))

    with pytest.raises(RuntimeError, match="rowwise quantization failed"):
        runtime.run_worker_service()
    assert closed == [True]
    assert "rowwise quantization failed" in runtime.coordinator.sent[-1][0]["traceback"]
