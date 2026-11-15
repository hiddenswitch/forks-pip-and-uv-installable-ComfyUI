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
    runtime._executor = None
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
    assert runtime.coordinator.broadcasts == [{"kind": "load_tensor", "load_spec": load_spec}]
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
    runtime._executor = None
    runtime._closed = False
    loaded = []
    commands = []

    def load_rank(load_spec, rank, device, operations):
        loaded.append((load_spec, rank, device, operations))
        return SimpleNamespace(model_size=lambda: 64, model=SimpleNamespace(diffusion_model=object()))

    def run_worker(operations, coordinator, patcher, span_name, close_commands=("close",)):
        del operations, coordinator, patcher, span_name
        assert close_commands == ("unload_tensor", "close")
        command = next(runtime.coordinator.commands)
        commands.append(command["kind"])
        return command["kind"]

    destroyed = []
    monkeypatch.setattr("comfy.tensor_parallel.loader.load_tensor_parallel_rank", load_rank)
    monkeypatch.setattr(external, "_run_worker", run_worker)
    monkeypatch.setattr("comfy.model_management.soft_empty_cache", lambda: None)
    monkeypatch.setattr(external, "destroy_independent_process_groups", destroyed.append)

    runtime.run_worker_service()

    assert [item[0] for item in loaded] == ["first", "second"]
    assert all(item[1:] == (1, torch.device("cpu"), runtime.operations) for item in loaded)
    assert [item for item in runtime.coordinator.sent if item[0]["kind"] == "ready"] == [
        ({"kind": "ready", "size": 64}, 0),
        ({"kind": "ready", "size": 64}, 0),
    ]
    assert commands == ["unload_tensor", "close"]
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
