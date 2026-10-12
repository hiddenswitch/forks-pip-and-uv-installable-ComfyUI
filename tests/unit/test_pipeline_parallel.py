import json
import ctypes
import struct
import types
import uuid

import pytest
import torch

import comfy.ops as comfy_ops
from comfy import memory_management, model_management
from comfy.distributed.config import DistributedConfiguration
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.flux.model import Flux
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.model_management_types import LoadingListItem
from comfy.model_patcher import ModelPatcher
from comfy.pipeline_parallel import (
    AbstractBaseDeviceRuntime,
    AbstractBasePendingPipelineIntermediate,
    AbstractBasePipelineTransport,
    AbstractBasePipelineStageRunner,
    DevicePipelineBufferPool,
    Flux2PipelineStageSpec,
    MiniMaxH3PipelineStageSpec,
    PipelineParallelConfig,
    QwenImagePipelineStageSpec,
    SafetensorsCheckpointReader,
    SingleProcessPipelineExecutor,
    TensorDescriptor,
    deserialize_pipeline_metadata,
    serialize_pipeline_metadata,
)
from comfy.pipeline_parallel.types import PipelineDeviceMemoryBudget, PipelineModelMemoryGeometry
from comfy.pipeline_parallel.types import PipelineIntermediateTensors, PipelinePartitionPlan, PipelineStageConfig, PipelineStagePlan, PipelineWorkerLoadSpec
from comfy.pipeline_parallel.types import pack_pipeline_value, unpack_pipeline_value
from comfy.pipeline_parallel.patcher import get_pipeline_model_patcher_class
from comfy.pipeline_parallel.memory import ComfyDynamicVRAMStageMemoryEstimator, ComfyPipelineMemoryCoordinator
from comfy.pipeline_parallel.memory import ExternalPipelineMemoryCoordinator
from comfy.pipeline_parallel.distributed import (
    RemotePipelineStageModel,
    TorchDistributedProcessGroupCoordinator,
    _DistributedCompletion,
)
from comfy.pipeline_parallel.external import (
    CudaExternalPipelinePlacementProvider,
    ExternalTorchDistributedRuntime,
)
from comfy.pipeline_parallel.operations import PipelineOperationsMux


def descriptor(nbytes):
    return TensorDescriptor((nbytes,), torch.uint8, nbytes)


def test_qwen_pipeline_partition_owns_entry_and_exit_keys():
    tensors = {
        "img_in.weight": descriptor(10),
        "transformer_blocks.0.weight": descriptor(100),
        "transformer_blocks.1.weight": descriptor(100),
        "transformer_blocks.2.weight": descriptor(100),
        "transformer_blocks.3.weight": descriptor(100),
        "norm_out.linear.weight": descriptor(20),
        "proj_out.weight": descriptor(30),
    }
    spec = QwenImagePipelineStageSpec()
    spec.block_count = 4
    plan = spec.plan(tensors, PipelineParallelConfig(("cpu", "meta")))

    assert [(stage.start_layer, stage.end_layer) for stage in plan.stages] == [(0, 2), (2, 4)]
    assert "img_in.weight" in plan.stages[0].owned_keys
    assert "norm_out.linear.weight" in plan.stages[1].owned_keys
    assert "proj_out.weight" in plan.stages[1].owned_keys


def test_minimax_partition_balances_checkpoint_bytes_not_layer_count():
    tensors = {"blocks.0.weight": descriptor(1000), "blocks.1.weight": descriptor(1000)}
    tensors.update({f"blocks.{index}.weight": descriptor(100) for index in range(2, 10)})
    spec = MiniMaxH3PipelineStageSpec()
    spec.block_count = 10
    plan = spec.plan(tensors, PipelineParallelConfig(("cpu", "meta")))

    assert (plan.stages[0].end_layer, 10 - plan.stages[0].end_layer) != (5, 5)
    assert max(stage.checkpoint_bytes for stage in plan.stages) <= 2000


def test_flux2_pipeline_partition_flattens_double_and_single_blocks():
    tensors = {
        "img_in.weight": descriptor(10),
        "final_layer.linear.weight": descriptor(10),
    }
    tensors.update(
        {
            f"double_blocks.{index}.img_attn.qkv.weight": descriptor(100)
            for index in range(8)
        }
    )
    tensors.update(
        {
            f"single_blocks.{index}.linear1.weight": descriptor(100)
            for index in range(48)
        }
    )
    spec = Flux2PipelineStageSpec()

    assert spec.block_index("double_blocks.7.img_attn.qkv.weight") == 7
    assert spec.block_index("single_blocks.0.linear1.weight") == 8
    assert spec.block_index("single_blocks.47.linear1.weight") == 55
    plan = spec.plan(tensors, PipelineParallelConfig(("cpu", "meta")))
    assert "img_in.weight" in plan.stages[0].owned_keys
    assert "final_layer.linear.weight" in plan.stages[-1].owned_keys


def test_flux2_pipeline_rejects_non_dev_block_geometry():
    tensors = {
        f"double_blocks.{index}.weight": descriptor(1)
        for index in range(8)
    }
    tensors.update(
        {
            f"single_blocks.{index}.weight": descriptor(1)
            for index in range(24)
        }
    )

    with pytest.raises(ValueError, match="currently supports Flux2 Dev"):
        Flux2PipelineStageSpec().plan(
            tensors,
            PipelineParallelConfig(("cpu", "meta")),
        )


def test_partition_tracks_dynamic_vram_device_capacity_instead_of_equal_bytes():
    spec = QwenImagePipelineStageSpec()
    spec.block_count = 8
    tensors = {f"transformer_blocks.{index}.weight": descriptor(100) for index in range(8)}
    config = PipelineParallelConfig(("cpu", "meta"))
    budgets = (
        PipelineDeviceMemoryBudget(torch.device("cpu"), 600),
        PipelineDeviceMemoryBudget(torch.device("meta"), 200),
    )

    plan = spec.plan(tensors, config, memory_budgets=budgets)

    assert [(stage.start_layer, stage.end_layer) for stage in plan.stages] == [(0, 6), (6, 8)]
    assert [stage.checkpoint_bytes for stage in plan.stages] == [600, 200]


def test_partition_uses_loaded_dynamic_vram_geometry_instead_of_checkpoint_bytes():
    spec = QwenImagePipelineStageSpec()
    spec.block_count = 4
    tensors = {
        "transformer_blocks.0.weight": descriptor(1000),
        "transformer_blocks.1.weight": descriptor(1000),
        "transformer_blocks.2.weight": descriptor(100),
        "transformer_blocks.3.weight": descriptor(100),
    }
    geometry = PipelineModelMemoryGeometry((100, 100, 1000, 1000), (0, 0))

    plan = spec.plan(
        tensors,
        PipelineParallelConfig(("cpu", "meta")),
        model_memory_geometry=geometry,
    )

    assert [(stage.start_layer, stage.end_layer) for stage in plan.stages] == [(0, 3), (3, 4)]


def test_dynamic_vram_stage_estimator_uses_materialization_geometry():
    spec = QwenImagePipelineStageSpec()
    spec.block_count = 2
    stages = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("meta"), 1, 2, 1, frozenset()),
    )

    class FakePatcher:
        def __init__(self, size, items):
            self.size = size
            self.items = items

        def _load_list(self, for_dynamic=False):
            assert for_dynamic
            return self.items

        def model_size(self):
            return self.size

    def item(name, stored, offload):
        return LoadingListItem(None, offload, stored, name, None, {})

    patchers = (
        FakePatcher(160, [
            item("diffusion_model.transformer_blocks.0.proj", 100, 300),
            item("diffusion_model.img_in", 50, 150),
        ]),
        FakePatcher(100, [
            item("diffusion_model.transformer_blocks.1.proj", 100, 100),
        ]),
    )

    geometry = ComfyDynamicVRAMStageMemoryEstimator().estimate(
        spec, PipelinePartitionPlan("qwen_image", stages), patchers
    )

    assert geometry.block_bytes == (200, 100)
    assert geometry.non_block_bytes == (110, 0)


def test_comfy_pipeline_memory_coordinator_uses_projected_dynamic_vram_accounting(monkeypatch):
    calls = []

    def available(devices):
        calls.append(tuple(devices))
        return {torch.device("cpu"): 100, torch.device("meta"): 50}

    monkeypatch.setattr("comfy.pipeline_parallel.memory.model_management.projected_dynamic_vram_available_memory", available)
    budgets = ComfyPipelineMemoryCoordinator().budgets((torch.device("cpu"), torch.device("meta")))

    assert calls == [(torch.device("cpu"), torch.device("meta"))]
    assert [budget.available_weight_bytes for budget in budgets] == [100, 50]


def test_external_pipeline_placement_uses_global_rank_as_logical_device():
    configuration = DistributedConfiguration(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        pipeline_parallel_size=2,
        externally_launched=True,
    )
    provider = CudaExternalPipelinePlacementProvider()

    assert provider.local_device(configuration) == torch.device("cuda:0")
    assert provider.logical_devices(configuration) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


def test_external_pipeline_memory_coordinator_combines_rank_local_budgets():
    class PlacementProvider:
        def available_weight_bytes(self, device):
            assert device == torch.device("cuda:0")
            return 700

    class Coordinator:
        def __init__(self):
            self.commands = []

        def broadcast_command(self, command):
            self.commands.append(command)

        def receive_object(self, rank):
            assert rank == 1
            return {"kind": "memory_budget", "available_weight_bytes": 300}

    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    runtime = object.__new__(ExternalTorchDistributedRuntime)
    runtime._configuration = DistributedConfiguration(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        pipeline_parallel_size=2,
        externally_launched=True,
    )
    runtime._device = torch.device("cuda:0")
    runtime._logical_devices = devices
    runtime.placement_provider = PlacementProvider()
    runtime.coordinator = Coordinator()

    budgets = ExternalPipelineMemoryCoordinator(runtime).budgets(devices)

    assert runtime.coordinator.commands == [{"kind": "probe_memory"}]
    assert [budget.available_weight_bytes for budget in budgets] == [700, 300]


def test_external_worker_resolves_logical_stage_to_rank_local_device(monkeypatch):
    plan = PipelinePartitionPlan(
        "qwen_image",
        (
            PipelineStagePlan(0, torch.device("cuda:0"), 0, 1, 1, frozenset()),
            PipelineStagePlan(1, torch.device("cuda:1"), 1, 2, 1, frozenset()),
        ),
    )
    load_spec = PipelineWorkerLoadSpec(
        checkpoint_path="model.safetensors",
        plan=plan,
        stage_index=1,
        model_options={},
        disable_dynamic=False,
        dtype=torch.bfloat16,
    )

    class Coordinator:
        def broadcast_command(self):
            return {"kind": "load_pipeline", "worker_load_specs": (load_spec,)}

        def send_object(self, value, rank):
            assert rank == 0
            assert value["kind"] == "ready"

    loaded = []

    def load_stage(resolved, rank):
        assert rank == 1
        loaded.append(resolved.plan.stages[1].device)
        return object(), object()

    monkeypatch.setattr("comfy.pipeline_parallel.external._load_worker_stage", load_stage)
    monkeypatch.setattr(
        "comfy.pipeline_parallel.external._run_worker_commands",
        lambda *_args: "close",
    )
    runtime = object.__new__(ExternalTorchDistributedRuntime)
    runtime._configuration = DistributedConfiguration(
        rank=1,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        pipeline_parallel_size=2,
        externally_launched=True,
    )
    runtime._device = torch.device("cuda:0")
    runtime.coordinator = Coordinator()
    runtime.close = lambda: None

    runtime.run_worker_service()

    assert loaded == [torch.device("cuda:0")]


def test_projected_dynamic_vram_capacity_includes_ejectable_loaded_models(monkeypatch):
    device = torch.device("cpu")
    loaded = types.SimpleNamespace(
        device=device,
        is_dead=lambda: False,
        model=types.SimpleNamespace(reclaimable_non_vbar_memory=lambda: 75),
    )
    monkeypatch.setattr(model_management, "current_loaded_models", [loaded])
    monkeypatch.setattr(model_management, "get_free_memory", lambda _device: 25)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)

    assert model_management.projected_dynamic_vram_available_memory((device,))[device] == 100


def test_manual_partition_must_cover_all_blocks():
    spec = QwenImagePipelineStageSpec()
    spec.block_count = 4
    tensors = {f"transformer_blocks.{index}.weight": descriptor(1) for index in range(4)}
    with pytest.raises(ValueError, match="covers 3 blocks"):
        spec.plan(tensors, PipelineParallelConfig(("cpu", "meta"), partition=(1, 2)))


def test_safetensors_reader_does_not_load_unselected_tensor(tmp_path, monkeypatch):
    header = {
        "large.weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "layer.comfy_quant": {"dtype": "U8", "shape": [2], "data_offsets": [16, 18]},
        "__metadata__": {"model": "test"},
    }
    encoded = json.dumps(header).encode("utf-8")
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(18))
    loaded = []

    def load_torch_file(_path, include_keys=None):
        loaded.extend(include_keys)
        return {key: torch.tensor([123, 125], dtype=torch.uint8) for key in include_keys}

    monkeypatch.setattr("comfy.pipeline_parallel.checkpoint.utils.load_torch_file", load_torch_file)
    reader = SafetensorsCheckpointReader(path)
    state_dict = reader.detection_state_dict()

    assert loaded == ["layer.comfy_quant"]
    assert state_dict["large.weight"].device.type == "meta"
    assert state_dict["layer.comfy_quant"].device.type == "cpu"
    assert reader.metadata == {"model": "test"}


@pytest.mark.parametrize("selection,mapped", [
    ((slice(1, 3), slice(None)), True),
    ((slice(1, 2), slice(2, 5)), True),
    ((slice(None), slice(1, 3)), False),
    ((slice(None, None, 2), slice(None)), False),
    ((slice(2, 2), slice(None)), False),
])
def test_checkpoint_shards_keep_bounded_file_backing(tmp_path, monkeypatch, selection, mapped):
    from safetensors.torch import save_file
    import comfy_aimdo.model_mmap

    class Mapping:
        # Exercise the loader's bounded storage on CPU-only CI too.
        def __init__(self, path):
            with open(path, "rb") as source:
                self.buffer = ctypes.create_string_buffer(source.read())

        def get(self):
            return ctypes.addressof(self.buffer)

        def get_file_handle(self):
            return self

    monkeypatch.setattr(comfy_aimdo.model_mmap, "ModelMMAP", Mapping)
    monkeypatch.setitem(memory_management.__dict__, "aimdo_enabled", True)
    weights = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    path = tmp_path / "shards.safetensors"
    save_file({"weight": weights, "unselected": torch.ones(3)}, path)
    actual = SafetensorsCheckpointReader(path).load_slices({"weight": selection})
    assert set(actual) == {"weight"}
    tensor = actual["weight"]
    torch.testing.assert_close(tensor, weights[selection])
    assert tensor.storage_offset() == 0
    assert tensor.untyped_storage().nbytes() == tensor.nbytes
    info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
    assert (info is not None) == mapped
    if mapped:
        assert info.size == tensor.nbytes
        with path.open("rb") as source:
            source.seek(info.offset)
            stored = torch.frombuffer(bytearray(source.read(info.size)), dtype=tensor.dtype)
        torch.testing.assert_close(stored.reshape(tensor.shape), tensor)


def test_checkpoint_shards_respect_disabled_mmap(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from comfy import utils

    monkeypatch.setitem(memory_management.__dict__, "aimdo_enabled", True)
    monkeypatch.setattr(utils, "DISABLE_MMAP", True)
    path = tmp_path / "shards.safetensors"
    save_file({"weight": torch.ones(4, 6)}, path)
    reader = SafetensorsCheckpointReader(path)
    value = reader.load_slices({"weight": (slice(2), slice(None))})["weight"]
    assert not hasattr(value.untyped_storage(), "_comfy_tensor_file_slice")
    with pytest.raises(KeyError, match="missing"):
        reader.load_slices({"missing": None})


def test_checkpoint_reader_loads_through_a_hugging_face_cache_symlink(tmp_path):
    # the HF hub cache names the file snapshots/<rev>/x.safetensors -> blobs/<sha256>, no extension
    from safetensors.torch import save_file

    blob = tmp_path / "blobs" / "0123abcd"
    blob.parent.mkdir()
    save_file({"layer.weight": torch.arange(4, dtype=torch.float32), "other.weight": torch.ones(2)}, str(blob))
    link = tmp_path / "snapshots" / "rev" / "model.safetensors"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)

    loaded = SafetensorsCheckpointReader(link).load_keys({"layer.weight"})

    # read as safetensors with include_keys, not torch.load (which raises IndexError on a large
    # safetensors header and otherwise falls back to loading every tensor)
    assert set(loaded) == {"layer.weight"}
    torch.testing.assert_close(loaded["layer.weight"], torch.arange(4, dtype=torch.float32))


class FakeDeviceRuntime(AbstractBaseDeviceRuntime):
    def __init__(self):
        self.allocations = 0

    def supports(self, device):
        return True

    def can_transfer(self, source, destination):
        return True

    def allocate(self, shape, dtype, device):
        self.allocations += 1
        return torch.empty(shape, dtype=dtype)

    def copy(self, source, destination):
        destination.copy_(source)
        return None

    def wait(self, completion, device):
        return None

    def memory_available(self, device):
        return 1 << 40


class FakePipelineTransport(AbstractBasePipelineTransport):
    def __init__(self, runtime):
        self.pool = DevicePipelineBufferPool(runtime)
        self.closed = False

    def transfer(self, tensors, schema, destination):
        output = self.pool.acquire(schema, destination)
        for name, tensor in tensors.items():
            output[name].copy_(tensor)
        return output

    def close(self):
        self.pool.clear()
        self.closed = True

    def transfer_metadata(self, metadata, destination):
        del destination
        return deserialize_pipeline_metadata(serialize_pipeline_metadata(metadata))


def test_single_process_executor_injects_transport_and_reuses_buffers():
    device = torch.device("cpu")
    plans = (
        PipelineStagePlan(0, device, 0, 1, 1, frozenset()),
        PipelineStagePlan(1, device, 1, 2, 1, frozenset()),
    )
    plan = PipelinePartitionPlan("test", plans)
    runtime = FakeDeviceRuntime()
    transport = FakePipelineTransport(runtime)

    def first(value):
        return PipelineIntermediateTensors({"hidden": value + 1}, {"offset": 2})

    def second(intermediate):
        return intermediate.tensors["hidden"] + intermediate.metadata["offset"]

    executor = SingleProcessPipelineExecutor(plan, (first, second), transport)
    assert torch.equal(executor.execute(torch.tensor([1])), torch.tensor([4]))
    assert torch.equal(executor.execute(torch.tensor([2])), torch.tensor([5]))
    assert runtime.allocations == 2
    executor.close()
    assert transport.closed


def test_single_process_executor_consumes_pending_intermediate_at_stage_boundary():
    events = []

    class Pending(AbstractBasePendingPipelineIntermediate):
        def __init__(self, intermediate):
            self.intermediate = intermediate

        def wait(self):
            events.append("wait")
            return self.intermediate

    class DeferredTransport(FakePipelineTransport):
        def begin_transfer(self, intermediate, destination):
            del destination
            events.append("begin")
            return Pending(intermediate)

    device = torch.device("cpu")
    plan = PipelinePartitionPlan("test", (
        PipelineStagePlan(0, device, 0, 1, 1, frozenset()),
        PipelineStagePlan(1, device, 1, 2, 1, frozenset()),
    ))

    def first(value):
        return PipelineIntermediateTensors({"hidden": value}, {})

    def second(intermediate):
        events.append("stage")
        return intermediate.tensors["hidden"]

    executor = SingleProcessPipelineExecutor(
        plan,
        (first, second),
        DeferredTransport(FakeDeviceRuntime()),
    )

    executor.execute(torch.tensor([1]))

    assert events == ["begin", "wait", "stage", "begin", "wait"]


def test_pipeline_transport_does_not_share_mutable_metadata_between_stages():
    device = torch.device("cpu")
    plan = PipelinePartitionPlan("test", (
        PipelineStagePlan(0, device, 0, 1, 1, frozenset()),
        PipelineStagePlan(1, device, 1, 2, 1, frozenset()),
    ))
    original_metadata = {"values": [1]}

    def first(value):
        return PipelineIntermediateTensors({"hidden": value}, original_metadata)

    def second(intermediate):
        intermediate.metadata["values"].append(2)
        return intermediate.tensors["hidden"]

    executor = SingleProcessPipelineExecutor(
        plan,
        (first, second),
        FakePipelineTransport(FakeDeviceRuntime()),
    )

    executor.execute(torch.tensor([1]))

    assert original_metadata == {"values": [1]}


def test_pipeline_transport_serializes_final_output_structure():
    device = torch.device("cpu")
    plan = PipelinePartitionPlan("test", (
        PipelineStagePlan(0, device, 0, 1, 1, frozenset()),
        PipelineStagePlan(1, device, 1, 2, 1, frozenset()),
    ))
    transport = FakePipelineTransport(FakeDeviceRuntime())
    executor = SingleProcessPipelineExecutor(
        plan,
        (
            lambda value: PipelineIntermediateTensors({"hidden": value}, {}),
            lambda intermediate: [intermediate.tensors["hidden"]],
        ),
        transport,
    )

    output = executor.execute(torch.tensor([1]))

    assert isinstance(output, list)
    assert torch.equal(output[0], torch.tensor([1]))


class RecordingStageRunner(AbstractBasePipelineStageRunner):
    def __init__(self):
        self.devices = []

    def run(self, stage, device, *args, **kwargs):
        self.devices.append(device)
        return stage(*args, **kwargs)


def test_single_process_executor_injects_stage_device_context():
    devices = (torch.device("cpu"), torch.device("meta"))
    plan = PipelinePartitionPlan("test", (
        PipelineStagePlan(0, devices[0], 0, 1, 1, frozenset()),
        PipelineStagePlan(1, devices[1], 1, 2, 1, frozenset()),
    ))
    runner = RecordingStageRunner()
    executor = SingleProcessPipelineExecutor(
        plan,
        (
            lambda value: PipelineIntermediateTensors({"hidden": value}, {}),
            lambda intermediate: intermediate.tensors["hidden"],
        ),
        FakePipelineTransport(FakeDeviceRuntime()),
        runner,
    )

    executor.execute(torch.tensor([1]))

    assert runner.devices == list(devices)


def test_pipeline_operations_mux_selects_first_supported_provider():
    class Provider:
        def __init__(self, supported):
            self.supported = supported

        def supports(self, devices):
            return self.supported and tuple(devices) == (torch.device("cpu"),)

    fallback = Provider(False)
    selected = Provider(True)

    assert PipelineOperationsMux((fallback, selected)).select((torch.device("cpu"),)) is selected


def test_distributed_completion_waits_for_nccl_work_before_compute_stream(monkeypatch):
    events = []

    class Work:
        def wait(self):
            events.append("work")

    class Stream:
        def wait_event(self, event):
            events.append(event)

    coordinator = object.__new__(TorchDistributedProcessGroupCoordinator)
    coordinator.device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: Stream())

    coordinator.wait((_DistributedCompletion("event", (Work(),)),))

    assert events == ["work", "event"]


def test_pipeline_metadata_uuid_is_encoded_without_shared_object_state():
    value = uuid.uuid4()
    tensors = {}

    encoded = pack_pipeline_value({"uuids": [value]}, tensors, "options")
    decoded = unpack_pipeline_value(encoded, tensors)

    assert decoded == {"uuids": [value]}
    assert decoded["uuids"][0] is not value


def test_pipeline_mapping_keys_preserve_type_and_none_wrapper_key():
    value = {
        "wrappers": {"predict_noise": {None: ["wrapper"]}},
        ("stage", 1): torch.device("cpu"),
    }
    tensors = {}

    encoded = pack_pipeline_value(value, tensors, "options")
    decoded = unpack_pipeline_value(encoded, tensors)

    assert decoded == value


def test_pipeline_patcher_routes_weight_patches_and_clone_children():
    class RootModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_weight = torch.nn.Parameter(torch.ones(1))

    class ChildModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child_weight = torch.nn.Parameter(torch.ones(1))

    device = torch.device("cpu")
    pipeline_class = get_pipeline_model_patcher_class(disable_dynamic=True)
    root = pipeline_class(RootModule(), device, device)
    child = ModelPatcher(ChildModule(), device, device, _force_core=True)
    root.set_additional_models(root.pipeline_additional_models_key, [child])
    clone = root.clone()

    applied = clone.add_patches({
        "root_weight": (torch.ones(1),),
        "child_weight": (torch.ones(1),),
    })

    assert set(applied) == {"root_weight", "child_weight"}
    assert "root_weight" in clone.patches
    assert "child_weight" in clone._pipeline_stage_patchers()[0].patches
    assert set(clone.model_state_dict()) == {"root_weight", "child_weight"}


def test_grouped_model_load_rollback_unloads_only_new_models(monkeypatch):
    unloaded = []

    class FakeLoadedModel:
        def __init__(self, name):
            self.name = name

        def model_unload(self):
            unloaded.append(self.name)

    existing = FakeLoadedModel("existing")
    first = FakeLoadedModel("first")
    second = FakeLoadedModel("second")
    monkeypatch.setattr(model_management, "current_loaded_models", [existing, first, second])

    model_management._rollback_newly_loaded_models([first, second])

    assert model_management.current_loaded_models == [existing]
    assert unloaded == ["second", "first"]


def test_remote_pipeline_stage_uses_model_manager_load_and_unload_contract():
    class FakeExecutor:
        def __init__(self):
            self.commands = []
            self.loaded = 0

        def remote_stage_command(self, rank, command):
            self.commands.append((rank, command))
            if command["kind"] == "load":
                self.loaded = min(100, self.loaded + int(command["extra_memory"]))
            else:
                self.loaded = max(0, self.loaded - int(command["memory_to_free"]))
            return {"kind": "stage_model", "loaded_size": self.loaded}

    executor = FakeExecutor()
    remote = RemotePipelineStageModel(
        executor,
        rank=1,
        device=torch.device("cuda:1"),
        size=100,
        dtype=torch.bfloat16,
        dynamic=True,
    )

    assert remote.partially_load(remote.load_device, 75) == 75
    assert remote.loaded_size() == 75
    assert remote.current_loaded_device() == remote.load_device
    assert remote.partially_unload(remote.offload_device, 25) == 25
    assert remote.loaded_size() == 50
    remote.detach()
    assert remote.loaded_size() == 0
    assert remote.current_loaded_device() == remote.offload_device
    assert [command[1]["kind"] for command in executor.commands] == ["load", "unload", "unload"]


def test_external_remote_pipeline_stage_delegates_memory_policy_to_rank():
    class FakeExecutor:
        def __init__(self):
            self.command = None

        def remote_stage_command(self, rank, command):
            assert rank == 1
            self.command = command
            return {"kind": "stage_model", "loaded_size": 100}

    executor = FakeExecutor()
    remote = RemotePipelineStageModel(
        executor,
        rank=1,
        device=torch.device("cuda:1"),
        size=100,
        dtype=torch.bfloat16,
        dynamic=True,
        self_managed_device=True,
    )

    remote.partially_load(remote.load_device, 10**12)

    assert remote.manages_own_device_memory()
    assert executor.command["manage_device_memory"] is True


def test_model_manager_does_not_query_remote_rank_logical_device(monkeypatch):
    class FakeExecutor:
        def remote_stage_command(self, _rank, command):
            return {
                "kind": "stage_model",
                "loaded_size": 100 if command["kind"] == "load" else 0,
            }

    remote = RemotePipelineStageModel(
        FakeExecutor(),
        rank=7,
        device=torch.device("cuda:7"),
        size=100,
        dtype=torch.bfloat16,
        dynamic=True,
        self_managed_device=True,
    )
    prepared = []
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    monkeypatch.setattr(model_management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(
        model_management,
        "prepare_device_model_loads",
        lambda required, **_kwargs: prepared.append(required) or [],
    )
    monkeypatch.setattr(
        model_management,
        "get_free_memory",
        lambda device: pytest.fail(f"queried rank-logical device {device}"),
    )

    model_management._load_models_gpu([remote], minimum_memory_required=1)

    assert prepared == [{}]
    assert remote.loaded_size() == 100


def test_pipeline_cleanup_keeps_injected_executor_alive_between_samples():
    class RootModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class FakeExecutor:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    pipeline_class = get_pipeline_model_patcher_class(disable_dynamic=True)
    patcher = pipeline_class(RootModule(), torch.device("cpu"), torch.device("cpu"))
    executor = FakeExecutor()
    patcher.set_attachments("pipeline_parallel_executor", executor)

    patcher.cleanup()

    assert not executor.closed


def test_qwen_two_stage_forward_matches_unpartitioned(monkeypatch):
    monkeypatch.setattr("comfy.ldm.qwen_image.model.optimized_attention_masked", _test_attention)
    kwargs = {
        "patch_size": 2,
        "in_channels": 4,
        "out_channels": 1,
        "num_layers": 2,
        "attention_head_dim": 6,
        "num_attention_heads": 2,
        "joint_attention_dim": 12,
        "pooled_projection_dim": 12,
        "axes_dims_rope": (2, 2, 2),
        "dtype": torch.float32,
        "device": "cpu",
        "operations": comfy_ops.disable_weight_init,
    }
    torch.manual_seed(7)
    full = QwenImageTransformer2DModel(**kwargs)
    for parameter in full.parameters():
        torch.nn.init.uniform_(parameter, -0.05, 0.05)

    first = QwenImageTransformer2DModel(**kwargs, pipeline_stage=PipelineStageConfig(0, 2, 0, 1))
    second = QwenImageTransformer2DModel(**kwargs, pipeline_stage=PipelineStageConfig(1, 2, 1, 2))
    first.load_state_dict(full.state_dict(), strict=False)
    second.load_state_dict(full.state_dict(), strict=False)

    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("qwen_image", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    x = torch.randn(1, 1, 1, 4, 4)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 3, 12)

    expected = full(x, timestep, context)
    actual = executor.execute(x, timestep, context)
    torch.testing.assert_close(actual, expected)


def _test_attention(query, key, value, heads, mask=None, **kwargs):
    del heads, kwargs
    from comfy.ldm.modules.attention import AttentionTensorContainer
    if isinstance(query, AttentionTensorContainer):
        query, key, value = query.take(), key.take(), value.take()
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return torch.matmul(scores.softmax(dim=-1), value).transpose(1, 2).flatten(2)


@pytest.mark.parametrize("pdd_heads", [1, 3])
def test_minimax_two_stage_forward_matches_unpartitioned(monkeypatch, pdd_heads):
    monkeypatch.setattr("comfy.ldm.minimax.model.Attention.forward", _test_minimax_attention)
    kwargs = {
        "hidden_size": 12,
        "num_layers": 2,
        "token_refiner_num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 6,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 4,
        "patch_size": (1, 2, 2),
        "text_dim": 12,
        "timestep_input_dim": 8,
        "time_embed_hidden_size": 12,
        "time_embed_dim": 6,
        "rope_inv_freq_len": 1,
        "dtype": torch.float32,
        "device": "cpu",
        "operations": comfy_ops.disable_weight_init,
    }
    torch.manual_seed(11)
    full = MiniMaxH3Model(**kwargs)
    for parameter in full.parameters():
        torch.nn.init.uniform_(parameter, -0.05, 0.05)
    for buffer in full.buffers():
        torch.nn.init.uniform_(buffer, -0.05, 0.05)

    first = MiniMaxH3Model(**kwargs, pipeline_stage=PipelineStageConfig(0, 2, 0, 1))
    second = MiniMaxH3Model(**kwargs, pipeline_stage=PipelineStageConfig(1, 2, 1, 2))

    def expand_pdd_heads(model):
        if pdd_heads == 1 or not hasattr(model.final_layer, "video_out"):
            return
        for head in (model.final_layer.video_out, model.final_layer.audio_out):
            head.weight = torch.nn.Parameter(head.weight.detach().repeat(pdd_heads, 1))
            head.bias = torch.nn.Parameter(head.bias.detach().repeat(pdd_heads))

    expand_pdd_heads(full)
    expand_pdd_heads(second)
    first.load_state_dict(full.state_dict(), strict=False)
    second.load_state_dict(full.state_dict(), strict=False)
    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("minimax_h3", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    video = torch.randn(1, 2, 1, 2, 2)
    audio = torch.randn(1, 4, 2, 2)
    context = torch.randn(1, 3, 12)
    timestep = torch.tensor([500.0])
    transformer_options = {
        "sample_sigmas": torch.tensor([0.5, 0.25, 0.0]),
    }

    expected = full([video, audio], timestep, context, transformer_options=transformer_options)
    actual = executor.execute(
        [video, audio],
        timestep,
        context,
        transformer_options=transformer_options,
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_flux2_two_stage_forward_matches_unpartitioned(monkeypatch):
    monkeypatch.setattr("comfy.ldm.flux.layers.attention", _test_flux_attention)
    kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "vec_in_dim": 4,
        "context_in_dim": 8,
        "hidden_size": 12,
        "mlp_ratio": 2.0,
        "num_heads": 2,
        "depth": 2,
        "depth_single_blocks": 2,
        "axes_dim": [2, 2, 2],
        "theta": 10000,
        "patch_size": 1,
        "qkv_bias": True,
        "guidance_embed": False,
        "txt_ids_dims": [],
        "global_modulation": True,
        "txt_norm": True,
        "dtype": torch.float32,
        "device": "cpu",
        "operations": comfy_ops.disable_weight_init,
    }
    torch.manual_seed(13)
    full = Flux(**kwargs)
    for parameter in full.parameters():
        torch.nn.init.uniform_(parameter, -0.05, 0.05)

    first = Flux(**kwargs, pipeline_stage=PipelineStageConfig(0, 2, 0, 3))
    second = Flux(**kwargs, pipeline_stage=PipelineStageConfig(1, 2, 3, 4))
    first.load_state_dict(full.state_dict(), strict=False)
    second.load_state_dict(full.state_dict(), strict=False)
    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 3, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 3, 4, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("flux2", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    image = torch.randn(1, 4, 2, 2)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 3, 8)
    pooled = torch.randn(1, 4)

    expected = full(image, timestep, context, pooled)
    actual = executor.execute(image, timestep, context, pooled)
    torch.testing.assert_close(actual, expected)


def _test_flux_attention(query, key, value, pe=None, mask=None, **kwargs):
    del pe, kwargs
    if isinstance(query, AttentionTensorContainer):
        query, key, value = query.take(), key.take(), value.take()
    return _test_attention(query, key, value, query.shape[1], mask=mask)


def _test_minimax_attention(self, x, rope_freqs=None, transformer_options=None):
    del rope_freqs, transformer_options
    sequence = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    q = q.view(1, sequence, self.heads, self.head_dim).transpose(1, 2)
    k = k.view(1, sequence, self.heads, self.head_dim).transpose(1, 2)
    v = v.view(1, sequence, self.heads, self.head_dim).transpose(1, 2)
    return self.out_proj(_test_attention(q, k, v, self.heads).squeeze(0))
