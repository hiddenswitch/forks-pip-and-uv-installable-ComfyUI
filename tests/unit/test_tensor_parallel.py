import threading
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import comfy.ops as comfy_ops
from comfy import interruption, memory_management, model_management
from comfy.ldm.minimax.model import Attention, MLP
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.model_base import BaseModel, MiniMaxH3
from comfy.tensor_parallel import (
    AbstractBaseTensorParallelOperations,
    full_domain_tensor_parallel_operations,
    TensorParallelConfig,
    TorchDistributedTensorParallelOperations,
    shard_minimax_h3_state_dict,
    shard_tensor_parallel_state_dict,
    tensor_parallel_operations,
)
from comfy.tensor_parallel.distributed import (
    RemoteModelParallelRankModel,
    TorchDistributedTensorParallelExecutor,
    _ObjectCoordinator,
    _broadcast_tensors,
    _without_execution_caches,
)
from comfy.tensor_parallel.operations import gathered_output_operations


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return args[0] * 2


class RecordingCoordinator:
    def broadcast_command(self, command):
        self.command = command

    def receive_object(self, rank):
        return {"kind": "done"}


class RecordingRootModel:
    def forward(self, value):
        self.value = value
        return value


class FailingRootModel:
    def forward(self):
        raise RuntimeError("rank zero failed")


class PreprocessingDiffusion(torch.nn.Module):
    dtype = torch.float32

    def preprocess_text_embeds(self, value):
        raise AssertionError("TP preprocessing must run through the rank executor")


class CompletedBroadcast:
    def __init__(self):
        self.waited = False
        self.stream_blocked = False

    def wait(self):
        self.waited = True

    def block_current_stream(self):
        self.stream_blocked = True


class ThreadCollective:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.values = {}

    def sum(self, index, tensor):
        with self.condition:
            values = self.values.setdefault(index, [])
            values.append(tensor)
            if len(values) == self.size:
                total = sum(value.clone() for value in values)
                for value in values:
                    value.copy_(total)
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(values) == self.size)
        return tensor

    def gather(self, index, rank, tensor):
        with self.condition:
            values = self.values.setdefault(("gather", index), {})
            values[rank] = tensor
            if len(values) == self.size:
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(values) == self.size)
            return torch.cat([values[rank].clone() for rank in range(self.size)], dim=-1)


class ThreadTensorParallelOperations(AbstractBaseTensorParallelOperations):
    def __init__(self, collective, rank):
        self.collective = collective
        self._rank = rank
        self.index = 0

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self.collective.size

    def sum(self, tensor):
        index = self.index
        self.index += 1
        return self.collective.sum(index, tensor)

    def gather(self, tensor):
        index = self.index
        self.index += 1
        return self.collective.gather(index, self._rank, tensor)


def _attention(query, key, value, heads, mask=None, **kwargs):
    del heads, kwargs
    if isinstance(query, AttentionTensorContainer):
        query, key, value = query.take(), key.take(), value.take()
    scores = torch.matmul(query, key.transpose(-2, -1)) * query.shape[-1] ** -0.5
    if mask is not None:
        scores = scores + mask
    return torch.matmul(scores.softmax(dim=-1), value).transpose(1, 2).flatten(2)


def _run_parallel(modules, value):
    outputs = [None] * len(modules)
    threads = [
        threading.Thread(target=lambda index=index: outputs.__setitem__(index, modules[index](value)))
        for index in range(len(modules))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outputs


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("input_act", ["swiglu", "gelu_tanh"])
@pytest.mark.parametrize("layout", ["row", "full_domain", "gathered_output"])
def test_fused_input_activation_preserves_parallel_forward(monkeypatch, size, input_act, layout):
    monkeypatch.setattr(comfy_ops, "_fp16_linear_wanted", lambda value: True)
    monkeypatch.setattr(model_management, "in_training", False)

    def fused_linear(value, weight, bias, activation, act_weight, act_eps, residual, residual_scale):
        activated = comfy_ops.INPUT_ACT_EAGER[activation](value)
        output = torch.nn.functional.linear(activated, weight, bias)
        return torch.addcmul(residual, output, residual_scale)

    monkeypatch.setattr(comfy_ops, "linear_input_act_", fused_linear)
    torch.manual_seed(42)
    value = torch.randn(3, 16 if input_act == "swiglu" else 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    residual = torch.randn(3, 8)
    scale = torch.randn(8)
    expected = fused_linear(value, weight, bias, input_act, None, 0.0, residual, scale)
    collective = ThreadCollective(size)
    calls = []
    for rank in range(size):
        config = TensorParallelConfig(ThreadTensorParallelOperations(collective, rank))
        if layout == "gathered_output":
            operations = gathered_output_operations(comfy_ops.disable_weight_init, config)
            linear = operations.Linear(8, 8)
            local_input = value
        else:
            decorator = tensor_parallel_operations if layout == "row" else full_domain_tensor_parallel_operations
            operations = decorator(comfy_ops.disable_weight_init, config)
            linear = operations.RowParallelLinear(8, 8)
            sections = value.chunk(2, dim=-1) if input_act == "swiglu" else (value,)
            local_input = torch.cat([section.chunk(size, dim=-1)[rank] for section in sections], dim=-1)
        local_weight = weight.chunk(size, dim=1 if layout == "row" else 0)[rank]
        linear.weight.detach().copy_(local_weight)
        if linear.bias is not None:
            linear.bias.detach().copy_(bias if layout == "row" else bias.chunk(size)[rank])
        calls.append(lambda _, linear=linear, local_input=local_input: comfy_ops.linear_input_act(
            linear, local_input, input_act, residual=residual, residual_scale=scale,
        ))
    for output in _run_parallel(calls, None):
        torch.testing.assert_close(output, expected)


def test_minimax_checkpoint_shards_megatron_projection_dimensions():
    state = {
        "blocks.0.attn.qkv_proj.weight": torch.arange(24).reshape(12, 2),
        "blocks.0.attn.qkv_proj.weight_scale": torch.arange(12).reshape(12, 1),
        "blocks.0.attn.out_proj.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.mlp.fc1.weight": torch.arange(32).reshape(16, 2),
        "blocks.0.mlp.fc1.weight_scale": torch.arange(16).reshape(16, 1),
        "blocks.0.mlp.fc2.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.norm.weight": torch.ones(4),
    }

    first = shard_minimax_h3_state_dict(state, 0, 2)
    second = shard_minimax_h3_state_dict(state, 1, 2)

    assert first["blocks.0.attn.qkv_proj.weight"][:, 0].tolist() == [0, 2, 8, 10, 16, 18]
    assert second["blocks.0.attn.qkv_proj.weight"][:, 0].tolist() == [4, 6, 12, 14, 20, 22]
    assert first["blocks.0.mlp.fc1.weight"][:, 0].tolist() == [0, 2, 4, 6, 16, 18, 20, 22]
    assert second["blocks.0.mlp.fc1.weight"][:, 0].tolist() == [8, 10, 12, 14, 24, 26, 28, 30]
    assert torch.equal(first["blocks.0.attn.out_proj.weight"], state["blocks.0.attn.out_proj.weight"][:, :4])
    assert torch.equal(second["blocks.0.mlp.fc2.weight"], state["blocks.0.mlp.fc2.weight"][:, 4:])
    assert first["blocks.0.norm.weight"] is state["blocks.0.norm.weight"]


def test_minimax_quantized_checkpoint_shards_row_outputs_for_full_domain_matmul():
    state = {
        "blocks.0.attn.out_proj.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.attn.out_proj.weight_scale": torch.arange(4).reshape(4, 1),
        "blocks.0.mlp.fc2.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.mlp.fc2.weight_scale": torch.arange(4).reshape(4, 1),
    }

    first = shard_minimax_h3_state_dict(state, 0, 2, full_domain_rows=True)
    second = shard_minimax_h3_state_dict(state, 1, 2, full_domain_rows=True)

    assert torch.equal(first["blocks.0.attn.out_proj.weight"], state["blocks.0.attn.out_proj.weight"][:2])
    assert torch.equal(second["blocks.0.attn.out_proj.weight"], state["blocks.0.attn.out_proj.weight"][2:])
    assert torch.equal(first["blocks.0.mlp.fc2.weight_scale"], state["blocks.0.mlp.fc2.weight_scale"][:2])
    assert torch.equal(second["blocks.0.mlp.fc2.weight_scale"], state["blocks.0.mlp.fc2.weight_scale"][2:])


@pytest.mark.parametrize(
    ("family", "state", "column_key", "expected_first_rows", "row_key"),
    [
        (
            "krea2",
            {
                "blocks.0.attn.wq.weight": torch.arange(32).reshape(8, 4),
                "blocks.0.attn.wq.weight_scale": torch.arange(8).reshape(8, 1),
                "blocks.0.attn.wo.weight": torch.arange(32).reshape(4, 8),
            },
            "blocks.0.attn.wq.weight",
            [0, 4, 8, 12],
            "blocks.0.attn.wo.weight",
        ),
        (
            "ideogram4",
            {
                "layers.0.attention.qkv.weight": torch.arange(96).reshape(24, 4),
                "layers.0.attention.qkv.weight_scale": torch.arange(24).reshape(24, 1),
                "layers.0.attention.o.weight": torch.arange(32).reshape(4, 8),
            },
            "layers.0.attention.qkv.weight",
            [0, 4, 8, 12, 32, 36, 40, 44, 64, 68, 72, 76],
            "layers.0.attention.o.weight",
        ),
        (
            "flux2",
            {
                "double_blocks.0.img_attn.qkv.weight": torch.arange(96).reshape(24, 4),
                "double_blocks.0.img_attn.proj.weight": torch.arange(32).reshape(4, 8),
            },
            "double_blocks.0.img_attn.qkv.weight",
            [0, 4, 8, 12, 32, 36, 40, 44, 64, 68, 72, 76],
            "double_blocks.0.img_attn.proj.weight",
        ),
    ],
)
def test_checkpoint_shards_supported_projection_layouts(
    family, state, column_key, expected_first_rows, row_key
):
    first = shard_tensor_parallel_state_dict(state, family, 0, 2)
    second = shard_tensor_parallel_state_dict(state, family, 1, 2)

    assert first[column_key][:, 0].tolist() == expected_first_rows
    assert first[column_key].shape == second[column_key].shape
    assert torch.equal(first[row_key], state[row_key][:, :4])
    assert torch.equal(second[row_key], state[row_key][:, 4:])


def test_flux2_single_stream_checkpoint_preserves_fused_projection_sections():
    weight = torch.arange(40 * 8).reshape(40, 8)
    state = {
        "single_blocks.0.linear1.weight": weight,
        "single_blocks.0.linear2.weight": torch.arange(128).reshape(8, 16),
    }

    first = shard_tensor_parallel_state_dict(state, "flux2", 0, 2)
    second = shard_tensor_parallel_state_dict(state, "flux2", 1, 2)

    expected_first = torch.cat((weight[0:4], weight[8:12], weight[16:20], weight[24:28], weight[32:36]))
    expected_second = torch.cat((weight[4:8], weight[12:16], weight[20:24], weight[28:32], weight[36:40]))
    assert torch.equal(first["single_blocks.0.linear1.weight"], expected_first)
    assert torch.equal(second["single_blocks.0.linear1.weight"], expected_second)


def test_minimax_conditioning_uses_tensor_parallel_executor():
    model = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.diffusion_model = PreprocessingDiffusion()
    model.pipeline_executor = RecordingExecutor()
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.latent_shapes = None

    cross_attn = torch.randn(1, 3, 4)
    conditions = model.extra_conds(cross_attn=cross_attn, device=torch.device("cpu"))

    assert model.pipeline_executor.calls[0][0] == "preprocess_text_embeds"
    torch.testing.assert_close(conditions["c_crossattn"].cond, cross_attn * 2)


def test_tensor_parallel_cpu_inputs_use_control_group(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, src, group, async_op, completion))
        return completion

    monkeypatch.setattr("comfy.tensor_parallel.distributed.dist.broadcast", broadcast)
    operations = type("Operations", (), {
        "control_process_group": "gloo",
        "process_group": "nccl",
    })()

    result = _broadcast_tensors(operations, tensors={"seed": torch.tensor(42)})

    assert result["seed"].device.type == "cpu"
    assert calls[0][2] == "gloo"
    assert calls[0][4].waited


def test_tensor_parallel_gpu_inputs_move_to_rank_device_before_collective(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, src, group, async_op, completion))
        return completion

    monkeypatch.setattr("comfy.tensor_parallel.distributed.dist.broadcast", broadcast)
    monkeypatch.setattr(
        "torch._subclasses.fake_tensor.init_gpu_context",
        lambda _: None,
    )
    operations = type("Operations", (), {
        "device": torch.device("cuda", 1),
        "control_process_group": "gloo",
        "process_group": "nccl",
    })()

    with FakeTensorMode():
        result = _broadcast_tensors(
            operations,
            tensors={"latent": torch.empty(2, device="cuda:0")},
        )

    assert result["latent"].device == torch.device("cuda", 1)
    assert calls[0][0].device == torch.device("cuda", 1)
    assert calls[0][2] == "nccl"
    assert calls[0][4].stream_blocked


def test_tensor_parallel_sum_reduces_owned_output_in_place(monkeypatch):
    calls = []

    def all_reduce(tensor, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, group, async_op, completion))
        return completion

    monkeypatch.setattr("torch.distributed.all_reduce", all_reduce)
    operations = TorchDistributedTensorParallelOperations(
        0, 2, torch.device("cpu"), "nccl"
    )
    tensor = torch.randn(3, 4)

    result = operations.sum(tensor)

    assert result is tensor
    assert calls[0][:3] == (tensor, "nccl", True)
    assert calls[0][3].stream_blocked


def test_tensor_parallel_root_uses_collective_prepared_inputs(monkeypatch):
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed._broadcast_tensors",
        lambda operations, tensors: {name: tensor + 1 for name, tensor in tensors.items()},
    )
    root = RecordingRootModel()
    executor = TorchDistributedTensorParallelExecutor(
        root,
        object(),
        RecordingCoordinator(),
        (),
        (),
        (torch.device("cpu"), torch.device("cpu")),
        (0, 0),
    )

    output = executor.execute(torch.tensor(2))

    assert output.item() == 3
    assert root.value.item() == 3


def test_model_parallel_root_failure_aborts_device_group(monkeypatch):
    aborted = []
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.dist.distributed_c10d._abort_process_group",
        aborted.append,
    )
    operations = type("Operations", (), {"process_group": "nccl"})()
    executor = TorchDistributedTensorParallelExecutor(
        FailingRootModel(), operations, RecordingCoordinator(), (), (),
        (torch.device("cpu"),), (0,),
    )

    with pytest.raises(RuntimeError, match="rank zero failed"):
        executor.execute()

    assert aborted == ["nccl"]
    assert executor._device_group_active is False


def test_model_parallel_cancel_finishes_collectives_and_allows_retry(monkeypatch):
    events = []
    monkeypatch.setattr("comfy.tensor_parallel.distributed._broadcast_tensors", lambda operations, tensors: tensors)
    monkeypatch.setattr("comfy.tensor_parallel.distributed.dist.distributed_c10d._abort_process_group", lambda group: events.append("abort"))

    class Root:
        def forward(self, cancel):
            if cancel:
                interruption.interrupt_current_processing()
            interruption.throw_exception_if_processing_interrupted()
            events.append("forward")
            return 7

    class Coordinator(RecordingCoordinator):
        def receive_object(self, rank):
            events.append("peer_done")
            return {"kind": "done"}

    executor = TorchDistributedTensorParallelExecutor(
        Root(), SimpleNamespace(process_group="nccl"), Coordinator(), (), (),
        (torch.device("cpu"), torch.device("cpu")), (0, 0),
    )
    try:
        with pytest.raises(interruption.InterruptProcessingException):
            executor.execute(True)
        assert sorted(events) == ["forward", "peer_done"]
        assert executor._device_group_active
        assert executor.execute(False) == 7
    finally:
        interruption.interrupt_current_processing(False)
        executor.close()


def test_model_parallel_transport_keeps_only_diffusion_model_wrappers():
    compile_wrapper = lambda executor, *args, **kwargs: executor(*args, **kwargs)
    extension_wrapper = object()
    value = {
        "transformer_options": {
            "wrappers": {
                "apply_model": {
                    "torch.compile": [compile_wrapper],
                    "extension": [extension_wrapper],
                },
                "diffusion_model": {"extension": [extension_wrapper]},
                "predict_noise": {None: [extension_wrapper]},
            },
            "layout": object(),
            "seed": 42,
        }
    }

    cleaned = _without_execution_caches(value)

    options = cleaned["transformer_options"]
    assert "layout" not in options
    assert options["wrappers"] == {"diffusion_model": {
        "extension": [extension_wrapper]
    }}
    assert options["seed"] == 42


def test_tensor_parallel_finish_execution_releases_all_ranks(monkeypatch):
    coordinator = RecordingCoordinator()
    finished = []
    monkeypatch.setattr(
        "comfy.model_prefetch.finish_model_execution",
        lambda: finished.append("root"),
    )
    executor = TorchDistributedTensorParallelExecutor(
        RecordingRootModel(), object(), coordinator, (), (),
        (torch.device("cpu"), torch.device("cpu")), (0, 0),
    )

    executor.finish_execution()

    assert coordinator.command["kind"] == "finish_execution"
    assert finished == ["root"]


def test_xdit_memory_estimate_uses_rank_local_sequence_share(monkeypatch):
    monkeypatch.setattr(
        "comfy.model_management.pytorch_attention_flash_attention",
        lambda: True,
    )
    monkeypatch.setattr("comfy.model_management.xformers_enabled", lambda: False)

    def estimate(parallel_size):
        diffusion_model = SimpleNamespace()
        if parallel_size is not None:
            diffusion_model.xdit_sequence_parallel = SimpleNamespace(
                size=parallel_size,
            )
        model = SimpleNamespace(
            diffusion_model=diffusion_model,
            memory_usage_factor_conds=(),
            memory_usage_shape_process={},
            memory_usage_factor=4.0,
            get_dtype_inference=lambda: torch.bfloat16,
        )
        return BaseModel.memory_required(model, [2, 16, 4, 8, 8])

    assert estimate(2) == estimate(None) / 2


def test_remote_rank_receives_rank_local_activation_reserve(monkeypatch):
    class FakeExecutor:
        def __init__(self):
            self.command = None

        def remote_rank_command(self, rank, command):
            assert rank == 1
            self.command = command
            return {"kind": "rank_model", "loaded_size": 100}

    executor = FakeExecutor()
    remote = RemoteModelParallelRankModel(
        executor,
        rank=1,
        device=torch.device("cuda:1"),
        size=100,
        dtype=torch.bfloat16,
        dynamic=True,
    )
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    monkeypatch.setattr(model_management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(
        model_management,
        "prepare_device_model_loads",
        lambda required, **_kwargs: [],
    )

    model_management._load_models_gpu(
        [remote],
        memory_required=400,
        minimum_memory_required=200,
    )

    assert executor.command["memory_required"] == 400
    assert executor.command["minimum_memory_required"] == 200


def test_pin_eviction_skips_remote_rank_stand_ins(monkeypatch):
    # A remote rank's stand-in is dynamic but holds no pins in this process. Pin pressure in the
    # driver must not raise on it: an exception there kills one rank mid-forward and the other
    # rank's all-reduce waits out the NCCL timeout.
    remote = RemoteModelParallelRankModel(
        object(), rank=1, device=torch.device("cuda:1"), size=100, dtype=torch.bfloat16, dynamic=True,
    )
    monkeypatch.setattr(model_management, "current_loaded_models", [SimpleNamespace(model=remote)])

    assert list(model_management.models_for_pin_eviction(None)) == []
    assert model_management.free_pins(1 << 20, evict_active=True, loaded=True) == 0


def test_dynamic_rank_load_flushes_stale_allocator_reservations(monkeypatch):
    emptied = []
    monkeypatch.setattr(model_management, "free_memory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(model_management, "get_free_memory", lambda _device: 10_000)
    # Restore absence as well as value: this attribute is a module property.
    monkeypatch.setitem(memory_management.__dict__, "aimdo_enabled", True)
    monkeypatch.setattr(
        model_management,
        "_soft_empty_cache",
        lambda force=False: emptied.append(force),
    )

    model_management.prepare_device_model_loads(
        {torch.device("cuda:0"): 100},
        extra_mem=20,
        minimum_memory_required=10,
        free_for_dynamic=True,
    )

    assert emptied == [True]


def test_tensor_parallel_commands_propagate_trace_context(monkeypatch):
    values_seen = []

    def inject(carrier, context):
        del context
        carrier["traceparent"] = "00-test"

    def broadcast_object_list(values, src):
        del src
        values_seen.extend(values)

    monkeypatch.setattr("comfy.distributed.tracing.propagate.inject", inject)
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.dist.broadcast_object_list",
        broadcast_object_list,
    )

    _ObjectCoordinator().broadcast_command({"kind": "execute"})

    assert values_seen == [{
        "kind": "execute",
        "trace_context": {"traceparent": "00-test"},
    }]


def test_model_parallel_commands_use_executor_owned_control_group(monkeypatch):
    calls = []

    def broadcast_object_list(values, src, group):
        calls.append((values, src, group))

    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.dist.broadcast_object_list",
        broadcast_object_list,
    )

    _ObjectCoordinator("executor-gloo").broadcast_command({"kind": "execute"})

    assert calls[0][1:] == (0, "executor-gloo")


def test_model_parallel_close_destroys_owned_groups_before_waiting_for_worker(
    monkeypatch,
):
    events = []

    class Coordinator:
        def broadcast_command(self, command):
            events.append(("broadcast", command["kind"]))

    class Future:
        def result(self, timeout):
            events.append(("wait", timeout))

    class Pool:
        def shutdown(self, wait):
            events.append(("shutdown", wait))

    groups = object()
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.destroy_independent_process_groups",
        lambda value: events.append(("destroy", value)),
    )
    executor = TorchDistributedTensorParallelExecutor(
        RecordingRootModel(),
        object(),
        Coordinator(),
        ((Pool(), Future()),),
        (),
        (torch.device("cpu"), torch.device("cpu")),
        (0, 0),
        process_groups=groups,
    )

    executor.close()

    assert events == [
        ("broadcast", "close"),
        ("destroy", groups),
        ("wait", 10),
        ("shutdown", True),
    ]


def test_minimax_tensor_parallel_attention_and_mlp_match_full_model(monkeypatch):
    monkeypatch.setattr("comfy.ldm.minimax.model.optimized_attention", _attention)
    torch.manual_seed(5)
    full_attention = Attention(8, 4, 2, 1e-5, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init)
    full_mlp = MLP(8, 12, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init)
    for module in (full_attention, full_mlp):
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)

    collective = ThreadCollective(2)
    attentions = []
    mlps = []
    for rank in range(2):
        config = TensorParallelConfig(ThreadTensorParallelOperations(collective, rank))
        operations = tensor_parallel_operations(comfy_ops.disable_weight_init, config)
        attention = Attention(8, 4, 2, 1e-5, dtype=torch.float32, device="cpu",
                              operations=operations)
        mlp = MLP(8, 12, dtype=torch.float32, device="cpu",
                  operations=operations)
        qkv = full_attention.qkv_proj.weight.reshape(3, 8, 8)
        attention.qkv_proj.weight.detach().copy_(qkv[:, rank * 4:(rank + 1) * 4].reshape(12, 8))
        attention.out_proj.weight.detach().copy_(full_attention.out_proj.weight[:, rank * 4:(rank + 1) * 4])
        attention.q_norm.weight.detach().copy_(full_attention.q_norm.weight)
        attention.k_norm.weight.detach().copy_(full_attention.k_norm.weight)
        fc1 = full_mlp.fc1.weight.reshape(2, 12, 8)
        mlp.fc1.weight.detach().copy_(fc1[:, rank * 6:(rank + 1) * 6].reshape(12, 8))
        mlp.fc2.weight.detach().copy_(full_mlp.fc2.weight[:, rank * 6:(rank + 1) * 6])
        attentions.append(attention)
        mlps.append(mlp)

    value = torch.randn(5, 8)
    expected_attention = full_attention(value)
    expected_mlp = full_mlp(value)
    attention_outputs = _run_parallel(attentions, value)
    mlp_outputs = _run_parallel(mlps, value)

    for output in attention_outputs:
        torch.testing.assert_close(output, expected_attention)
    for output in mlp_outputs:
        torch.testing.assert_close(output, expected_mlp)


def test_minimax_full_domain_tensor_parallel_matches_full_model(monkeypatch):
    monkeypatch.setattr("comfy.ldm.minimax.model.optimized_attention", _attention)
    torch.manual_seed(7)
    full_attention = Attention(
        8, 4, 2, 1e-5, dtype=torch.float32, device="cpu",
        operations=comfy_ops.disable_weight_init,
    )
    full_mlp = MLP(
        8, 12, dtype=torch.float32, device="cpu",
        operations=comfy_ops.disable_weight_init,
    )
    for module in (full_attention, full_mlp):
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)

    collective = ThreadCollective(2)
    attentions = []
    mlps = []
    for rank in range(2):
        config = TensorParallelConfig(ThreadTensorParallelOperations(collective, rank))
        operations = full_domain_tensor_parallel_operations(
            comfy_ops.disable_weight_init,
            config,
        )
        attention = Attention(
            8, 4, 2, 1e-5, dtype=torch.float32, device="cpu",
            operations=operations,
        )
        mlp = MLP(
            8, 12, dtype=torch.float32, device="cpu",
            operations=operations,
        )
        qkv = full_attention.qkv_proj.weight.reshape(3, 8, 8)
        attention.qkv_proj.weight.detach().copy_(
            qkv[:, rank * 4:(rank + 1) * 4].reshape(12, 8)
        )
        attention.out_proj.weight.detach().copy_(
            full_attention.out_proj.weight[rank * 4:(rank + 1) * 4]
        )
        attention.q_norm.weight.detach().copy_(full_attention.q_norm.weight)
        attention.k_norm.weight.detach().copy_(full_attention.k_norm.weight)
        fc1 = full_mlp.fc1.weight.reshape(2, 12, 8)
        mlp.fc1.weight.detach().copy_(
            fc1[:, rank * 6:(rank + 1) * 6].reshape(12, 8)
        )
        mlp.fc2.weight.detach().copy_(
            full_mlp.fc2.weight[rank * 4:(rank + 1) * 4]
        )
        attentions.append(attention)
        mlps.append(mlp)

    value = torch.randn(5, 8)
    expected_attention = full_attention(value)
    expected_mlp = full_mlp(value)
    attention_outputs = _run_parallel(attentions, value)
    mlp_outputs = _run_parallel(mlps, value)

    for output in attention_outputs:
        torch.testing.assert_close(output, expected_attention)
    for output in mlp_outputs:
        torch.testing.assert_close(output, expected_mlp)
