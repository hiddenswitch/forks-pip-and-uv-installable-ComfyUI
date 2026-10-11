from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
import threading
import sys
import weakref

import pytest
import torch
from safetensors.torch import save_file

from comfy import ops, model_management, model_prefetch
from comfy.interruption import InterruptProcessingException
from comfy.pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from comfy.tensor_parallel import TensorParallelConfig
from comfy.tensor_parallel.hunyuan_image3 import _load_state, _TextTransformer, _LayerPrefetch, _text_context
from comfy.tensor_parallel.operations import gathered_output_operations, tensor_parallel_operations
from comfy.tensor_parallel.kandinsky6 import _DistributedDiT, clear_magcache_after_sample, connect_piflow
from comfy.tensor_parallel.kandinsky6 import load_state as load_kandinsky_state, shard_model
from comfy.tensor_parallel import distributed
from comfy.ldm.kandinsky5.model import CrossAttention, FeedForward


class Collectives:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.values = {}

    def operation(self, index, rank, value, gather):
        with self.condition:
            values = self.values.setdefault(index, {})
            values[rank] = value
            self.condition.notify_all()
            assert self.condition.wait_for(lambda: len(values) == self.size, timeout=10)
            ordered = [values[i] for i in range(self.size)]
            return torch.cat(ordered, dim=-1) if gather else sum(ordered)


@pytest.mark.parametrize("cancel", [False, True])
def test_piflow_loads_peer_models_and_finishes_execution(monkeypatch, cancel):
    calls = []
    peer = object()
    patcher = SimpleNamespace(
        model=SimpleNamespace(memory_required=lambda shape: shape[2] * 1024),
        get_nested_additional_models=lambda: [peer],
    )
    executor = SimpleNamespace(root_patcher=patcher, finish_execution=lambda: calls.append("finish"))
    model = SimpleNamespace(n_grid=4)
    video = torch.zeros(1, 16, 5, 2, 2)

    def sample(dit, value):
        assert calls == [([patcher, peer], 5120)]
        assert isinstance(dit, _DistributedDiT)
        if cancel:
            raise InterruptProcessingException()
        return value

    sampling = ModuleType("test_k6.sampling")
    sampling.rollout = sample
    monkeypatch.setitem(sys.modules, "test_k6.kandinsky6.sampling", sampling)
    monkeypatch.setattr(model_management, "load_models_gpu", lambda models, memory_required: calls.append((models, memory_required)))
    connect_piflow(model, executor, "test_k6")
    if cancel:
        with pytest.raises(InterruptProcessingException):
            sampling.rollout(model, video)
    else:
        assert sampling.rollout(model, video) is video
    assert calls[-1] == "finish"


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_non_cuda_quantized_load_does_not_query_cuda_properties(monkeypatch, device):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)

    def unexpected(_):
        raise AssertionError("CPU/offload capability probe queried CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    assert not model_management.supports_nvfp4_compute(torch.device(device))
    assert not model_management.supports_mxfp8_compute(torch.device(device))


@pytest.mark.parametrize("device", [None, torch.device("cuda")])
def test_quantized_capability_probe_without_visible_cuda(monkeypatch, device):
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected(_):
        raise AssertionError("Capability probe initialized an unavailable CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    assert not model_management.supports_nvfp4_compute(device)
    assert not model_management.supports_mxfp8_compute(device)


class Rank:
    def __init__(self, collective, rank):
        self.collective, self.rank = collective, rank
        self.world_size = collective.size
        self.index = 0

    def _call(self, value, gather):
        index = self.index
        self.index += 1
        return self.collective.operation(index, self.rank, value, gather)

    def sum(self, value):
        return self._call(value, False)

    def gather(self, value):
        return self._call(value, True)


def test_magcache_releases_residuals_on_cancel():
    state = SimpleNamespace(_lanes={0: torch.ones(4)}, _finished=False)

    class CancelledSample:
        class_obj = SimpleNamespace(model_options={"transformer_options": {"k6_magcache": state}})

        def __call__(self):
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        clear_magcache_after_sample(CancelledSample())
    assert not state._lanes
    assert state._finished


def test_worker_releases_inputs_before_waiting_for_next_command(monkeypatch):
    references = []

    def forward(value):
        references.append(weakref.ref(value))

    class Coordinator:
        def broadcast_command(self):
            if references:
                assert references[0]() is None
                return {"kind": "close"}
            return {"kind": "execute", "method": "forward", "descriptors": {},
                    "structure": ("tuple", (("tuple", (("tensor", "x"),)), ("dict", ())))}

        def send_object(self, value, destination):
            assert value["kind"] == "done"

    monkeypatch.setattr(distributed, "_broadcast_tensors", lambda *args, **kwargs: {"x": torch.ones(4)})
    monkeypatch.setattr(distributed, "distributed_command_span", lambda *args: nullcontext())
    distributed._run_worker(SimpleNamespace(rank=1, world_size=2), Coordinator(),
                            SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(forward=forward))), "test")


@pytest.mark.parametrize("cancelled", [False, True])
def test_hunyuan_prefetch_ignores_unallocated_buffers_and_releases_on_exit(monkeypatch, cancelled):
    layer = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    layer[0]._v = None
    layer[1]._v = (object(), 0, 64)
    pinned, released = [], []
    monkeypatch.delenv("HUNYUAN_IMAGE_3_NO_LOOKAHEAD", raising=False)
    monkeypatch.setattr(model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(model_prefetch, "PREFETCH_QUEUES", [])
    monkeypatch.setattr(model_prefetch, "pin_modules", lambda modules, *args: (pinned.extend(modules), True))
    monkeypatch.setattr(model_prefetch, "cleanup_prefetched_modules", lambda module, modules: released.extend(modules))
    stream_lookup = model_management.get_offload_stream
    prefetch = _LayerPrefetch([layer], torch.device("cuda"), True)
    prefetch.before(0)
    assert pinned == [layer[1]]
    if not cancelled:
        prefetch.after(0)
    prefetch.abort()
    assert released == [layer[1]]
    assert model_management.get_offload_stream is stream_lookup
    assert prefetch.queue == []


@pytest.mark.parametrize("size", [2, 4])
def test_row_parallel_bias_is_added_once(size):
    torch.manual_seed(9)
    reference = torch.nn.Linear(16, 8)
    inputs = torch.randn(3, 16)
    collective = Collectives(size)
    modules = []
    for rank in range(size):
        operations = tensor_parallel_operations(ops.manual_cast, TensorParallelConfig(Rank(collective, rank)))
        module = operations.RowParallelLinear(16, 8)
        state = {"weight": reference.weight[:, rank * (16 // size):(rank + 1) * (16 // size)].clone()}
        if rank == 0:
            state["bias"] = reference.bias.detach().clone()
        module.load_state_dict(state)
        modules.append(module)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda rank: modules[rank](inputs.chunk(size, dim=-1)[rank]), range(size)))
    for output in outputs:
        torch.testing.assert_close(output, reference(inputs))


@pytest.mark.parametrize("size", [2, 4])
def test_gathered_projection_rows_preserve_full_input_and_output(size):
    torch.manual_seed(7)
    base = ops.mixed_precision_ops(compute_dtype=torch.float32)
    collective = Collectives(size)
    weights = torch.randn(16, 32)
    inputs = torch.randn(5, 32)
    modules = []
    for rank in range(size):
        operations = gathered_output_operations(base, TensorParallelConfig(Rank(collective, rank)))
        module = operations.Linear(32, 16, bias=False)
        module.load_state_dict({"weight": weights[rank * (16 // size):(rank + 1) * (16 // size)].clone()})
        modules.append(module)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda rank: modules[rank](inputs), range(size)))
    for output in outputs:
        torch.testing.assert_close(output, inputs @ weights.t())


@pytest.mark.parametrize("expert_codebook", [False, True])
def test_w4a8_expert_slices_keep_full_k_and_matching_codebook(tmp_path, expert_codebook):
    prefix = "model.layers.0.mlp.experts_gate_up_proj"
    state = {
        prefix + ".weight": torch.arange(2 * 16 * 32, dtype=torch.int64).reshape(2, 16, 32).to(torch.uint8),
        prefix + ".weight_s_rel": torch.randn(2, 16, 4),
        prefix + ".weight_s_channel": torch.randn(2, 16, 1),
        prefix + ".weight_codebook": torch.randn(2, 16) if expert_codebook else torch.randn(16),
        prefix + ".comfy_quant": torch.tensor([1, 2, 3], dtype=torch.uint8),
    }
    path = tmp_path / "model.safetensors"
    save_file(state, path)
    reader = SafetensorsCheckpointReader(path)
    parallel = TensorParallelConfig(Rank(Collectives(2), 1))
    actual = _load_state(reader, {prefix: 0}, parallel, ("lm_head.",))
    for suffix in ("weight", "weight_s_rel", "weight_s_channel"):
        torch.testing.assert_close(actual[prefix + "." + suffix], state[prefix + "." + suffix][1:])
        assert actual[prefix + "." + suffix]._base is None
    expected_codebook = state[prefix + ".weight_codebook"][1:] if expert_codebook else state[prefix + ".weight_codebook"]
    torch.testing.assert_close(actual[prefix + ".weight_codebook"], expected_codebook)
    torch.testing.assert_close(actual[prefix + ".comfy_quant"], state[prefix + ".comfy_quant"])


def test_direct_custom_node_calls_route_through_executor():
    calls = []

    def execute(method, *args, **kwargs):
        calls.append(method)
        if method == "tensor_parallel_text_forward":
            assert len(args) == 3  # No KV cache is transported.
            return args[0] + 1
        return args[0] + 2

    executor = SimpleNamespace(execute_method=execute)
    dit = _DistributedDiT(SimpleNamespace(n_grid=4), executor)
    assert dit.n_grid == 4
    assert dit(torch.zeros(1)).item() == 2
    transformer = _TextTransformer(SimpleNamespace(wte=None), executor)
    cache = [[None, None]]
    assert transformer(torch.zeros(1), None, None, cache).item() == 1
    assert cache == [[None, None]]
    assert calls == ["forward", "tensor_parallel_text_forward"]


@pytest.mark.parametrize("cancel", [False, True])
def test_text_context_keeps_prefix_locally_and_releases_it(cancel):
    references = []

    class Transformer:
        layers = [object()]

        def __call__(self, embeddings, frequencies, mask, cache):
            previous = cache[0][0]
            cache[0][0] = embeddings.clone() if previous is None else torch.cat((previous, embeddings))
            references.append(weakref.ref(cache[0][0]))
            return cache[0][0].sum()

    model = SimpleNamespace(model=Transformer())
    try:
        with _text_context(model) as context:
            assert context.tensor_parallel_text_forward(torch.tensor([2.0]), None, None).item() == 2
            assert context.tensor_parallel_text_forward(torch.tensor([3.0]), None, None).item() == 5
            if cancel:
                raise InterruptProcessingException()
    except InterruptProcessingException:
        assert cancel
    assert all(reference() is None for reference in references)
    with _text_context(model) as context:
        assert context.tensor_parallel_text_forward(torch.tensor([7.0]), None, None).item() == 7


@pytest.mark.parametrize("entry_fails", [False, True])
def test_worker_model_context_closes_and_next_execution_succeeds(monkeypatch, entry_fails):
    calls, responses = [], []

    @contextmanager
    def scope():
        if entry_fails:
            raise ValueError("entry failed")
        try:
            yield SimpleNamespace(forward=lambda: calls.append("scoped"))
        finally:
            calls.append("cleanup")

    execute = {"kind": "execute", "method": "forward", "descriptors": {},
               "structure": ("tuple", (("tuple", ()), ("dict", ())))}
    commands = [{"kind": "model_context", "method": "scope"}]
    if not entry_fails:
        commands.append(execute)
    commands += [{"kind": "end_model_context"}, execute, {"kind": "close"}]
    commands = iter(commands)
    coordinator = SimpleNamespace(broadcast_command=lambda: next(commands),
                                  send_object=lambda value, destination: responses.append(value["kind"]))
    model = SimpleNamespace(scope=scope, forward=lambda: calls.append("ordinary"))
    monkeypatch.setattr(distributed, "_broadcast_tensors", lambda *args, **kwargs: {})
    monkeypatch.setattr(distributed, "distributed_command_span", lambda *args: nullcontext())
    distributed._run_worker(SimpleNamespace(rank=1, world_size=2), coordinator,
                            SimpleNamespace(model=SimpleNamespace(diffusion_model=model)), "test")
    assert calls == (["ordinary"] if entry_fails else ["scoped", "cleanup", "ordinary"])
    assert responses == (["error", "done", "done"] if entry_fails else ["done"] * 4)


@pytest.mark.parametrize("failure", [None, "entry", "body"])
def test_root_model_context_drains_ranks_and_restores_model(failure):
    commands, received, cleanup = [], [], []

    @contextmanager
    def scope():
        try:
            yield object()
        finally:
            cleanup.append(True)

    def receive(rank):
        received.append(rank)
        if failure == "entry" and len(received) == 1:
            return {"kind": "error", "traceback": "entry failed"}
        return {"kind": "done"}

    original = SimpleNamespace(scope=scope)
    executor = distributed.TorchDistributedModelParallelExecutor(
        original, None, SimpleNamespace(broadcast_command=commands.append, receive_object=receive),
        [], [], ["cpu"] * 3, [],
    )
    with pytest.raises((RuntimeError, InterruptProcessingException)) if failure else nullcontext():
        with executor.model_context("scope"):
            assert executor.root_model is not original
            if failure == "body":
                raise InterruptProcessingException()
    assert executor.root_model is original
    assert cleanup == [True]
    assert received == [1, 2, 1, 2]
    assert [command["kind"] for command in commands] == ["model_context", "end_model_context"]


@pytest.mark.parametrize("size", [2, 4])
def test_kandinsky_attention_and_ffn_shards_match_full_forward(size, tmp_path, monkeypatch):
    # K6 reuses these K5 modules; its asymmetric attention has the same projection contract.
    extension = ModuleType("test_k6.kandinsky6.ldm.model")
    extension.AsymCrossAttention = CrossAttention
    registration = ModuleType("test_k6.kandinsky6.register")
    registration._native_key = lambda key: key
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    monkeypatch.setitem(sys.modules, registration.__name__, registration)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            settings = {"operations": ops.manual_cast, "device": "cpu", "dtype": torch.float32}
            self.attention = CrossAttention(32, 8, settings)
            self.feed_forward = FeedForward(32, 64, settings)

        def forward(self, x):
            return self.feed_forward(self.attention(x, x))

    torch.manual_seed(31)
    reference = Block()
    for parameter in reference.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.1)
    path = tmp_path / "k6.safetensors"
    save_file(reference.state_dict(), path)
    reader = SafetensorsCheckpointReader(path)
    collective = Collectives(size)
    modules = []
    for rank in range(size):
        parallel = TensorParallelConfig(Rank(collective, rank))
        module = Block()
        shards = shard_model(module, tensor_parallel_operations(ops.manual_cast, parallel), "test_k6")
        module.load_state_dict(load_kandinsky_state(reader, "", shards, parallel, "test_k6"))
        modules.append(module)
    inputs = torch.randn(1, 3, 32)
    expected = reference(inputs)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda module: module(inputs), modules))
    for output in outputs:
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
