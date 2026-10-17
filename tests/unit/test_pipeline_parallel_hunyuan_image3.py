"""Pipeline parallelism for the HunyuanImage3 custom node's transformer."""
import dataclasses
import importlib
import json
from importlib.metadata import distribution
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from comfy.pipeline_parallel import (
    PipelineParallelConfig,
    SafetensorsCheckpointReader,
    SingleProcessPipelineExecutor,
    get_pipeline_stage_spec,
)
from comfy.pipeline_parallel.types import (
    PipelinePartitionPlan,
    PipelineStageConfig,
    PipelineStagePlan,
    TensorDescriptor,
)
from tests.unit.test_pipeline_parallel import FakeDeviceRuntime, FakePipelineTransport


def _hunyuan_package():
    from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    entry = next(
        entry_point
        for entry_point in distribution("comfyui-hunyuanimage3").entry_points
        if entry_point.group == "comfyui.custom_nodes"
    ).load()
    root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
    prepare_vanilla_environment()
    _vanilla_load_custom_nodes_1(str(root))
    return root


@pytest.fixture(scope="module")
def hunyuan():
    root = _hunyuan_package()

    def module(name):
        return importlib.import_module(f"{root.name}.{name}")

    return module


def _params(hunyuan, model_type, **overrides):
    loader = hunyuan("hunyuan_image_3.loader")
    config = json.loads(Path(loader.CONFIG_PATH).read_text())
    config.update(loader.MODEL_TYPES[model_type], model_type=model_type)
    params = hunyuan("hunyuan_image_3.model").params_from_config(config)
    return dataclasses.replace(params, **overrides)


def _tiny_params(hunyuan, model_type):
    return _params(
        hunyuan,
        model_type,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=8,
        moe_intermediate_size=16,
        num_experts=8,
        moe_topk=2,
        num_shared_expert=1,
        vocab_size=64,
        pad_token_id=0,
        image_token_id=63,
        patch_embed_hidden_dim=32,
        vit_aligner={"projector_type": "mlp_gelu", "input_dim": 8, "n_embed": 32, "depth": 2},
    )


@pytest.mark.parametrize("model_type", ["instruct", "instruct_distil"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hunyuan_image3_two_stage_forward_matches_unpartitioned(hunyuan, model_type, batch_size):
    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    initialize = hunyuan("tests.op_module_init").init_op_modules
    params = _tiny_params(hunyuan, model_type)
    dtype = torch.float32

    full = upstream.HunyuanImage3(params, dtype=dtype, device="cpu", operations=operations)
    initialize(full, dtype, seed=3)
    state = full.state_dict()
    first = upstream.HunyuanImage3(
        params, dtype=dtype, device="cpu", operations=operations,
        pipeline_stage=PipelineStageConfig(0, 2, 0, 1),
    )
    second = upstream.HunyuanImage3(
        params, dtype=dtype, device="cpu", operations=operations,
        pipeline_stage=PipelineStageConfig(1, 2, 1, 2),
    )
    first_keys, second_keys = set(first.state_dict()), set(second.state_dict())
    assert first_keys.isdisjoint(second_keys)
    assert first_keys | second_keys == set(state)
    first.load_state_dict({key: state[key] for key in first_keys}, strict=True)
    second.load_state_dict({key: state[key] for key in second_keys}, strict=True)

    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("hunyuan_image3", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 63, 63, 63, 63, 8])
    if batch_size == 2:
        ids = (ids, ids[3:])
    x = torch.randn(batch_size, params.vae_latent_channels, 1, 2, 2)
    timestep = torch.full((batch_size,), 900.0)
    guidance = torch.tensor([2500.0]) if params.cfg_distilled else None
    options = {"sample_sigmas": torch.tensor([1.0, 0.9, 0.0]), "sigmas": torch.tensor([0.9])}

    expected = full(x, timestep, ids=ids, guidance=guidance, transformer_options=options)
    actual = executor.execute(x, timestep, ids=ids, guidance=guidance, transformer_options=options)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_hunyuan_image3_stage_layers_run_through_comfy_prefetch_queue(hunyuan, monkeypatch):
    # The node's LayerLookahead faults every module that has a `_v` attribute, and DynamicVRAM
    # leaves `_v` None on modules that are resident: on the chain the first step failed in
    # vbar_fault with "cannot unpack non-iterable NoneType". ComfyUI's own prefetch queue skips them.
    from comfy import model_prefetch

    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    params = _tiny_params(hunyuan, "instruct")
    stage = upstream.HunyuanImage3(params, dtype=torch.float32, device="cpu", operations=operations,
                                   pipeline_stage=PipelineStageConfig(1, 2, 1, 2))
    hunyuan("tests.op_module_init").init_op_modules(stage, torch.float32)
    events = []

    def make_prefetch_queue(queue, device, transformer_options):
        events.append(("queue", tuple(queue), transformer_options["prefetch_dynamic_vbars"]))
        return ["queue"]

    def prefetch_queue_pop(queue, device, module, *args, **kwargs):
        events.append(("pop", module))

    def lookahead(*args, **kwargs):
        raise AssertionError("the node's LayerLookahead faults resident modules")

    monkeypatch.setattr(model_prefetch, "make_prefetch_queue", make_prefetch_queue)
    monkeypatch.setattr(model_prefetch, "prefetch_queue_pop", prefetch_queue_pop)
    monkeypatch.setattr(upstream, "LayerLookahead", lookahead)
    hidden = torch.randn(1, 4, params.hidden_size)
    freqs = upstream.build_rope_freqs(4, params.attention_head_dim, [], params.rope_theta)

    stage._run_pipeline_layers(hidden, freqs, torch.zeros(1, 1, 4, 4), True)

    layer = stage.model.layers[1]
    assert events == [("queue", (layer,), True), ("pop", layer), ("pop", None)]


def _full_size_descriptors(hunyuan):
    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    params = _params(hunyuan, "instruct")
    model = upstream.HunyuanImage3(params, dtype=torch.bfloat16, device="meta", operations=operations)
    return {
        key: TensorDescriptor(tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size())
        for key, tensor in model.state_dict().items()
    }


def test_hunyuan_image3_stage_spec_covers_every_key_exactly_once(hunyuan):
    descriptors = _full_size_descriptors(hunyuan)
    spec = get_pipeline_stage_spec("hunyuan_image3")
    devices = tuple(torch.device("cuda", index) for index in range(8))

    plan = spec.plan(descriptors, PipelineParallelConfig(devices))

    owned = [stage.owned_keys for stage in plan.stages]
    assert sum(len(keys) for keys in owned) == len(descriptors)
    assert frozenset().union(*owned) == frozenset(descriptors)
    assert [(stage.start_layer, stage.end_layer) for stage in plan.stages] == [
        (index * 4, index * 4 + 4) for index in range(8)
    ]
    entry = ("model.wte.", "patch_embed.", "time_embed.", "timestep_emb.", "vision_aligner.")
    assert {key for key in descriptors if key.startswith(entry)} <= plan.stages[0].owned_keys
    assert {key for key in descriptors if key.startswith(("final_layer.", "time_embed_2."))} <= plan.stages[-1].owned_keys


def test_hunyuan_image3_stage_modules_hold_exactly_their_planned_keys(hunyuan):
    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    params = _params(hunyuan, "instruct")
    descriptors = _full_size_descriptors(hunyuan)
    devices = tuple(torch.device("cuda", index) for index in range(8))
    plan = get_pipeline_stage_spec("hunyuan_image3").plan(descriptors, PipelineParallelConfig(devices))

    for stage in plan.stages:
        module = upstream.HunyuanImage3(
            params, dtype=torch.bfloat16, device="meta", operations=operations,
            pipeline_stage=PipelineStageConfig(stage.index, plan.size, stage.start_layer, stage.end_layer),
        )
        assert set(module.state_dict()) == stage.owned_keys


def test_hunyuan_image3_stages_prune_by_structure_when_ops_create_weights_on_load(hunyuan):
    # mixed_precision_ops (quantized checkpoints) and DynamicVRAM's ops register a Linear's weight
    # only when a state dict arrives, so at construction time_embed_2 (Linears only) has no keys to
    # decide its stage by; it survived on stage 0 of the bf16 launch and failed the load there
    from comfy import ops
    from comfy.pipeline_parallel.types import PipelineMissingLayer

    upstream = hunyuan("hunyuan_image_3.model")
    params = _tiny_params(hunyuan, "instruct")
    operations = ops.mixed_precision_ops(compute_dtype=torch.float32)

    first = upstream.HunyuanImage3(params, dtype=torch.float32, device="cpu", operations=operations,
                                   pipeline_stage=PipelineStageConfig(0, 2, 0, 1))
    last = upstream.HunyuanImage3(params, dtype=torch.float32, device="cpu", operations=operations,
                                  pipeline_stage=PipelineStageConfig(1, 2, 1, 2))

    for name in ("final_layer", "time_embed_2"):
        assert isinstance(getattr(first, name), PipelineMissingLayer)
        assert not isinstance(getattr(last, name), PipelineMissingLayer)
    for name in ("patch_embed", "time_embed", "timestep_emb", "vision_aligner"):
        assert not isinstance(getattr(first, name), PipelineMissingLayer)
        assert isinstance(getattr(last, name), PipelineMissingLayer)
    assert isinstance(last.model.wte, PipelineMissingLayer)
    assert isinstance(first.model.layers[1], PipelineMissingLayer)
    assert isinstance(last.model.layers[0], PipelineMissingLayer)


def _write_shards(directory, shards):
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(directory / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in tensors})
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return index


def test_sharded_safetensors_reader_returns_only_requested_keys(tmp_path, monkeypatch):
    index = _write_shards(tmp_path, {
        "model-00001-of-00002.safetensors": {
            "a.weight": torch.arange(4, dtype=torch.float32),
            "b.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        },
        "model-00002-of-00002.safetensors": {"c.weight": torch.full((5,), 7.0)},
    })
    from comfy.pipeline_parallel import checkpoint

    real_load = checkpoint.utils.load_torch_file
    reads = []

    def load_torch_file(path, include_keys=None):
        reads.append((Path(path).name, frozenset(include_keys)))
        return real_load(path, include_keys=include_keys)

    monkeypatch.setattr("comfy.pipeline_parallel.checkpoint.utils.load_torch_file", load_torch_file)
    reader = SafetensorsCheckpointReader(index)

    assert set(reader.tensors) == {"a.weight", "b.weight", "c.weight"}
    assert reader.tensors["b.weight"] == TensorDescriptor((2, 3), torch.bfloat16, 12)
    assert reader.metadata == {"format": "pt"}

    loaded = reader.load_keys({"c.weight"})
    assert set(loaded) == {"c.weight"}
    torch.testing.assert_close(loaded["c.weight"], torch.full((5,), 7.0))
    assert reads == [("model-00002-of-00002.safetensors", frozenset({"c.weight"}))]

    reads.clear()
    loaded = reader.load_keys({"a.weight", "c.weight"})
    assert set(loaded) == {"a.weight", "c.weight"}
    torch.testing.assert_close(loaded["a.weight"], torch.arange(4, dtype=torch.float32))
    assert sorted(reads) == [
        ("model-00001-of-00002.safetensors", frozenset({"a.weight"})),
        ("model-00002-of-00002.safetensors", frozenset({"c.weight"})),
    ]
    with pytest.raises(KeyError):
        reader.load_keys({"missing.weight"})


def test_sharded_safetensors_reader_rejects_an_index_that_disagrees_with_its_shards(tmp_path):
    index = _write_shards(tmp_path, {"model-00001-of-00001.safetensors": {"a.weight": torch.ones(1)}})
    payload = json.loads(index.read_text())
    payload["weight_map"]["b.weight"] = "model-00001-of-00001.safetensors"
    index.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="b.weight"):
        SafetensorsCheckpointReader(index)


def test_hunyuan_image3_stage_loads_only_its_keys_from_the_checkpoint(hunyuan, tmp_path):
    from comfy.pipeline_parallel.hunyuan_image3 import HunyuanImage3PipelineStageLoader

    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    initialize = hunyuan("tests.op_module_init").init_op_modules
    params = _tiny_params(hunyuan, "instruct")
    full = upstream.HunyuanImage3(params, dtype=torch.bfloat16, device="cpu", operations=operations)
    initialize(full, torch.bfloat16, seed=5)
    state = full.state_dict()
    first_keys = [key for key in state if not key.startswith(("model.layers.1.", "final_layer.", "time_embed_2."))]
    index = _write_shards(tmp_path, {
        "model-00001-of-00002.safetensors": {key: state[key] for key in first_keys},
        "model-00002-of-00002.safetensors": {key: state[key] for key in state if key not in first_keys},
    })
    stage_loader = HunyuanImage3PipelineStageLoader(
        index, Path(hunyuan("hunyuan_image_3.loader").__file__).parent.parent,
        params=params, dtype=torch.bfloat16,
    )
    plan = PipelinePartitionPlan("hunyuan_image3", (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset(first_keys)),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset(set(state) - set(first_keys))),
    ))
    loaded = []
    original = stage_loader.reader.load_keys

    def load_keys(keys):
        loaded.append(frozenset(keys))
        return original(keys)

    stage_loader.reader.load_keys = load_keys

    patcher = stage_loader.load_stage(plan, plan.stages[1], torch.device("cpu"))

    assert loaded == [plan.stages[1].owned_keys]
    # a bf16 bank goes through ComfyUI's cast like any weight, so DynamicVRAM serves a resident
    # bank from VRAM; the node's own bank copied the whole bank from host memory on every forward
    from comfy import ops

    bank = patcher.model.diffusion_model.model.layers[1].mlp.experts_gate_up_proj
    assert isinstance(bank, ops.MixedPrecisionOp)
    stage_state = patcher.model.diffusion_model.state_dict()
    assert set(stage_state) == plan.stages[1].owned_keys
    for key, value in stage_state.items():
        assert torch.equal(value, state[key])


def test_pipeline_conditioning_does_not_run_the_full_model_prefix(hunyuan):
    from types import SimpleNamespace
    from comfy import ops

    upstream = hunyuan("hunyuan_image_3.model")
    params = _tiny_params(hunyuan, "instruct")
    stage = upstream.HunyuanImage3(params, dtype=torch.float32, device="cpu",
                                  operations=ops.mixed_precision_ops(compute_dtype=torch.float32),
                                  pipeline_stage=PipelineStageConfig(0, 2, 0, 1))

    def execute_diffusion_model(*args, **kwargs):
        raise AssertionError("prefix preparation traverses layers owned by other stages")

    model = SimpleNamespace(diffusion_model=stage, current_patcher=None,
                            execute_diffusion_model=execute_diffusion_model)
    result = hunyuan("hunyuan_image_3.model_base").HunyuanImage3Model.extra_conds(
        model, ids=torch.tensor([1, 2, 63]), noise=torch.zeros(1, params.vae_latent_channels, 1, 2, 2))
    assert "hy3_prefix" not in result


def _two_stage_model(hunyuan, model_type="instruct", seed=3):
    upstream = hunyuan("hunyuan_image_3.model")
    operations = hunyuan("hunyuan_image_3.ops").QuantlessMoEOps
    initialize = hunyuan("tests.op_module_init").init_op_modules
    params = _tiny_params(hunyuan, model_type)
    dtype = torch.float32
    full = upstream.HunyuanImage3(params, dtype=dtype, device="cpu", operations=operations)
    initialize(full, dtype, seed=seed)
    state = full.state_dict()
    first = upstream.HunyuanImage3(params, dtype=dtype, device="cpu", operations=operations,
                                   pipeline_stage=PipelineStageConfig(0, 2, 0, 1))
    second = upstream.HunyuanImage3(params, dtype=dtype, device="cpu", operations=operations,
                                    pipeline_stage=PipelineStageConfig(1, 2, 1, 2))
    first.load_state_dict({key: state[key] for key in first.state_dict()}, strict=True)
    second.load_state_dict({key: state[key] for key in second.state_dict()}, strict=True)
    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("hunyuan_image3", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    first._comfy_pipeline_executor = executor
    return params, full, first, second, executor


class _RecordingHead(torch.nn.Module):
    """A text head (hidden -> logits) that keeps every logits row it produced."""

    def __init__(self, hidden_size, vocab_size, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.logits = []

    def forward(self, hidden):
        logits = self.linear(hidden)
        self.logits.append(logits.detach().clone())
        return logits


def _text_sequence(params, with_image):
    # <bos> text, then (optionally) one conditioning block: <boi> <img_size> <img_ratio> <timestep>,
    # its 2x2 VAE run, <joint_img_sep>, its 2x2 tower run, <eoi>; then more text and the task tag
    image = params.image_token_id
    if not with_image:
        return {"ids": torch.tensor([1, 2, 3, 4, 5, 6, 7]), "cond_blocks": []}, None, None
    ids = [1, 2, 3, 10, 11, 12, 13] + [image] * 4 + [14] + [image] * 4 + [15, 4, 5, 6, 16]
    block = {"vae_slice": slice(7, 11), "vit_slice": slice(12, 16), "joint_slice": slice(7, 16),
             "timestep_position": 6, "token_height": 2, "token_width": 2, "patch_height": 2, "patch_width": 2}
    torch.manual_seed(11)
    cond_latent = [torch.randn(1, params.vae_latent_channels, 2, 2)]
    cond_vit = [torch.randn(1, 4, params.hidden_size)]
    return {"ids": torch.tensor(ids), "cond_blocks": [block]}, cond_latent, cond_vit


@pytest.mark.parametrize("with_image", [False, True])
def test_hunyuan_image3_text_generation_over_two_stages_matches_the_unsplit_model(hunyuan, with_image):
    # The prompt-rewriting stage (think / recaption) decodes autoregressively. Split over stages it
    # used to run only the driver's own layers (the others are identities there), silently; each
    # stage now keeps its own layers' KV cache and the hidden state flows stage to stage per token.
    rewrite = hunyuan("hunyuan_image_3.rewrite")
    params, full, first, _, _ = _two_stage_model(hunyuan)
    sequence, cond_latent, cond_vit = _text_sequence(params, with_image)
    settings = dict(stop_tokens=[], max_new_tokens=6, do_sample=False, cond_latent=cond_latent,
                    cond_vit=cond_vit, device=torch.device("cpu"), dtype=torch.float32)

    expected_head = _RecordingHead(params.hidden_size, params.vocab_size, seed=7)
    expected = rewrite.generate_text(full, expected_head, sequence, **settings)
    actual_head = _RecordingHead(params.hidden_size, params.vocab_size, seed=7)
    actual = rewrite.generate_text(first, actual_head, sequence, **settings)

    assert actual["tokens"] == expected["tokens"]
    # the prefill and one feed per generated token
    assert len(actual_head.logits) == len(expected_head.logits) == 7
    for got, want in zip(actual_head.logits, expected_head.logits):
        assert torch.equal(got, want)


def test_hunyuan_image3_text_generation_releases_every_stage_cache(hunyuan):
    rewrite = hunyuan("hunyuan_image_3.rewrite")
    params, _, first, second, _ = _two_stage_model(hunyuan)
    sequence, _, _ = _text_sequence(params, False)
    head = _RecordingHead(params.hidden_size, params.vocab_size, seed=7)

    rewrite.generate_text(first, head, sequence, stop_tokens=[], max_new_tokens=3, do_sample=False,
                          device=torch.device("cpu"), dtype=torch.float32)

    # the diffusion sampler runs next on the same stages: no prompt-length K/V may stay behind
    assert getattr(first, "_hy3_text_cache", None) is None
    assert getattr(second, "_hy3_text_cache", None) is None


def test_hunyuan_image3_denoising_is_unchanged_after_text_generation(hunyuan):
    rewrite = hunyuan("hunyuan_image_3.rewrite")
    params, full, first, _, executor = _two_stage_model(hunyuan)
    sequence, _, _ = _text_sequence(params, False)
    rewrite.generate_text(first, _RecordingHead(params.hidden_size, params.vocab_size, seed=7), sequence,
                          stop_tokens=[], max_new_tokens=2, do_sample=False, device=torch.device("cpu"),
                          dtype=torch.float32)
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 63, 63, 63, 63, 8])
    x = torch.randn(1, params.vae_latent_channels, 1, 2, 2)
    timestep = torch.full((1,), 900.0)
    options = {"sample_sigmas": torch.tensor([1.0, 0.9, 0.0]), "sigmas": torch.tensor([0.9])}

    expected = full(x, timestep, ids=ids, transformer_options=options)
    actual = executor.execute(x, timestep, ids=ids, transformer_options=options)

    assert torch.equal(actual, expected)


def test_each_stage_makes_room_for_its_text_prefill(hunyuan, monkeypatch):
    # After a denoising run a stage's DynamicVRAM weights filled its card, and the next prompt's
    # think_recaption prefill (the conditioning images' ~10k tokens, every expert bank resident) ran out
    # of memory in a stage's MLP. Each stage now frees room for its prefill before running it, as the
    # sampler does for a denoising step, and only for the prefill.
    from comfy import model_management

    rewrite = hunyuan("hunyuan_image_3.rewrite")
    params, _, first, second, _ = _two_stage_model(hunyuan)
    sequence, cond_latent, cond_vit = _text_sequence(params, True)
    requests = []
    monkeypatch.setattr(model_management, "free_memory",
                        lambda required, device, **kwargs: requests.append((required, device, kwargs)) or [])

    rewrite.generate_text(first, _RecordingHead(params.hidden_size, params.vocab_size, seed=7), sequence,
                          stop_tokens=[], max_new_tokens=3, do_sample=False, cond_latent=cond_latent,
                          cond_vit=cond_vit, device=torch.device("cpu"), dtype=torch.float32)

    assert len(requests) == 2
    length = sequence["ids"].shape[0]
    for stage, (required, device, kwargs) in zip((first, second), requests):
        assert device == torch.device("cpu") and kwargs == {"for_dynamic": True}
        layers = stage.pipeline_stage.end_layer - stage.pipeline_stage.start_layer
        cache = 2 * layers * params.num_key_value_heads * params.attention_head_dim * length * 4
        assert required >= cache + length * params.hidden_size * 4 + model_management.minimum_inference_memory()


def test_each_stage_makes_room_for_its_denoising_step(hunyuan, monkeypatch):
    # A worker stage's weights are loaded without the sampler's activation estimate (it reaches only the
    # driver's card), so a denoising step after a text decode ran out of memory on stage 1 with 8.2 GiB
    # of activations under ~15 GiB of resident weights. Each stage frees the sampler's own estimate for
    # the step's sequence before running its layers.
    from comfy import model_management

    params, _, _, _, executor = _two_stage_model(hunyuan)
    requests = []
    monkeypatch.setattr(model_management, "free_memory",
                        lambda required, device, **kwargs: requests.append((required, device, kwargs)) or [])
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 63, 63, 63, 63, 8])
    x = torch.randn(2, params.vae_latent_channels, 1, 2, 2)
    options = {"sample_sigmas": torch.tensor([1.0, 0.9, 0.0]), "sigmas": torch.tensor([0.9])}

    executor.execute(x, torch.full((2,), 900.0), ids=ids, transformer_options=options)

    assert len(requests) == 2
    length = ids.shape[0]
    for required, device, kwargs in requests:
        assert device == torch.device("cpu") and kwargs == {"for_dynamic": True}
        assert required >= 2 * (length * length * 4 + length * params.hidden_size * 24) + model_management.minimum_inference_memory()


class _RecordingTransport(FakePipelineTransport):
    """Records the tensor names of each stage-to-stage transfer."""

    def __init__(self, runtime):
        super().__init__(runtime)
        self.sent = []

    def transfer(self, tensors, schema, destination):
        self.sent.append(tuple(sorted(tensors)))
        return super().transfer(tensors, schema, destination)


def _recording_two_stage(hunyuan):
    params, full, first, second, _ = _two_stage_model(hunyuan)
    transport = _RecordingTransport(FakeDeviceRuntime())
    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 1, 2, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(PipelinePartitionPlan("hunyuan_image3", plans),
                                             (first, second.forward_pipeline_stage), transport)
    first._comfy_pipeline_executor = executor
    return params, full, first, executor, transport


def _denoise(model_or_executor, params, ids, steps, seed):
    torch.manual_seed(seed)
    x = torch.randn(len(ids), params.vae_latent_channels, 1, 2, 2)
    sigmas = torch.tensor([1.0, 0.9, 0.8, 0.0])
    outputs = []
    for step in range(steps):
        options = {"sample_sigmas": sigmas, "sigmas": sigmas[step:step + 1]}
        timestep = torch.full((len(ids),), float(sigmas[step]) * 1000)
        run = model_or_executor.execute if hasattr(model_or_executor, "execute") else model_or_executor
        # the transport's pooled buffer is the executor's output; the next step overwrites it
        outputs.append(run(x, timestep, ids=ids, transformer_options=options).clone())
    return outputs


def _stage_transfers(transport):
    # transfers into stage 1; the pipeline output's way back is the other entry per step
    return [names for names in transport.sent if "hidden_states" in names]


def test_a_prompts_mask_and_rotary_table_cross_the_pipeline_once_and_the_output_is_unchanged(hunyuan):
    # The dense mask is a CFG pair's prompt length squared (1.5 GB at 19k tokens) and was sent stage to
    # stage on every step; over a switched LAN that is most of a step. Within one prompt it never changes.
    params, full, _, executor, transport = _recording_two_stage(hunyuan)
    long = torch.tensor([1, 2, 3, 4, 5, 6, 7, 63, 63, 63, 63, 8])
    short = torch.tensor([1, 2, 3, 63, 63, 63, 63, 8])
    for ids, seed in (((long, long), 1), ((short, short), 2), ((long, long), 3)):
        before = len(_stage_transfers(transport))
        expected = _denoise(full, params, ids, 3, seed)
        actual = _denoise(executor, params, ids, 3, seed)
        for got, want in zip(actual, expected):
            assert torch.equal(got, want)
        sent = _stage_transfers(transport)[before:]
        assert "attention_mask" in sent[0] and "freqs" in sent[0]
        assert all("attention_mask" not in names and "freqs" not in names for names in sent[1:])


def test_a_text_decode_between_denoising_steps_sends_the_mask_again(hunyuan):
    # a text prefill drops the stages' denoising context to make room; the next step must not reuse it
    rewrite = hunyuan("hunyuan_image_3.rewrite")
    params, full, first, executor, transport = _recording_two_stage(hunyuan)
    ids = (torch.tensor([1, 2, 3, 4, 5, 6, 7, 63, 63, 63, 63, 8]),) * 2
    _denoise(executor, params, ids, 2, 4)
    sequence, _, _ = _text_sequence(params, False)
    rewrite.generate_text(first, _RecordingHead(params.hidden_size, params.vocab_size, seed=7), sequence,
                          stop_tokens=[], max_new_tokens=2, do_sample=False, device=torch.device("cpu"),
                          dtype=torch.float32)
    before = len(_stage_transfers(transport))
    expected = _denoise(full, params, ids, 2, 5)
    actual = _denoise(executor, params, ids, 2, 5)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
    after_text = [names for names in _stage_transfers(transport)[before:] if "timestep" in names]
    assert "attention_mask" in after_text[0]
