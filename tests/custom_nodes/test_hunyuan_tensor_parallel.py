"""Exercise the installed upstream transformer with rank-local weights."""
import dataclasses
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from comfy import memory_management, ops
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy.pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from comfy.tensor_parallel.hunyuan_image3 import _load_state, _shard_model
from comfy.tensor_parallel.kandinsky6 import load_state as load_kandinsky_state, shard_model as shard_kandinsky_model
from comfy.tensor_parallel.operations import tensor_parallel_operations
from comfy.tensor_parallel.types import TensorParallelConfig
from comfy.tensor_parallel.custom_node_state import CachedStateExecutor, install_state_transport
from comfy.pipeline_parallel.types import pack_pipeline_value, unpack_pipeline_value
from comfy_compatibility.vanilla import prepare_vanilla_environment
from tests.unit.test_custom_node_tensor_parallel import Collectives, Rank


@pytest.mark.parametrize("cache_kind", ["spectrum", "magcache"])
def test_custom_cache_state_survives_rank_transport(cache_kind):
    if cache_kind == "spectrum":
        entry = next(ep for ep in distribution("comfyui-hunyuanimage3").entry_points if ep.group == "comfyui.custom_nodes").load()
        root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
        suffix, key = ".hunyuan_image_3.spectrum", "hy3_spectrum"
    else:
        entry = next(ep for ep in distribution("kandinsky6").entry_points if ep.group == "comfyui.custom_nodes").load()
        root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "kandinsky-6"
        suffix, key = ".kandinsky6.magcache", "k6_magcache"
    prepare_vanilla_environment()
    assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
    source = importlib.import_module(root.name + suffix)
    if cache_kind == "spectrum":
        state = source.SpectrumState(num_steps=10, warmup_steps=2)
        classes = (source.SpectrumState, source.SpectrumForecaster, source.ChebyshevForecaster)
    else:
        state = source.K6MagCacheState(num_steps=10, threshold=0.1, max_skip_steps=2,
                                     retention_ratio=0, mag_ratios={"t2va": [1.0] * 20, "i2va": [1.0] * 20})
        classes = (source.K6MagCacheState, source._LaneState)

    class Model(torch.nn.Module):
        def forward(self, value, step, transformer_options):
            cache = transformer_options[key]
            if cache_kind == "spectrum":
                run = cache.should_run(step)
                cache.note_step(step, run)
                if run:
                    cache.store(0, step / 9, value)
                    return value
                return cache.predict(0, step / 9, step)
            decision = cache.begin(profile="t2va", timestep=torch.tensor([10 - step]),
                                   cond_or_uncond=[0], visual=value, audio=value)
            if decision.skip:
                return cache.apply_cached(value, value, decision)[0]
            cache.record_computed(value, value, value * 2, value * 2, decision)
            return value * 2

    ranks = [Model(), Model()]
    for model in ranks:
        install_state_transport(model, key, classes)

    def execute(method, *args, **kwargs):
        tensors = {}
        packed = pack_pipeline_value((args, kwargs), tensors, "test")
        results = []
        for model in ranks:
            rank_args, rank_kwargs = unpack_pipeline_value(packed, {k: v.clone() for k, v in tensors.items()})
            results.append(getattr(model, method)(*rank_args, **rank_kwargs))
        torch.testing.assert_close(results[0][0], results[1][0])
        return results[0]

    executor = CachedStateExecutor(SimpleNamespace(root_model=ranks[0], execute_method=execute))
    codec = ranks[0].tensor_parallel_state_codec
    reference_state = codec.decode(codec.encode(state))
    reference = Model()
    for step in range(10):
        value = torch.full((1, 3, 4), 1.0 + step / 10)
        actual = executor.execute(value, step, transformer_options={key: state})
        expected = reference(value, step, transformer_options={key: reference_state})
        torch.testing.assert_close(actual, expected)
    if cache_kind == "spectrum":
        assert state.skipped_steps == reference_state.skipped_steps > 0
        state.clear()
        assert state.peak_forecaster_bytes() == 0
    else:
        assert state.last_generation_stats == reference_state.last_generation_stats
        assert state.last_generation_stats["skipped_model_calls"] > 0
        assert all(lane.visual_residual is None for lane in state._lanes.values())


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("shard_dense", [False, True])
def test_installed_hunyuan_transformer_matches_replicated_model(size, tokens, shard_dense, tmp_path):
    entry = next(ep for ep in distribution("comfyui-hunyuanimage3").entry_points if ep.group == "comfyui.custom_nodes").load()
    root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
    prepare_vanilla_environment()
    assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
    upstream = importlib.import_module(root.name + ".hunyuan_image_3.model")
    loader = importlib.import_module(root.name + ".hunyuan_image_3.loader")
    initialize = importlib.import_module(root.name + ".tests.op_module_init").init_op_modules
    config = json.loads(Path(loader.CONFIG_PATH).read_text())
    config.update(loader.MODEL_TYPES["instruct_distil"], model_type="instruct_distil")
    params = dataclasses.replace(upstream.params_from_config(config),
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_head_dim=8, moe_intermediate_size=16, num_experts=8, moe_topk=2,
        num_shared_expert=1, vocab_size=64, pad_token_id=0,
    )
    base = ops.mixed_precision_ops(compute_dtype=torch.float32)
    frequencies = upstream.build_rope_freqs(tokens, 8, [], 10000.0, device="cpu")

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = upstream.HunyuanImage3Model(params, dtype=torch.float32, device="cpu", operations=base)

        def forward(self, value):
            return self.model(value, frequencies)

    reference = Transformer()
    initialize(reference, torch.float32)
    checkpoint = tmp_path / "transformer.safetensors"
    save_file(reference.state_dict(), checkpoint)
    reader = SafetensorsCheckpointReader(checkpoint)
    collective = Collectives(size)
    ranks = []
    reductions = [0] * size
    gathers = [0] * size

    class MeasuredRank(Rank):
        def sum(self, value):
            reductions[self.rank] += 1
            return super().sum(value)

        def gather(self, value):
            gathers[self.rank] += 1
            return super().gather(value)

    for rank in range(size):
        model = Transformer()
        parallel = TensorParallelConfig(MeasuredRank(collective, rank))
        shards = _shard_model(model, base, parallel, shard_dense=shard_dense)
        model.load_state_dict(_load_state(reader, shards, parallel, ()))
        ranks.append(model)
    inputs = torch.randn(1, tokens, 32)
    expected = reference(inputs)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda model: model(inputs), ranks))
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert reductions == [params.num_hidden_layers] * size
    assert gathers == [params.num_hidden_layers * 4 if shard_dense else 0] * size


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("lazy", [False, True])
def test_installed_kandinsky_asymmetric_attention_matches_full_model(size, lazy, tmp_path, monkeypatch):
    entry = next(ep for ep in distribution("kandinsky6").entry_points if ep.group == "comfyui.custom_nodes").load()
    root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "kandinsky-6"
    prepare_vanilla_environment()
    assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
    upstream = importlib.import_module(root.name + ".kandinsky6.ldm.model")
    context = torch.randn(1, 4, 16)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = upstream.AsymCrossAttention(32, 8, 16, {
                "operations": ops.manual_cast, "dtype": torch.float32, "device": "cpu",
            })

        def forward(self, value):
            return self.attention(value, context)

    reference = Attention()
    for parameter in reference.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path / "attention.safetensors"
    save_file(reference.state_dict(), checkpoint)
    reader = SafetensorsCheckpointReader(checkpoint)
    collective = Collectives(size)
    ranks = []
    for rank in range(size):
        with monkeypatch.context() as scoped:
            scoped.setattr(memory_management, "aimdo_enabled", lazy)
            model = Attention()
            parallel = TensorParallelConfig(Rank(collective, rank))
            shards = shard_kandinsky_model(model, tensor_parallel_operations(ops.manual_cast, parallel), root.name)
            model.load_state_dict(load_kandinsky_state(reader, "", shards, parallel, root.name), assign=lazy)
        ranks.append(model)
    inputs = torch.randn(1, 3, 32)
    expected = reference(inputs)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda model: model(inputs), ranks))
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
