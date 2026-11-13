import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest
import safetensors.torch
import torch

from comfy import sd

from . import magi2_reference
from .test_magi2_model import _randomize

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None, reason="needs MAGI2_REFERENCE_SRC")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "convert_magi2.py"


def _converter():
    spec = importlib.util.spec_from_file_location("convert_magi2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_checkpoint(directory, video_channels=8, audio_channels=6):
    """A tiny preview transformer saved in the release layout: shards plus model.safetensors.index.json."""
    ref = magi2_reference.load()
    cfg = ref.config
    config = cfg.ModelConfig(num_layers=3, hidden_size=256, head_dim=64, num_query_groups=4, video_in_channels=video_channels, audio_in_channels=audio_channels,
                             text_in_channels=5120, mm_layers=[0, 2], num_heads_q=4, num_heads_kv=4,
                             moe_config=cfg.MoEConfig(num_experts=8, top_k=6, moe_layers=[1], expert_intermediate_size=64, shared_expert_intermediate_size=32,
                                                      modality_specific_expert_intermediate_size=32, num_heads=4))
    with ref.eager():
        model = ref.preview.Transformer(config)
    _randomize(model)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[:len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for filename, keys in halves.items():
        safetensors.torch.save_file({key: state[key] for key in keys}, os.path.join(directory, filename))
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as handle:
        json.dump({"weight_map": {key: filename for filename, keys in halves.items() for key in keys}}, handle)
    return state


def _convert(converter, source, out, quantized, shard_gib):
    converter.transformer(argparse.Namespace(source=str(source), out=str(out), format="int8" if quantized else "bf16", device="cpu", shard_gib=shard_gib))


def test_converted_checkpoints_load_and_agree(tmp_path):
    release = tmp_path / "preview"
    release.mkdir()
    state = _release_checkpoint(release)
    converter = _converter()
    bf16, int8 = tmp_path / "magi2_bf16.safetensors", tmp_path / "magi2_int8_convrot.safetensors"
    _convert(converter, release, bf16, False, 40.0)
    _convert(converter, release, int8, True, 0.004)

    index = tmp_path / "magi2_int8_convrot.safetensors.index.json"
    assert index.exists() and len(json.loads(index.read_text())["weight_map"]) > 0
    shards = sorted(tmp_path.glob("magi2_int8_convrot-*-of-*.safetensors"))
    assert len(shards) > 1
    quantized = {}
    for shard in shards:
        quantized.update(safetensors.torch.load_file(shard))
    assert "block.layers.1.mlp.moe_mlp.router.bias" in quantized and "block.layers.1.mlp.moe_mlp.router.expert_bias" not in quantized
    torch.testing.assert_close(quantized["block.layers.1.mlp.moe_mlp.router.bias"], state["block.layers.1.mlp.moe_mlp.router.expert_bias_ema"])
    bank = quantized["block.layers.1.mlp.moe_mlp.gate_proj.weight"]
    assert bank.dtype == torch.int8 and bank.shape == (32, 64, 64)
    assert quantized["block.layers.1.mlp.moe_mlp.gate_proj.weight_scale"].shape == (32, 64, 1)
    assert json.loads(bytes(quantized["block.layers.1.mlp.moe_mlp.gate_proj.comfy_quant"].tolist())) == {
        "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64, "num_experts": 32}
    assert json.loads(bytes(quantized["block.layers.0.attention.linear_qkv.1.comfy_quant"].tolist())) == {
        "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    for kept in ("block.layers.0.attention.linear_g.0.weight", "block.layers.0.attention.sinks", "block.layers.1.mhc_phi_fused_mlp",
                 "pre_adapter.video_embedder.weight", "block.layers.1.mlp.moe_mlp.router.weight", "block.layers.0.attention.q_norm.weight"):
        assert quantized[kept].dtype != torch.int8 and kept.removesuffix(".weight") + ".comfy_quant" not in quantized

    reference_model = sd.load_diffusion_model(str(bf16))
    quantized_model = sd.load_diffusion_model(str(index))
    assert reference_model.model.diffusion_model.num_layers == quantized_model.model.diffusion_model.num_layers == 3
    generator = torch.Generator().manual_seed(0)
    x = [torch.randn(1, 8, 2, 2, 3, generator=generator), torch.randn(1, 6, 4, generator=generator)]
    context = torch.randn(1, 3, 5120, generator=generator)
    outputs = []
    for patcher in (reference_model, quantized_model):
        model = patcher.model.diffusion_model
        outputs.append(model([part.to(torch.float32) for part in x], torch.tensor([500.0]), context))
    for expected, actual in zip(*outputs):
        error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert error < 5e-2
