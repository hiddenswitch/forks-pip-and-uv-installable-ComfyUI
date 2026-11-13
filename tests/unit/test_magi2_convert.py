import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest
import safetensors.torch
import torch

from comfy import sd
from comfy.ldm.magi2.model import ieee_fp32_matmul
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8_utils import _build_hadamard

from . import magi2_reference
from .test_magi2_model import _randomize
from .test_magi2_moe import contract_inputs, kernel_contract, relative_error

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None, reason="needs MAGI2_REFERENCE_SRC")

INT8_MOE_TOLERANCE = 5e-4
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

    _assert_int8_experts_follow_the_kernel_contract(quantized_model.model.diffusion_model.block.layers[1].mlp.moe_mlp)


def _unrotated_experts(bank):
    """A ConvRot INT8 bank dequantized and rotated back to the checkpoint's basis, [E, out, in] in fp64."""
    weight = bank.weight
    qdata, scale = weight.layout_cls.get_plain_tensors(weight)
    group = weight.params.convrot_groupsize
    rotated = qdata.double().view(-1, bank.out_features, bank.in_features)
    rotated = rotated * (scale.double().view(-1, bank.out_features, 1) if scale.numel() > 1 else scale.double())
    return (rotated.unflatten(-1, (-1, group)) @ _build_hadamard(group, dtype=torch.float64)).flatten(-2)


def _assert_int8_experts_follow_the_kernel_contract(module):
    """The INT8 bank path computes the kernel's fp32 math on the dequantized weights.

    Dequantized weights are not bf16 values, so fp32 summation order now perturbs the pre-activations
    and occasionally flips a bf16 rounding of SwiGLU7's output or the result: the floor is about 1e-4,
    against about 4e-3 when the gate and up outputs are rounded to bf16. Both bank modes run: dequantized
    up front by the ops (full precision matmul) and kept quantized with the activations rotated instead.
    """
    banks = (module.gate_proj, module.up_proj, module.down_proj)
    assert all(isinstance(bank.weight, QuantizedTensor) for bank in banks)
    module.top_k = 1
    x = torch.randn(1, 160, module.heads * module.head_dim, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    inputs = contract_inputs(module, x[0], *(_unrotated_experts(bank) for bank in banks))
    # W8A16: SwiGLU7's output is rounded to the bf16 activation dtype, as the kernel rounds it to its bf16 W_down
    expected = kernel_contract(**inputs, hidden_dtype=torch.bfloat16).reshape(x.shape[1:])
    rounded = kernel_contract(**inputs, hidden_dtype=torch.bfloat16, gate_up_dtype=torch.bfloat16).reshape(x.shape[1:])
    assert relative_error(rounded, expected) > 4 * INT8_MOE_TOLERANCE
    for full_precision_mm in (True, False):
        for bank in banks:
            bank._full_precision_mm = full_precision_mm
        with ieee_fp32_matmul(), torch.no_grad():
            actual = module(x)[0]
        assert relative_error(actual, expected) < INT8_MOE_TOLERANCE, full_precision_mm


def test_wan_vae_is_kept_in_fp32_and_checked_against_the_fp16_copy(tmp_path, capsys):
    generator = torch.Generator().manual_seed(4)
    state = {"encoder.conv1.weight": torch.randn(4, 3, 3, 3, 3, generator=generator), "decoder.head.2.bias": torch.randn(3, generator=generator)}
    source = tmp_path / "vae"
    source.mkdir()
    torch.save(state, source / "Wan2.2_VAE.pth")
    reference = tmp_path / "wan2.2_vae.safetensors"
    safetensors.torch.save_file({key: value.half() for key, value in state.items()}, reference)
    out = tmp_path / "magi2_wan2.2_vae_fp32.safetensors"
    _converter().wan_vae(argparse.Namespace(source=str(source), reference=str(reference), out=str(out), shard_gib=40.0))
    assert "2 tensors, 0 differ" in capsys.readouterr().out
    converted = safetensors.torch.load_file(out)
    assert converted.keys() == state.keys()
    assert all(converted[key].dtype == torch.float32 and torch.equal(converted[key], value) for key, value in state.items())
