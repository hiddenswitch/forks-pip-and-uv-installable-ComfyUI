import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from comfy import ops, quant_ops
from comfy.tensor_parallel.checkpoint import shard_minimax_h3_state_dict, shard_tensor_parallel_state_dict
from comfy.tensor_parallel.operations import full_domain_tensor_parallel_operations, tensor_parallel_operations
from comfy.tensor_parallel.types import TensorParallelConfig
from comfy.tensor_parallel.hunyuan_image3 import _load_state
from comfy.pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from safetensors.torch import save_file
from types import SimpleNamespace
from unittest.mock import Mock
from comfy import supported_models
from comfy.tensor_parallel import loader
from comfy.ldm.flux.layers import ModulationOut, SingleStreamBlock
from comfy.ldm.krea2.model import SwiGLU
from comfy.ldm.lumina.model import FeedForward
from comfy.ldm.minimax.model import MLP
from tests.unit.test_custom_node_tensor_parallel import Collectives, Rank


PROJECTIONS = [
    ("minimax_h3", "blocks.0.attn.qkv_proj", 3, "blocks.0.attn.out_proj"),
    ("minimax_h3", "blocks.0.mlp.fc1", 2, "blocks.0.mlp.fc2"),
    ("flux2", "double_blocks.0.img_attn.qkv", 3, "double_blocks.0.img_attn.proj"),
    ("flux2", "double_blocks.0.txt_mlp.0", 2, "double_blocks.0.txt_mlp.2"),
    ("ideogram4", "layers.0.attention.qkv", 3, "layers.0.attention.o"),
    ("ideogram4", "layers.0.feed_forward.w1", 1, "layers.0.feed_forward.w2"),
    ("krea2", "blocks.0.attn.wq", 1, "blocks.0.attn.wo"),
    ("krea2", "txtfusion.refiner_blocks.0.mlp.gate", 1, "txtfusion.refiner_blocks.0.mlp.down"),
]


def metadata(format, **kwargs):
    return torch.tensor(list(json.dumps({"format": format, **kwargs}).encode()), dtype=torch.uint8)


def checkpoint(format, rows, columns=256):
    packed = format in ("nvfp4", "svdquant_w4a4", "awq_w4a16", "convrot_w4a4", "asym_w4a8_int8")
    width = columns // 2 if packed else columns * 3 // 4 if format == "w6a8_int8" else columns
    state = {"weight": torch.arange(rows * width).reshape(rows, width),
             "bias": torch.arange(rows), "comfy_quant": metadata(format)}
    if "input_scale" in quant_ops.QUANT_ALGOS[format]["parameters"]:
        state["input_scale"] = torch.tensor([0.5])
    if "pre_quant_scale" in quant_ops.QUANT_ALGOS[format]["parameters"]:
        state["pre_quant_scale"] = torch.arange(columns)
    if format in ("svdquant_w4a4", "awq_w4a16"):
        state["weight_scale"] = torch.arange(columns // 64 * rows).reshape(columns // 64, rows)
        if format == "awq_w4a16":
            state["weight_zeros"] = state["weight_scale"] + 1
        else:
            state.update(weight_proj_up=torch.arange(rows * 4).reshape(rows, 4),
                         weight_proj_down=torch.arange(columns * 4).reshape(columns, 4),
                         weight_smooth_factor=torch.arange(columns))
    elif format in ("asym_w4a8_int8", "w6a8_int8"):
        state.update(weight_s_rel=torch.arange(rows * columns // 16).reshape(rows, columns // 16),
                     weight_s_channel=torch.arange(rows))
        if format == "asym_w4a8_int8":
            state["weight_codebook"] = torch.linspace(-1, 1, 16)
    elif format in ("nvfp4", "mxfp8"):
        group = 16 if format == "nvfp4" else 32
        state["weight_scale"] = quant_ops.ck.to_blocked(
            torch.arange(rows * columns // group).reshape(rows, columns // group), flatten=False,
        )
        if format == "nvfp4":
            state["weight_scale_2"] = torch.tensor([0.25])
    elif format.startswith("float8"):
        state["weight_scale"] = torch.tensor([0.25])
    elif format == "convrot_w4a4":
        state["weight_scale"] = torch.arange(rows)
    else:
        state["weight_scale"] = torch.arange(rows).reshape(rows, 1)
    return state


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("format", ["float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "nvfp4", "mxfp8",
                                  "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "svdquant_w4a4", "awq_w4a16"])
@pytest.mark.parametrize("family,column,sections,row", PROJECTIONS)
def test_quantized_projection_weights_and_metadata_follow_output_channels(size, format, family, column, sections, row):
    source = {column: checkpoint(format, 128 * sections), row: checkpoint(format, 128)}
    state = {f"{module}.{key}": value for module, values in source.items() for key, value in values.items()}
    for rank in range(size):
        local = shard_tensor_parallel_state_dict(state, family, rank, size, full_domain_rows=True)
        for module, count in ((column, sections), (row, 1)):
            indices = torch.cat([torch.arange(i * 128 + rank * 128 // size, i * 128 + (rank + 1) * 128 // size)
                                 for i in range(count)])
            for name, value in source[module].items():
                actual = local[f"{module}.{name}"]
                if name == "weight_scale" and format in ("nvfp4", "mxfp8"):
                    columns = 256 // (16 if format == "nvfp4" else 32)
                    plain = quant_ops.ck.from_blocked(value, count * 128, columns)
                    expected = quant_ops.ck.to_blocked(plain[indices], flatten=False)
                elif name in ("weight_scale", "weight_zeros") and format in ("svdquant_w4a4", "awq_w4a16"):
                    expected = value[:, indices]
                elif name in ("weight", "bias", "weight_proj_up", "weight_s_rel", "weight_s_channel") or (
                    name == "weight_scale" and not format.startswith("float8")
                ):
                    expected = value[indices]
                else:
                    expected = value
                torch.testing.assert_close(actual, expected, msg=f"{module}.{name}, rank {rank}")


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("full_domain", [False, True])
def test_flux_packed_output_projection_matches_full_linear(size, full_domain):
    torch.manual_seed(17)
    hidden, mlp = 16, 32
    weight = torch.randn(hidden, hidden + mlp)
    state = {"single_blocks.0.linear1.weight": torch.randn(3 * hidden + 2 * mlp, hidden),
             "single_blocks.0.linear2.weight": weight, "single_blocks.0.linear2.bias": torch.randn(hidden)}
    input = torch.randn(3, hidden + mlp)
    expected = torch.nn.functional.linear(input, weight, state["single_blocks.0.linear2.bias"])
    collective = Collectives(size)
    modules, inputs = [], []
    for rank in range(size):
        decorator = full_domain_tensor_parallel_operations if full_domain else tensor_parallel_operations
        operations = decorator(ops.disable_weight_init, TensorParallelConfig(Rank(collective, rank)))
        module = operations.RowParallelLinear(hidden + mlp, hidden, section_sizes=(hidden, mlp))
        local = shard_tensor_parallel_state_dict(state, "flux2", rank, size, full_domain_rows=full_domain)
        prefix = "single_blocks.0.linear2."
        module.load_state_dict({key.removeprefix(prefix): value for key, value in local.items() if key.startswith(prefix)})
        modules.append(module)
        inputs.append(torch.cat([section.chunk(size, dim=-1)[rank] for section in input.split((hidden, mlp), dim=-1)], dim=-1))
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda rank: modules[rank](inputs[rank]), range(size)))
    for output in outputs:
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("size", [2, 4])
def test_flux_packed_four_bit_input_projection_uses_logical_hidden_width(size):
    state = {"single_blocks.0.linear1.weight": torch.arange(40 * 4).reshape(40, 4),
             "single_blocks.0.linear2.weight": torch.zeros(8, 8, dtype=torch.int8),
             "single_blocks.0.linear1.comfy_quant": metadata("convrot_w4a4"),
             "single_blocks.0.linear2.comfy_quant": metadata("convrot_w4a4")}
    for rank in range(size):
        local = shard_tensor_parallel_state_dict(state, "flux2", rank, size, full_domain_rows=True)
        expected = torch.cat([section.chunk(size)[rank] for section in state["single_blocks.0.linear1.weight"].split(8)])
        torch.testing.assert_close(local["single_blocks.0.linear1.weight"], expected)


@pytest.mark.parametrize("size", [2, 4])
def test_flux_output_shard_keeps_each_input_section(size):
    weight = torch.arange(8 * 24).reshape(8, 24)
    for rank in range(size):
        key = "single_blocks.0.linear2.weight"
        local = shard_tensor_parallel_state_dict({key: weight}, "flux2", rank, size)
        expected = torch.cat([section.chunk(size, dim=1)[rank] for section in weight.split((8, 16), dim=1)], dim=1)
        torch.testing.assert_close(local[key], expected)


@pytest.mark.parametrize("scalar", [torch.tensor(0.5), torch.tensor([0.5])])
def test_minimax_scalar_scale_remains_replicated(scalar):
    prefix = "blocks.0.attn.qkv_proj"
    state = {prefix + ".weight": torch.ones(24, 8), prefix + ".weight_scale": scalar}
    local = shard_minimax_h3_state_dict(state, 0, 2)
    torch.testing.assert_close(local[prefix + ".weight_scale"], scalar)


@pytest.mark.parametrize("family,column,sections,row", PROJECTIONS)
def test_row_parallel_checkpoint_does_not_load_bias_on_peer_ranks(family, column, sections, row):
    state = {row + ".weight": torch.ones(8, 16), row + ".bias": torch.ones(8)}
    assert row + ".bias" not in shard_tensor_parallel_state_dict(state, family, 1, 2)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("format", ["float8_e4m3fn", "nvfp4", "mxfp8", "int8_tensorwise", "asym_w4a8_int8", "svdquant_w4a4", "awq_w4a16"])
def test_hunyuan_dense_streamed_metadata_matches_shared_sharding(tmp_path, size, format):
    prefix = "model.layers.0.self_attn.o_proj"
    values = checkpoint(format, 128)
    state = {f"{prefix}.{key}": value.contiguous() for key, value in values.items()}
    path = tmp_path / "weights.safetensors"
    save_file(state, path)
    reader = SafetensorsCheckpointReader(path)
    for rank in range(size):
        parallel = TensorParallelConfig(Rank(Collectives(size), rank))
        actual = _load_state(reader, {prefix: 0}, parallel, ())
        native_prefix = "blocks.0.attn.out_proj"
        native = {key.replace(prefix, native_prefix): value for key, value in state.items()}
        expected = shard_tensor_parallel_state_dict(native, "minimax_h3", rank, size, full_domain_rows=True)
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key.replace(prefix, native_prefix)])


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("scalar", [False, True])
def test_hunyuan_expert_global_scales_and_per_expert_scales(tmp_path, size, scalar):
    prefix = "model.layers.0.mlp.experts_gate_up_proj"
    state = {f"{prefix}.weight": torch.zeros(8, 128, 128, dtype=torch.uint8),
             f"{prefix}.weight_scale": torch.arange(8 * 128 * 16).reshape(8, 128, 16),
             f"{prefix}.weight_scale_2": torch.tensor([0.5]) if scalar else torch.arange(8, dtype=torch.float32),
             f"{prefix}.comfy_quant": metadata("nvfp4", num_experts=8)}
    path = tmp_path / "experts.safetensors"
    save_file(state, path)
    for rank in range(size):
        parallel = TensorParallelConfig(Rank(Collectives(size), rank))
        actual = _load_state(SafetensorsCheckpointReader(path), {prefix: 0}, parallel, ())
        scale = state[f"{prefix}.weight_scale_2"]
        expected = scale if scalar else scale.chunk(size)[rank]
        torch.testing.assert_close(actual[f"{prefix}.weight_scale_2"], expected)
        torch.testing.assert_close(actual[f"{prefix}.weight_scale"], state[f"{prefix}.weight_scale"].chunk(size)[rank])


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("tiles", [1, 8])
def test_svdquant_tiled_weights_scales_and_low_rank_projection(size, tiles, tmp_path):
    prefix = "blocks.0.attn.out_proj"
    values = {"weight": torch.arange(tiles * 4 * 32 * 128).reshape(tiles, 4, 32, 128),
              "weight_scale": torch.randn(tiles, 4, 128), "weight_proj_up": torch.randn(tiles, 16, 128),
              "weight_proj_down": torch.randn(256, 16), "weight_smooth_factor": torch.randn(256),
              "bias": torch.randn(tiles * 128), "comfy_quant": metadata("svdquant_w4a4")}
    state = {f"{prefix}.{key}": value for key, value in values.items()}
    base = ops.mixed_precision_ops(compute_dtype=torch.float32)
    full = base.Linear(256, tiles * 128)
    full.load_state_dict(dict(values))
    expected = full.weight.dequantize()
    path = tmp_path / "tiled.safetensors"
    save_file(state, path)
    for rank in range(size):
        local = shard_tensor_parallel_state_dict(state, "minimax_h3", rank, size, full_domain_rows=True)
        parallel = TensorParallelConfig(Rank(Collectives(size), rank))
        streamed = _load_state(SafetensorsCheckpointReader(path), {prefix: 0}, parallel, ())
        for key, value in local.items():
            torch.testing.assert_close(streamed[key], value)
        linear = base.Linear(256, tiles * 128 // size)
        linear.load_state_dict({key.removeprefix(prefix + "."): value for key, value in local.items()})
        torch.testing.assert_close(linear.weight.dequantize(), expected.chunk(size)[rank])


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("family,column,sections,row", PROJECTIONS)
def test_int8_convrot_projection_forward_matches_unsharded(size, family, column, sections, row):
    torch.manual_seed(53)
    values = {"weight": torch.randint(-20, 20, (128, 256), dtype=torch.int8),
              "weight_scale": torch.rand(128, 1) * 0.01, "bias": torch.randn(128),
              "comfy_quant": metadata("int8_tensorwise", convrot=True, convrot_groupsize=256)}
    base = ops.mixed_precision_ops(compute_dtype=torch.float32)
    reference = base.Linear(256, 128)
    reference.load_state_dict(dict(values))
    inputs = torch.randn(3, 256)
    expected = reference(inputs)
    collective = Collectives(size)
    modules = []
    state = {f"{row}.{key}": value for key, value in values.items()}
    for rank in range(size):
        operations = full_domain_tensor_parallel_operations(base, TensorParallelConfig(Rank(collective, rank)))
        module = operations.RowParallelLinear(256, 128)
        local = shard_tensor_parallel_state_dict(state, family, rank, size, full_domain_rows=True)
        module.load_state_dict({key.removeprefix(row + "."): value for key, value in local.items()})
        modules.append(module)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda rank: modules[rank](inputs.chunk(size, dim=-1)[rank]), range(size)))
    for output in outputs:
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("model_type,family,row", [
    (supported_models.Flux2, "flux2", "double_blocks.0.img_attn.proj"),
    (supported_models.Krea2, "krea2", "blocks.0.attn.wo"),
    (supported_models.Ideogram4, "ideogram4", "layers.0.attention.o"),
    (supported_models.MiniMaxH3, "minimax_h3", "blocks.0.attn.out_proj"),
])
@pytest.mark.parametrize("legacy", [False, True])
def test_rank_loader_pairs_quantized_operations_and_checkpoint_layout(monkeypatch, model_type, family, row, legacy):
    config = model_type({"image_model": family, "hidden_size": 16})
    config.quant_config = {}
    state = {row + ".weight": torch.ones(8, 16, dtype=torch.float8_e4m3fn)}
    if legacy:
        state.update({row + ".scale_weight": torch.ones(8, 1), "scaled_fp8": torch.tensor(1.0)})
    else:
        state.update({row + ".weight_scale": torch.ones(8, 1), row + ".comfy_quant": metadata("float8_e4m3fn")})
    reader = Mock()
    reader.load_keys.return_value = state
    model = Mock()
    configurations = []

    def get_model(rank_config, state, prefix, device):
        configurations.append(rank_config.custom_operations)
        return model

    monkeypatch.setattr(model_type, "get_model", get_model)
    monkeypatch.setattr(loader, "get_model_patcher_class", lambda disabled: (
        lambda model, **kwargs: SimpleNamespace(is_dynamic=lambda: False)
    ))
    parallel = TensorParallelConfig(Rank(Collectives(2), 1))
    loader._load_rank(reader, config, {}, "", {key: key for key in state}, "model.safetensors", {},
                      True, torch.float32, torch.device("cpu"), parallel, family)
    loaded = model.load_model_weights.call_args.args[0]
    assert loaded[row + ".weight"].shape == (4, 16)
    assert loaded[row + ".weight_scale"].shape == (4, 1)
    linear = configurations[0].RowParallelLinear(16, 8)
    assert (linear.in_features, linear.out_features) == (16, 4)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("family,prefix", [
    ("minimax_h3", "blocks.0.mlp"), ("ideogram4", "layers.0.feed_forward"),
    ("krea2", "blocks.0.mlp"), ("flux2", "single_blocks.0"),
])
def test_model_blocks_match_unsharded_after_checkpoint_loading(size, quantized, family, prefix):
    def build(operations):
        settings = {"operations": operations, "dtype": torch.float32, "device": "cpu"}
        if family == "minimax_h3":
            return MLP(32, 64, **settings)
        if family == "ideogram4":
            return FeedForward(32, 64, 1, None, operation_settings=settings)
        if family == "krea2":
            return SwiGLU(32, 3, bias=True, multiple=32, **settings)
        return SingleStreamBlock(32, 4, mlp_ratio=2, modulation=False, mlp_silu_act=True, **settings)

    torch.manual_seed(91)
    state = {}
    for key, tensor in build(ops.disable_weight_init).state_dict().items():
        if quantized and key.endswith(".weight") and tensor.ndim == 2:
            state[key] = torch.randint(-12, 12, tensor.shape, dtype=torch.int8)
            module = key.removesuffix(".weight")
            state[module + ".weight_scale"] = torch.full((tensor.shape[0], 1), 0.01)
            state[module + ".comfy_quant"] = metadata("int8_tensorwise", convrot=True, convrot_groupsize=16)
        else:
            state[key] = torch.randn_like(tensor) * 0.1
    base = ops.mixed_precision_ops(compute_dtype=torch.float32) if quantized else ops.disable_weight_init
    reference = build(base)
    reference.load_state_dict(dict(state))
    inputs = torch.randn(1, 3, 32)
    args = (inputs,)
    if family == "flux2":
        args += (ModulationOut(torch.zeros(1, 1, 32), torch.zeros(1, 1, 32), torch.ones(1, 1, 32)), None)
    expected = reference(*args)
    collective = Collectives(size)
    ranks = []
    for rank in range(size):
        decorator = full_domain_tensor_parallel_operations if quantized else tensor_parallel_operations
        local = build(decorator(base, TensorParallelConfig(Rank(collective, rank))))
        prefixed = {prefix + "." + key: value for key, value in state.items()}
        sharded = shard_tensor_parallel_state_dict(prefixed, family, rank, size, full_domain_rows=quantized)
        local.load_state_dict({key.removeprefix(prefix + "."): value for key, value in sharded.items()})
        ranks.append(local)
    with ThreadPoolExecutor(size) as pool:
        outputs = list(pool.map(lambda model: model(*args), ranks))
    for output in outputs:
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("format", list(quant_ops.QUANT_ALGOS))
@pytest.mark.parametrize("family,row", [
    ("minimax_h3", "blocks.0.attn.out_proj"), ("flux2", "double_blocks.0.img_attn.proj"),
    ("krea2", "blocks.0.attn.wo"), ("ideogram4", "layers.0.attention.o"),
])
def test_every_quantized_layout_reconstructs_the_original_weight(size, format, family, row):
    torch.manual_seed(74)
    values = checkpoint(format, 128)
    values["weight"] = torch.randint(0, 100, values["weight"].shape).to(quant_ops.QUANT_ALGOS[format]["storage_t"])
    for name, tensor in list(values.items()):
        if name not in ("weight", "comfy_quant"):
            values[name] = torch.rand(tensor.shape) * 0.1 + 0.1
    if format == "nvfp4":
        values["weight_scale"] = values["weight_scale"].to(torch.float8_e4m3fn)
    if format == "mxfp8":
        values["weight_scale"] = values["weight_scale"].to(torch.float8_e8m0fnu)
    if format in ("asym_w4a8_int8", "w6a8_int8"):
        values["weight_s_rel"] = values["weight_s_rel"].to(torch.float8_e4m3fn)
    base = ops.mixed_precision_ops(compute_dtype=torch.float32)
    reference = base.Linear(256, 128)
    reference.load_state_dict(dict(values))
    expected = reference.weight.dequantize()
    assert torch.isfinite(expected).all()
    prefixed = {row + "." + key: value for key, value in values.items()}
    for rank in range(size):
        state = shard_tensor_parallel_state_dict(prefixed, family, rank, size, full_domain_rows=True)
        module = base.Linear(256, 128 // size)
        module.load_state_dict({key.removeprefix(row + "."): value for key, value in state.items()})
        torch.testing.assert_close(module.weight.dequantize(), expected.chunk(size)[rank])
