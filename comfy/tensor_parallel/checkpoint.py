from __future__ import annotations

from collections.abc import Mapping
import json
import re

import torch
from comfy_kitchen.backends.eager.svdquant import (
    _tile_packed_lora_up_to_natural,
    _tile_packed_scales_to_natural,
    _tile_packed_weight_to_row_major,
)

from .. import quant_ops


def _section_rows(tensor, section_sizes, rank, size, axis=0):
    if sum(section_sizes) != tensor.shape[axis] or any(section % size for section in section_sizes):
        raise ValueError(f"Sections {section_sizes} of shape {tuple(tensor.shape)} cannot divide {size} ranks")
    pieces = []
    offset = 0
    for section in section_sizes:
        width = section // size
        pieces.append(tensor.narrow(axis, offset + rank * width, width))
        offset += section
    return torch.cat(pieces, dim=axis).clone()


def shard_minimax_h3_state_dict(state_dict, rank, size, *, full_domain_rows=False):
    return shard_tensor_parallel_state_dict(
        state_dict, "minimax_h3", rank, size, full_domain_rows=full_domain_rows,
    )


def quantization_format(state, module):
    metadata = state.get(f"{module}.comfy_quant")
    return None if metadata is None else json.loads(metadata.numpy().tobytes())["format"]


def output_rows(weight, quant_format):
    if quant_format == "svdquant_w4a4":
        layout = quant_ops.get_layout_class(quant_ops.QUANT_ALGOS[quant_format]["comfy_tensor_layout"])
        return layout.get_out_features_from_storage(weight)
    return weight.shape[0]


def output_parameter_axis(parameter, shape, rows, quant_format):
    if parameter in ("weight_scale", "weight_zeros") and quant_format in ("svdquant_w4a4", "awq_w4a16"):
        return 1
    if parameter in ("weight", "bias", "weight_scale", "weight_s_rel", "weight_s_channel", "weight_proj_up"):
        if shape and shape[0] == rows:
            return 0
    return None


def shard_output_parameter(parameter, tensor, weight, quant_format, sections, rank, size):
    """Slice logical output channels, preserving input-side and global quantization data."""
    rows = sum(sections)
    if parameter == "weight_scale" and quant_format in ("nvfp4", "mxfp8"):
        group = quant_ops.QUANT_ALGOS[quant_format]["group_size"]
        columns = weight.shape[-1] * (2 if quant_format == "nvfp4" else 1) // group
        scales = quant_ops.ck.from_blocked(tensor, rows, columns)
        local = _section_rows(scales, sections, rank, size)
        return quant_ops.ck.to_blocked(local, flatten=tensor.ndim == 1)
    if (quant_format == "svdquant_w4a4" and len(weight.shape) == 4
            and parameter in ("weight", "weight_scale", "weight_proj_up") and tensor.ndim >= 3):
        block_rows = rows // weight.shape[0]
        if any(section % (size * block_rows) for section in sections):
            unpack = {
                "weight": _tile_packed_weight_to_row_major,
                "weight_scale": _tile_packed_scales_to_natural,
                "weight_proj_up": _tile_packed_lora_up_to_natural,
            }[parameter]
            return _section_rows(unpack(tensor), sections, rank, size, axis=1 if parameter == "weight_scale" else 0)
        return _section_rows(tensor, tuple(section // block_rows for section in sections), rank, size)
    axis = output_parameter_axis(parameter, tensor.shape, rows, quant_format)
    if axis is not None:
        return _section_rows(tensor, sections, rank, size, axis=axis)
    return tensor


_KREA2_COLUMN = re.compile(
    r"(?:^|\.)(?:blocks\.\d+|txtfusion\.(?:layerwise|refiner)_blocks\.\d+)\."
    r"(?:attn\.(?:wq|wk|wv|gate)|mlp\.(?:gate|up))$"
)
_KREA2_ROW = re.compile(
    r"(?:^|\.)(?:blocks\.\d+|txtfusion\.(?:layerwise|refiner)_blocks\.\d+)\."
    r"(?:attn\.wo|mlp\.down)$"
)
_IDEOGRAM4_QKV = re.compile(r"(?:^|\.)layers\.\d+\.attention\.qkv$")
_IDEOGRAM4_COLUMN = re.compile(r"(?:^|\.)layers\.\d+\.feed_forward\.(?:w1|w3)$")
_IDEOGRAM4_ROW = re.compile(r"(?:^|\.)layers\.\d+\.(?:attention\.o|feed_forward\.w2)$")
_FLUX2_QKV = re.compile(r"(?:^|\.)double_blocks\.\d+\.(?:img|txt)_attn\.qkv$")
_FLUX2_MLP_IN = re.compile(r"(?:^|\.)double_blocks\.\d+\.(?:img|txt)_mlp\.0$")
_FLUX2_ROW = re.compile(
    r"(?:^|\.)(?:double_blocks\.\d+\.(?:img|txt)_(?:attn\.proj|mlp\.2)|"
    r"single_blocks\.\d+\.linear2)$"
)
_FLUX2_SINGLE_IN = re.compile(r"(?:^|\.)single_blocks\.\d+\.linear1$")
_FLUX2_SINGLE_OUT = re.compile(r"(?:^|\.)single_blocks\.\d+\.linear2$")


def _column_sections(model_family, module, rows, weights, formats):
    if model_family == "minimax_h3":
        if module.endswith(".attn.qkv_proj"):
            return (rows // 3,) * 3
        if module.endswith(".mlp.fc1"):
            return (rows // 2,) * 2
    if model_family == "krea2" and _KREA2_COLUMN.search(module):
        return (rows,)
    if model_family == "ideogram4":
        if _IDEOGRAM4_QKV.search(module):
            return (rows // 3,) * 3
        if _IDEOGRAM4_COLUMN.search(module):
            return (rows,)
    if model_family == "flux2":
        if _FLUX2_QKV.search(module):
            return (rows // 3,) * 3
        if _FLUX2_MLP_IN.search(module):
            return (rows // 2,) * 2
        if _FLUX2_SINGLE_IN.search(module):
            output = module.removesuffix("linear1") + "linear2"
            hidden = output_rows(weights[output], formats[output])
            mlp = (rows - 3 * hidden) // 2
            return (hidden, hidden, hidden, mlp, mlp)
    return None


def _is_row_parallel(model_family: str, module: str) -> bool:
    if model_family == "minimax_h3":
        return module.endswith((".attn.out_proj", ".mlp.fc2"))
    if model_family == "krea2":
        return _KREA2_ROW.search(module) is not None
    if model_family == "ideogram4":
        return _IDEOGRAM4_ROW.search(module) is not None
    if model_family == "flux2":
        return _FLUX2_ROW.search(module) is not None
    return False


def shard_tensor_parallel_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    model_family: str,
    rank: int,
    size: int,
    *,
    full_domain_rows: bool = False,
) -> dict[str, torch.Tensor]:
    if size < 2:
        return dict(state_dict)
    weights = {
        key.removesuffix(".weight"): tensor
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and tensor.ndim >= 2
    }
    formats = {module: quantization_format(state_dict, module) for module in weights}
    sharded = {}
    for key, tensor in state_dict.items():
        module, _, parameter = key.rpartition(".")
        weight = weights.get(module)
        if weight is None:
            sharded[key] = tensor
            continue
        quant_format = formats[module]
        rows = output_rows(weight, quant_format)
        sections = _column_sections(model_family, module, rows, weights, formats)
        row_parallel = _is_row_parallel(model_family, module)
        if row_parallel and full_domain_rows:
            sections = (rows,)
        if sections is not None:
            sharded[key] = shard_output_parameter(parameter, tensor, weight, quant_format, sections, rank, size)
        elif row_parallel and parameter == "weight":
            if quant_format is not None:
                raise ValueError("Quantized tensor parallel projections require full_domain_rows=True")
            columns = (weight.shape[1],)
            if model_family == "flux2" and _FLUX2_SINGLE_OUT.search(module):
                columns = (rows, weight.shape[1] - rows)
            sharded[key] = _section_rows(tensor, columns, rank, size, axis=1)
        elif row_parallel and parameter == "bias" and rank != 0:
            continue
        else:
            sharded[key] = tensor
    return sharded
