from __future__ import annotations

from collections.abc import Mapping
import re

import torch


def _section_rows(tensor: torch.Tensor, section_sizes: tuple[int, ...], rank: int, size: int) -> torch.Tensor:
    rows = []
    offset = 0
    for section_size in section_sizes:
        rank_size = section_size // size
        rows.append(tensor.narrow(0, offset + rank * rank_size, rank_size))
        offset += section_size
    return torch.cat(rows).clone()


def _columns(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    rank_size = tensor.shape[1] // size
    return tensor.narrow(1, rank * rank_size, rank_size).clone()


def shard_minimax_h3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    rank: int,
    size: int,
    *,
    full_domain_rows: bool = False,
) -> dict[str, torch.Tensor]:
    """Return one Megatron-style MiniMax shard without retaining full-weight views."""
    if size < 2:
        return dict(state_dict)

    sharded = {}
    for key, tensor in state_dict.items():
        if key.endswith(".attn.qkv_proj.weight") or key.endswith(".attn.qkv_proj.weight_scale"):
            sharded[key] = _section_rows(tensor, (tensor.shape[0] // 3,) * 3, rank, size)
        elif key.endswith(".mlp.fc1.weight") or key.endswith(".mlp.fc1.weight_scale"):
            sharded[key] = _section_rows(tensor, (tensor.shape[0] // 2,) * 2, rank, size)
        elif key.endswith((
            ".attn.out_proj.weight",
            ".attn.out_proj.weight_scale",
            ".attn.out_proj.bias",
            ".mlp.fc2.weight",
            ".mlp.fc2.weight_scale",
            ".mlp.fc2.bias",
        )):
            if full_domain_rows:
                sharded[key] = _section_rows(tensor, (tensor.shape[0],), rank, size)
            elif key.endswith(".weight"):
                sharded[key] = _columns(tensor, rank, size)
            else:
                sharded[key] = tensor
        else:
            sharded[key] = tensor
    return sharded


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


def _module_key(key: str) -> str | None:
    for suffix in (".weight", ".weight_scale", ".bias"):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _column_sections(model_family: str, module: str, weight: torch.Tensor) -> tuple[int, ...] | None:
    if model_family == "krea2" and _KREA2_COLUMN.search(module):
        return (weight.shape[0],)
    if model_family == "ideogram4":
        if _IDEOGRAM4_QKV.search(module):
            return (weight.shape[0] // 3,) * 3
        if _IDEOGRAM4_COLUMN.search(module):
            return (weight.shape[0],)
    if model_family == "flux2":
        if _FLUX2_QKV.search(module):
            return (weight.shape[0] // 3,) * 3
        if _FLUX2_MLP_IN.search(module):
            return (weight.shape[0] // 2,) * 2
        if _FLUX2_SINGLE_IN.search(module):
            hidden = weight.shape[1]
            mlp = (weight.shape[0] - 3 * hidden) // 2
            return (hidden, hidden, hidden, mlp, mlp)
    return None


def _is_row_parallel(model_family: str, module: str) -> bool:
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
) -> dict[str, torch.Tensor]:
    if model_family == "minimax_h3":
        return shard_minimax_h3_state_dict(state_dict, rank, size)
    if size < 2:
        return dict(state_dict)

    weights = {
        key.removesuffix(".weight"): tensor
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and tensor.ndim == 2
    }
    sharded = {}
    for key, tensor in state_dict.items():
        module = _module_key(key)
        weight = weights.get(module)
        sections = None if weight is None else _column_sections(model_family, module, weight)
        if sections is not None and tensor.ndim > 0 and tensor.shape[0] == weight.shape[0]:
            sharded[key] = _section_rows(tensor, sections, rank, size)
        elif module is not None and _is_row_parallel(model_family, module) and key.endswith(".weight"):
            sharded[key] = _columns(tensor, rank, size)
        else:
            sharded[key] = tensor
    return sharded
