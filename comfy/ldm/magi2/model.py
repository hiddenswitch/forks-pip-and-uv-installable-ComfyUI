"""SandAI MAGI-2 Preview: a joint video and audio diffusion transformer.

Every token carries 4C channels split into four hyper-connection (mHC) streams. Each layer mixes
the streams into one branch input per attention / MLP branch and writes the branch output back
with a doubly stochastic stream mix. Attention has one learned sink logit per head and a
per-head sigmoid output gate, layers 0, 1 and the last two keep separate weights per modality,
and the middle layers route twelve 256-wide heads of every token through 256 experts each.

Tokens are laid out by modality: the generated video and reference-image patches, then the
audio frames, then the text tokens and one special token per reference image. Attention is
bidirectional and every other operation is per token, so this grouping only replaces the
reference's per-modality permutation; positions come from each token's coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

import comfy_kitchen
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

from ... import model_management, model_prefetch, patcher_extension
from ...pipeline_parallel import PipelineIntermediateTensors, PipelineMissingLayer, PipelineStageConfig
from ...pipeline_parallel.types import pack_pipeline_value, prepare_model_parallel_value, unpack_pipeline_value
from ...quant_ops import QuantizedTensor
from ...tensor_parallel.operations import column_parallel_linear, local_size, row_parallel_linear
from ..modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

TIME_CHANNELS = 64
SINK_PAD = 8
MHC_CHUNK = 8192
MOE_ROW_CHUNK = 1 << 16


if _HAS_TRITON:
    SWIGLU7_ALPHA = tl.constexpr(1.702)
    SWIGLU7_LIMIT = tl.constexpr(7.0)
    SWIGLU7_BIAS = tl.constexpr(1.0)

    @triton.jit
    def _swiglu7_fwd(gate, up, out_dtype: tl.constexpr):
        gate_clamped = tl.minimum(gate, SWIGLU7_LIMIT)
        up_clamped = tl.maximum(tl.minimum(up, SWIGLU7_LIMIT), -SWIGLU7_LIMIT)
        sig = tl.sigmoid(SWIGLU7_ALPHA * gate_clamped)
        swish = gate_clamped * sig
        return (swish * (up_clamped + SWIGLU7_BIAS)).to(out_dtype)

    @triton.jit
    def _binary_search_expert_id(
        cu_expert_tile_counts_ptr,
        tile_id,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
    ):
        lo = 0
        hi = NUM_EXPERTS
        for _ in tl.static_range(0, LOG2_NUM_EXPERTS + 1):
            mid = (lo + hi + 1) // 2
            below = tl.load(cu_expert_tile_counts_ptr + mid) <= tile_id
            lo = tl.where(below, mid, lo)
            hi = tl.where(below, hi, mid - 1)
        return lo

    @triton.jit
    def _split_fp32_to_bf16_pair_kernel(
        x_ptr, hi_ptr, lo_ptr, n_elements, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        hi = x.to(tl.bfloat16)
        lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
        tl.store(hi_ptr + offs, hi, mask=mask)
        tl.store(lo_ptr + offs, lo, mask=mask)

    @triton.jit
    def _mh_moe_fwd_kernel(
        x_ptr,
        wg_ptr,
        wu_ptr,
        wd_ptr,
        y_ptr,
        gather_ids_ptr,
        probs_ptr,
        expert_offsets_ptr,
        cu_expert_tile_counts_ptr,
        stride_x_s,
        stride_x_h,
        stride_x_dh,
        stride_wg_ne,
        stride_wg_dh,
        stride_wg_de,
        stride_wu_ne,
        stride_wu_dh,
        stride_wu_de,
        stride_wd_ne,
        stride_wd_de,
        stride_wd_dh,
        stride_y_s,
        stride_y_h,
        stride_y_dh,
        D_HEAD: tl.constexpr,
        D_EXPERT: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        BLOCK_DE: tl.constexpr,
        HIDDEN_DTYPE: tl.constexpr = tl.bfloat16,
        FP32_WEIGHTS: tl.constexpr = False,
        DETERMINISTIC: tl.constexpr = False,
    ):
        tile_id = tl.program_id(0)
        total_tiles = tl.load(cu_expert_tile_counts_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return

        expert_id = _binary_search_expert_id(
            cu_expert_tile_counts_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS
        )
        expert_id_i64 = expert_id.to(tl.int64)
        head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)

        tile_in_expert = tile_id - tl.load(cu_expert_tile_counts_ptr + expert_id)
        token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start

        block_dh_offs = tl.arange(0, BLOCK_DH)
        block_de_offs = tl.arange(0, BLOCK_DE)
        block_t_offs = tl.arange(0, BLOCK_T)
        dh_offs = tl.arange(0, D_HEAD)

        block_token_offs = token_start + block_t_offs
        block_token_mask = block_t_offs < n

        gather_ids = tl.load(
            gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0
        )
        probs = tl.load(
            probs_ptr + block_token_offs, mask=block_token_mask, other=0.0
        ).to(tl.float32)
        x_offs = gather_ids * stride_x_s + head_id * stride_x_h

        y_acc = tl.zeros([BLOCK_T, D_HEAD], dtype=tl.float32)
        for dexpert_start in tl.range(0, D_EXPERT, BLOCK_DE):
            dexpert_block_offs = dexpert_start + block_de_offs
            gate_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)
            up_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)
            for dhead_start in tl.static_range(0, D_HEAD, BLOCK_DH):
                dhead_block_offs = dhead_start + block_dh_offs
                x_block = tl.load(
                    x_ptr + x_offs[:, None] + dhead_block_offs[None, :] * stride_x_dh,
                    mask=block_token_mask[:, None],
                    other=0.0,
                )
                w_gate = tl.load(
                    wg_ptr
                    + expert_id_i64 * stride_wg_ne
                    + dhead_block_offs[:, None] * stride_wg_dh
                    + dexpert_block_offs[None, :] * stride_wg_de
                )
                w_up = tl.load(
                    wu_ptr
                    + expert_id_i64 * stride_wu_ne
                    + dhead_block_offs[:, None] * stride_wu_dh
                    + dexpert_block_offs[None, :] * stride_wu_de
                )
                if FP32_WEIGHTS:
                    x_bf16 = x_block.to(tl.bfloat16)
                    wg_hi = w_gate.to(tl.bfloat16)
                    wg_lo = (w_gate.to(tl.float32) - wg_hi.to(tl.float32)).to(tl.bfloat16)
                    wu_hi = w_up.to(tl.bfloat16)
                    wu_lo = (w_up.to(tl.float32) - wu_hi.to(tl.float32)).to(tl.bfloat16)
                    gate_acc += tl.dot(x_bf16, wg_hi) + tl.dot(x_bf16, wg_lo)
                    up_acc += tl.dot(x_bf16, wu_hi) + tl.dot(x_bf16, wu_lo)
                else:
                    gate_acc += tl.dot(x_block, w_gate)
                    up_acc += tl.dot(x_block, w_up)

            hidden = _swiglu7_fwd(gate_acc, up_acc, HIDDEN_DTYPE)
            w_down = tl.load(
                wd_ptr
                + expert_id_i64 * stride_wd_ne
                + dexpert_block_offs[:, None] * stride_wd_de
                + dh_offs[None, :] * stride_wd_dh
            )
            if FP32_WEIGHTS:
                h_bf16 = hidden.to(tl.bfloat16)
                wd_hi = w_down.to(tl.bfloat16)
                wd_lo = (w_down.to(tl.float32) - wd_hi.to(tl.float32)).to(tl.bfloat16)
                y_acc += tl.dot(h_bf16, wd_hi) + tl.dot(h_bf16, wd_lo)
            else:
                y_acc += tl.dot(hidden, w_down.to(HIDDEN_DTYPE))

        y_acc = y_acc * probs[:, None]
        if DETERMINISTIC:
            y_ptrs = (
                y_ptr
                + block_token_offs[:, None] * stride_y_s
                + dh_offs[None, :] * stride_y_dh
            )
            tl.store(
                y_ptrs,
                y_acc.to(y_ptr.dtype.element_ty),
                mask=block_token_mask[:, None],
            )
        else:
            y_offs = gather_ids * stride_y_s + head_id * stride_y_h
            y_ptrs = y_ptr + y_offs[:, None] + dh_offs[None, :] * stride_y_dh
            tl.atomic_add(
                y_ptrs,
                y_acc.to(y_ptr.dtype.element_ty),
                mask=block_token_mask[:, None],
            )

    @triton.jit
    def _mh_moe_gate_up_swiglu7_int8_kernel(
        x_hi_ptr,
        x_lo_ptr,
        qg_ptr,
        sg_ptr,
        qu_ptr,
        su_ptr,
        hidden_ptr,
        gather_ids_ptr,
        expert_offsets_ptr,
        cu_expert_tile_counts_ptr,
        stride_x_s,
        stride_x_h,
        stride_x_dh,
        stride_qg_ne,
        stride_qg_dh,
        stride_qg_de,
        stride_qu_ne,
        stride_qu_dh,
        stride_qu_de,
        stride_h_t,
        stride_h_de,
        D_HEAD: tl.constexpr,
        D_EXPERT: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        BLOCK_DE: tl.constexpr,
        SG_SCALAR: tl.constexpr,
        SU_SCALAR: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        de_block = tl.program_id(1)
        total_tiles = tl.load(cu_expert_tile_counts_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return

        expert_id = _binary_search_expert_id(
            cu_expert_tile_counts_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS
        )
        expert_id_i64 = expert_id.to(tl.int64)
        head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)

        tile_in_expert = tile_id - tl.load(cu_expert_tile_counts_ptr + expert_id)
        token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start

        block_dh_offs = tl.arange(0, BLOCK_DH)
        block_de_offs = de_block * BLOCK_DE + tl.arange(0, BLOCK_DE)
        de_mask = block_de_offs < D_EXPERT
        block_t_offs = tl.arange(0, BLOCK_T)

        block_token_offs = token_start + block_t_offs
        block_token_mask = block_t_offs < n

        gather_ids = tl.load(
            gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0
        )
        x_offs = gather_ids * stride_x_s + head_id * stride_x_h

        gate_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)

        for dhead_start in tl.static_range(0, D_HEAD, BLOCK_DH):
            dhead_block_offs = dhead_start + block_dh_offs
            offs_x = x_offs[:, None] + dhead_block_offs[None, :] * stride_x_dh
            x_hi = tl.load(x_hi_ptr + offs_x, mask=block_token_mask[:, None], other=0.0)
            x_lo = tl.load(x_lo_ptr + offs_x, mask=block_token_mask[:, None], other=0.0)
            qg = tl.load(
                qg_ptr
                + expert_id_i64 * stride_qg_ne
                + dhead_block_offs[:, None] * stride_qg_dh
                + block_de_offs[None, :] * stride_qg_de,
                mask=de_mask[None, :],
                other=0,
            ).to(tl.bfloat16)
            qu = tl.load(
                qu_ptr
                + expert_id_i64 * stride_qu_ne
                + dhead_block_offs[:, None] * stride_qu_dh
                + block_de_offs[None, :] * stride_qu_de,
                mask=de_mask[None, :],
                other=0,
            ).to(tl.bfloat16)
            gate_acc += tl.dot(x_hi, qg) + tl.dot(x_lo, qg)
            up_acc += tl.dot(x_hi, qu) + tl.dot(x_lo, qu)

        sg = (
            tl.load(sg_ptr).to(tl.float32)
            if SG_SCALAR
            else tl.load(
                sg_ptr + expert_id_i64 * D_EXPERT + block_de_offs,
                mask=de_mask,
                other=0.0,
            ).to(tl.float32)
        )
        su = (
            tl.load(su_ptr).to(tl.float32)
            if SU_SCALAR
            else tl.load(
                su_ptr + expert_id_i64 * D_EXPERT + block_de_offs,
                mask=de_mask,
                other=0.0,
            ).to(tl.float32)
        )
        gate_acc = gate_acc * sg
        up_acc = up_acc * su

        hidden = _swiglu7_fwd(gate_acc, up_acc, hidden_ptr.dtype.element_ty)
        tl.store(
            hidden_ptr
            + block_token_offs[:, None] * stride_h_t
            + block_de_offs[None, :] * stride_h_de,
            hidden,
            mask=block_token_mask[:, None] & de_mask[None, :],
        )

    @triton.jit
    def _mh_moe_down_int8_kernel(
        h_hi_ptr,
        h_lo_ptr,
        qd_ptr,
        sd_ptr,
        y_ptr,
        gather_ids_ptr,
        probs_ptr,
        expert_offsets_ptr,
        cu_expert_tile_counts_ptr,
        stride_h_t,
        stride_h_de,
        stride_qd_ne,
        stride_qd_de,
        stride_qd_dh,
        stride_y_s,
        stride_y_h,
        stride_y_dh,
        D_HEAD: tl.constexpr,
        D_EXPERT: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_DE: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        SD_SCALAR: tl.constexpr,
        DETERMINISTIC: tl.constexpr = False,
    ):
        tile_id = tl.program_id(0)
        dh_block = tl.program_id(1)
        total_tiles = tl.load(cu_expert_tile_counts_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return

        expert_id = _binary_search_expert_id(
            cu_expert_tile_counts_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS
        )
        expert_id_i64 = expert_id.to(tl.int64)
        head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)

        tile_in_expert = tile_id - tl.load(cu_expert_tile_counts_ptr + expert_id)
        token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start

        block_de_offs = tl.arange(0, BLOCK_DE)
        block_t_offs = tl.arange(0, BLOCK_T)
        dh_offs = dh_block * BLOCK_DH + tl.arange(0, BLOCK_DH)
        dh_mask = dh_offs < D_HEAD

        block_token_offs = token_start + block_t_offs
        block_token_mask = block_t_offs < n

        probs = tl.load(
            probs_ptr + block_token_offs, mask=block_token_mask, other=0.0
        ).to(tl.float32)

        y_acc = tl.zeros([BLOCK_T, BLOCK_DH], dtype=tl.float32)
        for dexpert_start in tl.range(0, D_EXPERT, BLOCK_DE):
            dexpert_block_offs = dexpert_start + block_de_offs
            offs_h = (
                block_token_offs[:, None] * stride_h_t
                + dexpert_block_offs[None, :] * stride_h_de
            )
            h_hi = tl.load(h_hi_ptr + offs_h, mask=block_token_mask[:, None], other=0.0)
            h_lo = tl.load(h_lo_ptr + offs_h, mask=block_token_mask[:, None], other=0.0)
            qd = tl.load(
                qd_ptr
                + expert_id_i64 * stride_qd_ne
                + dexpert_block_offs[:, None] * stride_qd_de
                + dh_offs[None, :] * stride_qd_dh,
                mask=dh_mask[None, :],
                other=0,
            ).to(tl.bfloat16)
            y_acc += tl.dot(h_hi, qd) + tl.dot(h_lo, qd)

        sd = (
            tl.load(sd_ptr).to(tl.float32)
            if SD_SCALAR
            else tl.load(
                sd_ptr + expert_id_i64 * D_HEAD + dh_offs, mask=dh_mask, other=0.0
            ).to(tl.float32)
        )
        y_acc = y_acc * (sd * probs[:, None])

        if DETERMINISTIC:
            y_ptrs = (
                y_ptr
                + block_token_offs[:, None] * stride_y_s
                + dh_offs[None, :] * stride_y_dh
            )
            tl.store(
                y_ptrs,
                y_acc.to(y_ptr.dtype.element_ty),
                mask=block_token_mask[:, None] & dh_mask[None, :],
            )
        else:
            gather_ids = tl.load(
                gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0
            )
            y_offs = gather_ids * stride_y_s + head_id * stride_y_h
            y_ptrs = y_ptr + y_offs[:, None] + dh_offs[None, :] * stride_y_dh
            tl.atomic_add(
                y_ptrs,
                y_acc.to(y_ptr.dtype.element_ty),
                mask=block_token_mask[:, None] & dh_mask[None, :],
            )

    @triton.jit
    def _dequant_rowwise_fp32_kernel(
        q_ptr, s_ptr, out_ptr, n_cols, s_is_scalar: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        row = tl.program_id(0)
        scale = tl.load(s_ptr).to(tl.float32) if s_is_scalar else tl.load(s_ptr + row).to(tl.float32)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < n_cols
        q = tl.load(q_ptr + row * n_cols + cols, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + row * n_cols + cols, q * scale, mask=mask)

    @triton.jit
    def _dequant_pair_rowwise_fp32_kernel(
        q0_ptr, s0_ptr, q1_ptr, s1_ptr, out_ptr,
        out_features, in_features,
        s0_scalar: tl.constexpr, s1_scalar: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        expert = pid // (2 * out_features)
        rem = pid % (2 * out_features)
        is_second = rem >= out_features
        row_in_expert = rem - tl.where(is_second, out_features, 0)
        src_row = expert * out_features + row_in_expert
        cols = tl.arange(0, BLOCK_C)
        mask = cols < in_features
        if is_second:
            scale = tl.load(s1_ptr).to(tl.float32) if s1_scalar else tl.load(s1_ptr + src_row).to(tl.float32)
            q = tl.load(q1_ptr + src_row * in_features + cols, mask=mask, other=0).to(tl.float32)
        else:
            scale = tl.load(s0_ptr).to(tl.float32) if s0_scalar else tl.load(s0_ptr + src_row).to(tl.float32)
            q = tl.load(q0_ptr + src_row * in_features + cols, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + pid * in_features + cols, q * scale, mask=mask)

    @triton.jit
    def _swiglu7_interleaved_kernel(
        x_ptr, out_ptr, stride_x, stride_o, out_cols, BLOCK_C: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < out_cols
        base = row * stride_x + cols * 2
        g = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        g = tl.minimum(g, 7.0)
        u = tl.minimum(tl.maximum(u, -7.0), 7.0) + 1.0
        sig = 1.0 / (1.0 + tl.exp(-1.702 * g))
        res = (g * sig * u).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_o + cols, res, mask=mask)

    @triton.jit
    def _swiglu7_pair_kernel(
        g_ptr, u_ptr, out_ptr, stride_g, stride_u, stride_o, n_cols, BLOCK_C: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < n_cols
        g = tl.load(g_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.minimum(g, 7.0)
        u = tl.minimum(tl.maximum(u, -7.0), 7.0) + 1.0
        sig = 1.0 / (1.0 + tl.exp(-1.702 * g))
        res = (g * sig * u).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_o + cols, res, mask=mask)


def swiglu7(x):
    """GPT-OSS clamped SwiGLU over interleaved [gate, linear] pairs, evaluated in fp32."""
    dtype = x.dtype
    if _HAS_TRITON and x.is_cuda and x.is_contiguous() and x.shape[-1] > 0 and x.shape[-1] <= 16384:
        out_cols = x.shape[-1] // 2
        flat = x.view(-1, x.shape[-1])
        out = torch.empty((flat.shape[0], out_cols), device=x.device, dtype=dtype)
        block_c = max(16, triton.next_power_of_2(out_cols))
        _swiglu7_interleaved_kernel[(flat.shape[0],)](
            flat, out, flat.stride(0), out.stride(0), out_cols, BLOCK_C=block_c, num_warps=8
        )
        return out.view(*x.shape[:-1], out_cols)
    x = x.float()
    gate = x[..., ::2].clamp(max=7.0)
    linear = x[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(dtype)


def swiglu7_pair(gate, up, out_dtype=None):
    """SwiGLU7 over separate fp32 gate and up projections; defaults to fp32 output."""
    target_dtype = gate.dtype if out_dtype is None else out_dtype
    if _HAS_TRITON and gate.is_cuda and up.is_cuda and gate.ndim == 2 and gate.shape[-1] > 0 and gate.shape[-1] <= 8192:
        n_rows, n_cols = gate.shape
        out = torch.empty((n_rows, n_cols), device=gate.device, dtype=target_dtype)
        block_c = max(16, triton.next_power_of_2(n_cols))
        _swiglu7_pair_kernel[(n_rows,)](
            gate, up, out, gate.stride(0), up.stride(0), out.stride(0), n_cols, BLOCK_C=block_c, num_warps=8
        )
        return out
    gate = gate.clamp_(max=7.0)
    up = up.clamp_(min=-7.0, max=7.0).add_(1.0)
    res = gate.mul_(torch.sigmoid(1.702 * gate)).mul_(up)
    return res if out_dtype is None else res.to(out_dtype)


def sinusoidal_embedding(t, dim=TIME_CHANNELS):
    """The reference embeds the normalized timestep scaled back by 1000, cos half first."""
    position = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = position.unsqueeze(-1) * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


def grid_coords(shape, ref_shape, offset=(0, 0, 0), device=None):
    """[T*H*W, 9] rows of (t, h, w, T, H, W, ref_T, ref_H, ref_W)."""
    axes = [torch.arange(size, dtype=torch.float32, device=device) + start for size, start in zip(shape, offset)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    meta = torch.tensor((*shape, *ref_shape), dtype=torch.float32, device=device)
    return torch.cat((grid, meta.expand(grid.shape[0], -1)), dim=-1)


def fourier_rope(coords, bands):
    """Element-wise Fourier RoPE: rotation table [N, 3 * bands, 2, 2] for the split-half layout."""
    xyz, sizes, refs = coords[:, :3], coords[:, 3:6], coords[:, 6:9]
    scales = (refs - 1) / (sizes - 1)
    scales = torch.where((refs == 1) & (sizes == 1), torch.ones_like(scales), scales)
    centers = (sizes - 1) / 2
    centers[:, 0] = 0
    projection = ((xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * bands).flatten(1)
    cos, sin = projection.cos(), projection.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))


def modality_linear_local(module, x, groups):
    """Apply one linear (or per-modality linears) without the row-parallel output collective."""
    if not isinstance(module, nn.ModuleList):
        return module.forward_local(x) if hasattr(module, "forward_local") else module(x)
    parts = x.split(groups, dim=1)
    outputs = [
        (linear.forward_local(part) if hasattr(linear, "forward_local") else linear(part))
        if part.shape[1] else None
        for linear, part in zip(module, parts)
    ]
    width = next(output.shape[-1] for output in outputs if output is not None)
    return torch.cat(
        [part.new_empty(part.shape[:-1] + (width,)) if output is None else output for output, part in zip(outputs, parts)],
        dim=1,
    )


def modality_linear(module, x, groups):
    """Apply one linear, or one linear per modality group of the token axis."""
    if not isinstance(module, nn.ModuleList):
        return module(x)
    first = module[0]
    if hasattr(first, "start_input_gather") and hasattr(first, "forward_local") and hasattr(first, "reduce_output"):
        full_x = first.start_input_gather(x)()
        return first.reduce_output(modality_linear_local(module, full_x, groups))
    parts = x.split(groups, dim=1)
    outputs = [linear(part) if part.shape[1] else None for linear, part in zip(module, parts)]
    # a parallel linear's output width is not its local out_features, so empty groups take a computed group's width
    width = next(output.shape[-1] for output in outputs if output is not None)
    return torch.cat([part.new_empty(part.shape[:-1] + (width,)) if output is None else output for output, part in zip(outputs, parts)], dim=1)


def make_modality_linear(num_modality, factory):
    if num_modality == 1:
        return factory()
    return nn.ModuleList(factory() for _ in range(num_modality))


class Float32Linear(nn.Module):
    """The reference's fp32 adapters: kept fp32 under every weight format."""

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device)) if bias else None

    def forward(self, x):
        weight = model_management.cast_to(self.weight, torch.float32, x.device)
        bias = None if self.bias is None else model_management.cast_to(self.bias, torch.float32, x.device)
        return F.linear(x, weight, bias)


class MultiModalityRMSNorm(nn.Module):
    """fp32 RMSNorm with a (weight + 1) scale per modality group."""

    def __init__(self, dim, num_modality=1, out_dtype=None, eps=1e-6, device=None):
        super().__init__()
        self.dim = dim
        self.num_modality = num_modality
        self.out_dtype = out_dtype
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(num_modality * dim, dtype=torch.float32, device=device))

    def scales(self, device):
        return (model_management.cast_to(self.weight, torch.float32, device) + 1.0).chunk(self.num_modality)

    def forward(self, x, groups=None):
        out_dtype = self.out_dtype or x.dtype
        scales = self.scales(x.device)
        x = x.float()
        if self.num_modality == 1:
            return F.rms_norm(x, (self.dim,), scales[0], self.eps).to(out_dtype)
        return torch.cat([F.rms_norm(part, (self.dim,), scale, self.eps) for part, scale in zip(x.split(groups, dim=1), scales)], dim=1).to(out_dtype)


def qk_norm_rope(q, k, q_norm, k_norm, freqs, groups):
    """fp32 per-head RMSNorm and partial split-half RoPE, rounded once to the input dtype."""
    dtype = q.dtype
    rot_dim = freqs.shape[-3] * 2
    q_scales, k_scales = q_norm.scales(q.device), k_norm.scales(k.device)
    q_parts, k_parts = [], []
    splits = groups if q_norm.num_modality > 1 else (q.shape[1],)
    for index, (qp, kp, fp) in enumerate(zip(q.split(splits, dim=1), k.split(splits, dim=1), freqs.split(splits, dim=1))):
        if not qp.shape[1]:
            q_parts.append(qp)
            k_parts.append(kp)
            continue
        qo, ko = comfy_kitchen.rms_rope_split_half(qp.float(), kp.float(), fp, q_scales[index], k_scales[index], epsilon=q_norm.eps, rot_dim=rot_dim)
        q_parts.append(qo.to(dtype))
        k_parts.append(ko.to(dtype))
    return torch.cat(q_parts, dim=1), torch.cat(k_parts, dim=1)


def sink_attention(q, k, v, sinks, comfy_attention, transformer_options):
    """Softmax over [scores, sink] with the sink column dropped, through the selected attention.

    q, k, v are [B, H, N, D]. The sink is one extra key whose score equals the head's sink logit for
    every query and whose value is zero: queries gain three constant 1.0 components, real keys three
    zeros, and the sink key carries the logit (divided by the softmax scale) split into three
    compute-dtype parts, so the fp32-accumulated product reproduces the fp32 logit.
    """
    batch, heads, tokens, head_dim = q.shape
    scale = head_dim ** -0.5
    logits = sinks.float() / scale
    hi = logits.to(q.dtype)
    mid = (logits - hi.float()).to(q.dtype)
    lo = (logits - hi.float() - mid.float()).to(q.dtype)
    pad = SINK_PAD - 3
    q = torch.cat((q, q.new_ones(batch, heads, tokens, 3), q.new_zeros(batch, heads, tokens, pad)), dim=-1)
    sink_key = torch.cat((k.new_zeros(heads, head_dim), torch.stack((hi, mid, lo), dim=-1), k.new_zeros(heads, pad)), dim=-1)
    k = torch.cat((torch.cat((k, k.new_zeros(batch, heads, tokens, SINK_PAD)), dim=-1), sink_key.view(1, heads, 1, -1).expand(batch, -1, -1, -1)), dim=2)
    v = torch.cat((v, v.new_zeros(batch, heads, tokens, SINK_PAD)), dim=-1)
    v = torch.cat((v, v.new_zeros(batch, heads, 1, head_dim + SINK_PAD)), dim=2)
    out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), heads,
                              preferred_attention=comfy_attention, skip_reshape=True, skip_output_reshape=True,
                              scale=scale, transformer_options=transformer_options)
    return out[..., :head_dim]


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim, num_modality, dtype=None, device=None, operations=None):
        super().__init__()
        self.head_dim = head_dim
        self.heads_q = local_size(operations, num_heads_q, "MAGI-2 attention heads")
        self.heads_kv = local_size(operations, num_heads_kv, "MAGI-2 key/value heads")
        q_size, kv_size = num_heads_q * head_dim, num_heads_kv * head_dim
        self.comfy_attention = ComfyAttention()
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality, device=device)
        self.linear_g = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, num_heads_q, bias=False, device=device, dtype=dtype))
        self.linear_qkv = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, q_size + 2 * kv_size, bias=False, section_sizes=(q_size, kv_size, kv_size), device=device, dtype=dtype))
        self.linear_proj = make_modality_linear(num_modality, lambda: row_parallel_linear(
            operations, q_size, hidden_size, bias=False, device=device, dtype=dtype))
        self.sinks = nn.Parameter(torch.empty(1, self.heads_q, dtype=torch.float32, device=device))
        self.q_norm = MultiModalityRMSNorm(head_dim, num_modality, out_dtype=torch.float32, device=device)
        self.k_norm = MultiModalityRMSNorm(head_dim, num_modality, out_dtype=torch.float32, device=device)

    def forward(self, x, freqs, groups, transformer_options):
        batch, tokens, _ = x.shape
        x = self.pre_norm(x, groups)
        gates = modality_linear(self.linear_g, x, groups)
        qkv = modality_linear(self.linear_qkv, x, groups)
        del x
        q, k, v = qkv.split((self.heads_q * self.head_dim, self.heads_kv * self.head_dim, self.heads_kv * self.head_dim), dim=-1)
        q = q.view(batch, tokens, self.heads_q, self.head_dim)
        k = k.view(batch, tokens, self.heads_kv, self.head_dim)
        q, k = qk_norm_rope(q, k, self.q_norm, self.k_norm, freqs, groups)
        v = v.reshape(batch, tokens, self.heads_kv, self.head_dim)
        sinks = model_management.cast_to(self.sinks, torch.float32, q.device)[0]
        out = sink_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), sinks, self.comfy_attention, transformer_options)
        out = out.transpose(1, 2) * torch.sigmoid(gates).unsqueeze(-1)
        return modality_linear(self.linear_proj, out.reshape(batch, tokens, -1), groups)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_modality, dtype=None, device=None, operations=None):
        super().__init__()
        local_size(operations, intermediate_size, "MAGI-2 MLP width")
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality, device=device)
        self.up_gate_proj = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype))
        self.down_proj = make_modality_linear(num_modality, lambda: row_parallel_linear(
            operations, intermediate_size, hidden_size, bias=False, device=device, dtype=dtype))

    def forward(self, x, groups):
        x = self.pre_norm(x, groups)
        x = swiglu7(modality_linear(self.up_gate_proj, x, groups))
        return modality_linear(self.down_proj, x, groups)


def _dequantize_rows(rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if _HAS_TRITON and rows.is_cuda and rows.is_contiguous() and scale.is_contiguous() and rows.shape[-1] <= 4096:
        out = torch.empty(rows.shape, device=rows.device, dtype=torch.float32)
        n_cols = rows.shape[-1]
        n_rows = rows.numel() // n_cols
        block_c = max(16, triton.next_power_of_2(n_cols))
        _dequant_rowwise_fp32_kernel[(n_rows,)](
            rows, scale, out, n_cols, s_is_scalar=(scale.numel() == 1), BLOCK_C=block_c, num_warps=4
        )
        return out
    return rows.float().mul_(scale.float())


def _expert_rows(bank, weight, start, count):
    """Experts [start, start + count) of a bank as a [count, out, in] fp32 tensor and its ConvRot group size.

    A ConvRot INT8 bank stays in its rotated basis: the caller rotates the activations instead, which
    is exact because the regular Hadamard used by ConvRot is its own inverse.
    """
    if not isinstance(weight, QuantizedTensor):
        return weight.view(-1, bank.out_features, bank.in_features)[start:start + count].float(), 0
    qdata, scale = weight.layout_cls.get_plain_tensors(weight)
    rows = qdata.view(-1, bank.out_features, bank.in_features)[start:start + count]
    if scale.numel() > 1:
        scale = scale.view(-1, bank.out_features, 1)[start:start + count]
    params = weight.params
    group = params.convrot_groupsize if getattr(params, "convrot", False) else 0
    return _dequantize_rows(rows, scale), group


def _expert_rows_pair(gate_entry, up_entry, start, count):
    """Fused [count, 2 * out, in] fp32 expert bank when gate and up share quantization and ConvRot group."""
    gate_bank, gate_weight = gate_entry
    up_bank, up_weight = up_entry
    out_f, in_f = gate_bank.out_features, gate_bank.in_features
    if gate_bank.out_features != up_bank.out_features or gate_bank.in_features != up_bank.in_features:
        return _expert_rows(gate_bank, gate_weight, start, count), _expert_rows(up_bank, up_weight, start, count)
    if not isinstance(gate_weight, QuantizedTensor) and not isinstance(up_weight, QuantizedTensor):
        r0 = gate_weight.view(-1, out_f, in_f)[start:start + count]
        r1 = up_weight.view(-1, out_f, in_f)[start:start + count]
        return (torch.cat((r0, r1), dim=1).float(), 0), None
    if isinstance(gate_weight, QuantizedTensor) and isinstance(up_weight, QuantizedTensor):
        g_params, u_params = gate_weight.params, up_weight.params
        g_group = g_params.convrot_groupsize if getattr(g_params, "convrot", False) else 0
        u_group = u_params.convrot_groupsize if getattr(u_params, "convrot", False) else 0
        if g_group == u_group:
            q0, s0 = gate_weight.layout_cls.get_plain_tensors(gate_weight)
            q1, s1 = up_weight.layout_cls.get_plain_tensors(up_weight)
            r0 = q0.view(-1, out_f, in_f)[start:start + count]
            r1 = q1.view(-1, out_f, in_f)[start:start + count]
            if s0.numel() > 1:
                s0 = s0.view(-1, out_f, 1)[start:start + count]
            if s1.numel() > 1:
                s1 = s1.view(-1, out_f, 1)[start:start + count]
            if _HAS_TRITON and r0.is_cuda and r0.is_contiguous() and r1.is_contiguous() and s0.is_contiguous() and s1.is_contiguous() and in_f <= 4096:
                out = torch.empty((count, 2 * out_f, in_f), device=r0.device, dtype=torch.float32)
                block_c = max(16, triton.next_power_of_2(in_f))
                _dequant_pair_rowwise_fp32_kernel[(count * 2 * out_f,)](
                    r0, s0, r1, s1, out, out_f, in_f,
                    s0_scalar=(s0.numel() == 1), s1_scalar=(s1.numel() == 1),
                    BLOCK_C=block_c, num_warps=4,
                )
                return (out, g_group), None
    return _expert_rows(gate_bank, gate_weight, start, count), _expert_rows(up_bank, up_weight, start, count)


def _prepare_grouped_input(x, group):
    x = x.float()
    if group:
        x = _rotate_activation(x, _build_hadamard(group, device=x.device, dtype=torch.float32), group)
    return x


def _grouped_projection(x, bank, offsets):
    """fp32 output of a grouped GEMM over compute-dtype rows.

    SandAI's fused kernel accumulates bf16 x bf16 products in fp32 and never rounds the gate, up or
    down outputs. Products of bf16 operands are exact in fp32, so an fp32 GEMM of the upcast operands
    is that accumulation.
    """
    weight, group = bank
    return torch._grouped_mm(_prepare_grouped_input(x, group), weight.transpose(1, 2), offs=offsets)


def _grouped_gate_up(x, gate_bank, up_bank, offsets):
    """Project gate and up from the same routed token rows, rotating activations only once when shared."""
    if up_bank is None:
        gu_weight, group = gate_bank
        gu = torch._grouped_mm(_prepare_grouped_input(x, group), gu_weight.transpose(1, 2), offs=offsets)
        return gu.chunk(2, dim=-1)
    gate_weight, gate_group = gate_bank
    up_weight, up_group = up_bank
    gate_in = _prepare_grouped_input(x, gate_group)
    up_in = gate_in if up_group == gate_group else _prepare_grouped_input(x, up_group)
    return (
        torch._grouped_mm(gate_in, gate_weight.transpose(1, 2), offs=offsets),
        torch._grouped_mm(up_in, up_weight.transpose(1, 2), offs=offsets),
    )


def _rotate_and_split_bf16(x: torch.Tensor, group: int, out_hi: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if not group:
        if x.dtype == torch.bfloat16:
            return (x if out_hi is None else out_hi.copy_(x)), torch.zeros_like(x)
        flat = x.reshape(-1)
        hi = torch.empty_like(flat, dtype=torch.bfloat16) if out_hi is None else out_hi.reshape(-1)
        lo = torch.empty_like(flat, dtype=torch.bfloat16)
        n = flat.numel()
        _split_fp32_to_bf16_pair_kernel[(triton.cdiv(n, 1024),)](
            flat, hi, lo, n, BLOCK=1024, num_warps=4
        )
        return hi.view_as(x), lo.view_as(x)
    h = _build_hadamard(group, device=x.device, dtype=torch.float32)
    flat = x.reshape(-1, group)
    hi = torch.empty_like(flat, dtype=torch.bfloat16) if out_hi is None else out_hi.reshape(-1, group)
    lo = torch.empty_like(flat, dtype=torch.bfloat16)
    chunk = max(1, (16 << 20) // (group * 4))
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        rot = flat[start:end].float() @ h
        n = rot.numel()
        _split_fp32_to_bf16_pair_kernel[(triton.cdiv(n, 1024),)](
            rot, hi[start:end], lo[start:end], n, BLOCK=1024, num_warps=4
        )
        del rot
    return hi.view_as(x), lo.view_as(x)


def _triton_block_dim(size: int) -> int:
    for block in (64, 32, 16):
        if size % block == 0:
            return block
    return 0


def _sort_routed_tokens(
    probs_slice: torch.Tensor,
    indices_slice: torch.Tensor,
    tokens_count: int,
    num_experts: int,
    block_t: int,
):
    h_cur, _, top_k = indices_slice.shape
    e_count = h_cur * num_experts
    device = indices_slice.device
    head_offset = torch.arange(h_cur, device=device).view(h_cur, 1, 1) * num_experts
    global_indices = (indices_slice + head_offset).reshape(-1)
    flat_probs = probs_slice.reshape(-1)
    flat_token_ids = (
        torch.arange(tokens_count, device=device)
        .view(1, tokens_count, 1)
        .expand(h_cur, tokens_count, top_k)
        .reshape(-1)
    )
    sorted_order = global_indices.argsort(stable=True)
    gather_ids = flat_token_ids[sorted_order].to(torch.int32)
    probs_sorted = flat_probs[sorted_order].float()
    expert_counts = torch.bincount(global_indices, minlength=e_count)
    expert_offsets = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int64), expert_counts.cumsum(0))
    )
    tile_counts = (expert_counts + block_t - 1) // block_t
    cu_tiles = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            tile_counts.cumsum(0, dtype=torch.int32),
        )
    )
    return gather_ids, probs_sorted, expert_counts, expert_offsets, cu_tiles


def _scatter_routed_tokens(
    y_sorted: torch.Tensor,
    gather_ids: torch.Tensor,
    expert_counts: torch.Tensor,
    tokens_count: int,
    h_cur: int,
    num_experts: int,
) -> torch.Tensor:
    total_routed, _, d_head = y_sorted.shape
    e_count = h_cur * num_experts
    head_ids = torch.repeat_interleave(
        torch.arange(e_count, device=y_sorted.device) // num_experts, expert_counts
    )
    scatter_idx = (
        (gather_ids.long() * h_cur + head_ids.long())
        .unsqueeze(-1)
        .expand(total_routed, d_head)
    )
    y = torch.zeros(tokens_count * h_cur, d_head, device=y_sorted.device, dtype=y_sorted.dtype)
    y.scatter_add_(0, scatter_idx, y_sorted.view(total_routed, d_head))
    return y.view(tokens_count, h_cur, d_head)


def _flash_mh_moe_unquantized_triton(
    x: torch.Tensor,
    all_probs: torch.Tensor,
    all_indices: torch.Tensor,
    gate_entry,
    up_entry,
    down_entry,
    num_experts: int,
) -> torch.Tensor:
    tokens_count, heads, d_head = x.shape
    top_k = all_indices.shape[-1]
    gate_bank, gate_weight = gate_entry
    _, up_weight = up_entry
    _, down_weight = down_entry
    d_expert = gate_bank.out_features

    wg = gate_weight.view(heads * num_experts, d_expert, d_head).transpose(1, 2)
    wu = up_weight.view(heads * num_experts, d_expert, d_head).transpose(1, 2)
    wd = down_weight.view(heads * num_experts, d_head, d_expert).transpose(1, 2)

    block_dh = _triton_block_dim(d_head)
    block_de = min(32, _triton_block_dim(d_expert))
    block_t = 64
    fp32_weights = wg.dtype == torch.float32 or x.dtype == torch.float32
    hidden_tl_dtype = tl.float16 if x.dtype == torch.float16 else tl.bfloat16
    head_chunk = max(1, MOE_ROW_CHUNK // max(1, tokens_count * top_k))

    out = torch.empty_like(x)
    for h_start in range(0, heads, head_chunk):
        h_end = min(h_start + head_chunk, heads)
        h_cur = h_end - h_start
        e_start = h_start * num_experts
        e_count = h_cur * num_experts
        gather_ids, probs_sorted, expert_counts, expert_offsets, cu_tiles = _sort_routed_tokens(
            all_probs[h_start:h_end], all_indices[h_start:h_end], tokens_count, num_experts, block_t
        )
        total_routed = gather_ids.size(0)
        if total_routed == 0:
            out[:, h_start:h_end].zero_()
            continue

        log2_e = max(1, math.ceil(math.log2(e_count + 1)))
        grid = ((total_routed + block_t - 1) // block_t + e_count,)
        x_slice = x[:, h_start:h_end]
        wg_slice = wg[e_start:e_start + e_count]
        wu_slice = wu[e_start:e_start + e_count]
        wd_slice = wd[e_start:e_start + e_count]
        y_sorted = torch.empty(total_routed, 1, d_head, device=x.device, dtype=x.dtype)

        _mh_moe_fwd_kernel[grid](
            x_slice,
            wg_slice,
            wu_slice,
            wd_slice,
            y_sorted,
            gather_ids,
            probs_sorted,
            expert_offsets,
            cu_tiles,
            x_slice.stride(0),
            x_slice.stride(1),
            x_slice.stride(2),
            wg_slice.stride(0),
            wg_slice.stride(1),
            wg_slice.stride(2),
            wu_slice.stride(0),
            wu_slice.stride(1),
            wu_slice.stride(2),
            wd_slice.stride(0),
            wd_slice.stride(1),
            wd_slice.stride(2),
            y_sorted.stride(0),
            y_sorted.stride(1),
            y_sorted.stride(2),
            D_HEAD=d_head,
            D_EXPERT=d_expert,
            NUM_HEADS=h_cur,
            NUM_EXPERTS=e_count,
            LOG2_NUM_EXPERTS=log2_e,
            BLOCK_T=block_t,
            BLOCK_DH=block_dh,
            BLOCK_DE=block_de,
            HIDDEN_DTYPE=hidden_tl_dtype,
            FP32_WEIGHTS=fp32_weights,
            DETERMINISTIC=True,
            num_stages=2,
            num_warps=4,
        )
        out[:, h_start:h_end] = _scatter_routed_tokens(
            y_sorted, gather_ids, expert_counts, tokens_count, h_cur, num_experts
        )
        del y_sorted
    return out


def _quant_bank_views(bank, weight, total_experts: int):
    qdata, scale = weight.layout_cls.get_plain_tensors(weight)
    q_3d = qdata.view(total_experts, bank.out_features, bank.in_features).transpose(1, 2)
    s_2d = scale.reshape(-1) if scale.numel() == 1 else scale.reshape(total_experts, bank.out_features)
    params = weight.params
    group = params.convrot_groupsize if getattr(params, "convrot", False) else 0
    return q_3d, s_2d.contiguous(), group, (scale.numel() == 1)


def _flash_mh_moe_int8_triton(
    x: torch.Tensor,
    all_probs: torch.Tensor,
    all_indices: torch.Tensor,
    banks,
    num_experts: int,
) -> torch.Tensor:
    tokens_count, heads, d_head = x.shape
    top_k = all_indices.shape[-1]
    total_experts = heads * num_experts
    d_expert = banks[0][0].out_features

    qg_t, sg, group_g, sg_scalar = _quant_bank_views(banks[0][0], banks[0][1], total_experts)
    qu_t, su, _, su_scalar = _quant_bank_views(banks[1][0], banks[1][1], total_experts)
    qd_t, sd, group_d, sd_scalar = _quant_bank_views(banks[2][0], banks[2][1], total_experts)

    block_dh = _triton_block_dim(d_head)
    block_de = _triton_block_dim(d_expert)
    block_t = 64
    hidden_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16
    head_chunk = max(1, MOE_ROW_CHUNK // max(1, tokens_count * top_k))

    out = torch.empty_like(x)
    for h_start in range(0, heads, head_chunk):
        h_end = min(h_start + head_chunk, heads)
        h_cur = h_end - h_start
        e_start = h_start * num_experts
        e_count = h_cur * num_experts
        gather_ids, probs_sorted, expert_counts, expert_offsets, cu_tiles = _sort_routed_tokens(
            all_probs[h_start:h_end], all_indices[h_start:h_end], tokens_count, num_experts, block_t
        )
        total_routed = gather_ids.size(0)
        if total_routed == 0:
            out[:, h_start:h_end].zero_()
            continue

        x_slice = x[:, h_start:h_end]
        x_hi, x_lo = _rotate_and_split_bf16(x_slice, group_g)
        log2_e = max(1, math.ceil(math.log2(e_count + 1)))
        grid_1d = (total_routed + block_t - 1) // block_t + e_count

        hidden = torch.empty(total_routed, d_expert, device=x.device, dtype=hidden_dtype)
        qg_slice = qg_t[e_start:e_start + e_count]
        qu_slice = qu_t[e_start:e_start + e_count]
        qd_slice = qd_t[e_start:e_start + e_count]
        sg_slice = sg if sg_scalar else sg[e_start:e_start + e_count]
        su_slice = su if su_scalar else su[e_start:e_start + e_count]
        sd_slice = sd if sd_scalar else sd[e_start:e_start + e_count]

        _mh_moe_gate_up_swiglu7_int8_kernel[(grid_1d, triton.cdiv(d_expert, block_de))](
            x_hi,
            x_lo,
            qg_slice,
            sg_slice,
            qu_slice,
            su_slice,
            hidden,
            gather_ids,
            expert_offsets,
            cu_tiles,
            x_hi.stride(0),
            x_hi.stride(1),
            x_hi.stride(2),
            qg_slice.stride(0),
            qg_slice.stride(1),
            qg_slice.stride(2),
            qu_slice.stride(0),
            qu_slice.stride(1),
            qu_slice.stride(2),
            hidden.stride(0),
            hidden.stride(1),
            D_HEAD=d_head,
            D_EXPERT=d_expert,
            NUM_HEADS=h_cur,
            NUM_EXPERTS=e_count,
            LOG2_NUM_EXPERTS=log2_e,
            BLOCK_T=block_t,
            BLOCK_DH=block_dh,
            BLOCK_DE=block_de,
            SG_SCALAR=sg_scalar,
            SU_SCALAR=su_scalar,
            num_stages=2,
            num_warps=4,
        )
        del x_hi, x_lo
        h_hi, h_lo = _rotate_and_split_bf16(
            hidden, group_d, out_hi=hidden if hidden.dtype == torch.bfloat16 else None
        )
        del hidden
        y_sorted = torch.empty(total_routed, 1, d_head, device=x.device, dtype=x.dtype)
        _mh_moe_down_int8_kernel[(grid_1d, triton.cdiv(d_head, block_dh))](
            h_hi,
            h_lo,
            qd_slice,
            sd_slice,
            y_sorted,
            gather_ids,
            probs_sorted,
            expert_offsets,
            cu_tiles,
            h_hi.stride(0),
            h_hi.stride(1),
            qd_slice.stride(0),
            qd_slice.stride(1),
            qd_slice.stride(2),
            y_sorted.stride(0),
            y_sorted.stride(1),
            y_sorted.stride(2),
            D_HEAD=d_head,
            D_EXPERT=d_expert,
            NUM_HEADS=h_cur,
            NUM_EXPERTS=e_count,
            LOG2_NUM_EXPERTS=log2_e,
            BLOCK_T=block_t,
            BLOCK_DE=block_de,
            BLOCK_DH=block_dh,
            SD_SCALAR=sd_scalar,
            DETERMINISTIC=True,
            num_stages=2,
            num_warps=4,
        )
        del h_hi, h_lo
        out[:, h_start:h_end] = _scatter_routed_tokens(
            y_sorted, gather_ids, expert_counts, tokens_count, h_cur, num_experts
        )
        del y_sorted
    return out


class MultiHeadMoE(nn.Module):
    """Twelve routed heads: each head's 256-wide slice of a token picks top_k of its own experts."""

    def __init__(self, hidden_size, num_heads, num_experts, top_k, expert_intermediate_size, route_scale, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = local_size(operations, num_heads, "MAGI-2 MoE heads")
        self.head_dim = hidden_size // num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        experts = self.heads * num_experts
        self.router = nn.Module()
        self.router.weight = nn.Parameter(torch.empty(experts, self.head_dim, dtype=torch.float32, device=device))
        self.router.bias = nn.Parameter(torch.empty(experts, dtype=torch.float32, device=device))
        self.gate_proj = operations.MoEExperts(experts, self.head_dim, expert_intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = operations.MoEExperts(experts, self.head_dim, expert_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.MoEExperts(experts, expert_intermediate_size, self.head_dim, bias=False, device=device, dtype=dtype)

    def route(self, x_head, head):
        """Sigmoid scores; the selection bias picks the experts but does not weight them."""
        start = head * self.num_experts
        weight = model_management.cast_to(self.router.weight[start:start + self.num_experts], torch.float32, x_head.device)
        bias = model_management.cast_to(self.router.bias[start:start + self.num_experts], torch.float32, x_head.device)
        scores = torch.sigmoid(x_head.float() @ weight.t())
        indices = torch.topk(scores + bias, self.top_k, dim=-1).indices
        probs = F.normalize(scores.gather(-1, indices), p=1, dim=-1, eps=1e-12) * self.route_scale
        return probs, indices

    def route_all(self, x):
        """Batched sigmoid routing across all local MoE heads [tokens, heads, head_dim]."""
        weight = model_management.cast_to(self.router.weight, torch.float32, x.device).view(
            self.heads, self.num_experts, self.head_dim
        )
        bias = model_management.cast_to(self.router.bias, torch.float32, x.device).view(
            self.heads, 1, self.num_experts
        )
        scores = torch.sigmoid(torch.bmm(x.float().transpose(0, 1), weight.transpose(1, 2)))
        indices = torch.topk(scores + bias, self.top_k, dim=-1).indices
        probs = F.normalize(scores.gather(-1, indices), p=1, dim=-1, eps=1e-12) * self.route_scale
        return probs, indices

    def forward(self, x):
        batch, tokens, _ = x.shape
        x = x.reshape(batch * tokens, self.heads, self.head_dim)
        all_probs, all_indices = self.route_all(x)
        # a quantized bank that the ops dequantize up front (full precision matmul) is dequantized in fp32, not rounded to bf16
        resident = lambda bank: x[:0].float() if isinstance(bank.weight, QuantizedTensor) else x
        with (self.gate_proj.bank_resident(resident(self.gate_proj)), self.up_proj.bank_resident(resident(self.up_proj)),
              self.down_proj.bank_resident(resident(self.down_proj))):
            banks = tuple((bank, bank._resident_bank[0]) for bank in (self.gate_proj, self.up_proj, self.down_proj))
            if (
                _HAS_TRITON
                and x.is_cuda
                and _triton_block_dim(self.head_dim) > 0
                and _triton_block_dim(self.gate_proj.out_features) > 0
                and self.gate_proj.out_features == self.up_proj.out_features
            ):
                if all(not isinstance(weight, QuantizedTensor) for _, weight in banks):
                    out = _flash_mh_moe_unquantized_triton(
                        x, all_probs, all_indices, banks[0], banks[1], banks[2], self.num_experts
                    )
                    return out.reshape(batch, tokens, self.heads * self.head_dim)
                if all(isinstance(weight, QuantizedTensor) for _, weight in banks):
                    g_group = banks[0][1].params.convrot_groupsize if getattr(banks[0][1].params, "convrot", False) else 0
                    u_group = banks[1][1].params.convrot_groupsize if getattr(banks[1][1].params, "convrot", False) else 0
                    if g_group == u_group:
                        out = _flash_mh_moe_int8_triton(x, all_probs, all_indices, banks, self.num_experts)
                        return out.reshape(batch, tokens, self.heads * self.head_dim)

            out = torch.empty_like(x)
            flat_indices = all_indices.flatten(1)
            all_orders = flat_indices.argsort(dim=-1, stable=True)
            for head in range(self.heads):
                x_head = x[:, head].contiguous()
                indices = flat_indices[head]
                order = all_orders[head]
                token_ids = order // self.top_k
                offsets = torch.bincount(indices, minlength=self.num_experts).cumsum(0).to(torch.int32)
                rows = x_head.index_select(0, token_ids)
                weights = all_probs[head].flatten()[order].unsqueeze(-1)
                start = head * self.num_experts
                if rows.shape[0] <= MOE_ROW_CHUNK:
                    gate_bank, up_bank = _expert_rows_pair(banks[0], banks[1], start, self.num_experts)
                    hidden = swiglu7_pair(*_grouped_gate_up(rows, gate_bank, up_bank, offsets), out_dtype=x.dtype)
                    del rows, gate_bank, up_bank
                    down_bank = _expert_rows(banks[2][0], banks[2][1], start, self.num_experts)
                    y = _grouped_projection(hidden, down_bank, offsets).mul_(weights).to(x.dtype)
                    del hidden, down_bank
                else:
                    gate_bank, up_bank = _expert_rows_pair(banks[0], banks[1], start, self.num_experts)
                    down_bank = _expert_rows(banks[2][0], banks[2][1], start, self.num_experts)
                    y = torch.empty_like(rows)
                    for chunk in range(0, rows.shape[0], MOE_ROW_CHUNK):
                        end = min(chunk + MOE_ROW_CHUNK, rows.shape[0])
                        chunk_offsets = offsets.clamp(min=chunk, max=end) - chunk
                        part = rows[chunk:end]
                        # the kernel rounds SwiGLU7's fp32 output once, to the down projection's dtype
                        hidden = swiglu7_pair(*_grouped_gate_up(part, gate_bank, up_bank, chunk_offsets), out_dtype=x.dtype)
                        y[chunk:end] = _grouped_projection(hidden, down_bank, chunk_offsets).mul_(weights[chunk:end])
                        del hidden
                    del rows, gate_bank, up_bank, down_bank
                head_out = torch.zeros_like(x_head)
                head_out.index_add_(0, token_ids, y)
                out[:, head] = head_out
        return out.reshape(batch, tokens, self.heads * self.head_dim)


class MoELayer(nn.Module):
    def __init__(self, hidden_size, moe_heads, num_experts, top_k, expert_intermediate_size, shared_expert_intermediate_size,
                 modality_shared_expert_intermediate_size, route_scale, dtype=None, device=None, operations=None):
        super().__init__()
        local_size(operations, shared_expert_intermediate_size, "MAGI-2 shared expert width")
        local_size(operations, modality_shared_expert_intermediate_size, "MAGI-2 modality expert width")
        self.pre_norm = MultiModalityRMSNorm(hidden_size, 3, device=device)
        self.split_linear = column_parallel_linear(operations, hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.moe_mlp = MultiHeadMoE(hidden_size, moe_heads, num_experts, top_k, expert_intermediate_size, route_scale, dtype=dtype, device=device, operations=operations)
        self.merge_linear = row_parallel_linear(operations, hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.shared_expert_fc1 = column_parallel_linear(operations, hidden_size, 2 * shared_expert_intermediate_size, bias=False, device=device, dtype=dtype)
        self.shared_expert_fc2 = row_parallel_linear(operations, shared_expert_intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.modality_specific_shared_expert_fc1 = make_modality_linear(3, lambda: column_parallel_linear(
            operations, hidden_size, 2 * modality_shared_expert_intermediate_size, bias=False, device=device, dtype=dtype))
        self.modality_specific_shared_expert_fc2 = make_modality_linear(3, lambda: row_parallel_linear(
            operations, modality_shared_expert_intermediate_size, hidden_size, bias=False, device=device, dtype=dtype))

    def _start_shared_input_gather(self, shared_act, specific_act):
        specific_first = self.modality_specific_shared_expert_fc2[0]
        if (
            shared_act.shape == specific_act.shape
            and getattr(self.shared_expert_fc2, "tensor_parallel_input_sections", None) == (shared_act.shape[-1],)
            and getattr(specific_first, "tensor_parallel_input_sections", None) == (specific_act.shape[-1],)
        ):
            finish_both = self.shared_expert_fc2.start_input_gather(torch.stack((shared_act, specific_act), dim=0))
            return lambda: finish_both().unbind(0)
        finish_shared = self.shared_expert_fc2.start_input_gather(shared_act)
        finish_specific = specific_first.start_input_gather(specific_act)
        return lambda: (finish_shared(), finish_specific())

    def forward(self, x, groups):
        normed = self.pre_norm(x, groups)
        moe_in = self.split_linear(normed)
        if not (hasattr(self.merge_linear, "start_input_gather") and hasattr(self.merge_linear, "forward_local") and hasattr(self.merge_linear, "reduce_output")):
            routed = self.merge_linear(self.moe_mlp(moe_in))
            shared = self.shared_expert_fc2(swiglu7(self.shared_expert_fc1(normed)))
            specific = modality_linear(self.modality_specific_shared_expert_fc2,
                                       swiglu7(modality_linear(self.modality_specific_shared_expert_fc1, normed, groups)), groups)
            return routed + (shared + specific)

        shared_act = swiglu7(self.shared_expert_fc1(normed))
        specific_act = swiglu7(modality_linear(self.modality_specific_shared_expert_fc1, normed, groups))
        finish_shared_gather = self._start_shared_input_gather(shared_act, specific_act)
        del normed, shared_act, specific_act

        moe_out = self.moe_mlp(moe_in)
        del moe_in
        finish_routed_gather = self.merge_linear.start_input_gather(moe_out)
        del moe_out

        shared_full, specific_full = finish_shared_gather()
        shared_local = self.shared_expert_fc2.forward_local(shared_full)
        del shared_full
        specific_local = modality_linear_local(self.modality_specific_shared_expert_fc2, specific_full, groups)
        del specific_full

        routed_local = self.merge_linear.forward_local(finish_routed_gather())
        return self.merge_linear.reduce_output(routed_local + (shared_local + specific_local))


def sinkhorn(logits, iterations, eps):
    matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
    return matrix


class TransformerLayer(nn.Module):
    def __init__(self, layer_index, hidden_size, head_dim, num_query_groups, multimodal_layers, moe_layers, num_streams,
                 intermediate_size, moe_heads, num_experts, top_k, expert_intermediate_size, shared_expert_intermediate_size,
                 modality_shared_expert_intermediate_size, route_scale, sinkhorn_iterations, dtype=None, device=None, operations=None):
        super().__init__()
        num_modality = 3 if layer_index in multimodal_layers else 1
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matmul_scale = 1.0 / math.sqrt(float(num_streams * hidden_size))
        self.attention = Attention(hidden_size, hidden_size // head_dim, num_query_groups, head_dim, num_modality, dtype=dtype, device=device, operations=operations)
        if layer_index in moe_layers:
            self.mlp = MoELayer(hidden_size, moe_heads, num_experts, top_k, expert_intermediate_size, shared_expert_intermediate_size,
                                modality_shared_expert_intermediate_size, route_scale, dtype=dtype, device=device, operations=operations)
        else:
            self.mlp = MLP(hidden_size, intermediate_size, num_modality, dtype=dtype, device=device, operations=operations)
        fused = 2 * num_streams + num_streams * num_streams
        for branch in ("attn", "mlp"):
            for coefficient in ("pre", "post", "res"):
                self.register_parameter(f"mhc_alpha_{coefficient}_{branch}", nn.Parameter(torch.empty(1, dtype=torch.float32, device=device)))
            self.register_parameter(f"mhc_bias_pre_{branch}", nn.Parameter(torch.empty(num_streams, dtype=torch.float32, device=device)))
            self.register_parameter(f"mhc_bias_post_{branch}", nn.Parameter(torch.empty(num_streams, dtype=torch.float32, device=device)))
            self.register_parameter(f"mhc_bias_res_{branch}", nn.Parameter(torch.empty(num_streams, num_streams, dtype=torch.float32, device=device)))
            self.register_parameter(f"mhc_phi_fused_{branch}", nn.Parameter(torch.empty(num_streams * hidden_size, fused, dtype=torch.float32, device=device)))
        self.mhc_norm = MultiModalityRMSNorm(num_streams * hidden_size, num_modality, out_dtype=torch.float32, device=device)

    def _param(self, name, device):
        return model_management.cast_to(getattr(self, name), torch.float32, device)

    def _logits(self, hidden, branch, groups):
        n = self.num_streams
        fused = self.mhc_norm(hidden, groups) @ self._param(f"mhc_phi_fused_{branch}", hidden.device)
        pre, post, res = fused.split((n, n, n * n), dim=-1)
        return pre, post, res.unflatten(-1, (n, n))

    def _branch_input(self, streams, pre, branch):
        alpha, bias = self._param(f"mhc_alpha_pre_{branch}", pre.device), self._param(f"mhc_bias_pre_{branch}", pre.device)
        weights = torch.sigmoid(alpha * self.matmul_scale * pre + bias).to(streams.dtype)
        return torch.matmul(weights.unsqueeze(-2), streams).squeeze(-2)

    def _connect(self, streams, branch_out, post, res, branch):
        """streams_i <- sum_j res_ij streams_j + post_i branch_out, accumulated in fp32 and rounded once."""
        device = post.device
        post = (2.0 * torch.sigmoid(self._param(f"mhc_alpha_post_{branch}", device) * self.matmul_scale * post
                                    + self._param(f"mhc_bias_post_{branch}", device))).to(streams.dtype)
        res = sinkhorn(self._param(f"mhc_alpha_res_{branch}", device) * self.matmul_scale * res.float()
                       + self._param(f"mhc_bias_res_{branch}", device), self.sinkhorn_iterations, 1e-12).to(streams.dtype)
        out = torch.empty_like(streams)
        for start in range(0, streams.shape[1], MHC_CHUNK):
            end = start + MHC_CHUNK
            mixed = torch.matmul(res[:, start:end].float(), streams[:, start:end].float())
            mixed.addcmul_(post[:, start:end].float().unsqueeze(-1), branch_out[:, start:end].float().unsqueeze(-2))
            out[:, start:end] = mixed
        return out

    def forward(self, x, freqs, groups, transformer_options={}):
        batch, tokens, _ = x.shape
        streams = x.view(batch, tokens, self.num_streams, self.hidden_size)
        pre, post, res = self._logits(x, "attn", groups)
        branch = self.attention(self._branch_input(streams, pre, "attn"), freqs, groups, transformer_options)
        streams = self._connect(streams, branch, post, res, "attn")
        pre, post, res = self._logits(streams.view(batch, tokens, -1), "mlp", groups)
        branch = self.mlp(self._branch_input(streams, pre, "mlp"), groups)
        streams = self._connect(streams, branch, post, res, "mlp")
        return streams.view(batch, tokens, -1)


@dataclass(frozen=True)
class Magi2Layout:
    """Token counts of one packed sample: [video, reference images | audio | text, reference tokens]."""
    video_shape: tuple[int, int, int]
    ref_shapes: tuple[tuple[int, int], ...]
    audio_len: int
    text_len: int

    @property
    def video_tokens(self):
        return math.prod(self.video_shape)

    @property
    def groups(self):
        return (self.video_tokens + sum(h * w for h, w in self.ref_shapes), self.audio_len, self.text_len + len(self.ref_shapes))

    def coords(self, device):
        t, h, w = self.video_shape
        video = [grid_coords((t, h, w), (t, h, w), device=device)]
        specials = []
        for index, (rh, rw) in enumerate(self.ref_shapes):
            # reference images sit past the clip on the time axis, one latent step apart
            offset = t + 2 + index
            video.append(grid_coords((1, rh, rw), (1, rh, rw), (offset, 0, 0), device=device))
            specials.append(torch.tensor([[offset, -1, -1, 1, rh, rw, 1, rh, rw]], dtype=torch.float32, device=device))
        audio = grid_coords((self.audio_len, 1, 1), ((self.audio_len - 1) // 8 + 1, 1, 1), device=device)
        text = grid_coords((self.text_len, 1, 1), (1, 1, 1), (-self.text_len, 0, 0), device=device)
        return torch.cat(video + [audio, text] + specials, dim=0)

    def timesteps(self, t):
        """Per-token normalized time: the noisy video and audio carry t, conditioning tokens carry 0."""
        batch = t.shape[0]
        zeros = lambda n: t.new_zeros(batch, n)
        ref_tokens = self.groups[0] - self.video_tokens
        return torch.cat((t.view(batch, 1).expand(batch, self.video_tokens), zeros(ref_tokens),
                          t.view(batch, 1).expand(batch, self.audio_len), zeros(self.groups[2])), dim=1)


class PreAdapter(nn.Module):
    def __init__(self, width, head_dim, video_in_channels, audio_in_channels, text_in_channels, device=None):
        super().__init__()
        self.video_embedder = Float32Linear(video_in_channels, width, device=device)
        self.text_embedder = Float32Linear(text_in_channels, width, device=device)
        self.audio_embedder = Float32Linear(audio_in_channels, width, device=device)
        self.rope = nn.Module()
        self.rope.register_buffer("bands", torch.empty(head_dim // 8, dtype=torch.float32, device=device))

    def forward(self, video, audio, text, time_embedding):
        width = self.video_embedder.weight.shape[0]
        out = torch.empty(video.shape[0], video.shape[1] + audio.shape[1] + text.shape[1], width, dtype=torch.float32, device=video.device)
        start = 0
        for tokens, embedder in ((video, self.video_embedder), (audio, self.audio_embedder), (text, self.text_embedder)):
            if tokens.shape[1]:
                out[:, start:start + tokens.shape[1]] = embedder(tokens.float())
            start += tokens.shape[1]
        if time_embedding is not None:
            out[..., :time_embedding.shape[-1]] = time_embedding
        return out


class PostAdapter(nn.Module):
    def __init__(self, width, video_in_channels, audio_in_channels, device=None):
        super().__init__()
        self.final_norm_video = MultiModalityRMSNorm(width, device=device)
        self.final_norm_audio = MultiModalityRMSNorm(width, device=device)
        self.final_linear_video = Float32Linear(width, video_in_channels, bias=False, device=device)
        self.final_linear_audio = Float32Linear(width, audio_in_channels, bias=False, device=device)

    def forward(self, video, audio):
        return (self.final_linear_video(self.final_norm_video(video.float())),
                self.final_linear_audio(self.final_norm_audio(audio.float())))


def video_tokens(latent):
    """[B, C, T, H, W] -> [B, T*H*W, C]."""
    return latent.flatten(2).transpose(1, 2)


class Magi2Model(nn.Module):
    """MAGI-2 Preview DiT: [video, audio] velocity for packed video, audio and text tokens."""

    def __init__(self, num_layers=40, hidden_size=3072, head_dim=128, num_query_groups=24, video_in_channels=48,
                 audio_in_channels=64, text_in_channels=5120, intermediate_size=8192, multimodal_layers=(0, 1, 38, 39),
                 moe_layers=tuple(range(2, 38)), num_streams=4, moe_heads=12, num_experts=256, top_k=6,
                 expert_intermediate_size=1280, shared_expert_intermediate_size=1280,
                 modality_shared_expert_intermediate_size=1280, route_scale=4.9, sinkhorn_iterations=20,
                 image_model=None, dtype=None, device=None, operations=None,
                 pipeline_stage: PipelineStageConfig | None = None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.audio_in_channels = audio_in_channels
        self.pipeline_stage = pipeline_stage
        width = num_streams * hidden_size
        is_first = pipeline_stage is None or pipeline_stage.is_first
        is_last = pipeline_stage is None or pipeline_stage.is_last
        self.pre_adapter = PreAdapter(width, head_dim, video_in_channels, audio_in_channels, text_in_channels, device=device) if is_first else PipelineMissingLayer()
        self.block = nn.Module()
        self.block.layers = nn.ModuleList([
            TransformerLayer(index, hidden_size, head_dim, num_query_groups, multimodal_layers, moe_layers, num_streams,
                             intermediate_size, moe_heads, num_experts, top_k, expert_intermediate_size,
                             shared_expert_intermediate_size, modality_shared_expert_intermediate_size, route_scale,
                             sinkhorn_iterations, dtype=dtype, device=device, operations=operations)
            if pipeline_stage is None or pipeline_stage.start_layer <= index < pipeline_stage.end_layer
            else PipelineMissingLayer()
            for index in range(num_layers)])
        self.post_adapter = PostAdapter(width, video_in_channels, audio_in_channels, device=device) if is_last else PipelineMissingLayer()

    def forward(self, x, timestep, context, transformer_options={}, magi2_refs=None, **kwargs):
        if self.pipeline_stage is not None and not self.pipeline_stage.is_first:
            raise RuntimeError("Only the first MAGI-2 pipeline stage accepts model inputs")
        video = x[0] if isinstance(x, (list, tuple)) else x
        compile_allocations = self.pipeline_stage is None and model_prefetch.malloc_graph_enabled(video.device)
        if compile_allocations:
            out = [torch.empty_like(part) for part in x] if isinstance(x, (list, tuple)) else torch.empty_like(x)
            model_prefetch.malloc_graph_begin(video.device)
        graph_out = patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timestep, context, transformer_options, magi2_refs=magi2_refs, **kwargs)
        if not compile_allocations:
            return graph_out
        if isinstance(out, list):
            for target, value in zip(out, graph_out):
                target.copy_(value)
        else:
            out.copy_(graph_out)
        del graph_out
        model_prefetch.malloc_graph_end()
        return out

    def _forward(self, x, timestep, context, transformer_options={}, magi2_refs=None, **kwargs):
        transformer_options = dict(transformer_options)
        video, audio = (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, None)
        batch, device = video.shape[0], video.device
        if audio is None:
            audio = video.new_zeros(batch, self.audio_in_channels, 0)
        refs = magi2_refs or []
        layout = Magi2Layout(tuple(video.shape[2:]), tuple(tuple(ref["latent"].shape[-2:]) for ref in refs), audio.shape[-1], context.shape[1])

        # The reference scheduler hands the model integer timesteps; the solver keeps the float sigmas.
        t = torch.trunc(timestep.to(torch.float64)).float() / 1000.0
        video_rows = torch.cat([video_tokens(video)] + [video_tokens(ref["latent"].to(video)).expand(batch, -1, -1) for ref in refs], dim=1)
        text_rows = torch.cat([context] + [ref["token"].to(context).view(1, 1, -1).expand(batch, -1, -1) for ref in refs], dim=1)
        hidden = self.pre_adapter(video_rows, audio.transpose(1, 2), text_rows, sinusoidal_embedding(layout.timesteps(t)))
        del video_rows, text_rows
        hidden = hidden.to(self.dtype)
        freqs = fourier_rope(layout.coords(device), model_management.cast_to(self.pre_adapter.rope.bands, torch.float32, device)).unsqueeze(0).unsqueeze(2)

        start = 0 if self.pipeline_stage is None else self.pipeline_stage.start_layer
        end = self.num_layers if self.pipeline_stage is None else self.pipeline_stage.end_layer
        hidden = self._run_layers(hidden, freqs, layout.groups, transformer_options, start, end)
        if self.pipeline_stage is not None and not self.pipeline_stage.is_last:
            tensors = {"hidden_states": hidden, "freqs": freqs}
            metadata = {
                "layout": pack_pipeline_value((layout.video_shape, layout.ref_shapes, layout.audio_len, layout.text_len), tensors, "layout"),
                "output_dtypes": pack_pipeline_value((video.dtype, audio.dtype), tensors, "output_dtypes"),
                "nested": isinstance(x, (list, tuple)),
                "transformer_options": pack_pipeline_value(prepare_model_parallel_value(transformer_options), tensors, "transformer_options"),
            }
            return PipelineIntermediateTensors(tensors, metadata)
        return self._output(hidden, layout, (video.dtype, audio.dtype), isinstance(x, (list, tuple)))

    def _run_layers(self, hidden, freqs, groups, transformer_options, start, end):
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        layers = [self.block.layers[index] for index in range(start, end)]
        prefetch_queue = model_prefetch.make_prefetch_queue(layers, hidden.device, transformer_options)
        for index in range(start, end):
            layer = self.block.layers[index]
            model_prefetch.prefetch_queue_pop(prefetch_queue, hidden.device, layer, malloc_scope="block")
            transformer_options["block_index"] = index
            if ("double_block", index) in blocks_replace:
                def block_wrap(args, layer=layer):
                    return {"img": layer(args["img"], args["freqs"], args["groups"], transformer_options=args["transformer_options"])}
                hidden = blocks_replace[("double_block", index)](
                    {"img": hidden, "freqs": freqs, "groups": groups, "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                hidden = layer(hidden, freqs, groups, transformer_options)
        model_prefetch.prefetch_queue_pop(prefetch_queue, hidden.device, None, malloc_scope="block")
        return hidden

    def _output(self, hidden, layout, dtypes, nested):
        t, h, w = layout.video_shape
        audio_start = layout.groups[0]
        video, audio = self.post_adapter(hidden[:, :layout.video_tokens], hidden[:, audio_start:audio_start + layout.audio_len])
        video = video.transpose(1, 2).unflatten(2, (t, h, w)).to(dtypes[0])
        if not nested:
            return video
        return [video, audio.transpose(1, 2).to(dtypes[1])]

    def forward_pipeline_stage(self, intermediate: PipelineIntermediateTensors):
        if self.pipeline_stage is None or self.pipeline_stage.is_first:
            raise RuntimeError("MAGI-2 pipeline continuation requires a non-first stage")
        tensors, metadata = intermediate.tensors, intermediate.metadata
        layout = Magi2Layout(*unpack_pipeline_value(metadata["layout"], tensors))
        transformer_options = unpack_pipeline_value(metadata["transformer_options"], tensors)
        hidden = self._run_layers(tensors["hidden_states"], tensors["freqs"], layout.groups, transformer_options,
                                  self.pipeline_stage.start_layer, self.pipeline_stage.end_layer)
        if not self.pipeline_stage.is_last:
            return PipelineIntermediateTensors({**tensors, "hidden_states": hidden}, metadata)
        return self._output(hidden, layout, unpack_pipeline_value(metadata["output_dtypes"], tensors), metadata["nested"])
