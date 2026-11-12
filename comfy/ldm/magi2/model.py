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

from ... import model_management, model_prefetch, patcher_extension
from ...pipeline_parallel import PipelineIntermediateTensors, PipelineMissingLayer, PipelineStageConfig
from ...pipeline_parallel.types import pack_pipeline_value, prepare_model_parallel_value, unpack_pipeline_value
from ...quant_ops import QuantizedTensor
from ...tensor_parallel.operations import column_parallel_linear, local_size, row_parallel_linear
from ..modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

TIME_CHANNELS = 64
SINK_PAD = 8
MHC_CHUNK = 8192


def swiglu7(x):
    """GPT-OSS clamped SwiGLU over interleaved [gate, linear] pairs, evaluated in fp32."""
    dtype = x.dtype
    x = x.float()
    gate = x[..., ::2].clamp(max=7.0)
    linear = x[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(dtype)


def swiglu7_pair(gate, up):
    dtype = gate.dtype
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


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


def modality_linear(module, x, groups):
    """Apply one linear, or one linear per modality group of the token axis."""
    if not isinstance(module, nn.ModuleList):
        return module(x)
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
        self.num_modality = num_modality
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


def _expert_rows(bank, weight, start, count, dtype):
    """Experts [start, start + count) of a bank as a [count, out, in] compute-dtype tensor and its ConvRot group size.

    A ConvRot INT8 bank stays in its rotated basis: the caller rotates the activations instead, which
    is exact because the regular Hadamard used by ConvRot is its own inverse.
    """
    if not isinstance(weight, QuantizedTensor):
        return weight.view(-1, bank.out_features, bank.in_features)[start:start + count], 0
    qdata, scale = weight.layout_cls.get_plain_tensors(weight)
    rows = qdata.view(-1, bank.out_features, bank.in_features)[start:start + count]
    if scale.numel() > 1:
        scale = scale.view(-1, bank.out_features, 1)[start:start + count]
    params = weight.params
    group = params.convrot_groupsize if getattr(params, "convrot", False) else 0
    return (rows.float() * scale.float()).to(dtype), group


def _grouped_projection(x, bank, offsets):
    weight, group = bank
    if group:
        x = _rotate_activation(x, _build_hadamard(group, device=x.device, dtype=x.dtype), group)
    return torch._grouped_mm(x, weight.transpose(1, 2), offs=offsets)


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

    def forward(self, x):
        batch, tokens, _ = x.shape
        x = x.reshape(batch * tokens, self.heads, self.head_dim)
        out = torch.empty_like(x)
        with self.gate_proj.bank_resident(x), self.up_proj.bank_resident(x), self.down_proj.bank_resident(x):
            banks = tuple((bank, bank._resident_bank[0]) for bank in (self.gate_proj, self.up_proj, self.down_proj))
            for head in range(self.heads):
                x_head = x[:, head].contiguous()
                probs, indices = self.route(x_head, head)
                order = indices.flatten().argsort(stable=True)
                token_ids = order // self.top_k
                offsets = torch.bincount(indices.flatten(), minlength=self.num_experts).cumsum(0).to(torch.int32)
                rows = x_head.index_select(0, token_ids)
                start = head * self.num_experts
                gate_bank, up_bank, down_bank = (_expert_rows(bank, weight, start, self.num_experts, x.dtype) for bank, weight in banks)
                hidden = swiglu7_pair(_grouped_projection(rows, gate_bank, offsets), _grouped_projection(rows, up_bank, offsets))
                del rows, gate_bank, up_bank
                y = _grouped_projection(hidden, down_bank, offsets)
                del hidden, down_bank
                y = (y.float() * probs.flatten()[order].unsqueeze(-1)).to(x.dtype)
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

    def forward(self, x, groups):
        normed = self.pre_norm(x, groups)
        routed = self.merge_linear(self.moe_mlp(self.split_linear(normed)))
        shared = self.shared_expert_fc2(swiglu7(self.shared_expert_fc1(normed)))
        specific = modality_linear(self.modality_specific_shared_expert_fc2,
                                   swiglu7(modality_linear(self.modality_specific_shared_expert_fc1, normed, groups)), groups)
        return routed + (shared + specific)


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
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.video_in_channels = video_in_channels
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
