"""Fibo 2.0 (briaai/Fibo-2.0-dev): an 8B single-stream flow-matching DiT on FLUX.2 VAE latents.

Supports both:
- diffusers' BriaFibo2Transformer2DModel (huggingface/diffusers de988b7e, Bria's reference implementation,
  appmana/Fibo-2.0-ComfyUI) with unfused to_q/to_k/to_v/to_out.0/norm_q/norm_k and w1/w2/w3, and
- Comfy-Org's repacked checkpoint format (repacked=True) with JointAttention (qkv, out, q_norm, k_norm),
  cross_attention (to_q, to_kv, to_out) and optionally fused w13 SwiGLU (fused_ffn=True).

Z-Image's blocks: a 2x2 latent patch is a token, two noise-refiner blocks run on the image tokens alone,
then 44 blocks run on one stream of [gist tokens, image tokens, context image tokens]. The gist tokens
come from a Perceiver resampler over Qwen3-VL hidden states (256 to 512 of them, more for longer
prompts); five of the blocks also cross-attend to Qwen3-VL hidden states of their own. Clean tokens (the
gist and the images to edit) are modulated at t = 0, the image being generated at its own t.

The conditioning is the six Qwen3-VL bundles of comfy.text_encoders.fibo2, each 3 x 2560 features,
concatenated on the feature axis: (B, text_len, 6 * 7680). The first is the Perceiver's, then one per
injection block in injection_layer_ids order.

ComfyUI's FLUX.2 latent (128 channels at 1/16) is the reference's normalized 32-channel latent at 1/8
patchified in its (channel, dy, dx) order, which is the reference's token feature order: a latent pixel
is a token.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.lumina.model import FinalLayer as LuminaFinalLayer, JointAttention, apply_gate, clamp_fp16, modulate, pos_ids_x
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention, optimized_attention_masked
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder

ADALN_EMBED_DIM = 256
NUM_BUNDLES = 6


def residual(x, y, gate=None):
    # in place inside the memory compiler region; out of place keeps x for backward when training
    if comfy.model_management.in_training:
        return x + y if gate is None else torch.addcmul(x, y, gate)
    return x.add_(y) if gate is None else x.addcmul_(y, gate)


def gist_count(text_len, max_gist):
    # 256 gist tokens up to 256 text tokens, rising linearly to max_gist at 2048, in steps of 64
    ratio = min(1.0, max(0.0, (text_len - 256) / (2048 - 256)))
    return max(256, min(max_gist, math.ceil((256 + ratio * (max_gist - 256)) / 64) * 64))


def compute_gist_count(text_lengths, min_gist, max_gist, gist_step, min_text_len, max_text_len):
    """Gist tokens per prompt from its length in tokens: [min_text_len, max_text_len] maps linearly onto
    [min_gist, max_gist], rounded up to a multiple of gist_step."""
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts


def key_padding_bias(keep, dtype):
    """An additive (B, 1, 1, N) attention bias from a (B, N) bool mask of the keys to keep: every backend
    takes it, where xformers would add a bool mask as 0/1."""
    if keep is None:
        return None
    return torch.zeros(keep.shape, dtype=dtype, device=keep.device).masked_fill(~keep, float("-inf"))[:, None, None, :]


def rms_norm(x, weight, eps):
    """diffusers' RMSNorm arithmetic: the inverse RMS in float32, the normalized input rounded to a half-precision
    weight's dtype before the weight multiplies it."""
    hidden = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)
    if weight is None:
        return hidden.to(x.dtype)
    if weight.dtype in (torch.float16, torch.bfloat16):
        hidden = hidden.to(weight.dtype)
    return hidden * weight


def reference_norms(operations):
    """operations.RMSNorm computing as the reference does (rms_norm), its weight cast and offloaded as ComfyUI's."""
    class RMSNorm(operations.RMSNorm):
        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = comfy.ops._cast_weight_bias(self, input)
            return comfy.ops._release_weight_bias(self, rms_norm(input, weight, self.eps), cast_state)

        def forward(self, input):
            comfy.ops.run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            return rms_norm(input, self.weight, self.eps)

    class Operations(operations):
        pass
    Operations.RMSNorm = RMSNorm
    return Operations


class RopeEmbedder:
    """diffusers' Z-Image rotary tables: per axis, angles from float64 positions and frequencies rounded to float32,
    as complex64."""

    def __init__(self, theta, axes_dims, axes_lens):
        self.theta, self.axes_dims, self.axes_lens = theta, tuple(axes_dims), tuple(axes_lens)
        self.tables = {}

    def table(self, device):
        if device not in self.tables:
            tables = []
            for d, e in zip(self.axes_dims, self.axes_lens):
                freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                freqs = torch.outer(torch.arange(e, dtype=torch.float64), freqs).float()
                tables.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64).to(device))
            self.tables[device] = tables
        return self.tables[device]

    def __call__(self, ids):
        tables = self.table(ids.device)
        return torch.cat([tables[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def apply_rotary_emb(x, freqs_cis):
    """x (B, N, H, D) rotated by freqs_cis (1 or B, N, D / 2) complex in float32, back in x's dtype."""
    rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(rotated * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)


def per_token(noisy, clean, noise_mask):
    """(B, N, D) taking `noisy` (B, D) where noise_mask (B, N) is True and `clean` elsewhere."""
    return torch.where(noise_mask[..., None], noisy[:, None, :], clean[:, None, :])


class FeedForward(nn.Module):
    # SwiGLU; fused files carry w1 and w3 as one w13, the hidden width zero padded to a multiple of 256 so int8 weights quantize swiglu(x) inside w2's GEMM
    def __init__(self, dim, hidden_dim, fused=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = operations.Linear(dim, 2 * hidden_dim, bias=False, dtype=dtype, device=device)
        else:
            self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
            self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if self.fused:
            return comfy.ops.linear_input_act(self.w2, self.w13(x), "swiglu")
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def ffn_dim(dim):
    return (dim * 8 // 3) // 64 * 64


class Attention(nn.Module):
    """Self-attention (RoPE when pe is given) or cross-attention to `context`, q/k RMS-normed per head."""

    def __init__(self, query_dim, heads, dim_head, kv_dim=None, eps=1e-5, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.comfy_attention = ComfyAttention()
        self.to_q = operations.Linear(query_dim, heads * dim_head, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(kv_dim, heads * dim_head, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(kv_dim, heads * dim_head, bias=False, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([operations.Linear(heads * dim_head, query_dim, bias=False, dtype=dtype, device=device)])
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)

    def forward(self, x, context=None, mask=None, pe=None, transformer_options={}):
        source = x if context is None else context
        q = self.to_q(x).unflatten(-1, (self.heads, -1))
        k = self.to_k(source).unflatten(-1, (self.heads, -1))
        v = self.to_v(source).unflatten(-1, (self.heads, -1))
        q = self.norm_q(q)
        k = self.norm_k(k)
        if pe is not None:
            q, k = apply_rotary_emb(q, pe), apply_rotary_emb(k, pe)
        q = AttentionTensorContainer(q.movedim(1, 2))
        k = AttentionTensorContainer(k.movedim(1, 2))
        v = AttentionTensorContainer(v.movedim(1, 2))
        out = optimized_attention_masked(q, k, v, self.heads, mask, skip_reshape=True, transformer_options=transformer_options,
                                         preferred_attention=self.comfy_attention)
        return self.to_out[0](out)


class RepackedAttention(nn.Module):
    # no RoPE: the cross-attention to the text, and the Perceiver's self-attention (repacked checkpoint format)
    def __init__(self, dim, n_heads, kv_dim, norm_eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.n_heads = n_heads
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_kv = operations.Linear(kv_dim, 2 * dim, bias=False, dtype=dtype, device=device)
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(dim // n_heads, eps=norm_eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim // n_heads, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, x, context, transformer_options={}):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.n_heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.n_heads, -1)).unbind(2)
        k = self.norm_k(k)
        q, k, v = (AttentionTensorContainer(t.movedim(1, 2)) for t in (q, k, v))
        return self.to_out(optimized_attention(q, k, v, self.n_heads, skip_reshape=True, transformer_options=transformer_options, preferred_attention=self.comfy_attention))


class Block(nn.Module):
    # Z-Image's block with in place residuals; an injection block also cross-attends to its own text bundle (repacked format)
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, norm_eps, cap_feat_dim=None, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.attention = JointAttention(dim, n_heads, None, True, operation_settings=operation_settings)
        self.attention.q_norm.eps = self.attention.k_norm.eps = norm_eps
        self.feed_forward = FeedForward(dim, ffn_dim, fused_ffn, dtype=dtype, device=device, operations=operations)
        self.attention_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.attention_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(operations.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True, dtype=dtype, device=device))
        self.cross_attention = None
        if cap_feat_dim is not None:
            self.cross_attn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attention = RepackedAttention(dim, n_heads, cap_feat_dim, norm_eps, dtype=dtype, device=device, operations=operations)
            self.cross_attn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attn_gate = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

    def forward(self, x, freqs_cis, adaln_input, timestep_zero_index=None, context=None, transformer_options={}):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
        y = self.attention(modulate(self.attention_norm1(x), scale_msa, timestep_zero_index=timestep_zero_index), None, freqs_cis, transformer_options=transformer_options)
        x = residual(x, apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context, transformer_options)
            x = residual(x, self.cross_attn_norm2(y), comfy.ops.cast_to_input(self.cross_attn_gate, y).tanh())
        y = self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp, timestep_zero_index=timestep_zero_index))
        return residual(x, apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))


class RepackedPerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.cross_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.cross_attention = RepackedAttention(dim, n_heads, cap_feat_dim, norm_eps, dtype=dtype, device=device, operations=operations)
        self.self_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.self_attention = RepackedAttention(dim, n_heads, dim, norm_eps, dtype=dtype, device=device, operations=operations)
        self.ffn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn = FeedForward(dim, ffn_dim, fused_ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x, context, transformer_options={}):
        x = x + self.cross_attention(self.cross_attn_norm(x), context, transformer_options)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y, transformer_options)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    # squeezes the text encoder's hidden states into gist tokens (repacked format)
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, num_queries, num_layers, norm_eps, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.latent_queries = nn.Parameter(torch.empty(num_queries, dim, dtype=dtype, device=device))
        self.layers = nn.ModuleList([RepackedPerceiverLayer(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps, operation_settings=operation_settings) for _ in range(num_layers)])
        self.norm_out = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, context, num_gist, transformer_options={}):
        x = comfy.ops.cast_to_input(self.latent_queries[:num_gist], context).unsqueeze(0).expand(context.shape[0], -1, -1)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), context.device, transformer_options)
        for layer in self.layers:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, context.device, layer, context.dtype)
            x = layer(x, context, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, context.device, None)
        return self.norm_out(x)


class TransformerBlock(nn.Module):
    """Z-Image's block; with cross_attention_dim an injection block that also reads its text bundle (diffusers format)."""

    def __init__(self, dim, n_heads, norm_eps, cross_attention_dim=None, hidden_dim=None, fused_ffn=False, dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps, operations=operations, **kw)
        self.feed_forward = FeedForward(dim, hidden_dim or ffn_dim(dim), fused=fused_ffn, operations=operations, **kw)
        self.attention_norm1 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.attention_norm2 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.adaLN_modulation = nn.Sequential(operations.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True, **kw))
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
            self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps,
                                             operations=operations, **kw)
            self.cross_attn_norm2 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1, **kw))
        else:
            self.cross_attention = None

    def modulation(self, adaln_noisy, adaln_clean, noise_mask):
        """(scale_msa, gate_msa, scale_mlp, gate_mlp), each (B, N or 1, dim): every token at t (noise_mask
        None), or the noisy tokens at t and the clean ones at t = 0."""
        if noise_mask is None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_noisy).unsqueeze(1).chunk(4, dim=2)
            return 1.0 + scale_msa, gate_msa.tanh(), 1.0 + scale_mlp, gate_mlp.tanh()
        noisy = self.adaLN_modulation(adaln_noisy).chunk(4, dim=1)
        clean = self.adaLN_modulation(adaln_clean).chunk(4, dim=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = (
            per_token(1.0 + noisy[0], 1.0 + clean[0], noise_mask), per_token(noisy[1].tanh(), clean[1].tanh(), noise_mask),
            per_token(1.0 + noisy[2], 1.0 + clean[2], noise_mask), per_token(noisy[3].tanh(), clean[3].tanh(), noise_mask))
        return scale_msa, gate_msa, scale_mlp, gate_mlp

    def forward(self, x, mask, pe, adaln_noisy, adaln_clean=None, noise_mask=None, context=None, context_mask=None,
                transformer_options={}):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.modulation(adaln_noisy, adaln_clean, noise_mask)
        attn = self.attention(self.attention_norm1(x) * scale_msa, mask=mask, pe=pe, transformer_options=transformer_options)
        x = x + gate_msa * self.attention_norm2(attn)
        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), context=context, mask=context_mask,
                                         transformer_options=transformer_options)
            gate = comfy.model_management.cast_to(self.cross_attn_gate, dtype=x.dtype, device=x.device)
            x = x + gate.tanh() * self.cross_attn_norm2(cross)
        return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))


class PerceiverLayer(nn.Module):
    """Gist tokens read the text (cross-attention), then each other, then an FFN (diffusers format)."""

    def __init__(self, dim, n_heads, cross_attention_dim, norm_eps, hidden_dim=None, fused_ffn=False, dtype=None, device=None, operations=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.cross_attn_norm = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps,
                                         operations=operations, **kw)
        self.self_attn_norm = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.self_attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps, operations=operations, **kw)
        self.ffn_norm = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, **kw)
        self.ffn = FeedForward(dim, hidden_dim or ffn_dim(dim), fused=fused_ffn, operations=operations, **kw)

    def forward(self, latents, context, context_mask=None, mask=None, transformer_options={}):
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), context=context, mask=context_mask,
                                                 transformer_options=transformer_options)
        latents = latents + self.self_attention(self.self_attn_norm(latents), mask=mask, transformer_options=transformer_options)
        return latents + self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim, num_queries, num_layers, norm_eps, hidden_dim=None, fused_ffn=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.empty(num_queries, dim, dtype=dtype, device=device))
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps, hidden_dim=hidden_dim, fused_ffn=fused_ffn,
                                                    dtype=dtype, device=device, operations=operations) for _ in range(num_layers)])
        self.norm_out = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)

    def forward(self, context, num_gist_tokens, context_mask=None, mask=None, transformer_options={}):
        queries = comfy.model_management.cast_to(self.latent_queries[:num_gist_tokens], dtype=context.dtype, device=context.device)
        latents = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, context, context_mask, mask, transformer_options=transformer_options)
        return self.norm_out(latents)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True,
                                                                         dtype=dtype, device=device))

    def forward(self, x, c):
        return self.linear(self.norm_final(x) * (1.0 + self.adaLN_modulation(c)).unsqueeze(1))


class Fibo2(nn.Module):
    def __init__(self, in_channels=32, patch_size=2, dim=3584, n_layers=44, n_refiner_layers=2, n_heads=28, norm_eps=1e-5,
                 cap_feat_dim=7680, injection_layer_ids=(6, 14, 22, 30, 38), injection_layers=None, perceiver_num_layers=2,
                 perceiver_layers=None, min_num_gist_tokens=256, max_num_gist_tokens=512, max_gist_tokens=None,
                 gist_step=64, gist_min_text_len=256, gist_max_text_len=2048, rope_theta=256.0,
                 t_scale=1000.0, axes_dims=(32, 48, 48), axes_lens=(2048, 512, 512), ffn_dim=None, fused_ffn=False,
                 repacked=False, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.repacked = repacked
        if injection_layers is not None:
            injection_layer_ids = injection_layers
        if perceiver_layers is not None:
            perceiver_num_layers = perceiver_layers
        if max_gist_tokens is not None:
            max_num_gist_tokens = max_gist_tokens

        self.cap_feat_dim = cap_feat_dim
        self.injection_layer_ids = tuple(injection_layer_ids)
        self.injection_layers = self.injection_layer_ids
        self.min_num_gist_tokens, self.max_num_gist_tokens, self.gist_step = min_num_gist_tokens, max_num_gist_tokens, gist_step
        self.max_gist_tokens = max_num_gist_tokens
        self.gist_min_text_len, self.gist_max_text_len = gist_min_text_len, gist_max_text_len
        self.t_scale = t_scale

        if repacked:
            operation_settings = {"operations": operations, "device": device, "dtype": dtype}
            hidden_dim = ffn_dim or globals()["ffn_dim"](dim)
            self.patch_channels = in_channels
            self.x_embedder = operations.Linear(self.patch_channels, dim, bias=True, dtype=dtype, device=device)
            self.t_embedder = TimestepEmbedder(1024, output_size=min(dim, ADALN_EMBED_DIM), **operation_settings)
            self.noise_refiner = nn.ModuleList([Block(dim, n_heads, hidden_dim, fused_ffn, norm_eps, operation_settings=operation_settings) for _ in range(n_refiner_layers)])
            self.perceiver = Perceiver(dim, n_heads, hidden_dim, fused_ffn, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps, operation_settings=operation_settings)
            self.layers = nn.ModuleList([Block(dim, n_heads, hidden_dim, fused_ffn, norm_eps, cap_feat_dim if i in self.injection_layers else None, operation_settings=operation_settings) for i in range(n_layers)])
            self.final_layer = LuminaFinalLayer(dim, 1, self.patch_channels, z_image_modulation=True, operation_settings=operation_settings)
            self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims))
        else:
            self.patch_channels = patch_size * patch_size * in_channels
            operations = reference_norms(operations)
            kw = dict(dtype=dtype, device=device, operations=operations)
            self.x_embedder = operations.Linear(self.patch_channels, dim, bias=True, dtype=dtype, device=device)
            self.t_embedder = TimestepEmbedder(1024, output_size=min(dim, ADALN_EMBED_DIM), **kw)
            self.rope_embedder = RopeEmbedder(rope_theta, axes_dims, axes_lens)
            self.noise_refiner = nn.ModuleList([TransformerBlock(dim, n_heads, norm_eps, hidden_dim=ffn_dim, fused_ffn=fused_ffn, **kw) for _ in range(n_refiner_layers)])
            self.perceiver = PerceiverResampler(dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps, hidden_dim=ffn_dim, fused_ffn=fused_ffn, **kw)
            self.layers = nn.ModuleList([
                TransformerBlock(dim, n_heads, norm_eps, cross_attention_dim=cap_feat_dim if i in self.injection_layer_ids else None, hidden_dim=ffn_dim, fused_ffn=fused_ffn, **kw)
                for i in range(n_layers)])
            self.final_layer = FinalLayer(dim, self.patch_channels, dtype=dtype, device=device, operations=operations)

    def rope(self, ids):
        """(1, N, head_dim / 2) complex rotations for (N, 3) integer positions."""
        return self.rope_embedder(ids).unsqueeze(0)

    @staticmethod
    def grid(size, start, device):
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1).flatten(0, 2)

    def gist_counts(self, text_mask, batch, text_len):
        lengths = text_mask.sum(dim=1).tolist() if text_mask is not None else [text_len] * batch
        return compute_gist_count(lengths, self.min_num_gist_tokens, self.max_num_gist_tokens, self.gist_step,
                                  self.gist_min_text_len, self.gist_max_text_len)

    def bundle(self, context, index):
        return context[..., index * self.cap_feat_dim:(index + 1) * self.cap_feat_dim].contiguous()

    def forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, ref_latents, transformer_options, **kwargs)

    def embed(self, x, timestep, ref_latents=None):
        """The image tokens and the clean context image tokens embedded, their rotations, and the t and t = 0
        modulation inputs. x and each ref latent: (B, patch_channels, h, w)."""
        B, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        adaln_noisy = self.t_embedder(timestep * self.t_scale, dtype)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep), dtype)
        # the image on RoPE plane max_num_gist_tokens + 1 whatever the gist count, context image k on the k-th after it
        image_plane = self.max_num_gist_tokens + 1
        tokens = [x.flatten(2).transpose(1, 2).contiguous()]
        ids = [self.grid((1, h, w), (image_plane, 0, 0), device)]
        for k, ref in enumerate(ref_latents or []):
            tokens.append(comfy.utils.repeat_to_batch_size(ref.to(dtype), B).flatten(2).transpose(1, 2).contiguous())
            ids.append(self.grid((1, ref.shape[2], ref.shape[3]), (image_plane + k + 1, 0, 0), device))
        hidden = self.x_embedder(torch.cat(tokens, dim=1))
        return hidden, self.rope(torch.cat(ids)), adaln_noisy, adaln_clean

    def _forward_repacked(self, x, timestep, context, transformer_options={}, ref_latents=None, **kwargs):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        ref_latents = ref_latents or []

        # the image and the reference tokens are noisy and clean: the second half of the adaln batch modulates the image
        adaln_mixed = self.t_embedder(torch.cat((torch.zeros_like(timestep), timestep * self.t_scale)), dtype)
        adaln_noisy = adaln_mixed[B:]

        # gist tokens on rope planes 1..num_gist, the image on plane max_gist_tokens + 1 whatever the gist count, reference k on the k-th plane after it
        num_gist = compute_gist_count([context.shape[1]], self.min_num_gist_tokens, self.max_num_gist_tokens, self.gist_step,
                                      self.gist_min_text_len, self.gist_max_text_len)[0]
        plane = self.max_gist_tokens + 1
        tokens = [x.flatten(2).transpose(1, 2)]
        ids = [torch.zeros(1, num_gist, 3, dtype=torch.float32, device=device), pos_ids_x(plane, H, W, 1, device, transformer_options=transformer_options)]
        ids[0][0, :, 0] = torch.arange(1, num_gist + 1, dtype=torch.float32, device=device)
        for k, ref in enumerate(ref_latents):
            tokens.append(comfy.utils.repeat_to_batch_size(ref, B).flatten(2).transpose(1, 2))
            ids.append(pos_ids_x(plane + 1 + k, ref.shape[-2], ref.shape[-1], 1, device, transformer_options=transformer_options))
        img = self.x_embedder(torch.cat(tokens, dim=1))
        freqs_cis = self.rope_embedder(torch.cat(ids, dim=1)).movedim(1, 2)
        del tokens

        n_img = H * W
        refiner_adaln, refiner_index = (adaln_mixed, [(0, n_img)]) if len(ref_latents) > 0 else (adaln_noisy, None)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.noise_refiner), device, transformer_options)
        for block in self.noise_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            img = block(img, freqs_cis[:, num_gist:], refiner_adaln, refiner_index, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        # the Perceiver reads the first text bundle, each injection block its own
        bundles = context.chunk(1 + len(self.injection_layers), dim=-1)
        gist = self.perceiver(bundles[0], num_gist, transformer_options)
        text = dict(zip(self.injection_layers, bundles[1:]))

        # allocated outside the memory compiler region, the blocks update it in place
        x = torch.cat((gist, img), dim=1)
        del gist, img
        image_index = [(num_gist, num_gist + n_img)]

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["pe"], args["vec"], image_index, text.get(i), args["transformer_options"])}
                x = blocks_replace[("single_block", i)]({"img": x, "vec": adaln_mixed, "pe": freqs_cis, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                x = block(x, freqs_cis, adaln_mixed, image_index, text.get(i), transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.final_layer(x[:, num_gist:num_gist + n_img], adaln_noisy)
        return out.transpose(1, 2).reshape(B, C, H, W)

    def _forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        if self.repacked:
            return self._forward_repacked(x, timestep, context, transformer_options=transformer_options, ref_latents=ref_latents, **kwargs)

        B, C, h, w = x.shape
        device = x.device
        dtype = x.dtype
        transformer_options = transformer_options.copy()
        num_image_tokens = h * w
        context = context.to(dtype)
        text_mask = None
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            text_mask = attention_mask.to(device=device, dtype=torch.bool)
        text_bias = key_padding_bias(text_mask, dtype)

        hidden, image_pe, adaln_noisy, adaln_clean = self.embed(x, timestep, ref_latents)

        # noise refiner: with context images only the generated image's tokens are noisy
        refiner_noise_mask = None
        if ref_latents:
            refiner_noise_mask = torch.zeros(B, hidden.shape[1], dtype=torch.bool, device=device)
            refiner_noise_mask[:, :num_image_tokens] = True
        for block in self.noise_refiner:
            hidden = block(hidden, None, image_pe, adaln_noisy, adaln_clean, refiner_noise_mask, transformer_options=transformer_options)

        # the gist tokens: a batch of prompts with different counts pads them to the largest, masked out
        gist_counts = self.gist_counts(text_mask, B, context.shape[1])
        num_gist = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist:
            gist_mask = torch.arange(num_gist, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(self.bundle(context, 0), num_gist, text_bias, key_padding_bias(gist_mask, dtype),
                              transformer_options=transformer_options)
        gist_pe = self.rope(self.grid((num_gist, 1, 1), (1, 0, 0), device))

        # one stream, gist first as trained; when gist tokens are padded the gist goes last so the padding ends
        # every row (positions come from RoPE, so the order changes nothing but rounding)
        num_context_tokens = hidden.shape[1] - num_image_tokens
        mask = None
        if gist_mask is None:
            image_start = num_gist
            hidden = torch.cat([gist, hidden], dim=1)
            pe = torch.cat([gist_pe, image_pe], dim=1)
        else:
            image_start = 0
            hidden = torch.cat([hidden, gist], dim=1)
            pe = torch.cat([image_pe, gist_pe], dim=1)
            keep = torch.cat([torch.ones(B, num_image_tokens + num_context_tokens, dtype=torch.bool, device=device), gist_mask], dim=1)
            mask = key_padding_bias(keep, dtype)
        noise_mask = torch.zeros(B, hidden.shape[1], dtype=torch.bool, device=device)
        noise_mask[:, image_start:image_start + num_image_tokens] = True

        bundle_index = {layer_id: i + 1 for i, layer_id in enumerate(self.injection_layer_ids)}
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
        for i, block in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            transformer_options["block_index"] = i
            text = self.bundle(context, bundle_index[i]) if i in bundle_index else None
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], mask, args["pe"], adaln_noisy, adaln_clean, noise_mask,
                                         context=text, context_mask=text_bias, transformer_options=args["transformer_options"])}
                hidden = blocks_replace[("single_block", i)]({"img": hidden, "vec": adaln_noisy, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                hidden = block(hidden, mask, pe, adaln_noisy, adaln_clean, noise_mask, context=text, context_mask=text_bias,
                               transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        out = self.final_layer(hidden[:, image_start:image_start + num_image_tokens], adaln_noisy)
        return out.transpose(1, 2).reshape(B, C, h, w)
