from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .. import common_dit
from ... import model_management
from ... import model_prefetch
from ... import ops
from ... import patcher_extension
from ... import quant_ops
from ...pipeline_parallel.types import (
    PipelineIntermediateTensors,
    PipelineMissingLayer,
    PipelineStageConfig,
    pack_pipeline_value,
    prepare_model_parallel_value,
    unpack_pipeline_value,
)
from ...tensor_parallel.operations import column_parallel_linear, local_size, row_parallel_linear
from ..flux import math as flux_math
from ..flux.layers import timestep_embedding
from ..lightricks.model import FeedForward
from ..modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

FFN_CHUNK = 8192


def modulated_norm(x, norm, scale, shift):
    if model_management.in_training or quant_ops.ck is None:
        return torch.addcmul(shift, norm(x), 1 + scale)
    return quant_ops.ck.adaln(x, scale, shift, norm.eps)


def gated_residual(x, y, gate):
    # in place inside the memory compiler region, out of place keeps x for backward when training
    if model_management.in_training:
        return torch.addcmul(x, y, gate)
    return x.addcmul_(y, gate)


def rms_rope(q, k, q_norm, k_norm, pe):
    if model_management.in_training or quant_ops.ck is None:
        return flux_math.apply_rope(q_norm(q), k_norm(k), pe)
    q_scale, _, q_stream = ops.cast_bias_weight(q_norm, q, offloadable=True)
    k_scale, _, k_stream = ops.cast_bias_weight(k_norm, k, offloadable=True)
    q, k = quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, q_norm.eps)
    ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
    ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    return q, k


def attention(q, k, v, heads, comfy_attention, transformer_options):
    q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, transformer_options=transformer_options, preferred_attention=comfy_attention)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=None):
        super().__init__()
        local_size(operations, dim_out, "PAN 2 refiner FFN width")
        self.proj = column_parallel_linear(operations, dim_in, 2 * dim_out, bias=False, dtype=dtype, device=device)


class RefinerFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(SwiGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations), nn.Identity(),
                                 row_parallel_linear(operations, inner_dim, dim, bias=False, dtype=dtype, device=device))

    def forward(self, x):
        # the projection is laid out [value | gate]
        value, gate = self.net[0].proj(x).chunk(2, dim=-1)
        return self.net[2](value * F.silu(gate))


class RefinerAttention(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = local_size(operations, num_heads, "PAN 2 refiner attention heads")
        self.to_q = column_parallel_linear(operations, dim, dim, bias=False, dtype=dtype, device=device)
        self.to_k = column_parallel_linear(operations, dim, dim, bias=False, dtype=dtype, device=device)
        self.to_v = column_parallel_linear(operations, dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(dim // num_heads, eps=1e-5, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim // num_heads, eps=1e-5, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([row_parallel_linear(operations, dim, dim, bias=False, dtype=dtype, device=device)])


class RefinerBlock(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.norm1 = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.attn = RefinerAttention(dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.ff = RefinerFeedForward(dim, 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, transformer_options={}):
        B, N, _ = x.shape
        y = self.norm1(x)
        q = self.attn.norm_q(self.attn.to_q(y).view(B, N, self.attn.heads, -1))
        k = self.attn.norm_k(self.attn.to_k(y).view(B, N, self.attn.heads, -1))
        v = self.attn.to_v(y)
        del y
        x = x + self.attn.to_out[0](attention(q, k, v, self.attn.heads, self.comfy_attention, transformer_options))
        return x + self.ff(self.norm2(x))


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = local_size(operations, num_heads, "PAN 2 joint attention heads")
        head_dim = dim // num_heads
        self.to_q = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.to_k = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.to_v = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.add_q_proj = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.add_k_proj = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.add_v_proj = column_parallel_linear(operations, dim, dim, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([row_parallel_linear(operations, dim, dim, dtype=dtype, device=device)])
        self.to_add_out = row_parallel_linear(operations, dim, dim, dtype=dtype, device=device)

    def qkv(self, img, txt, pe):
        # video q/k get the 3D RoPE, the caption q/k only their norm
        B, N, _ = img.shape
        q, k = rms_rope(self.to_q(img).view(B, N, self.heads, -1), self.to_k(img).view(B, N, self.heads, -1), self.norm_q, self.norm_k, pe)
        B, N, _ = txt.shape
        q_txt = self.norm_added_q(self.add_q_proj(txt).view(B, N, self.heads, -1))
        k_txt = self.norm_added_k(self.add_k_proj(txt).view(B, N, self.heads, -1))
        return torch.cat((q, q_txt), dim=1), torch.cat((k, k_txt), dim=1), torch.cat((self.to_v(img), self.add_v_proj(txt)), dim=1)


class Modulation(nn.Module):
    def __init__(self, dim, chunks, adaln_dim, adaln_dtype=None, device=None, operations=None):
        super().__init__()
        self.chunks = chunks
        self.linear = operations.Linear(adaln_dim, chunks * dim, dtype=adaln_dtype, device=device)

    def forward(self, vec, dtype):
        return self.linear(vec).to(dtype).unsqueeze(1).chunk(self.chunks, dim=-1)


class JointBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, context_mlp_ratio, adaln_dim, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.norm1 = Modulation(dim, 6, adaln_dim, adaln_dtype=adaln_dtype, device=device, operations=operations)
        # the caption modulation is a learned constant, the same at every timestep
        self.scale_shift_table_context = nn.Parameter(torch.empty(6 * dim, dtype=dtype, device=device))
        # affine-free, shared by video and caption tokens
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = JointAttention(dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(dim, dim, mult=mlp_ratio, dtype=dtype, device=device, operations=operations)
        self.ff_context = FeedForward(dim, dim, mult=context_mlp_ratio, dtype=dtype, device=device, operations=operations)

    def forward(self, img, txt, vec, pe, transformer_options={}):
        # (shift, scale, gate) for attention, then for the feed-forward
        mod = self.norm1(vec, img.dtype)
        mod_txt = ops.cast_to_input(self.scale_shift_table_context, txt).chunk(6)
        q, k, v = self.attn.qkv(modulated_norm(img, self.norm, mod[1], mod[0]), modulated_norm(txt, self.norm, mod_txt[1], mod_txt[0]), pe)
        out = attention(q, k, v, self.attn.heads, self.comfy_attention, transformer_options)
        del q, k, v
        n = img.shape[1]
        img = gated_residual(img, self.attn.to_out[0](out[:, :n]), mod[2])
        txt = gated_residual(txt, self.attn.to_add_out(out[:, n:]), mod_txt[2])
        del out
        if model_management.in_training:
            img = gated_residual(img, self.ff(modulated_norm(img, self.norm, mod[4], mod[3])), mod[5])
        else:
            # token chunks bound the 8x wide intermediate; activations are quantized per token, so the result is unchanged
            for x in img.split(FFN_CHUNK, dim=1):
                gated_residual(x, self.ff(modulated_norm(x, self.norm, mod[4], mod[3])), mod[5])
        txt = gated_residual(txt, self.ff_context(modulated_norm(txt, self.norm, mod_txt[4], mod_txt[3])), mod_txt[5])
        return img, txt


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(256, dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, t, dtype):
        return self.linear_2(F.silu(self.linear_1(timestep_embedding(t, 256).to(dtype))))


class OutputNorm(nn.Module):
    def __init__(self, dim, adaln_dim, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(adaln_dim, 2 * dim, dtype=adaln_dtype, device=device)
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x, vec):
        scale, shift = self.linear(vec).to(x.dtype).unsqueeze(1).chunk(2, dim=-1)
        return modulated_norm(x, self.norm, scale, shift)


class PAN2(nn.Module):
    def __init__(self, in_channels=97, out_channels=48, dim=4096, num_heads=32, num_layers=44, mlp_ratio=8.0, context_mlp_ratio=4.0,
                 text_dim=4096, num_refiner_layers=4, patch_size=(1, 2, 2), rope_axes_dim=(16, 56, 56), rope_theta=256.0, rope_temporal_scale=16 / 24,
                 adaln_curve_grid=None, adaln_curve_dim=None, image_model=None, dtype=None, device=None, operations=None,
                 pipeline_stage: PipelineStageConfig | None = None):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim
        self.rope_theta = rope_theta
        # frame positions in 16 fps units for the 24 fps data
        self.rope_temporal_scale = rope_temporal_scale
        self.use_adaln_curves = adaln_curve_grid is not None
        self.pipeline_stage = pipeline_stage
        self.num_layers = num_layers
        is_first_stage = pipeline_stage is None or pipeline_stage.is_first
        is_last_stage = pipeline_stage is None or pipeline_stage.is_last
        # curve-form checkpoints replace the time embedder and the full-width adaln weights with a small basis of the time-embedding curve silu(temb(t))
        adaln = {"adaln_dim": adaln_curve_dim, "adaln_dtype": torch.float32} if self.use_adaln_curves else {"adaln_dim": dim, "adaln_dtype": dtype}
        if is_first_stage:
            self.x_embedder = operations.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)
            if self.use_adaln_curves:
                self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, adaln_curve_dim, dtype=torch.float32, device=device))
            else:
                self.time_embedder = TimestepEmbedding(dim, dtype=dtype, device=device, operations=operations)
            self.context_embedder = nn.Sequential(operations.RMSNorm(text_dim, eps=1e-5, dtype=dtype, device=device), operations.Linear(text_dim, dim, dtype=dtype, device=device))
            self.context_refiner = nn.ModuleList([RefinerBlock(dim, num_heads, dtype=dtype, device=device, operations=operations) for _ in range(num_refiner_layers)])
            self.context_type_embedding = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        else:
            self.x_embedder = PipelineMissingLayer()
            self.time_embedder = PipelineMissingLayer()
            self.context_embedder = PipelineMissingLayer()
            self.context_refiner = nn.ModuleList([PipelineMissingLayer() for _ in range(num_refiner_layers)])
            self.context_type_embedding = None
        self.transformer_blocks = nn.ModuleList([
            JointBlock(dim, num_heads, mlp_ratio, context_mlp_ratio, **adaln, dtype=dtype, device=device, operations=operations)
            if pipeline_stage is None or pipeline_stage.start_layer <= i < pipeline_stage.end_layer
            else PipelineMissingLayer()
            for i in range(num_layers)
        ])
        if is_last_stage:
            self.norm_out = OutputNorm(dim, **adaln, dtype=dtype, device=device, operations=operations)
            self.proj_out = operations.Linear(dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2], dtype=dtype, device=device)
        else:
            self.norm_out = PipelineMissingLayer()
            self.proj_out = PipelineMissingLayer()

    def rope(self, t, h, w, device):
        ids = torch.stack(torch.meshgrid(torch.arange(t, device=device, dtype=torch.float32) * self.rope_temporal_scale,
                                         torch.arange(h, device=device, dtype=torch.float32),
                                         torch.arange(w, device=device, dtype=torch.float32), indexing="ij"), dim=-1).reshape(1, -1, 3)
        pe = torch.cat([flux_math.rope(ids[..., i], d, self.rope_theta) for i, d in enumerate(self.rope_axes_dim)], dim=-3)
        # (1, N, 1, ...) for (B, N, H, D) queries and keys
        return pe.unsqueeze(2)

    def time_embedding(self, timestep, dtype):
        if self.use_adaln_curves:
            # curve coordinates lerped from the table, t in [0, 1] -> fractional grid row
            table = model_management.cast_to(self.adaln_t_table, device=timestep.device)
            pos = timestep.float().clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        return F.silu(self.time_embedder(timestep, dtype))

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        if self.pipeline_stage is not None and not self.pipeline_stage.is_first:
            raise RuntimeError("Only the first PAN 2 pipeline stage accepts model inputs")
        if self.pipeline_stage is not None and patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options):
            raise ValueError("PAN 2 pipeline parallelism does not support diffusion-model wrappers")
        return patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _run_blocks(self, img, txt, vec, pe, transformer_options, start_layer, end_layer):
        device, dtype = img.device, img.dtype
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        blocks = [self.transformer_blocks[i] for i in range(start_layer, end_layer)]
        prefetch_queue = model_prefetch.make_prefetch_queue(blocks, device, transformer_options)
        model_prefetch.malloc_graph_begin(device)
        for i in range(start_layer, end_layer):
            block = self.transformer_blocks[i]
            model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(args["img"], args["txt"], args["vec"], args["pe"], args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})
                img, txt = out["img"], out["txt"]
            else:
                img, txt = block(img, txt, vec, pe, transformer_options)
        model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        model_prefetch.malloc_graph_end()
        return img, txt

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        if self.pipeline_stage is not None and (transformer_options.get("patches") or transformer_options.get("patches_replace")):
            raise ValueError("PAN 2 pipeline parallelism does not support transformer block patches")
        B, _, T, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        x = common_dit.pad_to_patch_size(x, self.patch_size)
        t, h, w = (s // p for s, p in zip(x.shape[2:], self.patch_size))
        # allocated outside the memory compiler region, the blocks update both streams in place
        img = self.x_embedder(x).flatten(2).transpose(1, 2).contiguous()
        del x
        vec = self.time_embedding(timestep, dtype)
        pe = self.rope(t, h, w, device)

        txt = self.context_embedder(context)
        prefetch_queue = model_prefetch.make_prefetch_queue(list(self.context_refiner), device, transformer_options)
        for block in self.context_refiner:
            model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            txt = block(txt, transformer_options)
        model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        txt = txt + ops.cast_to_input(self.context_type_embedding, txt)

        start_layer = 0 if self.pipeline_stage is None else self.pipeline_stage.start_layer
        end_layer = len(self.transformer_blocks) if self.pipeline_stage is None else self.pipeline_stage.end_layer
        img, txt = self._run_blocks(img, txt, vec, pe, transformer_options, start_layer, end_layer)

        if self.pipeline_stage is not None and not self.pipeline_stage.is_last:
            tensors = {
                "img": img,
                "txt": txt,
                "vec": vec,
                "pe": pe,
            }
            metadata = {
                "batch_size": B,
                "grid_size": (t, h, w),
                "target_size": (T, H, W),
                "transformer_options": pack_pipeline_value(
                    prepare_model_parallel_value(transformer_options),
                    tensors,
                    "transformer_options",
                ),
            }
            return PipelineIntermediateTensors(tensors, metadata)

        return self._forward_exit(img, vec, B, (t, h, w), (T, H, W))

    def forward_pipeline_stage(self, intermediate: PipelineIntermediateTensors):
        if self.pipeline_stage is None or self.pipeline_stage.is_first:
            raise RuntimeError("PAN 2 pipeline continuation requires a non-first stage")
        tensors = intermediate.tensors
        metadata = intermediate.metadata
        transformer_options = unpack_pipeline_value(metadata["transformer_options"], tensors)
        img, txt = self._run_blocks(
            tensors["img"],
            tensors["txt"],
            tensors["vec"],
            tensors["pe"],
            transformer_options,
            self.pipeline_stage.start_layer,
            self.pipeline_stage.end_layer,
        )
        if not self.pipeline_stage.is_last:
            tensors = dict(tensors)
            tensors["img"] = img
            tensors["txt"] = txt
            return PipelineIntermediateTensors(tensors, metadata)
        return self._forward_exit(
            img,
            tensors["vec"],
            metadata["batch_size"],
            metadata["grid_size"],
            metadata["target_size"],
        )

    def _forward_exit(self, img, vec, batch_size, grid_size, target_size):
        t, h, w = grid_size
        T, H, W = target_size
        img = self.proj_out(self.norm_out(img, vec))
        pt, ph, pw = self.patch_size
        img = img.view(batch_size, t, h, w, self.out_channels, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(batch_size, self.out_channels, t * pt, h * ph, w * pw)
        return img[:, :, :T, :H, :W]
