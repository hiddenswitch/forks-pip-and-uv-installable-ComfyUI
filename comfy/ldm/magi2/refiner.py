"""MAGI-2 refiner: the 1080p stage that re-noises an upsampled preview latent and denoises it in a few steps.

A plain residual transformer (no hyper-connections, sinks or experts) with gated grouped-query
attention. Video queries attend to the video tokens of the neighbouring blocks of a block grid
plus every text and audio token; text and audio queries attend to everything. The residual
stream is in the compute dtype until the first MLP output, which the reference keeps in fp32.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn

from ... import model_management, model_prefetch, patcher_extension
from ...pipeline_parallel import PipelineIntermediateTensors, PipelineMissingLayer, PipelineStageConfig
from ...pipeline_parallel.types import pack_pipeline_value, prepare_model_parallel_value, unpack_pipeline_value
from ...tensor_parallel.operations import column_parallel_linear, local_size, row_parallel_linear
from ..modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from .model import (Magi2Layout, MultiModalityRMSNorm, PostAdapter, PreAdapter, fourier_rope, ieee_fp32_matmul, make_modality_linear, modality_linear,
                    qk_norm_rope, swiglu7, video_tokens)

WINDOW_KEY_BUDGET = 1 << 19


def _attend(q, k, v, comfy_attention, transformer_options, mask=None):
    """q [B, Hq, Lq, D], k/v [B, Hkv, Lk, D] -> [B, Hq, Lq, D]."""
    return optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), q.shape[1],
                               preferred_attention=comfy_attention, mask=mask, skip_reshape=True, skip_output_reshape=True,
                               enable_gqa=k.shape[1] != q.shape[1], transformer_options=transformer_options)


def block_window_attention(q, k, v, video_shape, block_size, radius, comfy_attention, transformer_options):
    """Attention where video tokens see their block-grid neighbourhood plus all non-video tokens.

    q [B, N, Hq, D] and k/v [B, N, Hkv, D] hold the video tokens first in (t, h, w) raster order.
    The video grid is cut into blocks of block_size (edge blocks are partial); a video query in block
    (i, j, l) attends to the video keys of blocks within radius on every axis, clipped at the grid,
    and to every key after the video. Queries after the video attend to all keys.
    """
    batch, tokens, heads_q, head_dim = q.shape
    heads_kv = k.shape[2]
    t, h, w = video_shape
    video = t * h * w
    grid = tuple(math.ceil(size / block) for size, block in zip(video_shape, block_size))
    padded = tuple(count * block for count, block in zip(grid, block_size))
    block_tokens = math.prod(block_size)

    def blocks(x):
        heads = x.shape[2]
        x = x[:, :video].view(batch, t, h, w, heads, head_dim)
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded[2] - w, 0, padded[1] - h, 0, padded[0] - t))
        x = x.view(batch, grid[0], block_size[0], grid[1], block_size[1], grid[2], block_size[2], heads, head_dim)
        return x.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(batch, *grid, block_tokens, heads, head_dim)

    valid = torch.zeros(padded, dtype=torch.bool, device=q.device)
    valid[:t, :h, :w] = True
    valid = valid.view(grid[0], block_size[0], grid[1], block_size[1], grid[2], block_size[2]).permute(0, 2, 4, 1, 3, 5).reshape(*grid, block_tokens)
    # surround the block grid with empty blocks so every query block has the same neighbourhood shape
    pad = [r for r in reversed(radius) for _ in range(2)]
    key_blocks = torch.nn.functional.pad(blocks(k), (0, 0, 0, 0, 0, 0) + tuple(pad)).flatten(1, 3)
    value_blocks = torch.nn.functional.pad(blocks(v), (0, 0, 0, 0, 0, 0) + tuple(pad)).flatten(1, 3)
    key_valid = torch.nn.functional.pad(valid, (0, 0) + tuple(pad)).flatten(0, 2)
    query_blocks = blocks(q).flatten(1, 3)
    padded_grid = tuple(count + 2 * r for count, r in zip(grid, radius))
    offsets = [(dt * padded_grid[1] + dh) * padded_grid[2] + dw
               for dt, dh, dw in itertools.product(*(range(2 * r + 1) for r in radius))]
    offsets = torch.tensor(offsets, device=q.device)
    query_coords = torch.stack(torch.meshgrid(*(torch.arange(count, device=q.device) for count in grid), indexing="ij"), dim=-1).view(-1, 3)
    query_origin = (query_coords[:, 0] * padded_grid[1] + query_coords[:, 1]) * padded_grid[2] + query_coords[:, 2]

    tail_k, tail_v = k[:, video:], v[:, video:]
    tail = tokens - video
    window_keys = len(offsets) * block_tokens + tail
    chunk = max(1, WINDOW_KEY_BUDGET // (window_keys * batch))
    out = torch.empty_like(query_blocks)
    for start in range(0, query_blocks.shape[1], chunk):
        neighbours = query_origin[start:start + chunk, None] + offsets
        count = neighbours.shape[0]
        keys = torch.cat((key_blocks[:, neighbours].flatten(2, 3), tail_k.unsqueeze(1).expand(-1, count, -1, -1, -1)), dim=2)
        values = torch.cat((value_blocks[:, neighbours].flatten(2, 3), tail_v.unsqueeze(1).expand(-1, count, -1, -1, -1)), dim=2)
        mask = torch.cat((key_valid[neighbours].flatten(1), key_valid.new_ones(count, tail)), dim=1)
        queries = query_blocks[:, start:start + count].flatten(0, 1).transpose(1, 2)
        result = _attend(queries, keys.flatten(0, 1).transpose(1, 2), values.flatten(0, 1).transpose(1, 2), comfy_attention, transformer_options,
                         mask=mask.repeat(batch, 1).view(batch * count, 1, 1, -1))
        out[:, start:start + count] = result.transpose(1, 2).unflatten(0, (batch, count))
    out = out.view(batch, *grid, *block_size, heads_q, head_dim).permute(0, 1, 4, 2, 5, 3, 6, 7, 8).reshape(batch, *padded, heads_q, head_dim)
    video_out = out[:, :t, :h, :w].reshape(batch, video, heads_q, head_dim)
    if not tail:
        return video_out
    tail_out = _attend(q[:, video:].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), comfy_attention, transformer_options).transpose(1, 2)
    return torch.cat((video_out, tail_out), dim=1)


class RefinerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim, num_modality, block_size, radius, dtype=None, device=None, operations=None):
        super().__init__()
        self.head_dim = head_dim
        self.heads_q = local_size(operations, num_heads_q, "MAGI-2 refiner attention heads")
        self.heads_kv = local_size(operations, num_heads_kv, "MAGI-2 refiner key/value heads")
        self.block_size = block_size
        self.radius = radius
        q_size, kv_size = num_heads_q * head_dim, num_heads_kv * head_dim
        self.comfy_attention = ComfyAttention()
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality, device=device)
        self.linear_qkv = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, q_size + 2 * kv_size, bias=False, section_sizes=(q_size, kv_size, kv_size), device=device, dtype=dtype))
        self.linear_g = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, num_heads_q, bias=False, device=device, dtype=dtype))
        self.linear_proj = make_modality_linear(num_modality, lambda: row_parallel_linear(
            operations, q_size, hidden_size, bias=False, device=device, dtype=dtype))
        self.q_norm = MultiModalityRMSNorm(head_dim, num_modality, device=device)
        self.k_norm = MultiModalityRMSNorm(head_dim, num_modality, device=device)

    def forward(self, x, freqs, groups, video_shape, compute_dtype, transformer_options):
        batch, tokens, _ = x.shape
        x = self.pre_norm(x, groups).to(compute_dtype)
        qkv = modality_linear(self.linear_qkv, x, groups).float()
        gates = modality_linear(self.linear_g, x, groups).float()
        del x
        q, k, v = qkv.split((self.heads_q * self.head_dim, self.heads_kv * self.head_dim, self.heads_kv * self.head_dim), dim=-1)
        q, k = qk_norm_rope(q.view(batch, tokens, self.heads_q, self.head_dim), k.view(batch, tokens, self.heads_kv, self.head_dim),
                            self.q_norm, self.k_norm, freqs, groups)
        v = v.reshape(batch, tokens, self.heads_kv, self.head_dim)
        out = block_window_attention(q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype), video_shape, self.block_size, self.radius,
                                     self.comfy_attention, transformer_options)
        out = (out * torch.sigmoid(gates).unsqueeze(-1)).to(compute_dtype)
        return modality_linear(self.linear_proj, out.reshape(batch, tokens, -1), groups)


class RefinerMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_modality, dtype=None, device=None, operations=None):
        super().__init__()
        local_size(operations, intermediate_size, "MAGI-2 refiner MLP width")
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality, device=device)
        self.up_gate_proj = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype))
        self.down_proj = make_modality_linear(num_modality, lambda: row_parallel_linear(
            operations, intermediate_size, hidden_size, bias=False, device=device, dtype=dtype))

    def forward(self, x, groups, compute_dtype):
        x = modality_linear(self.up_gate_proj, self.pre_norm(x, groups).to(compute_dtype), groups).float()
        return modality_linear(self.down_proj, swiglu7(x).to(compute_dtype), groups).float()


class RefinerLayer(nn.Module):
    def __init__(self, layer_index, hidden_size, head_dim, num_query_groups, multimodal_layers, intermediate_size, block_size, radius,
                 dtype=None, device=None, operations=None):
        super().__init__()
        num_modality = 3 if layer_index in multimodal_layers else 1
        self.attention = RefinerAttention(hidden_size, hidden_size // head_dim, num_query_groups, head_dim, num_modality, block_size, radius,
                                          dtype=dtype, device=device, operations=operations)
        self.mlp = RefinerMLP(hidden_size, intermediate_size, num_modality, dtype=dtype, device=device, operations=operations)

    def forward(self, x, freqs, groups, video_shape, compute_dtype, transformer_options={}):
        x = x + self.attention(x, freqs, groups, video_shape, compute_dtype, transformer_options)
        return x + self.mlp(x, groups, compute_dtype)


class Magi2RefinerModel(nn.Module):
    """MAGI-2 refiner DiT: video (and optional audio) velocity for packed video, audio and text tokens."""

    def __init__(self, num_layers=30, hidden_size=4096, head_dim=128, num_query_groups=8, video_in_channels=48, audio_in_channels=64,
                 text_in_channels=5120, intermediate_size=10920, multimodal_layers=(0, 1, 28, 29), window_block_size=(8, 4, 4),
                 window_radius=(2, 2, 2), image_model=None, dtype=None, device=None, operations=None,
                 pipeline_stage: PipelineStageConfig | None = None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.audio_in_channels = audio_in_channels
        self.pipeline_stage = pipeline_stage
        is_first = pipeline_stage is None or pipeline_stage.is_first
        is_last = pipeline_stage is None or pipeline_stage.is_last
        self.pre_adapter = PreAdapter(hidden_size, head_dim, video_in_channels, audio_in_channels, text_in_channels, device=device) if is_first else PipelineMissingLayer()
        self.block = nn.Module()
        self.block.layers = nn.ModuleList([
            RefinerLayer(index, hidden_size, head_dim, num_query_groups, multimodal_layers, intermediate_size, tuple(window_block_size),
                         tuple(window_radius), dtype=dtype, device=device, operations=operations)
            if pipeline_stage is None or pipeline_stage.start_layer <= index < pipeline_stage.end_layer
            else PipelineMissingLayer()
            for index in range(num_layers)])
        self.post_adapter = PostAdapter(hidden_size, video_in_channels, audio_in_channels, device=device) if is_last else PipelineMissingLayer()

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        if self.pipeline_stage is not None and not self.pipeline_stage.is_first:
            raise RuntimeError("Only the first MAGI-2 refiner pipeline stage accepts model inputs")
        video = x[0] if isinstance(x, (list, tuple)) else x
        compile_allocations = self.pipeline_stage is None and model_prefetch.malloc_graph_enabled(video.device)
        if compile_allocations:
            out = [torch.empty_like(part) for part in x] if isinstance(x, (list, tuple)) else torch.empty_like(x)
            model_prefetch.malloc_graph_begin(video.device)
        with ieee_fp32_matmul():
            graph_out = patcher_extension.WrapperExecutor.new_class_executor(
                self._forward, self,
                patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
            ).execute(x, timestep, context, transformer_options, **kwargs)
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

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        transformer_options = dict(transformer_options)
        video, audio = (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, None)
        if audio is None:
            audio = video.new_zeros(video.shape[0], self.audio_in_channels, 0)
        layout = Magi2Layout(tuple(video.shape[2:]), (), audio.shape[-1], context.shape[1])
        device = video.device
        # the refiner carries no timestep input
        hidden = self.pre_adapter(video_tokens(video), audio.transpose(1, 2), context, None).to(self.dtype)
        freqs = fourier_rope(layout.coords(device), model_management.cast_to(self.pre_adapter.rope.bands, torch.float32, device)).unsqueeze(0).unsqueeze(2)
        start = 0 if self.pipeline_stage is None else self.pipeline_stage.start_layer
        end = self.num_layers if self.pipeline_stage is None else self.pipeline_stage.end_layer
        hidden = self._run_layers(hidden, freqs, layout, transformer_options, start, end)
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

    def _run_layers(self, hidden, freqs, layout, transformer_options, start, end):
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        layers = [self.block.layers[index] for index in range(start, end)]
        prefetch_queue = model_prefetch.make_prefetch_queue(layers, hidden.device, transformer_options)
        for index in range(start, end):
            layer = self.block.layers[index]
            model_prefetch.prefetch_queue_pop(prefetch_queue, hidden.device, layer, malloc_scope="block")
            transformer_options["block_index"] = index
            if ("double_block", index) in blocks_replace:
                def block_wrap(args, layer=layer):
                    return {"img": layer(args["img"], args["freqs"], layout.groups, layout.video_shape, self.dtype,
                                         transformer_options=args["transformer_options"])}
                hidden = blocks_replace[("double_block", index)](
                    {"img": hidden, "freqs": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                hidden = layer(hidden, freqs, layout.groups, layout.video_shape, self.dtype, transformer_options)
        model_prefetch.prefetch_queue_pop(prefetch_queue, hidden.device, None, malloc_scope="block")
        return hidden

    def _output(self, hidden, layout, dtypes, nested):
        t, h, w = layout.video_shape
        video, audio = self.post_adapter(hidden[:, :layout.video_tokens], hidden[:, layout.groups[0]:layout.groups[0] + layout.audio_len])
        video = video.transpose(1, 2).unflatten(2, (t, h, w)).to(dtypes[0])
        if not nested:
            return video
        return [video, audio.transpose(1, 2).to(dtypes[1])]

    def forward_pipeline_stage(self, intermediate: PipelineIntermediateTensors):
        if self.pipeline_stage is None or self.pipeline_stage.is_first:
            raise RuntimeError("MAGI-2 refiner pipeline continuation requires a non-first stage")
        tensors, metadata = intermediate.tensors, intermediate.metadata
        layout = Magi2Layout(*unpack_pipeline_value(metadata["layout"], tensors))
        transformer_options = unpack_pipeline_value(metadata["transformer_options"], tensors)
        with ieee_fp32_matmul():
            hidden = self._run_layers(tensors["hidden_states"], tensors["freqs"], layout, transformer_options,
                                      self.pipeline_stage.start_layer, self.pipeline_stage.end_layer)
            if not self.pipeline_stage.is_last:
                return PipelineIntermediateTensors({**tensors, "hidden_states": hidden}, metadata)
            return self._output(hidden, layout, unpack_pipeline_value(metadata["output_dtypes"], tensors), metadata["nested"])
