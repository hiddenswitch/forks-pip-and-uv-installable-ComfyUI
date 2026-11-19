"""MAGI-2 refiner: the 1080p stage that re-noises an upsampled preview latent and denoises it in a few steps.

A plain residual transformer (no hyper-connections, sinks or experts) with gated grouped-query
attention. Video queries attend to the video tokens of the neighbouring blocks of a block grid
plus every text and audio token; text and audio queries attend to everything. The residual
stream is in the compute dtype until the first MLP output, which the reference keeps in fp32.
"""

from __future__ import annotations

import itertools
import math
import mmap
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from ... import model_management, model_prefetch, patcher_extension
from ...pipeline_parallel import PipelineIntermediateTensors, PipelineMissingLayer, PipelineStageConfig
from ...pipeline_parallel.types import pack_pipeline_value, prepare_model_parallel_value, unpack_pipeline_value
from ...tensor_parallel.operations import column_parallel_linear, local_size, row_parallel_linear
from ..modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from .model import (MLP_GATE_CHUNK_BYTES, Magi2Layout, MultiModalityRMSNorm, PostAdapter, PreAdapter, fourier_rope, make_modality_linear, modality_linear,
                    qk_norm_rope, swiglu7, video_tokens)

QKV_PROJECTION_CHUNK_BYTES = 128 * 1024**2
WINDOW_KEY_BUDGET = 1 << 19
WINDOW_KV_CHUNK_BYTES = 256 * 1024**2


def _host_residual(shape, dtype):
    # Reserve enough RAM for the next fp32 residual as well. A file mapping lets
    # the OS reclaim completed token ranges when a whole pair cannot fit.
    count = math.prod(shape)
    if model_management.get_free_memory(torch.device("cpu")) >= count * (dtype.itemsize + 4):
        return torch.empty(shape, dtype=dtype, device="cpu")
    from ...cmd import folder_paths

    directory = Path(folder_paths.get_temp_directory())
    directory.mkdir(parents=True, exist_ok=True)
    size = count * dtype.itemsize
    with tempfile.TemporaryFile(dir=directory) as scratch:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(scratch.fileno(), 0, size)
        else:
            scratch.truncate(size)
        mapping = mmap.mmap(scratch.fileno(), size, access=mmap.ACCESS_WRITE)
    return torch.frombuffer(mapping, dtype=dtype).view(shape)


def _attend(q, k, v, comfy_attention, transformer_options, mask=None):
    """q [B, Hq, Lq, D], k/v [B, Hkv, Lk, D] -> [B, Hq, Lq, D]."""
    return optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), q.shape[1],
                               preferred_attention=comfy_attention, mask=mask, skip_reshape=True, skip_output_reshape=True,
                               enable_gqa=k.shape[1] != q.shape[1], transformer_options=transformer_options)


def block_window_attention(q, k, v, video_shape, block_size, radius, comfy_attention, transformer_options, out=None):
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

    # Pad token indices instead of the query, key and value tensors. Edge
    # queries scatter to distinct scratch positions beyond the returned tokens.
    padding = math.prod(padded) - video
    indices = torch.full(padded, -1, dtype=torch.long, device=k.device)
    indices[:t, :h, :w] = torch.arange(video, device=k.device).view(t, h, w)
    if padding:
        indices.masked_scatter_(indices < 0, torch.arange(tokens, tokens + padding, device=k.device))
    block_map = indices.view(grid[0], block_size[0], grid[1], block_size[1], grid[2], block_size[2])
    block_map = block_map.permute(0, 2, 4, 1, 3, 5).reshape(*grid, block_tokens)
    query_map = block_map.flatten(0, 2)
    pad = tuple(r for r in reversed(radius) for _ in range(2))
    key_map = torch.nn.functional.pad(block_map, (0, 0) + pad, value=-1).flatten(0, 2)
    padded_grid = tuple(count + 2 * r for count, r in zip(grid, radius))
    offsets = [(dt * padded_grid[1] + dh) * padded_grid[2] + dw
               for dt, dh, dw in itertools.product(*(range(2 * r + 1) for r in radius))]
    offsets = torch.tensor(offsets, device=k.device)
    query_coords = torch.stack(torch.meshgrid(*(torch.arange(count, device=k.device) for count in grid), indexing="ij"), dim=-1).view(-1, 3)
    query_origin = (query_coords[:, 0] * padded_grid[1] + query_coords[:, 1]) * padded_grid[2] + query_coords[:, 2]

    tail_indices = torch.arange(video, tokens, device=k.device)
    window_keys = len(offsets) * block_tokens + tokens - video
    bytes_per_key = heads_kv * head_dim * (k.element_size() + v.element_size())
    gather_bytes = WINDOW_KV_CHUNK_BYTES
    if k.is_cuda:
        # Attention may repack both gathered tensors and needs room for its score tiles.
        gather_bytes = min(gather_bytes, model_management.get_free_memory(k.device) // 4)
    key_budget = min(WINDOW_KEY_BUDGET, gather_bytes // bytes_per_key)
    chunk = max(1, key_budget // (window_keys * batch))
    if out is None:
        out = q.new_empty((batch, tokens + padding, heads_q, head_dim))
    for start in range(0, query_map.shape[0], chunk):
        neighbours = query_origin[start:start + chunk, None] + offsets
        count = neighbours.shape[0]
        key_ids = torch.cat((key_map[neighbours].flatten(1), tail_indices.expand(count, -1)), dim=1)
        mask = (key_ids >= 0) & (key_ids < tokens)
        key_ids.masked_fill_(~mask, 0)
        keys = k[:, key_ids].flatten(0, 1).transpose(1, 2)
        values = v[:, key_ids].flatten(0, 1).transpose(1, 2)
        dest = query_map[start:start + count]
        query_ids = dest.to(q.device)
        queries = q[:, query_ids.clamp_max(video - 1)].to(k.device).flatten(0, 1).transpose(1, 2)
        result = _attend(queries, keys, values, comfy_attention, transformer_options,
                         mask=mask.repeat(batch, 1).view(batch * count, 1, 1, -1))
        out[:, dest.flatten().to(out.device)] = result.transpose(1, 2).reshape(batch, -1, heads_q, head_dim).to(out.device)
        del queries, keys, values, result
    if tokens > video:
        out[:, video:tokens] = _attend(q[:, video:].to(k.device).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       comfy_attention, transformer_options).transpose(1, 2)
    return out[:, :tokens]


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

    def _project(self, x, freqs, groups, compute_dtype):
        batch, tokens, _ = x.shape
        x = self.pre_norm(x, groups).to(compute_dtype)
        qkv = modality_linear(self.linear_qkv, x, groups)
        # Q/K normalization promotes each chunk to fp32. Keep V and the packed
        # projection in compute precision until then.
        if qkv.dtype != compute_dtype:
            qkv = qkv.float()
        gates = modality_linear(self.linear_g, x, groups).float()
        del x
        q, k, v = qkv.split((self.heads_q * self.head_dim, self.heads_kv * self.head_dim, self.heads_kv * self.head_dim), dim=-1)
        q, k = qk_norm_rope(q.view(batch, tokens, self.heads_q, self.head_dim), k.view(batch, tokens, self.heads_kv, self.head_dim),
                            self.q_norm, self.k_norm, freqs, groups)
        v = v.reshape(batch, tokens, self.heads_kv, self.head_dim)
        return q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype), gates

    def _token_chunks(self, tokens, groups, chunk_size):
        splits = groups if self.pre_norm.num_modality > 1 else (tokens,)
        offset = 0
        for index, length in enumerate(splits):
            for start in range(offset, offset + length, chunk_size):
                end = min(start + chunk_size, offset + length)
                chunk_groups = [0] * len(splits)
                chunk_groups[index] = end - start
                yield slice(start, end), chunk_groups
            offset += length

    def _project_inputs(self, x, freqs, groups, compute_dtype, query_buffer=None):
        batch, tokens, _ = x.shape
        width = (self.heads_q + 2 * self.heads_kv) * self.head_dim
        chunk_size = max(1, QKV_PROJECTION_CHUNK_BYTES // (batch * width * 4))
        if tokens <= chunk_size and query_buffer is None:
            return self._project(x.to(freqs.device), freqs, groups, compute_dtype)

        # Write the normalized projections directly to the window attention inputs.
        # A full packed QKV copy at 1088p takes another 5.88 GiB.
        q = query_buffer if query_buffer is not None else freqs.new_empty((batch, tokens, self.heads_q, self.head_dim), dtype=compute_dtype)
        k = freqs.new_empty((batch, tokens, self.heads_kv, self.head_dim), dtype=compute_dtype)
        v = torch.empty_like(k)
        gates = freqs.new_empty((batch, tokens, self.heads_q), dtype=torch.float32)
        for span, chunk_groups in self._token_chunks(tokens, groups, chunk_size):
            q[:, span], k[:, span], v[:, span], gates[:, span] = self._project(
                x[:, span].to(freqs.device), freqs[:, span], chunk_groups, compute_dtype)
        return q, k, v, gates

    def _project_output(self, attended, gates, groups, compute_dtype, reuse_attended=False):
        batch, tokens, heads, dim = attended.shape
        chunk_size = max(1, QKV_PROJECTION_CHUNK_BYTES // (batch * heads * dim * 4))

        def project(values, weights, splits):
            gated = (values.to(weights.device) * weights.sigmoid().unsqueeze(-1)).to(compute_dtype)
            return modality_linear(self.linear_proj, gated.flatten(2), splits)

        if tokens <= chunk_size:
            return project(attended, gates, groups)
        # Preserve fp32 gating and the post-projection dtype without a full
        # fp32 video tensor or an extra concatenation of the modality outputs.
        output = None
        for span, chunk_groups in self._token_chunks(tokens, groups, chunk_size):
            part = project(attended[:, span], gates[:, span], chunk_groups)
            if output is None:
                if reuse_attended and heads * dim == part.shape[-1] and attended.dtype == part.dtype and attended.device == part.device:
                    output = attended.flatten(2)
                else:
                    output = part.new_empty((batch, tokens, part.shape[-1]))
            output[:, span] = part
        return output

    def forward(self, x, freqs, groups, video_shape, compute_dtype, transformer_options):
        buffer = None
        if not torch.is_grad_enabled() and x.nbytes > QKV_PROJECTION_CHUNK_BYTES:
            # Each query appears in one window. Once gathered, its slot can hold
            # that window's result. The tail still attends to the separate K/V tensors.
            padded_video = math.prod(math.ceil(n / b) * b for n, b in zip(video_shape, self.block_size))
            capacity = x.shape[1] + padded_video - math.prod(video_shape)
            shape = (x.shape[0], capacity, self.heads_q, self.head_dim)
            query_bytes = math.prod(shape) * compute_dtype.itemsize
            kv_bytes = 2 * self.heads_kv * self.head_dim * compute_dtype.itemsize
            kv_and_gates = x.shape[0] * x.shape[1] * (kv_bytes + self.heads_q * 4)
            required = query_bytes + kv_and_gates + WINDOW_KV_CHUNK_BYTES + QKV_PROJECTION_CHUNK_BYTES
            if x.device != freqs.device and required > model_management.get_free_memory(freqs.device):
                # Each video query is gathered once. Keep K/V resident for overlapping windows.
                buffer = _host_residual(shape, compute_dtype)
            else:
                buffer = freqs.new_empty(shape, dtype=compute_dtype)
        q, k, v, gates = self._project_inputs(x, freqs, groups, compute_dtype,
                                              None if buffer is None else buffer[:, :x.shape[1]])
        del x
        out = block_window_attention(q, k, v, video_shape, self.block_size, self.radius,
                                     self.comfy_attention, transformer_options, out=buffer)
        del q, k, v
        return self._project_output(out, gates, groups, compute_dtype, reuse_attended=buffer is not None)


class RefinerMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_modality, dtype=None, device=None, operations=None):
        super().__init__()
        self.gate_width = 2 * local_size(operations, intermediate_size, "MAGI-2 refiner MLP width")
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality, device=device)
        self.up_gate_proj = make_modality_linear(num_modality, lambda: column_parallel_linear(
            operations, hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype))
        self.down_proj = make_modality_linear(num_modality, lambda: row_parallel_linear(
            operations, intermediate_size, hidden_size, bias=False, device=device, dtype=dtype))

    def _project(self, x, groups, compute_dtype):
        x = modality_linear(self.up_gate_proj, self.pre_norm(x, groups).to(compute_dtype), groups).float()
        return modality_linear(self.down_proj, swiglu7(x).to(compute_dtype), groups).float()

    def forward(self, x, groups, compute_dtype, attention_output=None, allow_host_residual=False):
        def project(tokens, splits, attention):
            if attention is None:
                return self._project(tokens, splits, compute_dtype)
            residual = tokens.to(attention.device) + attention
            return residual + self._project(residual, splits, compute_dtype)

        # The fp32 gate and its activation workspace coexist during SwiGLU.
        bytes_per_token = x.shape[0] * self.gate_width * 4 * 2
        chunk_size = max(1, MLP_GATE_CHUNK_BYTES // max(1, bytes_per_token))
        host_stream = attention_output is not None and x.device != attention_output.device
        if allow_host_residual and not host_stream and x.is_cuda:
            # Attention and resident weights can consume the headroom measured before the first layer.
            output_bytes = x.numel() * torch.float32.itemsize
            host_stream = output_bytes + MLP_GATE_CHUNK_BYTES > model_management.get_free_memory(x.device)
        if x.shape[1] <= chunk_size and not host_stream:
            return project(x, groups, attention_output)

        # The gate expands each token independently. Keep only the fp32 residual output
        # across chunks and preserve each modality's normalization and projections.
        out = _host_residual(x.shape, torch.float32) if host_stream else torch.empty_like(x, dtype=torch.float32)
        offset = 0
        splits = groups if self.pre_norm.num_modality > 1 else (x.shape[1],)
        for index, part in enumerate(x.split(splits, dim=1)):
            if part.shape[1] == 0:
                continue
            chunk_groups = [0] * len(splits)
            for chunk in part.split(chunk_size, dim=1):
                length = chunk.shape[1]
                chunk_groups[index] = length
                span = slice(offset, offset + length)
                attention = None if attention_output is None else attention_output[:, span]
                out[:, span] = project(chunk, chunk_groups, attention)
                offset += length
        return out


class RefinerLayer(nn.Module):
    def __init__(self, layer_index, hidden_size, head_dim, num_query_groups, multimodal_layers, intermediate_size, block_size, radius,
                 dtype=None, device=None, operations=None):
        super().__init__()
        num_modality = 3 if layer_index in multimodal_layers else 1
        self.attention = RefinerAttention(hidden_size, hidden_size // head_dim, num_query_groups, head_dim, num_modality, block_size, radius,
                                          dtype=dtype, device=device, operations=operations)
        self.mlp = RefinerMLP(hidden_size, intermediate_size, num_modality, dtype=dtype, device=device, operations=operations)

    def forward(self, x, freqs, groups, video_shape, compute_dtype, transformer_options={}):
        attention = self.attention(x, freqs, groups, video_shape, compute_dtype, transformer_options)
        return self.mlp(x, groups, compute_dtype, attention_output=attention,
                        allow_host_residual=transformer_options.get("magi2_refiner_host_residual", False))


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
        shape = (video.shape[0], sum(layout.groups), self.pre_adapter.video_embedder.weight.shape[0])
        # An fp32 input and output plus the attention result coexist in each layer.
        residual_dtype = self.dtype or torch.get_default_dtype()
        residual_bytes = math.prod(shape) * (8 + residual_dtype.itemsize)
        host = None
        if (device.type == "cuda" and self.pipeline_stage is None and not torch.is_grad_enabled()
                and not transformer_options.get("patches_replace", {}).get("dit")
                and not model_prefetch.malloc_graph_enabled(device)
                and not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
                and residual_bytes > QKV_PROJECTION_CHUNK_BYTES and not model_management.is_unified_memory_device(device)):
            transformer_options["magi2_refiner_host_residual"] = True
            if residual_bytes + WINDOW_KV_CHUNK_BYTES > model_management.get_free_memory(device):
                host = _host_residual(shape, residual_dtype)
        hidden = self.pre_adapter(video_tokens(video), audio.transpose(1, 2), context, None, self.dtype, out=host)
        del host
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
        return self._output(hidden, layout, (video.dtype, audio.dtype), isinstance(x, (list, tuple)), device)

    def _run_layers(self, hidden, freqs, layout, transformer_options, start, end):
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        layers = [self.block.layers[index] for index in range(start, end)]
        prefetch_queue = model_prefetch.make_prefetch_queue(layers, freqs.device, transformer_options)
        for index in range(start, end):
            layer = self.block.layers[index]
            model_prefetch.prefetch_queue_pop(prefetch_queue, freqs.device, layer, malloc_scope="block")
            transformer_options["block_index"] = index
            if ("double_block", index) in blocks_replace:
                def block_wrap(args, layer=layer):
                    return {"img": layer(args["img"], args["freqs"], layout.groups, layout.video_shape, self.dtype,
                                         transformer_options=args["transformer_options"])}
                hidden = blocks_replace[("double_block", index)](
                    {"img": hidden, "freqs": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                hidden = layer(hidden, freqs, layout.groups, layout.video_shape, self.dtype, transformer_options)
        model_prefetch.prefetch_queue_pop(prefetch_queue, freqs.device, None, malloc_scope="block")
        return hidden

    def _output(self, hidden, layout, dtypes, nested, device=None):
        t, h, w = layout.video_shape
        video = hidden[:, :layout.video_tokens]
        audio = hidden[:, layout.groups[0]:layout.groups[0] + layout.audio_len]
        if device is not None and hidden.device != device:
            chunk = max(1, QKV_PROJECTION_CHUNK_BYTES // (hidden.shape[0] * hidden.shape[-1] * 4))
            pairs = [self.post_adapter(video[:, i:i + chunk].to(device), audio[:, i:i + chunk].to(device))
                     for i in range(0, max(1, video.shape[1], audio.shape[1]), chunk)]
            video, audio = (torch.cat(parts, dim=1) for parts in zip(*pairs))
        else:
            video, audio = self.post_adapter(video, audio)
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
        hidden = self._run_layers(tensors["hidden_states"], tensors["freqs"], layout, transformer_options,
                                  self.pipeline_stage.start_layer, self.pipeline_stage.end_layer)
        if not self.pipeline_stage.is_last:
            return PipelineIntermediateTensors({**tensors, "hidden_states": hidden}, metadata)
        return self._output(hidden, layout, unpack_pipeline_value(metadata["output_dtypes"], tensors), metadata["nested"])
