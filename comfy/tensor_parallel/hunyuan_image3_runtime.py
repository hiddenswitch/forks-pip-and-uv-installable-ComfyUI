"""Batching and execution-local decoding for the installed Hunyuan node."""
from functools import wraps
import math
import sys

import torch
import torch.nn.functional as F

from .. import conds, memory_management, model_management, quant_ops
from ..ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_masked


class KVCache:
    def __init__(self):
        self.key = self.value = None
        self.length = 0

    def append(self, key, value):
        end = self.length + key.shape[2]
        if self.key is None or end > self.key.shape[2]:
            capacity = ((end + 255) // 256) * 256
            shape = (*key.shape[:2], capacity, key.shape[-1])
            new_key, new_value = key.new_empty(shape), value.new_empty(shape)
            if self.length:
                new_key[:, :, :self.length] = self.key[:, :, :self.length]
                new_value[:, :, :self.length] = self.value[:, :, :self.length]
            self.key, self.value = new_key, new_value
        self.key[:, :, self.length:end] = key
        self.value[:, :, self.length:end] = value
        self.length = end
        return self.key[:, :, :end], self.value[:, :, :end]


class PromptCondition(conds.CONDRegular):
    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(tuple(self.cond[index % len(self.cond)] for index in range(batch_size)))

    def can_concat(self, other):
        return isinstance(other, PromptCondition)

    def concat(self, others):
        return self.cond + tuple(prompt for other in others for prompt in other.cond)

    def size(self):
        return [len(self.cond), max(prompt.numel() for prompt in self.cond)]


def attention_forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None):
    source = sys.modules[type(self).__module__]
    batch, length, _ = hidden_states.shape
    query, key, value = source._split_qkv(
        self.qkv_proj(hidden_states), self.num_key_value_heads, self.num_key_value_groups, self.head_dim,
    )
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    query, key = quant_ops.ck.apply_rope_split_half(query, key, freqs)
    query, key = self.query_layernorm(query), self.key_layernorm(key)
    if isinstance(kv_cache, KVCache):
        key, value = kv_cache.append(key, value)
    elif kv_cache is not None:
        if kv_cache[0] is not None:
            key = torch.cat((kv_cache[0], key), dim=2)
            value = torch.cat((kv_cache[1], value), dim=2)
        kv_cache[0], kv_cache[1] = key, value
    out = optimized_attention_masked(
        AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
        self.num_heads, attention_mask, skip_reshape=True, skip_output_reshape=True,
        enable_gqa=True, preferred_attention=self.comfy_attention,
    )
    return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


def batched_forward(original, self, x, timestep, context=None, transformer_options=None,
                    guidance=None, ids=None, cond_latent=None, cond_vit=None, cond_vit_grid=None, **kwargs):
    options = transformer_options or {}
    if ids is None:
        return original(self, x, timestep, context=context, transformer_options=options, guidance=guidance,
                        cond_latent=cond_latent, cond_vit=cond_vit, cond_vit_grid=cond_vit_grid, **kwargs)
    if not isinstance(ids, (list, tuple)):
        ids = (ids,) * x.shape[0]
    # Spectrum owns a separate forecast for each CFG lane.
    if options.get("hy3_spectrum") is not None:
        lanes = options.get("cond_or_uncond", [0])
        return torch.cat([
            original(self, x[index:index + 1], timestep[index:index + 1], context=context,
                     transformer_options=dict(options, cond_or_uncond=[lanes[index * len(lanes) // len(ids)]]),
                     guidance=guidance, ids=prompt, cond_latent=cond_latent, cond_vit=cond_vit,
                     cond_vit_grid=cond_vit_grid, **kwargs)
            for index, prompt in enumerate(ids)
        ])
    source = sys.modules[type(self).__module__]
    pipeline = sys.modules[source.__package__ + ".pipeline"]
    latents = x[:, :, 0] if x.dim() == 5 else x
    schedule, sigmas = options["sample_sigmas"], options["sigmas"]
    step_index = torch.where(sigmas.flatten()[0] == schedule)[0][0]
    timestep_r = (schedule[step_index + 1] * pipeline.NUM_TRAIN_TIMESTEPS).reshape(1).to(x.device, torch.float32)
    steps, embeddings, masks, frequencies = [], [], [], []
    for index, prompt in enumerate(ids):
        step = source._sequence_from_ids(prompt.flatten(), latents[index:index + 1], self.config,
                                         cond_latent=cond_latent, cond_patch_grid=cond_vit_grid)
        length = step["ids"].shape[0]
        steps.append(step)
        embeddings.append(pipeline.build_input_embeddings(
            self, step, latents[index:index + 1], timestep[index:index + 1], guidance, timestep_r,
            cond_latent=cond_latent, cond_vit=cond_vit,
        ))
        masks.append(pipeline.build_attention_mask(step, length, x.dtype, x.device))
        frequencies.append(source.build_rope_freqs(length, self.config.attention_head_dim,
                                                   step["rope_image_info"], self.config.rope_theta, device=x.device))
    length = max(value.shape[1] for value in embeddings)
    for index, value in enumerate(embeddings):
        padding = length - value.shape[1]
        if padding:
            embeddings[index] = F.pad(value, (0, 0, 0, padding))
            masks[index] = F.pad(masks[index], (0, padding, 0, padding), value=float("-inf"))
            # Padding queries attend to a real token; their outputs are discarded.
            masks[index][:, :, -padding:, 0] = 0
            frequencies[index] = torch.cat((frequencies[index], frequencies[index][:, :, -1:].expand(
                -1, -1, padding, -1, -1, -1)), dim=2)
    hidden = self.model(torch.cat(embeddings), torch.cat(frequencies), torch.cat(masks), transformer_options=options)
    image_hidden = torch.stack([hidden[index, step["image_slice"]] for index, step in enumerate(steps)])
    pred = self.final_layer(image_hidden, self.time_embed_2(timestep), steps[0]["token_height"], steps[0]["token_width"])
    return pred.unsqueeze(2) if x.dim() == 5 else pred


def install(model, base):
    initialize = model.HunyuanImage3Attention.__init__

    @wraps(initialize)
    def attention_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.comfy_attention = ComfyAttention()

    model.HunyuanImage3Attention.__init__ = attention_init
    model.HunyuanImage3Attention.forward = attention_forward
    forward = model.HunyuanImage3.forward

    @wraps(forward)
    def diffusion_forward(self, *args, **kwargs):
        return batched_forward(forward, self, *args, **kwargs)

    model.HunyuanImage3.forward = diffusion_forward
    extra_conds = base.HunyuanImage3Model.extra_conds

    @wraps(extra_conds)
    def conditioning(self, **kwargs):
        result = extra_conds(self, **kwargs)
        if kwargs.get("ids") is not None:
            result["ids"] = PromptCondition((kwargs["ids"],))
        return result

    base.HunyuanImage3Model.extra_conds = conditioning
    memory_required = base.HunyuanImage3Model.memory_required

    @wraps(memory_required)
    def memory(self, input_shape, cond_shapes=None):
        cond_shapes = cond_shapes or {}
        config = self.diffusion_model.config
        batch = input_shape[0]
        lengths = [shape[-1] for shape in cond_shapes.get("ids", ())]
        length = max(lengths) if lengths else math.prod(input_shape[2:]) + 1024
        element_size = model_management.dtype_size(self.get_dtype_inference())
        # Dense mixed attention mask, QKV/residuals, router and expert outputs.
        activations = batch * (length * length * element_size + length * config.hidden_size * 24)
        banks = self.diffusion_model.model.layers[0].mlp
        largest_bank = max(memory_management.vram_aligned_size([bank.weight, bank.bias])
                           for bank in (banks.experts_gate_up_proj, banks.experts_down_proj))
        # The prefetch queue can have a current cast and the next bank in flight.
        return max(memory_required(self, input_shape, cond_shapes), activations + 2 * largest_bank)

    base.HunyuanImage3Model.memory_required = memory
