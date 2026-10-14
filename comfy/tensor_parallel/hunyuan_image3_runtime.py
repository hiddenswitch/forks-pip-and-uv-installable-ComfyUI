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


class PrefixCondition(PromptCondition):
    def size(self):
        return [len(self.cond), max(prompt["mask"].shape[-1] for prompt in self.cond)]


class PrefixCache:
    def __init__(self, key, value):
        self.key, self.value = key, value

    def append(self, key, value):
        return torch.cat((self.key, key), dim=2), torch.cat((self.value, value), dim=2)


def transformer_forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None, transformer_options=None):
    source = sys.modules[type(self).__module__]
    diffusion = kv_cache is None or isinstance(kv_cache[0], PrefixCache)
    lookahead = source.LayerLookahead(self.layers, hidden_states.device,
                                    diffusion and (transformer_options or {}).get("prefetch_dynamic_vbars", False))
    try:
        for index, layer in enumerate(self.layers):
            lookahead.before(index)
            hidden_states = layer(hidden_states, freqs, attention_mask,
                                  None if kv_cache is None else kv_cache[index])
            lookahead.after(index)
    finally:
        lookahead.abort()
    return hidden_states


def prepare_condition(self, ids, noise, cond_latent=None, cond_vit=None, cond_vit_grid=None):
    source = sys.modules[type(self).__module__]
    pipeline = sys.modules[source.__package__ + ".pipeline"]
    step = source._sequence_from_ids(ids.to(noise.device).flatten(), noise, self.config,
                                    cond_latent=cond_latent, cond_patch_grid=cond_vit_grid)
    length, prefix = step["ids"].numel(), step["timestep_position"]
    freqs = source.build_rope_freqs(length, self.config.attention_head_dim, step["rope_image_info"],
                                    self.config.rope_theta, device=noise.device)
    mask = pipeline.build_attention_mask(step, length, self.dtype, noise.device)
    embeds = pipeline.build_input_embeddings(self, dict(step, ids=step["ids"][:prefix]), None, None, None, None,
                                             cond_latent=cond_latent, cond_vit=cond_vit)
    cache = [[None, None] for _ in self.model.layers]
    self.model(embeds, freqs[:, :, :prefix], mask[:, :, :prefix, :prefix], cache)
    packed = torch.stack([torch.stack(layer) for layer in cache])
    parallel = getattr(self, "tensor_parallel", None)
    if parallel is not None and packed.shape[3] != self.config.num_key_value_heads:
        # Conditioning is transported as full GQA heads; each rank selects its
        # own heads when consuming it. No peer retains state between calls.
        packed = parallel.operations.gather(packed.movedim(3, -1)).movedim(-1, 3).contiguous()
    return {"kv": packed, "length": prefix, "freqs": freqs[:, :, prefix:].clone(),
            "mask": mask[:, :, prefix:].clone()}


def attention_forward(self, hidden_states, freqs, attention_mask=None, kv_cache=None):
    source = sys.modules[type(self).__module__]
    batch, length, _ = hidden_states.shape
    query, key, value = source._split_qkv(
        self.qkv_proj(hidden_states), self.num_key_value_heads, self.num_key_value_groups, self.head_dim,
    )
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    query, key = quant_ops.ck.apply_rope_split_half(query, key, freqs)
    query, key = self.query_layernorm(query), self.key_layernorm(key)
    if isinstance(kv_cache, (KVCache, PrefixCache)):
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


def prepare_batched_inputs(self, x, timestep, options, guidance, ids, cond_latent=None,
                           cond_vit=None, cond_vit_grid=None, hy3_prefix=None):
    source = sys.modules[type(self).__module__]
    pipeline = sys.modules[source.__package__ + ".pipeline"]
    latents = x[:, :, 0] if x.dim() == 5 else x
    schedule, sigmas = options["sample_sigmas"], options["sigmas"]
    step_index = torch.where(sigmas.flatten()[0] == schedule)[0][0]
    timestep_r = (schedule[step_index + 1] * pipeline.NUM_TRAIN_TIMESTEPS).reshape(1).to(x.device, torch.float32)
    steps, embeddings, masks, frequencies = [], [], [], []
    prefixes = []
    max_prefix = max(state["length"] for state in hy3_prefix) if hy3_prefix is not None else 0
    for index, prompt in enumerate(ids):
        step = source._sequence_from_ids(prompt.flatten(), latents[index:index + 1], self.config,
                                         cond_latent=cond_latent, cond_patch_grid=cond_vit_grid)
        length = step["ids"].shape[0]
        if hy3_prefix is not None:
            state = hy3_prefix[index]
            prefix = state["length"]
            step = dict(step, ids=step["ids"][prefix:], cond_blocks=[],
                        image_slice=slice(step["image_slice"].start - prefix, step["image_slice"].stop - prefix))
            for name in ("timestep_position", "guidance_position", "timestep_r_position"):
                if step[name] is not None:
                    step[name] -= prefix
            padding = max_prefix - prefix
            masks.append(torch.cat((F.pad(state["mask"][..., :prefix], (0, padding), value=float("-inf")),
                                    state["mask"][..., prefix:]), dim=-1) if padding else state["mask"])
            frequencies.append(state["freqs"])
            prefixes.append(F.pad(state["kv"], (0, 0, 0, padding)) if padding else state["kv"])
        else:
            masks.append(pipeline.build_attention_mask(step, length, x.dtype, x.device))
            frequencies.append(source.build_rope_freqs(length, self.config.attention_head_dim,
                                                       step["rope_image_info"], self.config.rope_theta, device=x.device))
        steps.append(step)
        embeddings.append(pipeline.build_input_embeddings(
            self, step, latents[index:index + 1], timestep[index:index + 1], guidance, timestep_r,
            cond_latent=cond_latent if hy3_prefix is None else None,
            cond_vit=cond_vit if hy3_prefix is None else None,
        ))
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
    cache = None
    if prefixes:
        packed = torch.cat(prefixes, dim=2)
        parallel = getattr(self, "tensor_parallel", None)
        heads = self.model.layers[0].self_attn.num_key_value_heads
        start = parallel.rank * heads if parallel is not None and heads != packed.shape[3] else 0
        cache = [PrefixCache(layer[0, :, start:start + heads], layer[1, :, start:start + heads]) for layer in packed]
    return steps, torch.cat(embeddings), torch.cat(frequencies), torch.cat(masks), cache


def batched_forward(original, self, x, timestep, context=None, transformer_options=None,
                    guidance=None, ids=None, cond_latent=None, cond_vit=None, cond_vit_grid=None,
                    hy3_prefix=None, **kwargs):
    options = transformer_options or {}
    if ids is None:
        return original(self, x, timestep, context=context, transformer_options=options, guidance=guidance,
                        cond_latent=cond_latent, cond_vit=cond_vit, cond_vit_grid=cond_vit_grid, **kwargs)
    if not isinstance(ids, (list, tuple)):
        ids = (ids,) * x.shape[0]
    ids = tuple(prompt.to(device=x.device) for prompt in ids)
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
    steps, embeddings, frequencies, masks, cache = prepare_batched_inputs(
        self, x, timestep, options, guidance, ids, cond_latent, cond_vit, cond_vit_grid, hy3_prefix)
    hidden = self.model(embeddings, frequencies, masks, kv_cache=cache, transformer_options=options)
    image_hidden = torch.stack([hidden[index, step["image_slice"]] for index, step in enumerate(steps)])
    pred = self.final_layer(image_hidden, self.time_embed_2(timestep), steps[0]["token_height"], steps[0]["token_width"])
    return pred.unsqueeze(2) if x.dim() == 5 else pred


def install(model, base):
    model.HunyuanImage3.supports_prefix_cache = True
    initialize = model.HunyuanImage3Attention.__init__

    @wraps(initialize)
    def attention_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.comfy_attention = ComfyAttention()

    model.HunyuanImage3Attention.__init__ = attention_init
    model.HunyuanImage3Attention.forward = attention_forward
    model.HunyuanImage3Model.forward = transformer_forward
    model.HunyuanImage3.prepare_condition = prepare_condition
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
            options = self.current_patcher.model_options if self.current_patcher is not None else {}
            spectrum = options.get("transformer_options", {}).get("hy3_spectrum")
            if (self.diffusion_model.supports_prefix_cache and not self.diffusion_model.config.cfg_distilled and spectrum is None
                    and kwargs.get("hooks") is None and kwargs.get("noise") is not None
                    and kwargs.get("cond_latent") is None):
                prefix = self.execute_diffusion_model("prepare_condition", kwargs["ids"], kwargs["noise"],
                                                     kwargs.get("cond_latent"), kwargs.get("cond_vit"), kwargs.get("cond_vit_grid"))
                result["hy3_prefix"] = PrefixCondition((prefix,))
        return result

    base.HunyuanImage3Model.extra_conds = conditioning
    extra_conds_shapes = base.HunyuanImage3Model.extra_conds_shapes

    @wraps(extra_conds_shapes)
    def conditioning_shapes(self, **kwargs):
        result = extra_conds_shapes(self, **kwargs)
        if kwargs.get("ids") is not None:
            result["ids"] = [1, kwargs["ids"].numel()]
        return result

    base.HunyuanImage3Model.extra_conds_shapes = conditioning_shapes
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
        if not config.cfg_distilled:
            prefix = max(0, length - math.prod(input_shape[2:]))
            # Stored K/V, unequal-prefix padding and the concatenated CFG batch.
            activations += 3 * batch * prefix * config.num_hidden_layers * 2 * config.num_key_value_heads * config.attention_head_dim * element_size
        banks = self.diffusion_model.model.layers[0].mlp
        largest_bank = max(memory_management.vram_aligned_size([bank.weight, bank.bias])
                           for bank in (banks.experts_gate_up_proj, banks.experts_down_proj))
        # The prefetch queue can have a current cast and the next bank in flight.
        return max(memory_required(self, input_shape, cond_shapes), activations + 2 * largest_bank)

    base.HunyuanImage3Model.memory_required = memory
