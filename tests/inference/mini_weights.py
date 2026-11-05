"""ComfyUI constructor sizing; checkpoint substitution itself is framework-independent."""
import contextlib
import dataclasses
import copy
import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from comfy import controlnet, hooks, lora, model_base, ops, quant_ops, sd, sd1_clip, utils
from comfy.ldm.models import autoencoder
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.text_encoders import llama, qwen35
from tests.mini_weights import miniature_checkpoints, random_state
from tests.inference.mini_gguf import MiniatureGGUF


@contextlib.contextmanager
def miniature_loras():
    """Initialize adapter factors at the native LoRA loading boundary."""
    class SizedKeys(dict):
        def __init__(self, keys):
            super().__init__(keys)
            self.shapes = {}

    def mapper(original):
        def mapped(model, key_map=None):
            keys = original(model, key_map)
            if not isinstance(keys, SizedKeys):
                keys = SizedKeys(keys)
            keys.shapes.update((name, tuple(p.shape)) for name, p in model.named_parameters())
            return keys
        return mapped

    load = lora.load_lora

    def load_factors(state, keys, *args, **kwargs):
        patches = load(state, keys, *args, **kwargs)
        generator = torch.Generator().manual_seed(0)
        for key, adapter in patches.items():
            if not isinstance(adapter, LoRAAdapter) or not any(isinstance(t, torch.Tensor) and t.is_meta for t in adapter.weights):
                continue
            name = key[0] if isinstance(key, tuple) else key
            shape = list(keys.shapes[name])
            if isinstance(key, tuple):
                dim, _, length = key[1]
                shape[dim] = length
            up, down, alpha, mid, dora, reshape = adapter.weights
            rank = min(8, down.shape[0])

            def factor(size):
                return torch.randn(size, generator=generator) / math.sqrt(math.prod(size[1:]))

            kernel = math.prod(down.shape[2:]) * (math.prod(mid.shape[2:]) if mid is not None else 1)
            new_up = factor((shape[0], rank, *up.shape[2:]))
            new_down = factor((rank, math.prod(shape[1:]) // kernel, *down.shape[2:]))
            new_mid = factor((rank, rank, *mid.shape[2:])) if mid is not None else None
            new_dora = torch.ones((shape[0], *dora.shape[1:])) if dora is not None else None
            adapter.weights = (new_up, new_down, alpha * rank / down.shape[0] if alpha is not None else None,
                               new_mid, new_dora, tuple(shape) if reshape is not None else None)
        return patches

    with contextlib.ExitStack() as stack:
        for name in ('model_lora_keys_clip', 'model_lora_keys_unet'):
            wrapped = mapper(getattr(lora, name))
            for owner in (lora, sd, hooks):
                stack.enter_context(patch.object(owner, name, wrapped))
        for owner in (lora, sd, hooks):
            stack.enter_context(patch.object(owner, 'load_lora', load_factors))
        yield


@contextlib.contextmanager
def miniature_transformers(loaded, width, depth, initialization_device):
    """Use the native Transformers loader with reduced configs and checkpoint tensors."""
    from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
    from copy import deepcopy

    from_pretrained = PreTrainedModel.from_pretrained.__func__
    load_pretrained = PreTrainedModel._load_pretrained_model
    handles = []

    def resize(config):
        config = deepcopy(config)
        values = _smaller(config.to_dict(), width, depth)
        if isinstance(values.get('d_model'), int):
            old = values['d_model']
            head_keys = [key for key in ('encoder_attention_heads', 'decoder_attention_heads') if key in values]
            quantum = math.lcm(*(old // values[key] for key in head_keys)) if head_keys else 64
            values['d_model'] = min(old, max(quantum, width // quantum * quantum))
            for key in head_keys:
                values[key] = values['d_model'] // (old // values[key])
        for key in ('encoder_layers', 'decoder_layers'):
            if isinstance(values.get(key), int):
                values[key] = min(depth, values[key])
        for key in ('encoder_ffn_dim', 'decoder_ffn_dim'):
            if isinstance(values.get(key), int):
                values[key] = min(width * 2, values[key])
        for key, value in values.items():
            if isinstance(getattr(config, key, None), PretrainedConfig):
                value = resize(getattr(config, key))
            setattr(config, key, value)
        if hasattr(config, 'text_config') and hasattr(config, 'vision_config'):
            # Qwen vision mergers emit the language model's embedding width.
            for key in ('hidden_size', 'out_hidden_size'):
                if key in values.get('vision_config', {}) and values['vision_config'][key] == values.get('text_config', {}).get('hidden_size'):
                    setattr(config.vision_config, key, config.text_config.hidden_size)
        return config

    def from_checkpoint(cls, path, *args, **kwargs):
        config = kwargs.get('config')
        if config is None:
            config = AutoConfig.from_pretrained(path, **{k: kwargs[k] for k in
                ('subfolder', 'revision', 'local_files_only', 'trust_remote_code') if k in kwargs})
        kwargs['config'] = resize(config)
        return from_pretrained(cls, path, *args, **kwargs)

    def load_checkpoint(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        if state_dict is None:
            state_dict = {}
            for filename in checkpoint_files:
                state_dict.update(utils.load_torch_file(filename))
        weights = random_state(state_dict, model.state_dict(), max_bytes=1024**3, device=initialization_device)
        result = load_pretrained(model, weights, None, load_config, expected_keys)
        loaded.append(type(model).__name__)
        def check_logits(module, inputs, output):
            logits = getattr(output, 'logits', None)
            if logits is not None:
                assert logits.numel() and torch.isfinite(logits).all(), 'Non-finite miniature generation logits'
                loaded.append(type(module).__name__ + ':finite logits')
        handles.append(model.register_forward_hook(check_logits))
        return result

    with patch.object(PreTrainedModel, 'from_pretrained', classmethod(from_checkpoint)), \
         patch.object(PreTrainedModel, '_load_pretrained_model', staticmethod(load_checkpoint)):
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def _smaller(config, width, depth, alignment=1):
    _original_counts = (config.get("n_double_layers", 0), config.get("n_layers", 0))
    config = config.copy()
    for key in ('depth', 'depth_single_blocks', 'num_layers', 'num_single_layers', 'num_double_stream_layers', 'num_refiner_layers', 'num_blocks', 'n_layers', 'n_double_layers', 'num_hidden_layers', 'num_text_blocks', 'num_visual_blocks', 'layers', 'patch_depth', 'pixel_depth', 'vace_layers'):
        if isinstance(config.get(key), int):
            config[key] = min(depth, config[key])
    for key in ('transformer_depth', 'transformer_depth_output'):
        if isinstance(config.get(key), list):
            config[key] = [min(depth, n) for n in config[key]]
    if isinstance(config.get('transformer_depth_middle'), int):
        config['transformer_depth_middle'] = min(depth, config['transformer_depth_middle'])
    if config.get('injection_layers') is not None:
        config['injection_layers'] = tuple(range(min(depth, len(config['injection_layers']))))
    if 'block_out_channels' in config:
        config['block_out_channels'] = [min(width, n) for n in config['block_out_channels']]
    if 'dim_head' in config and isinstance(config.get('heads'), int):
        config['heads'] = min(config['heads'], max(1, width // config['dim_head']))
    widths = ('hidden_size', 'dim', 'hidden_dim', 'model_dim', 'model_channels', 'features')
    if not any(isinstance(config.get(k), int) for k in widths) and 'attention_head_dim' in config:
        inner = config['num_attention_heads'] * config['attention_head_dim']
        config['num_attention_heads'] = min(config['num_attention_heads'], max(1, width // config['attention_head_dim']))
        # A caption projection inside the model must agree with cross attention.
        if 'caption_channels' in config and not config.get('caption_proj_before_connector') and config.get('cross_attention_dim') == inner:
            config['cross_attention_dim'] = config['num_attention_heads'] * config['attention_head_dim']
    for key in widths:
        # Ming also receives unprojected direct conditioning at its native width.
        if key == 'dim' and config.get('image_model') == 'ming_image':
            continue
        if key not in config:
            continue
        old = config[key]
        if not isinstance(old, int):
            continue
        heads_key = next((k for k in ('num_heads', 'n_heads', 'num_attention_heads', 'heads', 'num_groups') if isinstance(config.get(k), int) and config[k] > 0), None)
        head_dim = old // config[heads_key] if heads_key else 64
        for axes in ('axes_dim', 'axes_dims'):
            if axes in config:
                head_dim = sum(config[axes])
        quantum = math.lcm(32, head_dim) if key == 'model_channels' else head_dim
        kv_key = next((k for k in ('num_key_value_heads', 'num_kv_heads', 'n_kv_heads', 'kvheads') if isinstance(config.get(k), int) and config[k] > 0), None)
        groups = config[heads_key] // config[kv_key] if heads_key and kv_key else 1
        # Do not impose storage alignment stronger than the original valid width.
        # Plain FP8, for example, need not satisfy ConvRot's 256-wide blocks.
        quantum = math.lcm(quantum, head_dim * groups, math.gcd(alignment, old))
        cap = width
        if key == 'model_channels' and config.get('channel_mult'):
            cap //= max(config['channel_mult'])
        small = min(old, max(quantum, cap // quantum * quantum))
        config[key] = small
        if heads_key:
            config[heads_key] = small // head_dim
            if kv_key:
                config[kv_key] = config[heads_key] // groups
    for key in ('ffn_dim', 'ff_dim', 'ffn_hidden_size', 'intermediate_size', 'd_ff', 'time_embed_hidden_size'):
        if isinstance(config.get(key), int):
            config[key] = min(config[key], width * 2)
    if 'context_embedder_config' in config and 'depth' in config:
        embedder = config['context_embedder_config']
        config['context_embedder_config'] = {**embedder, 'params': {**embedder['params'], 'out_features': config['depth'] * 64}}
    if 'n_double_layers' in config and 'n_layers' in config:
        config['n_layers'] = config['n_double_layers'] + min(depth, _original_counts[1] - _original_counts[0])
    if isinstance(config.get('llama_layers'), list) and 'num_single_layers' in config:
        config['llama_layers'] = config['llama_layers'][:config['num_layers'] + config['num_single_layers']]
    if config.get('image_model') == 'chroma':
        config['out_dim'] = config['hidden_size']
    return config


@contextlib.contextmanager
def miniature_models(*, sources=None, width=256, depth=2, initialization_device="cpu"):
    gguf = MiniatureGGUF(utils.load_torch_file)
    if sources is None:
        # Saved conditioning and textual inversions are inputs, not model parameters.
        sources = lambda path: "embeddings" not in Path(path).parts
    handles = []
    initialize_model = model_base.BaseModel.__init__
    signature = inspect.signature(initialize_model)
    read_text_config = sd1_clip.get_path_as_dict
    initialize_text = sd1_clip.SDClipModel.__init__
    text_signature = inspect.signature(initialize_text)
    read_control_config = controlnet.controlnet_config
    instantiate_vae = autoencoder.instantiate_from_config

    @contextlib.contextmanager
    def component_configs():
        # These components own their configs rather than receiving BaseModel's config.
        # Only constructor inputs change; native classes still construct every block.
        from comfy.ldm.hidream_o1 import model as hidream
        from comfy.ldm.minimax import vae as minimax_vae, controlnet as minimax_control
        from comfy.ldm.minimax_music import dit as music_dit
        from comfy.ldm.yue2 import model as yue_model
        from comfy.text_encoders import yue2 as yue_text
        from comfy.text_encoders import minimax_music, ming_image, qwen_vl

        def sized_constructor(original):
            sig = inspect.signature(original)

            def construct(self, *args, **kwargs):
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                operations = bound.arguments.get('operations', bound.arguments.get('ops'))
                alignment = 256 if getattr(operations, '_quant_config', None) else 1
                resized = _smaller(dict(bound.arguments), width, depth, alignment)
                bound.arguments.update(resized)
                return original(*bound.args, **bound.kwargs)
            return construct

        yue_config = yue_model.model_config
        def yue_components(**kwargs):
            config = yue_config(**kwargs)
            return type(config)(**_smaller(dataclasses.asdict(config), width, depth))

        initialize_music = music_dit.ContinuousTransformer.__init__
        def music_transformer(model, *args, **kwargs):
            initialize_music(model, *args, **kwargs)
            # Native constructor hardcodes 36 identical blocks; retain the real block loop.
            model.layers = model.layers[:depth]

        text_config = hidream.HiDreamO1TextConfig
        def hidream_text(**kwargs):
            values = dataclasses.asdict(text_config(**kwargs))
            return text_config(**_smaller(values, width, depth))

        connector_config = ming_image.MingConnectorConfig
        def ming_connector(**kwargs):
            return connector_config(**_smaller(dataclasses.asdict(connector_config(**kwargs)), width, depth))

        ming_config = ming_image.BailingMoeV2Config
        def ming_text(**kwargs):
            config = ming_config(**kwargs)
            config = ming_config(**_smaller(dataclasses.asdict(config), width, max(ming_image.MingImageEncoder.capture_layers) + 1))
            per_group = max(2, math.ceil(config.num_experts_per_tok / config.topk_group))
            config.num_experts = min(config.num_experts, config.n_group * per_group)
            return config

        music_config = dict(minimax_music.MODEL_CONFIG)
        # Music conditioning has a public hidden width shared with the diffusion model.
        for key in ('num_hidden_layers', 'decoder_num_layers'):
            music_config[key] = min(depth, music_config[key])
        for key in ('intermediate_size', 'decoder_intermediate_size'):
            music_config[key] = min(width * 2, music_config[key])
        with patch.object(hidream, 'HiDreamO1TextConfig', hidream_text), \
             patch.object(hidream, 'QWEN3VL_VISION_DEFAULTS', _smaller(hidream.QWEN3VL_VISION_DEFAULTS, width, depth)), \
             patch.object(minimax_vae.ViT3DDecoder, '__init__', sized_constructor(minimax_vae.ViT3DDecoder.__init__)), \
             patch.object(minimax_control.MiniMaxH3FunControl, '__init__', sized_constructor(minimax_control.MiniMaxH3FunControl.__init__)), \
             patch.object(minimax_music, 'MODEL_CONFIG', music_config), \
             patch.object(ming_image, 'BailingMoeV2Config', ming_text), \
             patch.object(ming_image, 'MingConnectorConfig', ming_connector), \
             patch.object(qwen_vl.Qwen2VLVisionTransformer, '__init__', sized_constructor(qwen_vl.Qwen2VLVisionTransformer.__init__)), \
             patch.object(music_dit.ContinuousTransformer, '__init__', music_transformer), \
             patch.object(yue_model, 'model_config', yue_components), \
             patch.object(yue_text, 'model_config', yue_components):
            yield

    def model_config(self, config, *args, **kwargs):
        bound = signature.bind(self, config, *args, **kwargs)
        bound.apply_defaults()
        defaults = {k: p.default for k, p in inspect.signature(bound.arguments['unet_model']).parameters.items()
                    if k not in ('device', 'operations', 'dtype') and p.default is not inspect.Parameter.empty}
        # Quantized matrix layouts require complete storage/rotation blocks.
        alignment = 256 if getattr(config, 'quant_config', None) else 1
        config.unet_config = _smaller({**defaults, **config.unet_config}, width, depth, alignment)
        initialize_model(self, config, *args, **kwargs)
        def check_output(module, inputs, output):
            from torch.utils._pytree import tree_leaves
            for value in tree_leaves(output):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert torch.isfinite(value).all(), f'Non-finite miniature {type(module).__name__} output'
        handles.append(self.diffusion_model.register_forward_hook(check_output))

    def text_model(self, *args, **kwargs):
        bound = text_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        layer = bound.arguments['layer']
        taps = layer if isinstance(layer, list) else [bound.arguments['layer_idx']]
        minimum = max(4, depth, *(abs(n) + 1 for n in taps if isinstance(n, int)))

        def text_config(*args, **kwargs):
            config = read_text_config(*args, **kwargs).copy()
            # Preserve external width/vocabulary and every requested intermediate layer.
            counts = [k for k in ('num_layers', 'num_hidden_layers') if k in config]
            if layer != 'all':
                for key in counts:
                    config[key] = min(minimum, config.get(key, minimum))
            for key in ('d_ff', 'intermediate_size'):
                if key in config or key == 'intermediate_size':
                    config[key] = min(width * 2, config.get(key, width * 2))
            return config

        def text_constructor(original):
            def initialize(model, config, *a, **kw):
                config = copy.copy(config)
                if layer != 'all':
                    config.num_hidden_layers = min(minimum, config.num_hidden_layers)
                config.intermediate_size = min(width * 2, config.intermediate_size)
                if hasattr(config, 'layer_types'):
                    config.layer_types = config.layer_types[:config.num_hidden_layers]
                original(model, config, *a, **kw)
            return initialize

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sd1_clip, 'get_path_as_dict', text_config))
            for cls in (llama.Llama2_, qwen35.Qwen35Transformer):
                stack.enter_context(patch.object(cls, '__init__', text_constructor(cls.__init__)))
            initialize_text(self, *args, **kwargs)

    def control_config(*args, **kwargs):
        result = read_control_config(*args, **kwargs)
        result[0].unet_config = _smaller(result[0].unet_config, width, depth)
        return result

    def vae_component(config):
        return instantiate_vae({**config, 'params': _smaller(config.get('params', {}), width, depth)})

    def state_shapes(module, state):
        targets = module.state_dict()
        gguf.shapes(module, state, targets)
        descriptors = {}
        for key, value in state.items():
            if key not in targets and (value.ndim <= 1 or key.endswith('weight_scale')):
                descriptors.setdefault(key.rpartition('.')[0], []).append((key, value))
        for name, child in module.named_modules():
            if not (isinstance(child, ops.MixedPrecisionOp) or isinstance(child, torch.nn.Embedding) and hasattr(child, '_disabled_storage_formats')):
                continue
            prefix = name + '.' if name else ''
            key = prefix + 'weight'
            if key not in state:
                continue
            if prefix + 'comfy_quant' in state:
                descriptor = json.loads(state[prefix + 'comfy_quant'].numpy().tobytes())
                if descriptor['format'] not in ('float8_e4m3fn', 'float8_e5m2', 'int8_tensorwise', 'mxfp8', 'nvfp4'):
                    raise ValueError(f"Miniature quantization unsupported: {descriptor['format']} ({key})")
            shape = child._orig_shape if hasattr(child, '_orig_shape') else tuple(child.weight.shape)
            compute_dtype = child.factory_kwargs['dtype'] if hasattr(child, 'factory_kwargs') else child.weight.dtype
            dtype = compute_dtype
            targets.update((prefix + k, v) for k, v in child.named_parameters(recurse=False))
            targets[key] = torch.empty(shape, device='meta', dtype=dtype)
            source_rows = state[key].shape[0]
            for key, value in descriptors.get(name, ()):
                if key in targets:
                    continue
                if value.is_floating_point() and "scale" in key:
                    shape = (targets[prefix + 'weight'].shape[0], *value.shape[1:]) if value.ndim and value.shape[0] == source_rows else value.shape
                    if key.endswith('pre_quant_scale') and value.ndim == 1:
                        shape = (targets[prefix + 'weight'].shape[-1],)
                    value = state[key] = torch.ones(shape, dtype=value.dtype)
                targets[key] = value
        return targets

    def encode_weights(module, state):
        gguf.encode(state)
        for name, child in module.named_modules():
            if not (isinstance(child, ops.MixedPrecisionOp) or isinstance(child, torch.nn.Embedding) and hasattr(child, '_disabled_storage_formats')):
                continue
            prefix = name + '.' if name else ''
            descriptor = state.get(prefix + 'comfy_quant')
            if descriptor is None:
                continue
            config = json.loads(descriptor.numpy().tobytes())
            if config['format'] not in ('float8_e4m3fn', 'float8_e5m2', 'int8_tensorwise', 'mxfp8', 'nvfp4'):
                continue
            if prefix + 'weight_scale' not in state:
                continue  # Let the native loader report a missing checkpoint scale.
            rotation = config.get('params', {}).get('convrot', config.get('convrot', False))
            group = config.get('convrot_groupsize', config.get('params', {}).get('convrot_groupsize', 256))
            layout = quant_ops.get_layout_class(quant_ops.QUANT_ALGOS[config['format']]['comfy_tensor_layout'])
            options = dict(per_channel=True, convrot=rotation, convrot_groupsize=group) if config['format'] == 'int8_tensorwise' else {}
            if config['format'].startswith('float8_'):
                options = {'scale': 'recalculate'}
            value = state[prefix + 'weight']
            bank_shape = value.shape if value.ndim == 3 and hasattr(child, 'num_experts') else None
            if bank_shape:
                # Each expert is a matrix, not a convolution's channel/kernel dimensions.
                value = value.mul_(math.sqrt(bank_shape[1])).flatten(0, 1)
            weight, params = layout.quantize(value, **options)
            for suffix, tensor in layout.state_dict_tensors(weight, params).items():
                if bank_shape and tensor.ndim > 1 and tensor.shape[0] == bank_shape[0] * bank_shape[1]:
                    tensor = tensor.reshape(*bank_shape[:2], *tensor.shape[1:])
                state[prefix + 'weight' + suffix] = tensor

    with component_configs(), miniature_loras(), \
         patch.object(model_base.BaseModel, '__init__', model_config), \
         patch.object(sd1_clip.SDClipModel, '__init__', text_model), \
         patch.object(controlnet, 'controlnet_config', control_config), \
         patch.object(autoencoder, 'instantiate_from_config', vae_component), \
         miniature_checkpoints(utils, 'load_torch_file', sources=sources, max_bytes=8 * 1024**3, shapes=state_shapes, encode=encode_weights, describe=gguf.describe, device=initialization_device) as loaded:
        try:
            with miniature_transformers(loaded, width, depth, initialization_device):
                yield loaded
        finally:
            for handle in handles:
                handle.remove()
