"""Pipeline parallelism for the HunyuanImage3 custom node (`comfyui-hunyuanimage3`).

The node builds its own `HunyuanImage3` transformer and loader rather than a ComfyUI model config,
so the stage spec cannot be reached through model detection. `install` adds pipeline stages to the
node's classes when ComfyUI imports it, and routes the node's loader through `load_pipeline_model`
whenever this process spans more than one pipeline device, exactly as `load_diffusion_model` does
for the built-in families.

A stage is the node's own module with the submodules other stages own dropped before any weight
is loaded, so a rank never holds another stage's 4.6 GiB decoder layers. The first stage builds each step's sequence (token embeddings, patch embedding, timestep and
guidance embedders, conditioning images) and with it the attention mask and rotary table, which
travel with the hidden states: rebuilding them costs a stage more than the transfer does. The last
stage applies `final_layer` to the image rows.

The node's prompt-rewriting stage (think / recaption, `rewrite.generate_text`) decodes text
autoregressively, and runs the same way: the driver embeds the prompt (conditioning images included)
or the next token, the hidden state flows stage to stage, each stage keeps the K/V of its own layers
for the prefill and every decoded token, and the last stage hands back the last position's hidden
state, which the driver's text head turns into logits. The caches are dropped when the decode ends.
"""
from __future__ import annotations

from functools import partial, wraps
import importlib
import json
import logging
import os
import re
import sys

import torch

from .. import model_management, model_prefetch, ops, utils
from ..model_patcher import get_model_patcher_class

from ..tensor_parallel.hunyuan_image3_runtime import KVCache, prepare_batched_inputs
from .checkpoint import SafetensorsCheckpointReader
from .loader import AbstractBasePipelineStageLoader, load_pipeline_model, resolve_pipeline_devices
from .patcher import get_pipeline_model_patcher_class
from .stages import HunyuanImage3PipelineStageSpec
from .types import (
    PipelineIntermediateTensors,
    PipelineMissingLayer,
    PipelineParallelConfig,
    PipelineStageConfig,
    PipelineStagePlan,
    PipelineWorkerLoadSpec,
)

logger = logging.getLogger(__name__)

MODEL_FAMILY = HunyuanImage3PipelineStageSpec.model_family
_SUPPORTED_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# the per-expert layout of Tencent's own shards; the node's converted checkpoints stack each layer's
# experts into one bank per projection
_PER_EXPERT_KEY = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.")


def _upstream(package: str, name: str):
    return importlib.import_module(f"{package}.{name}")


def _prune(module: torch.nn.Module, prefix: str, stage: PipelineStageConfig) -> torch.nn.Module:
    """`module` with every submodule another stage owns replaced.

    Ownership is decided by module name, the way the stage spec decides it for a key: a decoder
    layer by its index, anything else by its prefix. A module's parameters cannot decide it, since
    ops that create a weight only when a state dict arrives (mixed precision, DynamicVRAM) leave a
    Linear-only module such as `time_embed_2` with no keys at construction.
    """
    spec = HunyuanImage3PipelineStageSpec()
    plan = PipelineStagePlan(stage.index, torch.device("meta"), stage.start_layer, stage.end_layer, 0, frozenset())
    holds_layers = prefix == "" or any(
        spec.block_index(f"{prefix}{name}.") is not None for name, _ in module.named_modules() if name
    )
    if spec.block_index(prefix) is not None or not holds_layers:
        return module if spec.owns_key(prefix, plan, stage.count) else PipelineMissingLayer()
    for name, child in list(module.named_children()):
        replacement = _prune(child, f"{prefix}{name}.", stage)
        if replacement is not child:
            module._modules[name] = replacement
    if next(module.parameters(recurse=False), None) is not None or next(module.buffers(recurse=False), None) is not None:
        raise ValueError(f"{prefix or 'the model'} holds tensors of its own beside decoder layers of several stages")
    return module


def _stage_init(original):
    @wraps(original)
    def __init__(self, config, dtype=None, device=None, operations=None, pipeline_stage=None):
        if pipeline_stage is None:
            original(self, config, dtype=dtype, device=device, operations=operations)
            self.pipeline_stage = None
            self.supports_prefix_cache = True
            return
        # Built whole, then pruned. ComfyUI's ops create Linear and expert-bank weights only when a
        # state dict arrives, so the whole module costs its norms, embeddings and convolutions
        # (about 1.6 GiB, never written) until the other stages' modules are dropped. It cannot be
        # built on the meta device instead: mixed-precision ops keep the construction device for
        # the weights they create at load.
        original(self, config, dtype=dtype, device=device, operations=operations)
        self.pipeline_stage = pipeline_stage
        self.supports_prefix_cache = False
        _prune(self, "", pipeline_stage)

    return __init__


def _run_pipeline_layers(self, hidden_states, freqs, attention_mask, prefetch):
    """This stage's decoder layers, computed as the node's `HunyuanImage3Model.forward` computes them.

    Weights are faulted ahead through ComfyUI's prefetch queue, as the built-in families' block
    loops do, rather than the node's LayerLookahead: that faults every module carrying a `_v`
    attribute, and DynamicVRAM leaves `_v` None on a module it keeps resident.
    """
    stage = self.pipeline_stage
    device = hidden_states.device
    layers = [self.model.layers[index] for index in range(stage.start_layer, stage.end_layer)]
    queue = model_prefetch.make_prefetch_queue(layers, device, {"prefetch_dynamic_vbars": prefetch})
    for layer in layers:
        model_prefetch.prefetch_queue_pop(queue, device, layer, malloc_scope="block")
        hidden_states = layer(hidden_states, freqs, attention_mask, None)
    model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
    return hidden_states


def _first_stage_forward(self, x, timestep, transformer_options=None, guidance=None, ids=None,
                         cond_latent=None, cond_vit=None, cond_vit_grid=None):
    """The node's `HunyuanImage3.forward` up to this stage's last layer."""
    source = sys.modules[type(self).__module__]
    if not self.pipeline_stage.is_first:
        raise RuntimeError("Only the first HunyuanImage3 pipeline stage accepts model inputs")
    transformer_options = transformer_options or {}
    if transformer_options.get("hy3_spectrum") is not None:
        raise ValueError("HunyuanImage3 pipeline parallelism does not support the Spectrum node")
    if ids is None:
        raise ValueError("HunyuanImage3 requires token ids; the conditioning must carry them")

    if not isinstance(ids, (tuple, list)):
        ids = (ids,) * x.shape[0]
    ids = tuple(prompt.to(x.device) for prompt in ids)
    steps, embeds, freqs, mask, _ = prepare_batched_inputs(
        self, x, timestep, transformer_options, guidance, ids, source._as_list(cond_latent),
        source._as_list(cond_vit), cond_vit_grid)
    prefetch = bool(transformer_options.get("prefetch_dynamic_vbars", False))
    hidden_states = self._run_pipeline_layers(embeds, freqs, mask, prefetch)
    return PipelineIntermediateTensors(
        {"hidden_states": hidden_states, "freqs": freqs, "attention_mask": mask, "timestep": timestep},
        {
            "image_slices": [(step["image_slice"].start, step["image_slice"].stop) for step in steps],
            "token_height": int(steps[0]["token_height"]),
            "token_width": int(steps[0]["token_width"]),
            "latent_dims": x.dim(),
            "prefetch_dynamic_vbars": prefetch,
        },
    )


# A pipelined text step: what each stage does with its layers' K/V cache.
TEXT_STEP = "hy3_text"
TEXT_PREFILL, TEXT_DECODE, TEXT_RELEASE = "prefill", "decode", "release"


def _run_text_layers(self, hidden_states, freqs, attention_mask, phase):
    """This stage's layers over a text step, each with its own K/V cache: a prefill starts new caches,
    a decode step appends to them, a release drops them and computes nothing."""
    stage = self.pipeline_stage
    if phase == TEXT_RELEASE:
        self._hy3_text_cache = None
        return hidden_states
    if phase == TEXT_PREFILL:
        self._hy3_text_cache = {index: KVCache() for index in range(stage.start_layer, stage.end_layer)}
    elif getattr(self, "_hy3_text_cache", None) is None:
        raise RuntimeError("a HunyuanImage3 text decode step arrived before its prefill")
    for index in range(stage.start_layer, stage.end_layer):
        hidden_states = self.model.layers[index](hidden_states, freqs, attention_mask, self._hy3_text_cache[index])
    return hidden_states


def _text_intermediate(hidden_states, freqs, attention_mask, phase):
    tensors = {"hidden_states": hidden_states, "freqs": freqs}
    if attention_mask is not None:
        tensors["attention_mask"] = attention_mask
    return PipelineIntermediateTensors(tensors, {TEXT_STEP: phase})


def _first_stage_text(self, embeddings, freqs, attention_mask, phase):
    if not self.pipeline_stage.is_first:
        raise RuntimeError("Only the first HunyuanImage3 pipeline stage accepts text steps")
    hidden_states = _run_text_layers(self, embeddings, freqs, attention_mask, phase)
    return _text_intermediate(hidden_states, freqs, attention_mask, phase)


def _stage_forward(original):
    @wraps(original)
    def forward(self, x, timestep, context=None, transformer_options=None, guidance=None, ids=None,
                cond_latent=None, cond_vit=None, cond_vit_grid=None, **kwargs):
        if getattr(self, "pipeline_stage", None) is not None and TEXT_STEP in kwargs:
            # a text step: x is the step's embeddings and timestep its rotary table
            return _first_stage_text(self, x, timestep, kwargs.get("text_attention_mask"), kwargs[TEXT_STEP])
        if getattr(self, "pipeline_stage", None) is None:
            return original(self, x, timestep, context=context, transformer_options=transformer_options,
                            guidance=guidance, ids=ids, cond_latent=cond_latent, cond_vit=cond_vit,
                            cond_vit_grid=cond_vit_grid, **kwargs)
        return _first_stage_forward(self, x, timestep, transformer_options=transformer_options,
                                    guidance=guidance, ids=ids, cond_latent=cond_latent, cond_vit=cond_vit,
                                    cond_vit_grid=cond_vit_grid)

    return forward


def forward_pipeline_stage(self, intermediate: PipelineIntermediateTensors):
    stage = self.pipeline_stage
    if stage is None or stage.is_first:
        raise RuntimeError("HunyuanImage3 pipeline continuation requires a non-first stage")
    tensors, metadata = intermediate.tensors, intermediate.metadata
    phase = metadata.get(TEXT_STEP)
    if phase is not None:
        hidden_states = _run_text_layers(self, tensors["hidden_states"], tensors["freqs"],
                                         tensors.get("attention_mask"), phase)
        if not stage.is_last:
            return _text_intermediate(hidden_states, tensors["freqs"], tensors.get("attention_mask"), phase)
        # only the last position predicts the next token
        return hidden_states[:, -1:].contiguous()
    hidden_states = self._run_pipeline_layers(
        tensors["hidden_states"], tensors["freqs"], tensors["attention_mask"], metadata["prefetch_dynamic_vbars"],
    )
    if not stage.is_last:
        tensors = dict(tensors)
        tensors["hidden_states"] = hidden_states
        return PipelineIntermediateTensors(tensors, metadata)
    image_hidden = torch.stack([hidden_states[index, start:stop]
                                for index, (start, stop) in enumerate(metadata["image_slices"])])
    pred = self.final_layer(image_hidden, self.time_embed_2(tensors["timestep"]),
                            metadata["token_height"], metadata["token_width"])
    return pred.unsqueeze(2) if metadata["latent_dims"] == 5 else pred


class _PipelineTextTransformer:
    """`model.model` for `rewrite.generate_text` on the driver: the call goes through every stage. The
    caller's per-layer cache is unused; each stage keeps its own layers' caches."""

    def __init__(self, model, executor):
        self.wte = model.model.wte
        self.executor = executor
        self.hidden_size = model.config.hidden_size
        self.prefilled = False
        self.device = None

    def __call__(self, embeddings, freqs, attention_mask, cache):
        phase = TEXT_DECODE if self.prefilled else TEXT_PREFILL
        self.prefilled, self.device = True, embeddings.device
        return self.executor.execute(embeddings, freqs, **{TEXT_STEP: phase, "text_attention_mask": attention_mask})

    def release(self):
        """Every stage drops its caches; the step carries a one-position placeholder and computes nothing."""
        placeholder = torch.zeros(1, 1, self.hidden_size, device=self.device)
        self.executor.execute(placeholder, torch.zeros(1, device=self.device),
                              **{TEXT_STEP: TEXT_RELEASE, "text_attention_mask": None})


class _PipelineTextModel:
    def __init__(self, model, executor):
        self._model = model
        self.model = _PipelineTextTransformer(model, executor)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _wrap_generate_text(rewrite):
    generate = rewrite.generate_text
    # the decode loop itself, under any other wrapper (tensor parallelism routes `model.model` its own way)
    decode = getattr(generate, "__wrapped__", generate)

    @wraps(generate)
    def generate_text(model, *args, **kwargs):
        executor = getattr(model, "_comfy_pipeline_executor", None)
        if getattr(model, "pipeline_stage", None) is None or executor is None:
            return generate(model, *args, **kwargs)
        proxy = _PipelineTextModel(model, executor)
        try:
            return decode(proxy, *args, **kwargs)
        finally:
            if proxy.model.prefilled:
                proxy.model.release()
            executor.finish_execution()

    generate_text._comfy_pipeline_parallel = True
    rewrite.generate_text = generate_text


class HunyuanImage3PipelineStageLoader(AbstractBasePipelineStageLoader):
    """Build stages of the node's `HunyuanImage3Model` from a checkpoint in the node's key layout.

    Which checkpoint it is (Instruct, Instruct-Distil, base) is read off the weights the way the
    node's own loader reads it, and selects the sampling contract. `params` and `dtype` override
    that detection and the device's dtype choice.
    """

    def __init__(self, checkpoint_path, extension_path, disable_dynamic=False, params=None, dtype=None):
        self.extension_path = os.fspath(extension_path)
        package = os.path.basename(self.extension_path)
        self.upstream_loader = _upstream(package, "hunyuan_image_3.loader")
        self.upstream_model = _upstream(package, "hunyuan_image_3.model")
        self.upstream_base = _upstream(package, "hunyuan_image_3.model_base")
        self.upstream_ops = _upstream(package, "hunyuan_image_3.ops")
        self.latent_formats = _upstream(package, "latent_formats")
        self._checkpoint_path = os.fspath(checkpoint_path)
        self.disable_dynamic = disable_dynamic
        self.dtype = dtype

        self.reader = SafetensorsCheckpointReader(checkpoint_path)
        per_expert = next((key for key in self.reader.tensors if _PER_EXPERT_KEY.match(key)), None)
        if per_expert is not None:
            raise ValueError(
                f"{self._checkpoint_path} stores routed experts one tensor each ({per_expert}); "
                "HunyuanImage3 pipeline stages load the node's converted checkpoints, which stack "
                "each layer's experts into banks"
            )
        self._descriptors = {
            key: descriptor for key, descriptor in self.reader.tensors.items()
            if not key.startswith(self.upstream_model.DEFERRED_PREFIXES)
        }
        detection = self.reader.detection_state_dict()
        self.quant_config = utils.detect_layer_quantization(detection, "")
        if params is None:
            fingerprint = self.upstream_loader.FINGERPRINT_KEY
            if fingerprint in self.reader.tensors:
                detection.update(self.reader.load_keys([fingerprint]))
            model_type = self.upstream_loader.detect_model_type(detection, self._checkpoint_path)
            with open(self.upstream_loader.CONFIG_PATH) as handle:
                config = json.load(handle)
            config.update(self.upstream_loader.MODEL_TYPES[model_type])
            config["model_type"] = model_type
            params = self.upstream_model.params_from_config(config)
            logger.info("HunyuanImage3: %s is the %s checkpoint", os.path.basename(self._checkpoint_path), model_type)
        self.params = params

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def model_family(self) -> str:
        return MODEL_FAMILY

    @property
    def descriptors(self):
        return self._descriptors

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self.dtype is None:
            self.dtype = model_management.unet_dtype(device=device, supported_dtypes=_SUPPORTED_DTYPES)
        return self.dtype

    def _operations(self, device):
        if self.quant_config is None:
            # ComfyUI's mixed-precision ops hold an unquantized bank as a plain compute-dtype
            # [E, out, in] weight and cast it like any other, so DynamicVRAM serves a resident
            # bank from VRAM. The node's QuantlessMoEOps bank copies itself from host memory on
            # every forward, which a stage that fits on its card never needs. With no quantized
            # layer there are no formats to disable.
            return ops.mixed_precision_ops({}, self.dtype)
        return ops.pick_operations(self.dtype, self.dtype, load_device=device,
                                   model_config=self.upstream_loader._OpsConfig(self.quant_config))

    def load_stage(self, plan, stage, offload_device):
        model_config = self.upstream_base.HunyuanImage3ModelConfig(
            self.params, self.latent_formats.HunyuanImage3(), {"shift": self.upstream_loader.SHIFT},
            quant_config=self.quant_config, dtype=self.dtype, custom_operations=self._operations(stage.device),
        )
        model_config.unet_config["pipeline_stage"] = PipelineStageConfig(
            stage.index, plan.size, stage.start_layer, stage.end_layer,
        )
        model = self.upstream_base.HunyuanImage3Model(model_config, device=offload_device)
        patcher_class = (get_pipeline_model_patcher_class(self.disable_dynamic) if stage.index == 0
                         else get_model_patcher_class(self.disable_dynamic))
        patcher = patcher_class(model, load_device=stage.device, offload_device=offload_device,
                                ckpt_name=os.path.basename(self._checkpoint_path))
        state = self.reader.load_keys(stage.owned_keys)
        self.upstream_loader.shared_codebooks(state)
        loaded = model.diffusion_model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
        del state
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(
                f"HunyuanImage3 stage {stage.index} does not match its checkpoint keys: "
                f"missing {loaded.missing_keys[:8]}, unexpected {loaded.unexpected_keys[:8]}"
            )
        if stage.index == 0:
            from tokenizers import Tokenizer

            model.tokenizer = Tokenizer.from_file(self.upstream_loader.TOKENIZER_PATH)
        return patcher

    def worker_load_spec(self, plan, stage) -> PipelineWorkerLoadSpec:
        return PipelineWorkerLoadSpec(
            checkpoint_path=self._checkpoint_path,
            plan=plan,
            stage_index=stage.index,
            model_options={},
            disable_dynamic=self.disable_dynamic,
            dtype=self.dtype,
            custom_node_path=self.extension_path,
        )


def _import_custom_node(path: str) -> None:
    """Import the node in a rank that did not load custom nodes, applying `install` as it does."""
    if os.path.basename(path) in sys.modules:
        return
    from comfy_compatibility.vanilla import prepare_vanilla_environment
    from ..nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1

    prepare_vanilla_environment()
    if not _vanilla_load_custom_nodes_1(path).NODE_CLASS_MAPPINGS:
        raise RuntimeError(f"Could not import the HunyuanImage3 custom node at {path}")


def load_worker_stage(load_spec: PipelineWorkerLoadSpec):
    if load_spec.custom_node_path is None:
        raise ValueError("A HunyuanImage3 pipeline worker needs the custom node's path")
    _import_custom_node(load_spec.custom_node_path)
    stage_loader = HunyuanImage3PipelineStageLoader(
        load_spec.checkpoint_path, load_spec.custom_node_path, load_spec.disable_dynamic, dtype=load_spec.dtype,
    )
    return stage_loader.load_stage(load_spec.plan, load_spec.plan.stages[load_spec.stage_index], torch.device("cpu"))


def load_pipeline(checkpoint_path, extension_path, disable_dynamic=False):
    """The node's model as pipeline stages over this process's pipeline devices."""
    placement = resolve_pipeline_devices()
    if placement is None:
        raise RuntimeError("HunyuanImage3 pipeline loading needs more than one pipeline device")
    devices, memory_coordinator = placement
    root = load_pipeline_model(
        HunyuanImage3PipelineStageLoader(checkpoint_path, extension_path, disable_dynamic),
        PipelineParallelConfig(devices),
        disable_dynamic=disable_dynamic,
        memory_coordinator=memory_coordinator,
    )
    root.cached_patcher_init = (partial(load_pipeline, disable_dynamic=disable_dynamic), (checkpoint_path, extension_path))
    # the prompt-rewriting stage is handed the diffusion module alone; it reaches the stages through this
    root.model.diffusion_model._comfy_pipeline_executor = root.model.pipeline_executor
    return root


def install(extension) -> None:
    package = extension.__name__
    upstream_loader = _upstream(package, "hunyuan_image_3.loader")
    upstream_model = _upstream(package, "hunyuan_image_3.model")
    transformer = upstream_model.HunyuanImage3
    if getattr(transformer, "_comfy_pipeline_parallel", False):
        return
    transformer.__init__ = _stage_init(transformer.__init__)
    transformer.forward = _stage_forward(transformer.forward)
    transformer.forward_pipeline_stage = forward_pipeline_stage
    transformer._run_pipeline_layers = _run_pipeline_layers
    transformer._comfy_pipeline_parallel = True

    extension_path = os.path.dirname(os.path.abspath(extension.__file__))
    original = upstream_loader.load_hunyuan_image_3

    @wraps(original)
    def load_hunyuan_image_3(checkpoint_path, disable_dynamic=False):
        if resolve_pipeline_devices() is None:
            return original(checkpoint_path, disable_dynamic=disable_dynamic)
        patcher = load_pipeline(checkpoint_path, extension_path, disable_dynamic)
        return patcher, patcher.model.tokenizer

    upstream_loader.load_hunyuan_image_3 = load_hunyuan_image_3
    _upstream(package, "nodes").load_hunyuan_image_3 = load_hunyuan_image_3

    rewrite = _upstream(package, "hunyuan_image_3.rewrite")
    if not getattr(rewrite.generate_text, "_comfy_pipeline_parallel", False):
        _wrap_generate_text(rewrite)
