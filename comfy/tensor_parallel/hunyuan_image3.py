from __future__ import annotations

import importlib
import contextlib
import json
import logging
import os
import sys
from functools import partial, wraps
from types import MethodType, SimpleNamespace

import torch
from tokenizers import Tokenizer

from .. import model_management, model_prefetch, ops, utils
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class
from ..pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from .distributed import RemoteTensorParallelRankModel, launch_tensor_parallel
from .custom_node_state import CachedStateExecutor, install_state_transport
from .operations import gathered_output_operations
from .types import TensorParallelConfig


def _modules(extension):
    return tuple(importlib.import_module(extension + ".hunyuan_image_3." + name) for name in ("loader", "model", "model_base"))


class _LayerPrefetch:
    """Adapt the upstream block loop to Comfy's allocation-aware prefetch queue."""
    def __init__(self, layers, device, enabled):
        self.layers = layers
        self.device = device
        self.queue = model_prefetch.make_prefetch_queue(list(layers), device, {
            "prefetch_dynamic_vbars": enabled and not os.environ.get("HUNYUAN_IMAGE_3_NO_LOOKAHEAD"),
        })

    def before(self, index):
        model_prefetch.prefetch_queue_pop(self.queue, self.device, self.layers[index])

    def after(self, index):
        if index == len(self.layers) - 1:
            model_prefetch.prefetch_queue_pop(self.queue, self.device, None)

    def abort(self):
        if self.queue is None:
            return
        for pending in self.queue:
            if isinstance(pending, tuple):
                stream, (module, modules) = pending
                if stream is not None:
                    stream.wait_stream(model_management.current_stream(self.device))
                if modules is not None:
                    model_prefetch.cleanup_prefetched_modules(module, modules)
        self.queue.clear()


def _moe_forward(self, hidden_states):
    source = sys.modules[type(self).__module__]
    bsz, seq_len, hidden_size = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_size)
    weights, indices = self.gate(flat)
    weights = weights.to(hidden_states.dtype)
    expert_mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
    count = self.experts_gate_up_proj.num_experts
    start = self.tensor_parallel.rank * count
    local_mask = expert_mask[start:start + count]
    expert_hit = (local_mask.sum(dim=(-1, -2)) > 0).nonzero()
    combined = torch.zeros((flat.shape[0] * self.top_k, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
    full_bank = len(expert_hit) * 2 >= count or hasattr(self.experts_gate_up_proj, "_prefetch")
    gate_bank = self.experts_gate_up_proj.bank_resident(flat) if full_bank else contextlib.nullcontext(self.experts_gate_up_proj)
    down_bank = self.experts_down_proj.bank_resident(flat) if full_bank else contextlib.nullcontext(self.experts_down_proj)
    linear = source._bank_linear if full_bank else source.expert_linear_sliced
    with gate_bank as gate_experts, down_bank as down_experts:
        for expert in expert_hit:
            index = int(expert.item())
            position, token = torch.where(local_mask[index])
            gate_up = linear(gate_experts, flat[token], index)
            output = linear(down_experts, source._swiglu(gate_up), index)
            combined[token * self.top_k + position] = (output * weights[token, position, None]).to(combined.dtype)
    # Each routing slot belongs to one rank. Merge slots before the top-k sum
    # to preserve upstream's reduction order and avoid an extra bf16 rounding.
    combined = self.tensor_parallel.operations.sum(combined)
    routed = combined.view(bsz, seq_len, self.top_k, hidden_size).sum(dim=2)
    return self.shared_mlp(hidden_states) + routed


def _shard_model(model, base_operations, parallel):
    operations = gathered_output_operations(base_operations, parallel)

    class AttentionOutput(operations.Linear):
        def forward(self, input, *args, **kwargs):
            # Restore the full activation domain before W4A8 quantization.
            return super().forward(parallel.operations.gather(input), *args, **kwargs)

    shards = {}
    attention = set()
    for name, module in list(model.named_modules()):
        if not name.startswith("model.layers.") or name.endswith(".gate.wg"):
            continue
        if name.endswith(".self_attn"):
            if module.num_key_value_heads % parallel.size == 0:
                module.num_heads //= parallel.size
                module.num_key_value_heads //= parallel.size
                attention.add(name)
            continue
        if name.endswith(".mlp"):
            module.tensor_parallel = parallel
            module.forward = MethodType(_moe_forward, module)
            continue
        if isinstance(module, base_operations.MoEExperts):
            if module.num_experts % parallel.size:
                raise ValueError(f"{name}: {module.num_experts} experts must divide {parallel.size} ranks")
            replacement = base_operations.MoEExperts(module.num_experts // parallel.size, module.in_features, module.out_features, bias=module.bias is not None, device="cpu")
            axis = 0
        elif isinstance(module, base_operations.Linear):
            owner, _, projection = name.rpartition(".")
            if owner in attention and projection == "qkv_proj":
                replacement = base_operations.Linear(module.in_features, module.out_features // parallel.size, bias=module.bias is not None, device="cpu")
            else:
                linear = AttentionOutput if owner in attention and projection == "o_proj" else operations.Linear
                replacement = linear(module.in_features, module.out_features, bias=module.bias is not None, device="cpu")
            axis = 0
        else:
            continue
        model.set_submodule(name, replacement)
        shards[name] = axis
    return shards


def _load_state(reader, shards, parallel, head_prefixes):
    selections = {}
    row_parameters = {"weight", "bias", "weight_scale", "weight_s_rel", "weight_s_channel"}
    for key, descriptor in reader.tensors.items():
        if key.startswith(head_prefixes):
            continue
        module, _, parameter = key.rpartition(".")
        axis = shards.get(module)
        selection = None
        expert_codebook = parameter == "weight_codebook" and axis == 0 and len(descriptor.shape) == 2
        if axis is not None and (parameter in row_parameters or expert_codebook) and len(descriptor.shape) > axis:
            width, remainder = divmod(descriptor.shape[axis], parallel.size)
            if remainder:
                raise ValueError(f"{key} cannot be split across {parallel.size} ranks")
            selection = tuple(slice(parallel.rank * width, (parallel.rank + 1) * width) if i == axis else slice(None) for i in range(len(descriptor.shape)))
        selections[key] = selection
    return reader.load_slices(selections)


@contextlib.contextmanager
def _text_context(self):
    cache = [[None, None] for _ in self.model.layers]

    def forward(embeddings, frequencies, mask):
        return self.model(embeddings, frequencies, mask, cache)

    try:
        yield SimpleNamespace(tensor_parallel_text_forward=forward)
    finally:
        cache.clear()


class _TextTransformer:
    def __init__(self, model, executor):
        self.wte = model.wte
        self.executor = executor

    def __call__(self, embeddings, frequencies, mask, cache):
        # The upstream caller's cache is unused: each rank owns its cache inside
        # the bounded model context, so growing prefixes never cross the wire.
        return self.executor.execute_method(
            "tensor_parallel_text_forward", embeddings, frequencies, mask,
        )


class _TextModel:
    def __init__(self, model, executor):
        self._model = model
        self.model = _TextTransformer(model.model, executor)

    def __getattr__(self, name):
        return getattr(self._model, name)


def load_rank(load_spec, device, tensor_operations):
    extension = os.path.basename(load_spec.custom_node_path)
    upstream_loader, upstream_model, upstream_base = _modules(extension)
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection = reader.detection_state_dict()
    detection.update(reader.load_keys([upstream_loader.FINGERPRINT_KEY]))
    model_type = upstream_loader.detect_model_type(detection, load_spec.checkpoint_path)
    with open(upstream_loader.CONFIG_PATH) as source:
        config = json.load(source)
    config.update(upstream_loader.MODEL_TYPES[model_type])
    config["model_type"] = model_type
    quantization = utils.detect_layer_quantization(detection, "")
    if quantization is None:
        raise ValueError("HunyuanImage3 tensor parallelism requires a quantized checkpoint")
    operations = ops.pick_operations(load_spec.dtype, load_spec.dtype, load_device=device, model_config=upstream_loader._OpsConfig(quantization))
    config = upstream_base.HunyuanImage3ModelConfig(
        upstream_model.params_from_config(config), upstream_loader.HunyuanImage3(),
        {"shift": upstream_loader.SHIFT}, quant_config=quantization,
        dtype=load_spec.dtype, custom_operations=operations,
    )
    model = upstream_base.HunyuanImage3Model(config, device=torch.device("cpu"))
    parallel = TensorParallelConfig(tensor_operations)
    shards = _shard_model(model.diffusion_model, operations, parallel)
    state = _load_state(reader, shards, parallel, upstream_model.HEAD_PREFIXES)
    upstream_loader.shared_codebooks(state)
    patcher = get_model_patcher_class(load_spec.disable_dynamic)(
        model, load_device=device, offload_device=torch.device("cpu"),
        ckpt_name=os.path.basename(load_spec.checkpoint_path),
    )
    loaded = model.diffusion_model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(f"HunyuanImage3 checkpoint mismatch: {loaded}")
    model.tokenizer = Tokenizer.from_file(upstream_loader.TOKENIZER_PATH)
    model.diffusion_model.tensor_parallel_text_context = MethodType(_text_context, model.diffusion_model)
    spectrum = importlib.import_module(extension + ".hunyuan_image_3.spectrum")
    install_state_transport(model.diffusion_model, "hy3_spectrum", (
        spectrum.SpectrumState, spectrum.SpectrumForecaster, spectrum.ChebyshevForecaster,
    ))
    return patcher


def load_model(checkpoint_path, extension_path, disable_dynamic=False):
    from .loader import TensorParallelWorkerLoadSpec

    configuration = current_execution_context().configuration
    size = int(configuration.tensor_parallel_size or 1)
    current = model_management.get_torch_device()
    devices = tuple([current] + [device for device in model_management.get_all_torch_devices() if device != current])
    if size > len(devices):
        raise ValueError(f"Tensor parallel size {size} exceeds {len(devices)} available devices")
    devices = devices[:size]
    dtype = model_management.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    spec = TensorParallelWorkerLoadSpec(os.fspath(checkpoint_path), "hunyuan_image3", {}, disable_dynamic, dtype, extension_path)
    root, executor = launch_tensor_parallel(spec, devices, lambda operations: load_rank(spec, devices[0], operations))
    root.set_additional_models("tensor_parallel", [
        RemoteTensorParallelRankModel(executor, rank, devices[rank], executor.rank_sizes[rank], dtype, not disable_dynamic, os.path.basename(checkpoint_path))
        for rank in range(1, size)
    ])
    root.model.pipeline_executor = CachedStateExecutor(executor)
    root.model.diffusion_model._comfy_tensor_parallel_executor = executor
    root.set_attachments("tensor_parallel_executor", executor)
    root.cached_patcher_init = (partial(load_model, disable_dynamic=disable_dynamic), (checkpoint_path, extension_path))
    logging.info("Loaded hunyuan_image3 with tensor-parallel ranks %s", ", ".join(
        f"{device}:{executor.rank_sizes[rank] / (1024 ** 3):.2f} GiB" for rank, device in enumerate(devices)
    ))
    return root


def install(extension):
    upstream_loader, upstream_model, _ = _modules(extension.__name__)
    if getattr(upstream_loader.load_hunyuan_image_3, "_comfy_tensor_parallel", False):
        return
    original = upstream_loader.load_hunyuan_image_3
    extension_path = os.path.dirname(extension.__file__)
    upstream_model.LayerLookahead = _LayerPrefetch

    @wraps(original)
    def load(checkpoint_path, disable_dynamic=False):
        if int(current_execution_context().configuration.tensor_parallel_size or 1) == 1:
            return original(checkpoint_path, disable_dynamic=disable_dynamic)
        patcher = load_model(checkpoint_path, extension_path, disable_dynamic)
        return patcher, patcher.model.tokenizer

    load._comfy_tensor_parallel = True
    upstream_loader.load_hunyuan_image_3 = load
    importlib.import_module(extension.__name__ + ".nodes").load_hunyuan_image_3 = load

    rewrite = importlib.import_module(extension.__name__ + ".hunyuan_image_3.rewrite")
    generate = rewrite.generate_text

    @wraps(generate)
    def generate_text(model, *args, **kwargs):
        executor = getattr(model, "_comfy_tensor_parallel_executor", None)
        if executor is None:
            return generate(model, *args, **kwargs)
        model_management.load_models_gpu(executor.root_patcher.get_nested_additional_models())
        try:
            with executor.model_context("tensor_parallel_text_context"):
                return generate(_TextModel(model, executor), *args, **kwargs)
        finally:
            executor.finish_execution()

    rewrite.generate_text = generate_text
