from __future__ import annotations

import importlib
from functools import wraps

from .. import model_management
from ..ldm.kandinsky5.model import FeedForward, SelfAttention
from .custom_node_state import install_state_transport


def install_magcache_transport(model, extension_module):
    cache = importlib.import_module(extension_module + ".kandinsky6.magcache")
    install_state_transport(model, "k6_magcache", (cache.K6MagCacheState, cache._LaneState))


def clear_magcache_after_sample(executor, *args, **kwargs):
    state = executor.class_obj.model_options.get("transformer_options", {}).get("k6_magcache")
    try:
        return executor(*args, **kwargs)
    finally:
        if state is not None:
            state._lanes.clear()
            state._finished = True


class _DistributedDiT:
    def __init__(self, model, executor):
        self.n_grid = model.n_grid
        self.executor = executor

    def __call__(self, *args, **kwargs):
        return self.executor.execute_method("forward", *args, **kwargs)


def connect_piflow(model, executor, extension_module):
    sampling = importlib.import_module(extension_module + ".kandinsky6.sampling")
    model._comfy_tensor_parallel_executor = executor
    if getattr(sampling.rollout, "_comfy_tensor_parallel", False):
        return
    original = sampling.rollout

    @wraps(original)
    def rollout(dit, video, *args, **kwargs):
        distributed = getattr(dit, "_comfy_tensor_parallel_executor", None)
        if distributed is None:
            return original(dit, video, *args, **kwargs)
        patcher = distributed.root_patcher
        model_management.load_models_gpu(
            [patcher, *patcher.get_nested_additional_models()],
            memory_required=patcher.model.memory_required(video.shape),
        )
        try:
            return original(_DistributedDiT(dit, distributed), video, *args, **kwargs)
        finally:
            distributed.finish_execution()

    rollout._comfy_tensor_parallel = True
    sampling.rollout = rollout


def shard_model(model, operations, extension_module):
    """Partition K6 attention heads and FFN channels before checkpoint loading."""
    external = importlib.import_module(extension_module + ".kandinsky6.ldm.model")
    shards = {}
    parallel = operations.tensor_parallel
    for name, module in list(model.named_modules()):
        if isinstance(module, (SelfAttention, external.AsymCrossAttention)):
            if module.num_heads % parallel.size:
                raise ValueError(f"{name}: {module.num_heads} heads must divide {parallel.size} ranks")
            module.num_heads //= parallel.size
            columns, rows = ("to_query", "to_key", "to_value"), ("out_layer",)
        elif isinstance(module, FeedForward):
            columns, rows = ("in_layer",), ("out_layer",)
        else:
            continue
        for attribute in (*columns, *rows):
            linear = getattr(module, attribute)
            axis = 0 if attribute in columns else 1
            factory = operations.ColumnParallelLinear if axis == 0 else operations.RowParallelLinear
            lazy = linear.weight is None
            replacement = factory(
                linear.in_features, linear.out_features,
                bias=linear.comfy_need_lazy_init_bias if lazy else linear.bias is not None,
                device="cpu", dtype=linear.weight_comfy_model_dtype if lazy else linear.weight.dtype,
            )
            setattr(module, attribute, replacement)
            shards[f"{name}.{attribute}"] = axis
    return shards


def load_state(reader, prefix, shards, parallel, extension_module):
    registration = importlib.import_module(extension_module + ".kandinsky6.register")
    selections, names = {}, {}
    for key, descriptor in reader.tensors.items():
        if prefix and not key.startswith(prefix):
            continue
        name = registration._native_key(key[len(prefix):])
        module, _, parameter = name.rpartition(".")
        axis = shards.get(module)
        selection = None
        if axis is not None and parameter == "weight":
            if descriptor.shape[axis] % parallel.size:
                raise ValueError(f"{key} cannot be split across {parallel.size} ranks")
            width = descriptor.shape[axis] // parallel.size
            selection = tuple(slice(parallel.rank * width, (parallel.rank + 1) * width) if i == axis else slice(None) for i in range(len(descriptor.shape)))
        elif axis == 0 and parameter == "bias":
            width = descriptor.shape[0] // parallel.size
            selection = (slice(parallel.rank * width, (parallel.rank + 1) * width),)
        elif axis == 1 and parameter == "bias" and parallel.rank != 0:
            continue
        selections[key] = selection
        names[key] = name
    return {names[key]: value for key, value in reader.load_slices(selections).items()}
