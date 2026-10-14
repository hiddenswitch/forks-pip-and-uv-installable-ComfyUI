from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
import os
import sys

import torch

from .. import model_detection, model_management, ops, utils
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class
from ..pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from ..pipeline_parallel.loader import _normalize_detection_state, _normalized_descriptors
from .checkpoint import shard_tensor_parallel_state_dict
from .distributed import RemoteTensorParallelRankModel, launch_tensor_parallel
from .types import TensorParallelConfig
from .operations import tensor_parallel_operations


logger = logging.getLogger(__name__)

SUPPORTED_MODEL_FAMILIES = frozenset(("flux2", "ideogram4", "krea2", "minimax_h3", "kandinsky6"))


@dataclass(frozen=True)
class TensorParallelWorkerLoadSpec:
    checkpoint_path: str
    model_family: str
    model_options: dict
    disable_dynamic: bool
    dtype: torch.dtype
    custom_node_path: str | None = None


def _load_rank(
    reader,
    model_config,
    metadata,
    prefix,
    original_keys,
    checkpoint_path,
    model_options,
    disable_dynamic,
    dtype,
    device,
    parallel_config,
    model_family,
    operations_decorator=tensor_parallel_operations,
):
    rank_config = type(model_config)(model_config.unet_config)
    rank_config.quant_config = utils.deepcopy_list_dict(model_config.quant_config) if model_config.quant_config is not None else None
    rank_config.custom_operations = model_config.custom_operations
    rank_config.optimizations = model_config.optimizations.copy()
    manual_cast_dtype = model_management.unet_manual_cast(
        None if rank_config.quant_config is not None else dtype,
        device,
        rank_config.supported_inference_dtypes,
    )
    rank_config.set_inference_dtype(dtype, manual_cast_dtype, device=device)
    if model_options.get("custom_operations") is not None:
        rank_config.custom_operations = model_options["custom_operations"]
    if model_options.get("fp8_optimizations", False):
        rank_config.optimizations["fp8"] = True
    base_operations = rank_config.custom_operations
    if base_operations is None:
        base_operations = ops.pick_operations(
            rank_config.unet_config.get("dtype"),
            rank_config.manual_cast_dtype,
            fp8_optimizations=rank_config.optimizations.get("fp8", False),
            model_config=rank_config,
        )
    rank_config.custom_operations = operations_decorator(
        base_operations,
        parallel_config,
    )

    if model_family == "kandinsky6":
        from .kandinsky6 import install_magcache_transport, load_state, shard_model

        if rank_config.quant_config is not None:
            raise ValueError("Kandinsky 6 tensor parallelism currently requires unquantized checkpoints")
        extension_module = type(model_config).__module__.split(".")[0]
        model = rank_config.get_model({}, "", device=torch.device("cpu"))
        install_magcache_transport(model.diffusion_model, extension_module)
        shards = shard_model(model.diffusion_model, rank_config.custom_operations, extension_module)
        state = load_state(reader, prefix, shards, parallel_config, extension_module)
        patcher = get_model_patcher_class(disable_dynamic)(
            model, load_device=device, offload_device=torch.device("cpu"),
            ckpt_name=os.path.basename(checkpoint_path),
        )
        loaded = model.diffusion_model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(f"Kandinsky 6 checkpoint mismatch: {loaded}")
        return patcher

    state = reader.load_keys(original_keys.values())
    if prefix:
        state = utils.state_dict_prefix_replace(state, {prefix: ""}, filter_keys=True)
    state = shard_tensor_parallel_state_dict(
        state,
        model_family,
        parallel_config.rank,
        parallel_config.size,
    )
    if model_options.get("custom_operations") is None:
        state, _ = utils.convert_old_quants(state, "", metadata=dict(metadata))

    model = rank_config.get_model(state, "", device=torch.device("cpu"))
    patcher = get_model_patcher_class(disable_dynamic)(
        model,
        load_device=device,
        offload_device=torch.device("cpu"),
        ckpt_name=os.path.basename(checkpoint_path),
    )
    model.load_model_weights(state, "", assign=patcher.is_dynamic())
    return patcher


def load_tensor_parallel_rank(load_spec, rank, device, tensor_operations):
    del rank
    if load_spec.custom_node_path is not None:
        from comfy_compatibility.vanilla import prepare_vanilla_environment
        from ..nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1

        prepare_vanilla_environment()
        exported = _vanilla_load_custom_nodes_1(load_spec.custom_node_path)
        if not exported.NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"Cannot register tensor-parallel custom node: {load_spec.custom_node_path}")
    if load_spec.model_family == "hunyuan_image3":
        from .hunyuan_image3 import load_rank

        return load_rank(load_spec, device, tensor_operations)
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    model_family = None if model_config is None else model_config.unet_config.get("image_model")
    if model_family != load_spec.model_family:
        raise RuntimeError(
            f"Tensor-parallel worker detected {model_family!r}, expected "
            f"{load_spec.model_family!r} in {load_spec.checkpoint_path}"
        )
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)
    return _load_rank(
        reader, model_config, metadata, prefix, original_keys,
        load_spec.checkpoint_path, load_spec.model_options, load_spec.disable_dynamic,
        load_spec.dtype, device, TensorParallelConfig(tensor_operations), model_family,
    )


def load_diffusion_model_tensor_parallel(unet_path, devices, model_options=None, disable_dynamic=False):
    model_options = dict(model_options or {})
    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    model_family = None if model_config is None else model_config.unet_config.get("image_model")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise ValueError(
            f"Tensor parallelism does not support model family {model_family!r}; "
            f"supported families: {supported}"
        )
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)

    parameters = utils.calculate_parameters(detection_state)
    weight_dtype = None if model_config.quant_config is not None else utils.weight_dtype(detection_state)
    dtype = model_options.get("dtype") or model_management.unet_dtype(
        device=devices[0], model_params=parameters,
        supported_dtypes=list(model_config.supported_inference_dtypes),
        weight_dtype=weight_dtype,
    )
    custom_node_path = None
    if model_family == "kandinsky6":
        extension_module = type(model_config).__module__.split(".")[0]
        custom_node_path = os.path.dirname(sys.modules[extension_module].__file__)
    load_spec = TensorParallelWorkerLoadSpec(
        os.fspath(unet_path), model_family, model_options, disable_dynamic, dtype, custom_node_path
    )

    def load_root(tensor_operations):
        return _load_rank(
            reader, model_config, metadata, prefix, original_keys, unet_path,
            model_options,
            disable_dynamic,
            dtype,
            devices[0],
            TensorParallelConfig(tensor_operations),
            model_family,
        )

    root, executor = launch_tensor_parallel(load_spec, devices, load_root)
    remotes = [
        RemoteTensorParallelRankModel(
            executor, rank, devices[rank], executor.rank_sizes[rank], dtype,
            not disable_dynamic, os.path.basename(unet_path),
        )
        for rank in range(1, len(devices))
    ]
    root.set_additional_models("tensor_parallel", remotes)
    root.model.pipeline_executor = executor
    if model_family == "kandinsky6":
        from .custom_node_state import CachedStateExecutor
        from .kandinsky6 import clear_magcache_after_sample, connect_piflow
        from ..patcher_extension import WrappersMP

        connect_piflow(root.model.diffusion_model, executor, extension_module)
        root.model.pipeline_executor = CachedStateExecutor(executor)
        root.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "tensor_parallel_magcache", clear_magcache_after_sample)
    root.set_attachments("tensor_parallel_executor", executor)
    root.cached_patcher_init = (
        partial(load_diffusion_model_tensor_parallel, disable_dynamic=disable_dynamic),
        (unet_path, devices, model_options),
    )
    logger.info(
        "Loaded %s with tensor-parallel ranks %s",
        model_family,
        ", ".join(
            f"{device}:{executor.rank_sizes[index] / (1024 ** 3):.2f} GiB"
            for index, device in enumerate(devices)
        ),
    )
    return root


def try_load_diffusion_model_tensor_parallel(unet_path, model_options=None, disable_dynamic=False):
    configuration = current_execution_context().configuration
    size = int(configuration.tensor_parallel_size or 1)
    if size == 1:
        return None
    if os.path.splitext(os.fspath(unet_path))[1].lower() not in (".safetensors", ".sft"):
        raise ValueError("Tensor parallel loading requires a safetensors checkpoint")
    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, _prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    model_family = None if model_config is None else model_config.unet_config.get("image_model")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        # Tensor parallelism is a default on machines with identical GPUs, so a
        # checkpoint outside the supported families keeps loading through the
        # pipeline-parallel and single-device loaders instead of failing.
        logger.warning(
            "Tensor parallelism does not support model family %r (%s); loading it without tensor parallelism. Supported families: %s",
            model_family,
            os.path.basename(os.fspath(unet_path)),
            ", ".join(sorted(SUPPORTED_MODEL_FAMILIES)),
        )
        return None
    current = model_management.get_torch_device()
    available = model_management.get_all_torch_devices()
    devices = tuple([current] + [device for device in available if device != current])
    if size > len(devices):
        raise ValueError(f"Tensor parallel size {size} exceeds {len(devices)} available devices")
    return load_diffusion_model_tensor_parallel(
        unet_path, devices[:size], model_options=model_options, disable_dynamic=disable_dynamic
    )
