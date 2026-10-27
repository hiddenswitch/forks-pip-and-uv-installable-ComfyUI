from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

from .. import model_detection, model_management, utils
from ..execution_context import current_execution_context
from ..pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from ..pipeline_parallel.loader import (
    _normalize_detection_state,
    _normalized_descriptors,
)
from ..tensor_parallel.distributed import (
    RemoteModelParallelRankModel,
    launch_model_parallel,
)
from ..tensor_parallel.loader import _load_rank
from .operations import xdit_sequence_parallel_operations
from .types import XDiTSequenceParallelConfig


logger = logging.getLogger(__name__)
SUPPORTED_MODEL_FAMILIES = frozenset(("flux2", "qwen_image", "minimax_h3"))


@dataclass(frozen=True)
class XDiTSequenceParallelWorkerLoadSpec:
    checkpoint_path: str
    model_options: dict
    disable_dynamic: bool
    dtype: torch.dtype
    strategy: str


def _replicate_state(state, model_family, rank, size):
    del model_family, rank, size
    return dict(state)


def _load_xdit_rank(
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
    operations,
    strategy,
):
    return _load_rank(
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
        XDiTSequenceParallelConfig(operations, strategy),
        model_config.unet_config["image_model"],
        operations_decorator=xdit_sequence_parallel_operations,
        state_transform=_replicate_state,
    )


def load_xdit_sequence_parallel_rank(load_spec, rank, device, operations):
    del rank
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(
        detection_state,
        "",
        metadata=metadata,
    )
    family = None if model_config is None else model_config.unet_config.get("image_model")
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise RuntimeError(
            f"xDiT worker could not detect a supported model in "
            f"{load_spec.checkpoint_path}"
        )
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)
    return _load_xdit_rank(
        reader,
        model_config,
        metadata,
        prefix,
        original_keys,
        load_spec.checkpoint_path,
        load_spec.model_options,
        load_spec.disable_dynamic,
        load_spec.dtype,
        device,
        operations,
        load_spec.strategy,
    )


def load_diffusion_model_xdit_sequence_parallel(
    unet_path,
    devices,
    model_options=None,
    disable_dynamic=False,
    strategy="ulysses",
):
    model_options = dict(model_options or {})
    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(
        detection_state,
        "",
        metadata=metadata,
    )
    family = None if model_config is None else model_config.unet_config.get("image_model")
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            "xDiT sequence parallelism currently supports Flux2, Qwen Image, and MiniMax H3"
        )
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)

    parameters = utils.calculate_parameters(detection_state)
    weight_dtype = (
        None
        if model_config.quant_config is not None
        else utils.weight_dtype(detection_state)
    )
    dtype = model_options.get("dtype") or model_management.unet_dtype(
        device=devices[0],
        model_params=parameters,
        supported_dtypes=list(model_config.supported_inference_dtypes),
        weight_dtype=weight_dtype,
    )
    load_spec = XDiTSequenceParallelWorkerLoadSpec(
        os.fspath(unet_path),
        model_options,
        disable_dynamic,
        dtype,
        strategy,
    )

    def load_root(operations):
        return _load_xdit_rank(
            reader,
            model_config,
            metadata,
            prefix,
            original_keys,
            unet_path,
            model_options,
            disable_dynamic,
            dtype,
            devices[0],
            operations,
            strategy,
        )

    root, executor = launch_model_parallel(
        load_spec,
        devices,
        load_root,
        f"xdit_{strategy}",
    )
    remotes = [
        RemoteModelParallelRankModel(
            executor,
            rank,
            devices[rank],
            executor.rank_sizes[rank],
            dtype,
            not disable_dynamic,
            os.path.basename(unet_path),
        )
        for rank in range(1, len(devices))
    ]
    root.set_additional_models("xdit_sequence_parallel", remotes)
    root.model.pipeline_executor = executor
    root.set_attachments("xdit_sequence_parallel_executor", executor)
    root.cached_patcher_init = (
        load_diffusion_model_xdit_sequence_parallel,
        (unet_path, devices, model_options, disable_dynamic, strategy),
    )
    logger.info(
        "Loaded %s with xDiT %s ranks %s",
        family,
        strategy,
        ", ".join(
            f"{device}:{executor.rank_sizes[index] / (1024 ** 3):.2f} GiB"
            for index, device in enumerate(devices)
        ),
    )
    return root


def try_load_diffusion_model_xdit_sequence_parallel(
    unet_path,
    model_options=None,
    disable_dynamic=False,
):
    configuration = current_execution_context().configuration
    degree = int(configuration.ulysses_degree or 1)
    ring_degree = int(configuration.ring_degree or 1)
    if degree == 1 and ring_degree == 1:
        return None
    if degree != 1 and ring_degree != 1:
        raise NotImplementedError(
            "Combined xDiT Ulysses and Ring parallelism is not implemented yet"
        )
    if os.path.splitext(os.fspath(unet_path))[1].lower() not in (
        ".safetensors",
        ".sft",
    ):
        raise ValueError("xDiT sequence-parallel loading requires a safetensors checkpoint")
    current = model_management.get_torch_device()
    available = model_management.get_all_torch_devices()
    devices = tuple([current] + [device for device in available if device != current])
    strategy = "ring" if ring_degree > 1 else "ulysses"
    size = ring_degree if strategy == "ring" else degree
    if size > len(devices):
        raise ValueError(
            f"xDiT {strategy} degree {size} exceeds {len(devices)} available devices"
        )
    return load_diffusion_model_xdit_sequence_parallel(
        unet_path,
        devices[:size],
        model_options=model_options,
        disable_dynamic=disable_dynamic,
        strategy=strategy,
    )
