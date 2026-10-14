from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Mapping

import torch

from .. import model_detection, model_management, utils
from ..distributed.config import resolve_distributed_configuration
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class

from .checkpoint import SafetensorsCheckpointReader, is_safetensors_checkpoint
from .distributed import RemotePipelineStageModel
from .memory import (
    AbstractBasePipelineMemoryCoordinator,
    AbstractBasePipelineStageMemoryEstimator,
    ComfyDynamicVRAMStageMemoryEstimator,
    ComfyPipelineMemoryCoordinator,
    ExternalPipelineMemoryCoordinator,
)
from .patcher import get_pipeline_model_patcher_class
from .operations import select_pipeline_operations
from .runtime import AbstractBasePipelineOperations
from .stages import get_pipeline_stage_spec
from .types import (
    PipelineModelMemoryGeometry,
    PipelineParallelConfig,
    PipelinePartitionPlan,
    PipelineStageConfig,
    PipelineStagePlan,
    PipelineWorkerLoadSpec,
    TensorDescriptor,
)

logger = logging.getLogger(__name__)


def _normalize_detection_state(reader: SafetensorsCheckpointReader):
    state_dict = reader.detection_state_dict()
    metadata = dict(reader.metadata)
    state_dict, metadata = utils.convert_old_quants(state_dict, "", metadata=metadata)
    prefix = model_detection.unet_prefix_from_state_dict(state_dict)
    normalized = utils.state_dict_prefix_replace(state_dict, {prefix: ""}, filter_keys=True)
    if not normalized:
        return state_dict, metadata, ""
    normalized, metadata = utils.convert_old_quants(normalized, "", metadata=metadata)
    return normalized, metadata, prefix


def _normalized_descriptors(reader: SafetensorsCheckpointReader, prefix: str):
    descriptors = {}
    original_keys = {}
    for key, descriptor in reader.tensors.items():
        if prefix:
            if not key.startswith(prefix):
                continue
            normalized = key[len(prefix):]
        else:
            normalized = key
        descriptors[normalized] = descriptor
        original_keys[normalized] = key
    return descriptors, original_keys


def _load_pipeline_stage(
    reader,
    model_config,
    metadata,
    prefix,
    original_keys,
    unet_path,
    stage_plan,
    stage,
    model_options,
    disable_dynamic,
    dtype,
    offload_device,
):
    stage_model_config = type(model_config)(model_config.unet_config)
    stage_model_config.quant_config = utils.deepcopy_list_dict(model_config.quant_config) if model_config.quant_config is not None else None
    stage_model_config.custom_operations = model_config.custom_operations
    stage_model_config.optimizations = model_config.optimizations.copy()
    manual_cast_dtype = model_management.unet_manual_cast(
        None if stage_model_config.quant_config is not None else dtype,
        stage.device,
        stage_model_config.supported_inference_dtypes,
    )
    stage_model_config.set_inference_dtype(dtype, manual_cast_dtype, device=stage.device)
    if model_options.get("custom_operations") is not None:
        stage_model_config.custom_operations = model_options["custom_operations"]
    if model_options.get("fp8_optimizations", False):
        stage_model_config.optimizations["fp8"] = True
    stage_model_config.unet_config["pipeline_stage"] = PipelineStageConfig(
        stage.index, stage_plan.size, stage.start_layer, stage.end_layer
    )

    raw_keys = {original_keys[key] for key in stage.owned_keys}
    stage_state = reader.load_keys(raw_keys)
    if prefix:
        stage_state = utils.state_dict_prefix_replace(stage_state, {prefix: ""}, filter_keys=True)
    if model_options.get("custom_operations") is None:
        stage_state, _ = utils.convert_old_quants(stage_state, "", metadata=dict(metadata))
    stage_spec = get_pipeline_stage_spec(model_config.unet_config.get("image_model"))
    stage_state = {
        key: value for key, value in stage_state.items()
        if stage_spec.owns_key(key, stage, stage_plan.size)
    }

    model = stage_model_config.get_model(stage_state, "", device=offload_device)
    patcher_class = get_pipeline_model_patcher_class(disable_dynamic) if stage.index == 0 else get_model_patcher_class(disable_dynamic)
    patcher = patcher_class(
        model,
        load_device=stage.device,
        offload_device=offload_device,
        ckpt_name=os.path.basename(unet_path),
    )
    model.load_model_weights(stage_state, "", assign=patcher.is_dynamic())
    return patcher


class AbstractBasePipelineStageLoader(ABC):
    """How one model family builds a pipeline stage from its checkpoint.

    `load_pipeline_model` owns everything the families share: the memory budgets, the
    checkpoint-geometry plan, the DynamicVRAM repartition and the worker ranks. A family supplies
    the keys the stage spec plans over and the construction of one stage's model.
    """

    @property
    @abstractmethod
    def checkpoint_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_family(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def descriptors(self) -> Mapping[str, TensorDescriptor]:
        """The checkpoint tensors the stage spec partitions, by the keys stages own."""
        raise NotImplementedError

    @abstractmethod
    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        """Choose the stages' compute dtype once, for the first stage's device."""
        raise NotImplementedError

    @abstractmethod
    def load_stage(self, plan: PipelinePartitionPlan, stage: PipelineStagePlan, offload_device: torch.device):
        """A model patcher holding exactly `stage`'s keys."""
        raise NotImplementedError

    @abstractmethod
    def worker_load_spec(self, plan: PipelinePartitionPlan, stage: PipelineStagePlan) -> PipelineWorkerLoadSpec:
        raise NotImplementedError


class ComfyDetectedPipelineStageLoader(AbstractBasePipelineStageLoader):
    """A diffusion model ComfyUI detects from its checkpoint and builds through its model config."""

    def __init__(self, unet_path, model_options=None, disable_dynamic=False, dtype=None):
        self.unet_path = unet_path
        self.model_options = dict(model_options or {})
        self.disable_dynamic = disable_dynamic
        self.reader = SafetensorsCheckpointReader(unet_path)
        detection_state, self.metadata, self.prefix = _normalize_detection_state(self.reader)
        self.model_config = model_detection.model_config_from_unet(detection_state, "", metadata=self.metadata)
        if self.model_config is None:
            raise RuntimeError(f"Pipeline parallel loading could not detect the model type of {unet_path}")
        self._descriptors, self.original_keys = _normalized_descriptors(self.reader, self.prefix)
        self.parameters = utils.calculate_parameters(detection_state)
        self.weight_dtype = None if self.model_config.quant_config is not None else utils.weight_dtype(detection_state)
        self.dtype = dtype

    @property
    def checkpoint_path(self) -> str:
        return os.fspath(self.unet_path)

    @property
    def model_family(self) -> str:
        return self.model_config.unet_config.get("image_model")

    @property
    def descriptors(self) -> Mapping[str, TensorDescriptor]:
        return self._descriptors

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        dtype = self.model_options.get("dtype")
        if dtype is None:
            dtype = model_management.unet_dtype(
                device=device,
                model_params=self.parameters,
                supported_dtypes=list(self.model_config.supported_inference_dtypes),
                weight_dtype=self.weight_dtype,
            )
        self.dtype = dtype
        return dtype

    def load_stage(self, plan, stage, offload_device):
        return _load_pipeline_stage(
            self.reader,
            self.model_config,
            self.metadata,
            self.prefix,
            self.original_keys,
            self.unet_path,
            plan,
            stage,
            self.model_options,
            self.disable_dynamic,
            self.dtype,
            offload_device,
        )

    def worker_load_spec(self, plan, stage) -> PipelineWorkerLoadSpec:
        return PipelineWorkerLoadSpec(
            checkpoint_path=self.checkpoint_path,
            plan=plan,
            stage_index=stage.index,
            model_options=self.model_options,
            disable_dynamic=self.disable_dynamic,
            dtype=self.dtype,
        )


def load_pipeline_worker_stage(load_spec: PipelineWorkerLoadSpec):
    if load_spec.plan.model_family == "hunyuan_image3":
        from .hunyuan_image3 import load_worker_stage

        return load_worker_stage(load_spec)
    stage_loader = ComfyDetectedPipelineStageLoader(
        load_spec.checkpoint_path,
        load_spec.model_options,
        load_spec.disable_dynamic,
        dtype=load_spec.dtype,
    )
    if stage_loader.model_family != load_spec.plan.model_family:
        raise RuntimeError(
            f"Pipeline worker detected {stage_loader.model_family!r}, expected "
            f"{load_spec.plan.model_family!r} in {load_spec.checkpoint_path}"
        )
    return stage_loader.load_stage(
        load_spec.plan,
        load_spec.plan.stages[load_spec.stage_index],
        torch.device("cpu"),
    )


def load_diffusion_model_pipeline(
    unet_path,
    pipeline_config: PipelineParallelConfig,
    model_options=None,
    disable_dynamic=False,
    pipeline_operations: AbstractBasePipelineOperations | None = None,
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
    stage_memory_estimator: AbstractBasePipelineStageMemoryEstimator | None = None,
):
    model_options = dict(model_options or {})
    root = load_pipeline_model(
        ComfyDetectedPipelineStageLoader(unet_path, model_options, disable_dynamic),
        pipeline_config,
        disable_dynamic=disable_dynamic,
        pipeline_operations=pipeline_operations,
        memory_coordinator=memory_coordinator,
        stage_memory_estimator=stage_memory_estimator,
        offload_device=model_options.get("offload_device", torch.device("cpu")),
    )
    root.cached_patcher_init = (
        load_diffusion_model_pipeline,
        (unet_path, pipeline_config, model_options, disable_dynamic),
    )
    return root


def load_pipeline_model(
    stage_loader: AbstractBasePipelineStageLoader,
    pipeline_config: PipelineParallelConfig,
    disable_dynamic=False,
    pipeline_operations: AbstractBasePipelineOperations | None = None,
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
    stage_memory_estimator: AbstractBasePipelineStageMemoryEstimator | None = None,
    offload_device: torch.device = torch.device("cpu"),
):
    """Load `stage_loader`'s model as pipeline stages and return the first stage's patcher."""
    image_model = stage_loader.model_family
    stage_spec = get_pipeline_stage_spec(image_model)
    descriptors = stage_loader.descriptors
    memory_coordinator = memory_coordinator or ComfyPipelineMemoryCoordinator()
    stage_memory_estimator = stage_memory_estimator or ComfyDynamicVRAMStageMemoryEstimator()
    budgets = memory_coordinator.budgets(pipeline_config.devices)
    plan = stage_spec.plan(descriptors, pipeline_config, memory_budgets=budgets)
    pipeline_operations = select_pipeline_operations(pipeline_config.devices, pipeline_operations)
    dtype = stage_loader.resolve_dtype(plan.stages[0].device)
    checkpoint_name = os.path.basename(stage_loader.checkpoint_path)

    def load_stages(stage_plan):
        return [stage_loader.load_stage(stage_plan, stage, offload_device) for stage in stage_plan.stages]

    def load_root(stage_plan):
        return stage_loader.load_stage(stage_plan, stage_plan.stages[0], offload_device)

    def worker_load_specs(stage_plan):
        return tuple(stage_loader.worker_load_spec(stage_plan, stage) for stage in stage_plan.stages[1:])

    def worker_geometry(stage_plan, root_patcher, worker_geometries):
        geometries = (
            stage_memory_estimator.estimate_stage(stage_spec, stage_plan.stages[0], root_patcher),
            *worker_geometries,
        )
        block_bytes = [0] * stage_spec.block_count
        non_block_bytes = [0] * stage_plan.size
        for stage, geometry in zip(stage_plan.stages, geometries, strict=True):
            for index, size in geometry.block_bytes.items():
                block_bytes[index] += size
            non_block_bytes[stage.index] += geometry.non_block_bytes
        return PipelineModelMemoryGeometry(tuple(block_bytes), tuple(non_block_bytes))

    executor = None

    if pipeline_operations.uses_worker_processes:
        root = load_root(plan)
        executor = pipeline_operations.create_executor(
            plan,
            [root.model.diffusion_model],
            worker_load_specs(plan),
        )
        model_memory_geometry = worker_geometry(plan, root, executor.worker_geometries)
        measured_plan = stage_spec.plan(
            descriptors,
            pipeline_config,
            memory_budgets=budgets,
            model_memory_geometry=model_memory_geometry,
        )
        initial_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in plan.stages)
        measured_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in measured_plan.stages)
        if initial_boundaries != measured_boundaries:
            logger.info(
                "Repartitioning %s process ranks from checkpoint estimate %s to DynamicVRAM geometry %s",
                image_model,
                initial_boundaries,
                measured_boundaries,
            )
            executor.close()
            del root
            plan = measured_plan
            root = load_root(plan)
            executor = pipeline_operations.create_executor(
                plan,
                [root.model.diffusion_model],
                worker_load_specs(plan),
            )
        else:
            plan = measured_plan
        stage_patchers = [root]
    else:
        stage_patchers = load_stages(plan)
        model_memory_geometry = stage_memory_estimator.estimate(stage_spec, plan, stage_patchers)
        measured_plan = stage_spec.plan(
            descriptors,
            pipeline_config,
            memory_budgets=budgets,
            model_memory_geometry=model_memory_geometry,
        )
        initial_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in plan.stages)
        measured_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in measured_plan.stages)
        if initial_boundaries != measured_boundaries:
            logger.info(
                "Repartitioning %s from checkpoint estimate %s to DynamicVRAM geometry %s",
                image_model,
                initial_boundaries,
                measured_boundaries,
            )
            del stage_patchers
            stage_patchers = load_stages(measured_plan)
        plan = measured_plan

    root = stage_patchers[0]
    if executor is None:
        stages = [root.model.diffusion_model]
        stages.extend(patcher.model.diffusion_model.forward_pipeline_stage for patcher in stage_patchers[1:])
        executor = pipeline_operations.create_executor(plan, stages)
    if pipeline_operations.uses_worker_processes:
        self_managed_remote_devices = resolve_distributed_configuration(
            current_execution_context().configuration
        ).externally_launched
        remote_stages = [
            RemotePipelineStageModel(
                executor,
                stage.index,
                stage.device,
                model_memory_geometry.stage_bytes(stage),
                dtype,
                not disable_dynamic,
                checkpoint_name,
                self_managed_device=self_managed_remote_devices,
            )
            for stage in plan.stages[1:]
        ]
        root.set_additional_models(root.pipeline_additional_models_key, remote_stages)
        del stage_patchers[1:]
    else:
        root.set_additional_models(root.pipeline_additional_models_key, stage_patchers[1:])
    root.model.pipeline_executor = executor
    root.set_attachments("pipeline_parallel_plan", plan)
    root.set_attachments("pipeline_parallel_executor", executor)
    logger.info(
        "Loaded %s with pipeline stages %s",
        image_model,
        ", ".join(
            f"{stage.device}:{stage.start_layer}-{stage.end_layer} "
            f"({model_memory_geometry.stage_bytes(stage) / (1024 ** 3):.2f} GiB DynamicVRAM geometry, "
            f"{budgets[stage.index].available_weight_bytes / (1024 ** 3):.2f} GiB available)"
            for stage in plan.stages
        ),
    )
    return root


def resolve_pipeline_devices(
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
):
    """The stage devices of a transparently pipelined load and the coordinator for their budgets.

    None when this process should load a model ordinarily: one device, or pipeline parallelism
    explicitly set to one stage. An external launch spans every launched rank.
    """
    runtime_configuration = current_execution_context().configuration
    distributed = resolve_distributed_configuration(runtime_configuration)
    requested_size = distributed.pipeline_parallel_size
    if distributed.externally_launched:
        from .external import get_external_pipeline_runtime

        runtime = get_external_pipeline_runtime()
        if runtime is None:
            raise RuntimeError("External-launcher pipeline runtime has not been initialized")
        devices = runtime.logical_devices
        memory_coordinator = memory_coordinator or ExternalPipelineMemoryCoordinator(runtime)
    else:
        current = model_management.get_torch_device()
        available = model_management.get_all_torch_devices()
        devices = tuple([current] + [device for device in available if device != current])
        if requested_size > 1:
            if requested_size > len(devices):
                raise RuntimeError(
                    f"Pipeline parallel size {requested_size} exceeds the "
                    f"{len(devices)} available devices"
                )
            devices = devices[:requested_size]
        elif (
            runtime_configuration is not None
            and runtime_configuration.pipeline_parallel_size == 1
        ):
            return None
    if len(devices) < 2:
        return None
    return devices, memory_coordinator


def try_load_diffusion_model_pipeline(
    unet_path,
    model_options=None,
    disable_dynamic=False,
    pipeline_operations: AbstractBasePipelineOperations | None = None,
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
    stage_memory_estimator: AbstractBasePipelineStageMemoryEstimator | None = None,
):
    """Load a supported checkpoint transparently when multiple devices are selected."""
    if not is_safetensors_checkpoint(unet_path):
        return None
    placement = resolve_pipeline_devices(memory_coordinator)
    if placement is None:
        return None
    devices, memory_coordinator = placement

    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, _prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None or model_config.unet_config.get("image_model") not in (
        "qwen_image",
        "minimax_h3",
        "flux2",
    ):
        return None

    return load_diffusion_model_pipeline(
        unet_path,
        PipelineParallelConfig(devices),
        model_options=model_options,
        disable_dynamic=disable_dynamic,
        pipeline_operations=pipeline_operations,
        memory_coordinator=memory_coordinator,
        stage_memory_estimator=stage_memory_estimator,
    )
