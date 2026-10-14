from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import timedelta
import gc
import logging
import traceback

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration
from ..distributed.process_group import create_device_process_group

from .distributed import (
    TorchDistributedPipelineExecutor,
    TorchDistributedProcessGroupCoordinator,
    _load_worker_stage,
    _run_worker_commands,
)
from .runtime import AbstractBasePipelineOperations
from .types import PipelineDeviceMemoryBudget


logger = logging.getLogger(__name__)
_runtime: ExternalTorchDistributedRuntime | None = None
# A launcher rank waits on the control group for its driver's next command for as long as the
# driver is busy elsewhere (idle between prompts, loading its own stage, waiting on another rank's
# load), and gloo fails a pending receive once the group timeout passes. The bound only has to
# exceed any such wait; gloo adds it to a nanosecond steady clock, so it cannot be unbounded.
CONTROL_PLANE_TIMEOUT = timedelta(days=30)


class AbstractBaseExternalPipelinePlacementProvider(ABC):
    """Map launcher ranks to logical stages and rank-local accelerators."""

    @abstractmethod
    def local_device(self, configuration: DistributedConfiguration) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def logical_devices(
        self,
        configuration: DistributedConfiguration,
    ) -> tuple[torch.device, ...]:
        raise NotImplementedError

    @abstractmethod
    def available_weight_bytes(self, device: torch.device) -> int:
        raise NotImplementedError


class CudaExternalPipelinePlacementProvider(
    AbstractBaseExternalPipelinePlacementProvider
):
    """CUDA placement using canonical TorchElastic rank identities."""

    def local_device(self, configuration: DistributedConfiguration) -> torch.device:
        return torch.device("cuda", configuration.local_rank)

    def logical_devices(
        self,
        configuration: DistributedConfiguration,
    ) -> tuple[torch.device, ...]:
        # These are stable planner identities, not devices rank zero may access.
        # Every stage resolves its identity to its own LOCAL_RANK before loading.
        return tuple(torch.device("cuda", rank) for rank in range(configuration.world_size))

    def available_weight_bytes(self, device: torch.device) -> int:
        from .. import model_management

        projected = model_management.projected_dynamic_vram_available_memory((device,))
        return max(1, int(projected[device]))


class AbstractBaseExternalPipelineRuntime(ABC):
    @property
    @abstractmethod
    def configuration(self) -> DistributedConfiguration:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_driver(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def logical_devices(self) -> tuple[torch.device, ...]:
        raise NotImplementedError

    @abstractmethod
    def probe_memory_budgets(
        self,
        devices,
    ) -> tuple[PipelineDeviceMemoryBudget, ...]:
        raise NotImplementedError

    @abstractmethod
    def run_worker_service(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_executor(self, plan, first_stage, worker_load_specs):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ExternalTorchDistributedRuntime(AbstractBaseExternalPipelineRuntime):
    """Long-lived process groups and rank service created by torchrun et al."""

    def __init__(
        self,
        configuration: DistributedConfiguration,
        placement_provider: AbstractBaseExternalPipelinePlacementProvider | None = None,
    ):
        if not configuration.externally_launched:
            raise ValueError("External runtime requires a launcher-provided process identity")
        if configuration.tensor_parallel_size != 1:
            raise NotImplementedError(
                "External-launcher tensor parallelism is not implemented; "
                "use the internal multiprocessing executor"
            )
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("External pipeline launching requires CUDA and NCCL")

        self._configuration = configuration
        self.placement_provider = (
            placement_provider or CudaExternalPipelinePlacementProvider()
        )
        self._device = self.placement_provider.local_device(configuration)
        self._logical_devices = self.placement_provider.logical_devices(configuration)
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                rank=configuration.rank,
                world_size=configuration.world_size,
                timeout=CONTROL_PLANE_TIMEOUT,
            )
        elif (
            dist.get_rank() != configuration.rank
            or dist.get_world_size() != configuration.world_size
        ):
            raise RuntimeError(
                "Existing torch.distributed process group does not match canonical configuration"
            )
        self.device_group = create_device_process_group(
            range(configuration.world_size),
            self.device,
        )
        self.coordinator = TorchDistributedProcessGroupCoordinator(
            configuration.rank,
            self.device,
            self.device_group,
        )
        self._closed = False

        from .. import model_management

        model_management.set_torch_device(self.device)
        logger.info(
            "Initialized external pipeline rank %d/%d on %s",
            configuration.rank,
            configuration.world_size,
            self.device,
        )

    @property
    def configuration(self) -> DistributedConfiguration:
        return self._configuration

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_driver(self) -> bool:
        return self.configuration.is_first_pipeline_stage

    @property
    def logical_devices(self) -> tuple[torch.device, ...]:
        return self._logical_devices

    def _local_available_weight_bytes(self) -> int:
        return self.placement_provider.available_weight_bytes(self.device)

    def probe_memory_budgets(
        self,
        devices,
    ) -> tuple[PipelineDeviceMemoryBudget, ...]:
        devices = tuple(devices)
        if not self.is_driver:
            raise RuntimeError("Only pipeline rank zero can probe all stage budgets")
        if devices != self.logical_devices:
            raise ValueError(
                f"External pipeline devices {devices} do not match launcher stages "
                f"{self.logical_devices}"
            )
        self.coordinator.broadcast_command({"kind": "probe_memory"})
        available = [self._local_available_weight_bytes()]
        for rank in range(1, self.configuration.world_size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") != "memory_budget":
                raise RuntimeError(
                    f"External pipeline rank {rank} failed to report memory: {response!r}"
                )
            available.append(int(response["available_weight_bytes"]))
        return tuple(
            PipelineDeviceMemoryBudget(device, max(1, size))
            for device, size in zip(devices, available, strict=True)
        )

    def run_worker_service(self) -> None:
        if self.is_driver:
            raise RuntimeError("Pipeline rank zero is the ComfyUI driver, not a worker")
        rank = self.configuration.pipeline_rank
        while True:
            command = self.coordinator.broadcast_command()
            if command["kind"] == "close":
                break
            if command["kind"] == "probe_memory":
                self.coordinator.send_object(
                    {
                        "kind": "memory_budget",
                        "available_weight_bytes": self._local_available_weight_bytes(),
                    },
                    0,
                )
                continue
            if command["kind"] != "load_pipeline":
                raise RuntimeError(
                    f"External pipeline rank {rank} expected a load command, got {command!r}"
                )
            try:
                load_specs = command["worker_load_specs"]
                load_spec = load_specs[rank - 1]
                if load_spec.stage_index != rank:
                    raise RuntimeError(
                        f"External pipeline rank {rank} received stage {load_spec.stage_index}"
                    )
                stages = list(load_spec.plan.stages)
                stages[rank] = replace(stages[rank], device=self.device)
                load_spec = replace(
                    load_spec,
                    plan=replace(load_spec.plan, stages=tuple(stages)),
                )
                patcher, geometry = _load_worker_stage(load_spec, rank)
                self.coordinator.send_object(
                    {"kind": "ready", "geometry": geometry}, 0
                )
                result = _run_worker_commands(
                    self.coordinator,
                    rank,
                    self.configuration.world_size,
                    self.device,
                    patcher,
                )
                del patcher
                gc.collect()
                if result == "close":
                    break
            except Exception:
                self.coordinator.send_object(
                    {"kind": "error", "traceback": traceback.format_exc()}, 0
                )
                raise
        self.close()

    def load_executor(self, plan, first_stage, worker_load_specs):
        if not self.is_driver:
            raise RuntimeError("Only pipeline rank zero can create the driver executor")
        if plan.size != self.configuration.pipeline_parallel_size:
            raise ValueError(
                f"Plan size {plan.size} does not match launched pipeline size "
                f"{self.configuration.pipeline_parallel_size}"
            )
        self.coordinator.broadcast_command(
            {
                "kind": "load_pipeline",
                "worker_load_specs": tuple(worker_load_specs),
            }
        )
        worker_geometries = []
        for rank in range(1, plan.size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") != "ready":
                raise RuntimeError(
                    f"External pipeline rank {rank} failed to load:\n{response['traceback']}"
                )
            worker_geometries.append(response["geometry"])
        return TorchDistributedPipelineExecutor(
            plan,
            first_stage,
            self.coordinator,
            (),
            (),
            self.device_group,
            worker_geometries,
            close_command="unload_pipeline",
            destroy_process_group=False,
        )

    def close(self) -> None:
        global _runtime
        if self._closed:
            return
        self._closed = True
        if self.is_driver and dist.is_initialized():
            try:
                self.coordinator.broadcast_command({"kind": "close"})
            except Exception:
                logger.debug("Could not stop external pipeline ranks", exc_info=True)
        if dist.is_initialized():
            try:
                dist.destroy_process_group(self.device_group)
            finally:
                dist.destroy_process_group()
        if _runtime is self:
            _runtime = None


class ExternalTorchDistributedPipelineOperations(AbstractBasePipelineOperations):
    def __init__(self, runtime: AbstractBaseExternalPipelineRuntime):
        self.runtime = runtime

    @property
    def uses_worker_processes(self) -> bool:
        return True

    def supports(self, devices) -> bool:
        return (
            self.runtime.is_driver
            and len(devices) == self.runtime.configuration.pipeline_parallel_size
            and all(device.type == "cuda" for device in devices)
        )

    def create_transport(self):
        raise RuntimeError("External pipeline transport is owned by its rank runtime")

    def create_stage_runner(self):
        raise RuntimeError("External pipeline stages run in launcher processes")

    def create_executor(self, plan, stages, worker_load_specs=()):
        if len(stages) != 1:
            raise ValueError("External pipeline driver loads only its first stage")
        return self.runtime.load_executor(plan, stages[0], worker_load_specs)


def initialize_external_pipeline_runtime(
    configuration: DistributedConfiguration,
) -> ExternalTorchDistributedRuntime | None:
    global _runtime
    if not configuration.externally_launched:
        return None
    if _runtime is None:
        _runtime = ExternalTorchDistributedRuntime(configuration)
    elif _runtime.configuration != configuration:
        raise RuntimeError("External pipeline runtime was already initialized differently")
    return _runtime


def get_external_pipeline_runtime() -> ExternalTorchDistributedRuntime | None:
    return _runtime
