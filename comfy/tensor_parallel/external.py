from __future__ import annotations

from datetime import timedelta
import gc
import logging
import traceback

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration
from ..distributed.process_group import (
    create_independent_process_groups,
    destroy_independent_process_groups,
)
from .distributed import (
    TorchDistributedModelParallelExecutor,
    _ObjectCoordinator,
    _run_worker,
)
from .runtime import TorchDistributedTensorParallelOperations


logger = logging.getLogger(__name__)
_runtime: ExternalTensorParallelRuntime | None = None
CONTROL_PLANE_TIMEOUT = timedelta(days=30)


class ExternalTensorParallelRuntime:
    """A tensor-parallel rank service created by an external launcher."""

    def __init__(self, configuration: DistributedConfiguration):
        if not configuration.externally_launched:
            raise ValueError("External tensor parallelism requires a launcher-provided process identity")
        if configuration.pipeline_parallel_size != 1:
            raise ValueError("External tensor parallelism requires pipeline_parallel_size=1")
        if configuration.world_size != configuration.tensor_parallel_size:
            raise ValueError("External tensor parallelism requires WORLD_SIZE == tensor_parallel_size")
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("External tensor parallelism requires CUDA and NCCL")

        self._configuration = configuration
        self.device = torch.device("cuda", configuration.local_rank)
        torch.cuda.set_device(self.device)
        self.logical_devices = tuple(
            torch.device("cuda", rank) for rank in range(configuration.world_size)
        )
        self.process_groups = create_independent_process_groups(
            f"tcp://{configuration.master_addr}:{configuration.master_port}",
            configuration.rank,
            configuration.world_size,
            self.device,
            "comfy-external-tensor",
            timeout=CONTROL_PLANE_TIMEOUT,
        )
        self.operations = TorchDistributedTensorParallelOperations(
            configuration.rank,
            configuration.world_size,
            self.device,
            self.process_groups.device_process_group,
            self.process_groups.control_process_group,
        )
        self.coordinator = _ObjectCoordinator(self.process_groups.control_process_group)
        self._executor: TorchDistributedModelParallelExecutor | None = None
        self._closed = False

        from .. import model_management

        model_management.set_torch_device(self.device)
        logger.info(
            "Initialized external tensor-parallel rank %d/%d on %s",
            configuration.rank,
            configuration.world_size,
            self.device,
        )

    @property
    def configuration(self) -> DistributedConfiguration:
        return self._configuration

    @property
    def is_driver(self) -> bool:
        return self.configuration.rank == 0

    def load_executor(self, load_spec, load_root):
        if not self.is_driver:
            raise RuntimeError("Only tensor-parallel rank zero can create the driver executor")
        if self._executor is not None and not self._executor._closed:
            self._executor.close()

        self.coordinator.broadcast_command({"kind": "load_tensor", "load_spec": load_spec})
        rank_sizes = []
        for rank in range(1, self.configuration.world_size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") != "ready":
                raise RuntimeError(
                    f"External tensor-parallel rank {rank} failed to load:\n"
                    f"{response.get('traceback', response)!s}"
                )
            rank_sizes.append(int(response["size"]))

        root = load_root(self.operations)
        self._executor = TorchDistributedModelParallelExecutor(
            root.model.diffusion_model,
            self.operations,
            self.coordinator,
            (),
            (),
            self.logical_devices,
            (root.model_size(), *rank_sizes),
            root_patcher=root,
            close_command="unload_tensor",
            preserve_process_groups=True,
        )
        return root, self._executor

    def run_worker_service(self) -> None:
        if self.is_driver:
            raise RuntimeError("Tensor-parallel rank zero is the ComfyUI driver, not a worker")
        from .loader import load_tensor_parallel_rank

        while True:
            command = self.coordinator.broadcast_command()
            if command["kind"] == "close":
                break
            if command["kind"] != "load_tensor":
                self.coordinator.send_object(
                    {"kind": "error", "traceback": f"Unexpected command: {command!r}"},
                    0,
                )
                raise RuntimeError(
                    f"External tensor-parallel rank {self.configuration.rank} expected load_tensor, "
                    f"got {command!r}"
                )
            try:
                patcher = load_tensor_parallel_rank(
                    command["load_spec"],
                    self.configuration.tensor_rank,
                    self.device,
                    self.operations,
                )
                self.coordinator.send_object(
                    {"kind": "ready", "size": patcher.model_size()}, 0
                )
                result = _run_worker(
                    self.operations,
                    self.coordinator,
                    patcher,
                    "External Tensor Parallel Rank Command",
                    close_commands=("unload_tensor", "close"),
                )
                del patcher
                gc.collect()
                from .. import model_management

                model_management.soft_empty_cache()
                if result == "close":
                    break
            except Exception:
                self.coordinator.send_object(
                    {"kind": "error", "traceback": traceback.format_exc()}, 0
                )
                raise
        self.close()

    def close(self) -> None:
        global _runtime
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.close()
            self._executor = None
        if self.is_driver:
            try:
                self.coordinator.broadcast_command({"kind": "close"})
            except Exception:
                logger.debug("Could not stop external tensor-parallel ranks", exc_info=True)
        destroy_independent_process_groups(self.process_groups)
        if _runtime is self:
            _runtime = None


def initialize_external_tensor_parallel_runtime(
    configuration: DistributedConfiguration,
) -> ExternalTensorParallelRuntime | None:
    global _runtime
    if not configuration.externally_launched or configuration.tensor_parallel_size == 1:
        return None
    if _runtime is None:
        _runtime = ExternalTensorParallelRuntime(configuration)
    elif _runtime.configuration != configuration:
        raise RuntimeError("External tensor-parallel runtime was already initialized differently")
    return _runtime


def get_external_tensor_parallel_runtime() -> ExternalTensorParallelRuntime | None:
    return _runtime
