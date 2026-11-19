from __future__ import annotations

import atexit
import dataclasses
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
from ..distributed.tracing import distributed_command_span
from .distributed import (
    TorchDistributedModelParallelExecutor,
    _ObjectCoordinator,
    _abort_device_group,
    _broadcast_tensors,
    _send_worker_error,
    unpack_pipeline_value,
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
        self._executors: dict[int, TorchDistributedModelParallelExecutor] = {}
        self._next_model_id = 0
        self._closed = False
        if self.is_driver:
            atexit.register(self.close)

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
        self._next_model_id += 1
        model_id = self._next_model_id

        self.coordinator.broadcast_command({
            "kind": "load_tensor",
            "model_id": model_id,
            "load_spec": load_spec,
        })
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
        executor = TorchDistributedModelParallelExecutor(
            root.model.diffusion_model,
            self.operations,
            _ObjectCoordinator(self.process_groups.control_process_group, model_id=model_id),
            (),
            (),
            self.logical_devices,
            (root.model_size(), *rank_sizes),
            root_patcher=root,
            close_command="unload_tensor",
            preserve_process_groups=True,
        )
        self._executors[model_id] = executor
        return root, executor

    def run_worker_service(self) -> None:
        if self.is_driver:
            raise RuntimeError("Tensor-parallel rank zero is the ComfyUI driver, not a worker")
        from .loader import load_tensor_parallel_rank
        from .. import model_management

        loaded_models: dict[int, list] = {}
        forward_failure = None
        while True:
            command = self.coordinator.broadcast_command()
            if command["kind"] == "close":
                break
            model_id = int(command.get("model_id", 1))
            if command["kind"] == "load_tensor":
                try:
                    load_spec = command["load_spec"]
                    ckpt_path = getattr(load_spec, "checkpoint_path", None)
                    if ckpt_path:
                        from ..model_downloader import resolve_worker_checkpoint

                        load_spec = dataclasses.replace(load_spec, checkpoint_path=resolve_worker_checkpoint(ckpt_path))
                    patcher = load_tensor_parallel_rank(
                        load_spec,
                        self.configuration.tensor_rank,
                        self.device,
                        self.operations,
                    )
                    loaded_models[model_id] = [patcher, patcher.model.diffusion_model, []]
                    self.coordinator.send_object(
                        {"kind": "ready", "size": patcher.model_size()}, 0
                    )
                except Exception:
                    self.coordinator.send_object(
                        {"kind": "error", "traceback": traceback.format_exc()}, 0
                    )
                    raise
                continue
            if command["kind"] == "unload_tensor":
                entry = loaded_models.pop(model_id, None)
                if entry is not None:
                    del entry
                    gc.collect()
                    if forward_failure is None:
                        model_management.soft_empty_cache()
                continue
            if model_id not in loaded_models:
                self.coordinator.send_object(
                    {"kind": "error", "traceback": f"Unknown model_id {model_id} in {command!r}"},
                    0,
                )
                continue
            patcher, execution_model, ctx_stack = loaded_models[model_id]
            with distributed_command_span(
                command,
                "External Tensor Parallel Rank Command",
                {
                    "distributed.rank": self.operations.rank,
                    "distributed.world_size": self.operations.world_size,
                    "comfy.command.kind": command["kind"],
                },
            ):
                if command["kind"] == "model_context":
                    try:
                        cm = getattr(execution_model, command["method"])()
                        scoped_model = cm.__enter__()
                        ctx_stack.append((cm, execution_model))
                        loaded_models[model_id][1] = scoped_model
                        self.coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        _send_worker_error(self.coordinator, self.operations.rank)
                    continue
                if command["kind"] == "end_model_context":
                    try:
                        if ctx_stack:
                            cm, prev_model = ctx_stack.pop()
                            loaded_models[model_id][1] = prev_model
                            cm.__exit__(None, None, None)
                        self.coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        _send_worker_error(self.coordinator, self.operations.rank)
                    continue
                if command["kind"] == "rank_model":
                    if command["rank"] != self.operations.rank:
                        continue
                    try:
                        model_command = command["command"]
                        if model_command["kind"] == "load":
                            if model_command.get("manage_device_memory", False):
                                model_management.load_models_gpu(
                                    [patcher],
                                    memory_required=model_command.get("memory_required", 0),
                                    minimum_memory_required=model_command.get("minimum_memory_required"),
                                    force_full_load=model_command.get("force_full_load", False),
                                    force_patch_weights=model_command["force_patch_weights"],
                                )
                            else:
                                patcher.partially_load(
                                    self.operations.device,
                                    model_command["extra_memory"],
                                    force_patch_weights=model_command["force_patch_weights"],
                                )
                        elif model_command["kind"] == "unload":
                            patcher.partially_unload(patcher.offload_device, model_command["memory_to_free"])
                        elif model_command["kind"] == "reset_dynamic_buffers":
                            patcher.reset_dynamic_buffers()
                        elif model_command["kind"] == "configure_torch_compile":
                            from comfy_api.torch_helpers.torch_compile import compile_model_for_execution
                            new_exec_model, strategy = compile_model_for_execution(
                                patcher, execution_model, model_command["compile_kwargs"]
                            )
                            loaded_models[model_id][1] = new_exec_model
                            self.coordinator.send_object({"kind": "rank_model", "strategy": strategy}, 0)
                            continue
                        self.coordinator.send_object({"kind": "rank_model", "loaded_size": patcher.loaded_size()}, 0)
                    except Exception:
                        _send_worker_error(self.coordinator, self.operations.rank)
                    continue
                if command["kind"] == "finish_execution":
                    try:
                        from .. import model_prefetch
                        model_prefetch.finish_model_execution()
                        self.coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        _send_worker_error(self.coordinator, self.operations.rank)
                    continue
                try:
                    tensors = _broadcast_tensors(self.operations, descriptors=command["descriptors"])
                    args, kwargs = unpack_pipeline_value(command["structure"], tensors)
                    getattr(execution_model, command["method"])(*args, **kwargs)
                    self.coordinator.send_object({"kind": "done"}, 0)
                except Exception as error:
                    if forward_failure is None:
                        forward_failure = f"{type(error).__name__}: {error}"
                    _abort_device_group(self.operations.process_group)
                    _send_worker_error(self.coordinator, self.operations.rank)
                finally:
                    tensors = args = kwargs = None
        loaded_models.clear()
        gc.collect()
        self.close()
        if forward_failure is not None:
            raise RuntimeError(f"Tensor-parallel rank {self.operations.rank} failed: {forward_failure}")

    def close(self) -> None:
        global _runtime
        if self._closed:
            return
        self._closed = True
        for executor in list(self._executors.values()):
            executor.close()
        self._executors.clear()
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
