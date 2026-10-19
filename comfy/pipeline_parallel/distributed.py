from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from datetime import timedelta
import logging
import multiprocessing.connection
import os
import secrets
import socket
import traceback
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration, resolve_distributed_configuration
from ..distributed.device import AcceleratorDeviceIdentity, accelerator_device_provider
from ..distributed.executors import ContextVarProcessPoolExecutor
from ..distributed.process_group import create_device_process_group
from ..distributed.tracing import distributed_command_span, inject_trace_context
from ..model_management_types import ModelManageableStub

from .runtime import AbstractBasePipelineExecutor, AbstractBasePipelineOperations, LatestShapeBuffers
from .types import (
    PipelineIntermediateTensors,
    PipelinePartitionPlan,
    PipelineStageMemoryGeometry,
    PipelineWorkerLoadSpec,
    TensorDescriptor,
    deserialize_pipeline_metadata,
    pack_pipeline_value,
    serialize_pipeline_metadata,
    unpack_pipeline_value,
)

logger = logging.getLogger(__name__)


class _PipelineAbort(Exception):
    pass


class AbstractBaseProcessGroupCoordinator(ABC):
    """Rank-local control and tensor transport used by a pipeline executor."""

    @abstractmethod
    def broadcast_command(self, command) -> object:
        raise NotImplementedError

    @abstractmethod
    def send_object(self, value, destination_rank: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def receive_object(self, source_rank: int):
        raise NotImplementedError

    @abstractmethod
    def send_tensors(self, tensors: Mapping[str, torch.Tensor], destination_rank: int) -> Sequence[object]:
        raise NotImplementedError

    @abstractmethod
    def receive_tensors(self, descriptors: Mapping[str, TensorDescriptor], source_rank: int) -> tuple[dict[str, torch.Tensor], Sequence[object]]:
        raise NotImplementedError

    @abstractmethod
    def wait(self, work: Sequence[object]) -> None:
        raise NotImplementedError

    def is_open(self) -> bool:
        """Whether the control plane can still carry a command."""
        return dist.is_initialized()


class TorchDistributedProcessGroupCoordinator(AbstractBaseProcessGroupCoordinator):
    """Gloo control plane plus an injected torch.distributed device group.

    The control plane is `control_group` when given, else torch.distributed's default group.
    """

    def __init__(self, rank: int, device: torch.device, device_group, control_group=None):
        self.rank = rank
        self.device = device
        self.device_group = device_group
        self.control_group = control_group
        self._buffers = LatestShapeBuffers(per_stream=False)
        self._communication_stream = torch.cuda.Stream(device=device)
        self._closed = False

    def is_open(self) -> bool:
        if self._closed:
            return False
        return self.control_group is not None or dist.is_initialized()

    def close(self) -> None:
        """Mark the control plane closed before its owner destroys the groups."""
        self._closed = True

    def _control(self) -> dict:
        return {} if self.control_group is None else {"group": self.control_group}

    def broadcast_command(self, command=None):
        if command is not None:
            command = inject_trace_context(command)
        values = [command]
        dist.broadcast_object_list(values, src=0, **self._control())
        return values[0]

    def send_object(self, value, destination_rank: int) -> None:
        dist.send_object_list([value], dst=destination_rank, **self._control())

    def receive_object(self, source_rank: int):
        values = [None]
        dist.recv_object_list(values, src=source_rank, **self._control())
        return values[0]

    def send_tensors(self, tensors: Mapping[str, torch.Tensor], destination_rank: int) -> Sequence[object]:
        main_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._communication_stream):
            self._communication_stream.wait_stream(main_stream)
            prepared = tuple(tensor.contiguous() for tensor in tensors.values())
            operations = [
                dist.P2POp(dist.isend, tensor, destination_rank, self.device_group)
                for tensor in prepared
            ]
            work = tuple(dist.batch_isend_irecv(operations) if operations else ())
            event = self._communication_stream.record_event()
            for tensor in prepared:
                tensor.record_stream(self._communication_stream)
        return (_DistributedCompletion(event, work, prepared),)

    def receive_tensors(self, descriptors: Mapping[str, TensorDescriptor], source_rank: int) -> tuple[dict[str, torch.Tensor], Sequence[object]]:
        buffers = self._buffers.get(descriptors, lambda descriptors: {
            name: torch.empty(descriptor.shape, dtype=descriptor.dtype, device=self.device)
            for name, descriptor in descriptors.items()
        })
        with torch.cuda.stream(self._communication_stream):
            operations = [
                dist.P2POp(dist.irecv, tensor, source_rank, self.device_group)
                for tensor in buffers.values()
            ]
            work = tuple(dist.batch_isend_irecv(operations) if operations else ())
            event = self._communication_stream.record_event()
        return buffers, (_DistributedCompletion(event, work),)

    def wait(self, work: Sequence[object]) -> None:
        for item in work:
            if isinstance(item, _DistributedCompletion):
                for operation in item.work:
                    operation.wait()
                torch.cuda.current_stream(self.device).wait_event(item.event)
            else:
                item.wait()


@dataclass
class _DistributedCompletion:
    event: torch.cuda.Event
    work: Sequence[object]
    references: Sequence[torch.Tensor] = ()


class _PendingTensorPacket:
    def __init__(self, coordinator, tensors, payload, work):
        self.coordinator = coordinator
        self.tensors = tensors
        self.payload = payload
        self.work = work

    def wait(self):
        self.coordinator.wait(self.work)
        result = self.tensors, self.payload
        self.work = ()
        return result


def _descriptors(tensors: Mapping[str, torch.Tensor]) -> dict[str, TensorDescriptor]:
    return {
        name: TensorDescriptor(tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size())
        for name, tensor in tensors.items()
    }


def _send_tensor_packet(coordinator, tensors, payload, destination_rank):
    coordinator.send_object(
        {"kind": "tensors", "descriptors": _descriptors(tensors), "payload": payload},
        destination_rank,
    )
    work = coordinator.send_tensors(tensors, destination_rank)
    return _PendingTensorPacket(
        coordinator,
        tensors,
        payload,
        tuple(work),
    )


def _receive_tensor_packet(coordinator, source_rank):
    header = coordinator.receive_object(source_rank)
    if header.get("kind") == "abort":
        raise _PipelineAbort
    if header.get("kind") == "error":
        raise RuntimeError(f"Pipeline rank {source_rank} failed:\n{header['traceback']}")
    if header.get("kind") != "tensors":
        raise RuntimeError(f"Invalid pipeline tensor packet from rank {source_rank}: {header!r}")
    tensors, work = coordinator.receive_tensors(header["descriptors"], source_rank)
    return _PendingTensorPacket(
        coordinator,
        tensors,
        header["payload"],
        tuple(work),
    )


class TorchDistributedPipelineExecutor(AbstractBasePipelineExecutor):
    def __init__(
        self,
        plan,
        first_stage,
        coordinator,
        workers,
        ready_connections,
        device_group,
        worker_geometries,
        *,
        close_command="close",
        destroy_process_group=True,
    ):
        self._plan = plan
        self.first_stage = first_stage
        self.coordinator = coordinator
        self.workers = tuple(workers)
        self.ready_connections = tuple(ready_connections)
        self.device_group = device_group
        self.worker_geometries = tuple(worker_geometries)
        self.close_command = close_command
        self.destroy_process_group = destroy_process_group
        self._closed = False

    def remote_stage_command(self, rank, command):
        self.coordinator.broadcast_command({"kind": "stage_model", "rank": rank, "command": command})
        response = self.coordinator.receive_object(rank)
        if response.get("kind") == "error":
            raise RuntimeError(f"Pipeline rank {rank} failed to update model residency:\n{response['traceback']}")
        return response

    @property
    def plan(self) -> PipelinePartitionPlan:
        return self._plan

    def execute(self, *args, **kwargs):
        self.coordinator.broadcast_command({"kind": "execute"})
        try:
            output = self.first_stage(*args, **kwargs)
        except Exception:
            self.coordinator.send_object({"kind": "abort"}, 1)
            raise
        if not isinstance(output, PipelineIntermediateTensors):
            raise TypeError("A non-final pipeline stage must return PipelineIntermediateTensors")
        schema = output.schema()
        schema.validate(output.tensors, output.metadata)
        pending_send = _send_tensor_packet(
            self.coordinator,
            output.tensors,
            serialize_pipeline_metadata(output.metadata),
            1,
        )
        tensors, structure = _receive_tensor_packet(
            self.coordinator,
            self.plan.size - 1,
        ).wait()
        pending_send.wait()
        return unpack_pipeline_value(structure, tensors)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.coordinator.broadcast_command({
            "kind": "add_patches",
            "patches": patches,
            "strength_patch": strength_patch,
            "strength_model": strength_model,
        })
        applied = set()
        for rank in range(1, self.plan.size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") == "error":
                raise RuntimeError(f"Pipeline rank {rank} failed to apply patches:\n{response['traceback']}")
            applied.update(response["applied"])
        return list(applied)

    def finish_execution(self) -> None:
        self.coordinator.broadcast_command({"kind": "finish_execution"})
        from .. import model_prefetch

        model_prefetch.finish_model_execution()
        for rank in range(1, self.plan.size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") == "error":
                raise RuntimeError(
                    f"Pipeline rank {rank} failed to finish execution:\n"
                    f"{response['traceback']}"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.coordinator.is_open():
                self.coordinator.broadcast_command({"kind": self.close_command})
                if self.close_command == "unload_pipeline":
                    for rank in range(1, self.plan.size):
                        self.coordinator.receive_object(rank)
        except Exception:
            logger.debug("Could not send pipeline worker shutdown", exc_info=True)
        for worker_pool, worker_future in self.workers:
            try:
                worker_future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                worker_pool.shutdown(wait=False)
            except Exception:
                worker_pool.shutdown(wait=True)
            else:
                worker_pool.shutdown(wait=True)
        if dist.is_initialized() and self.destroy_process_group:
            try:
                dist.destroy_process_group(self.device_group)
            finally:
                dist.destroy_process_group()

    def __del__(self):
        self.close()


class RemotePipelineStageModel(ModelManageableStub):
    """Model-manager view of a stage whose weights live in another rank."""

    def __init__(
        self,
        executor,
        rank,
        device,
        size,
        dtype,
        dynamic,
        ckpt_name=None,
        self_managed_device=False,
    ):
        self.executor = executor
        self.rank = rank
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.model = torch.nn.Module()
        self.size = int(size)
        self.dtype = dtype
        self.dynamic = dynamic
        self.ckpt_name = ckpt_name
        self.self_managed_device = self_managed_device
        self._loaded_size = 0
        self._current_device = self.offload_device

    def model_size(self):
        return self.size

    def model_dtype(self):
        return self.dtype

    def model_mmap_residency(self, free=False):
        del free
        return self.size, self.size

    def current_loaded_device(self):
        return self._current_device

    @property
    def current_device(self):
        return self._current_device

    def loaded_size(self):
        return self._loaded_size

    def reclaimable_non_vbar_memory(self):
        return self._loaded_size

    def loaded_ram_size(self):
        return self.size

    def is_dynamic(self):
        return self.dynamic

    def manages_own_device_memory(self):
        return self.self_managed_device

    def reset_dynamic_buffers(self):
        self.executor.remote_stage_command(self.rank, {"kind": "reset_dynamic_buffers"})

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        del patches, strength_patch, strength_model
        return []

    def model_state_dict(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_key_patches(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_additional_models(self):
        return []

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        del load_weights
        self.partially_load(device_to or self.load_device, lowvram_model_memory, force_patch_weights)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        del unpatch_weights
        self.partially_unload(device_to or self.offload_device, self._loaded_size)
        return self.model

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        response = self.executor.remote_stage_command(self.rank, {
            "kind": "load",
            "extra_memory": extra_memory,
            "force_patch_weights": force_patch_weights,
            "manage_device_memory": self.self_managed_device,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        self._current_device = device_to if self._loaded_size else self.offload_device
        return self._loaded_size - previous

    def partially_unload(self, device_to, memory_to_free=0):
        response = self.executor.remote_stage_command(self.rank, {
            "kind": "unload",
            "memory_to_free": memory_to_free,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        self._current_device = self.load_device if self._loaded_size else device_to
        return previous - self._loaded_size

    def __str__(self):
        return f"<RemotePipelineStage rank={self.rank} device={self.load_device}>"


def _load_worker_stage(load_spec: PipelineWorkerLoadSpec, rank: int):
    from .loader import load_pipeline_worker_stage
    from .memory import ComfyDynamicVRAMStageMemoryEstimator
    from .stages import get_pipeline_stage_spec

    patcher = load_pipeline_worker_stage(load_spec)
    stage_spec = get_pipeline_stage_spec(load_spec.plan.model_family)
    geometry = ComfyDynamicVRAMStageMemoryEstimator().estimate_stage(
        stage_spec,
        load_spec.plan.stages[rank],
        patcher,
    )
    return patcher, geometry


def _run_worker_commands(coordinator, rank, world_size, device, patcher):
    stage = patcher.model.diffusion_model.forward_pipeline_stage
    while True:
        command = coordinator.broadcast_command()
        with distributed_command_span(
            command,
            "Pipeline Parallel Rank Command",
            {
                "distributed.rank": rank,
                "distributed.world_size": world_size,
                "comfy.command.kind": command["kind"],
            },
        ):
            if command["kind"] in ("close", "unload_pipeline"):
                if command["kind"] == "unload_pipeline":
                    try:
                        patcher.detach()
                        coordinator.send_object({"kind": "unloaded"}, 0)
                    except Exception:
                        coordinator.send_object(
                            {"kind": "error", "traceback": traceback.format_exc()}, 0
                        )
                return command["kind"]
            if command["kind"] == "stage_model":
                if command["rank"] != rank:
                    continue
                try:
                    stage_command = command["command"]
                    if stage_command["kind"] == "load":
                        if stage_command.get("manage_device_memory", False):
                            from .. import model_management

                            model_management.load_models_gpu(
                                [patcher],
                                force_patch_weights=stage_command["force_patch_weights"],
                            )
                        else:
                            patcher.partially_load(
                                device,
                                stage_command["extra_memory"],
                                force_patch_weights=stage_command["force_patch_weights"],
                            )
                    elif stage_command["kind"] == "unload":
                        patcher.partially_unload(
                            patcher.offload_device,
                            stage_command["memory_to_free"],
                        )
                    elif stage_command["kind"] == "reset_dynamic_buffers":
                        patcher.reset_dynamic_buffers()
                    else:
                        raise RuntimeError(
                            f"Unknown pipeline stage model command: {stage_command!r}"
                        )
                    coordinator.send_object(
                        {"kind": "stage_model", "loaded_size": patcher.loaded_size()}, 0
                    )
                except Exception:
                    coordinator.send_object(
                        {"kind": "error", "traceback": traceback.format_exc()}, 0
                    )
                continue
            if command["kind"] == "add_patches":
                try:
                    applied = patcher.add_patches(
                        command["patches"],
                        command["strength_patch"],
                        command["strength_model"],
                    )
                    coordinator.send_object(
                        {"kind": "patches", "applied": applied}, 0
                    )
                except Exception:
                    coordinator.send_object(
                        {"kind": "error", "traceback": traceback.format_exc()}, 0
                    )
                continue
            if command["kind"] == "finish_execution":
                try:
                    from .. import model_prefetch

                    model_prefetch.finish_model_execution()
                    coordinator.send_object({"kind": "done"}, 0)
                except Exception:
                    coordinator.send_object(
                        {"kind": "error", "traceback": traceback.format_exc()}, 0
                    )
                continue
            if command["kind"] != "execute":
                raise RuntimeError(f"Unknown pipeline worker command: {command!r}")

            try:
                tensors, metadata = _receive_tensor_packet(coordinator, rank - 1).wait()
                intermediate = PipelineIntermediateTensors(
                    dict(tensors),
                    deserialize_pipeline_metadata(metadata),
                )
                output = stage(intermediate)
                if rank < world_size - 1:
                    if not isinstance(output, PipelineIntermediateTensors):
                        raise TypeError(
                            "A non-final pipeline stage must return PipelineIntermediateTensors"
                        )
                    pending_send = _send_tensor_packet(
                        coordinator,
                        output.tensors,
                        serialize_pipeline_metadata(output.metadata),
                        rank + 1,
                    )
                else:
                    tensors = {}
                    structure = pack_pipeline_value(output, tensors, "pipeline_output")
                    pending_send = _send_tensor_packet(
                        coordinator, tensors, structure, 0
                    )
                pending_send.wait()
            except _PipelineAbort:
                if rank < world_size - 1:
                    coordinator.send_object({"kind": "abort"}, rank + 1)
                continue
            except Exception:
                destination_rank = rank + 1 if rank < world_size - 1 else 0
                coordinator.send_object(
                    {"kind": "error", "traceback": traceback.format_exc()},
                    destination_rank,
                )


def _worker_main(
    rank: int,
    world_size: int,
    init_method: str,
    load_spec: PipelineWorkerLoadSpec,
    device_identity: AcceleratorDeviceIdentity,
    ready: multiprocessing.connection.Connection,
):
    device_provider = accelerator_device_provider(device_identity.device_type)
    device = device_provider.resolve(device_identity)
    stages = list(load_spec.plan.stages)
    stages[rank] = replace(stages[rank], device=device)
    load_spec = replace(
        load_spec,
        plan=replace(load_spec.plan, stages=tuple(stages)),
    )
    try:
        distributed = resolve_distributed_configuration()
        if distributed.rank != rank or distributed.world_size != world_size:
            raise RuntimeError(
                "Pipeline worker identity does not match its canonical environment: "
                f"rank={rank}/{world_size}, environment={distributed.rank}/{distributed.world_size}"
            )
        if distributed.pipeline_rank != load_spec.stage_index:
            raise RuntimeError(
                f"Pipeline worker rank {distributed.pipeline_rank} received stage {load_spec.stage_index}"
            )
        device_provider.select(device)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
        )
        device_group = create_device_process_group(range(world_size), device)
        coordinator = TorchDistributedProcessGroupCoordinator(rank, device, device_group)

        from .. import model_management

        model_management.set_torch_device(device)
        from .. import aimdo_integration  # noqa: F401

        model_management.share_pinned_memory_budget(world_size)
        patcher, geometry = _load_worker_stage(load_spec, rank)
        ready.send({"kind": "ready", "geometry": geometry})
        _run_worker_commands(coordinator, rank, world_size, device, patcher)
    except Exception:
        try:
            ready.send({"kind": "error", "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise
    finally:
        ready.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _pipeline_worker_entry(host: str, port: str, authkey: str) -> None:
    distributed = resolve_distributed_configuration(environment=os.environ)
    connection = multiprocessing.connection.Client(
        (host, int(port)),
        authkey=bytes.fromhex(authkey),
    )
    rank = distributed.rank
    connection.send(rank)
    world_size, init_method, load_spec, device_identity = connection.recv()
    _worker_main(
        rank,
        world_size,
        init_method,
        load_spec,
        device_identity,
        connection,
    )


def _free_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class TorchDistributedPipelineOperations(AbstractBasePipelineOperations):
    @property
    def uses_worker_processes(self) -> bool:
        return True

    def supports(self, devices) -> bool:
        return (
            len(devices) >= 2
            and all(device.type == "cuda" for device in devices)
            and dist.is_available()
            and dist.is_nccl_available()
            and not dist.is_initialized()
        )

    def create_transport(self):
        raise RuntimeError("Distributed pipeline transport is owned by its rank coordinator")

    def create_stage_runner(self):
        raise RuntimeError("Distributed pipeline stages run in rank workers")

    def create_executor(self, plan, stages, worker_load_specs=()):
        if len(worker_load_specs) != plan.size - 1:
            raise ValueError("Distributed pipeline execution requires one load specification per worker rank")
        init_method = _free_tcp_init_method()
        listener_authkey = secrets.token_bytes(32)
        listener = multiprocessing.connection.Listener(("127.0.0.1", 0), authkey=listener_authkey)
        listener_host, listener_port = listener.address
        master_host, master_port = init_method.removeprefix("tcp://").rsplit(":", 1)
        from .. import model_management

        model_management.share_pinned_memory_budget(plan.size)
        workers = []
        ready_connections = [None] * (plan.size - 1)
        device_provider = accelerator_device_provider(plan.stages[0].device.type)
        device_identities = tuple(
            device_provider.identify(stage.device) for stage in plan.stages
        )
        for spec in worker_load_specs:
            stage = plan.stages[spec.stage_index]
            worker_environment = DistributedConfiguration(
                rank=spec.stage_index,
                world_size=plan.size,
                local_rank=spec.stage_index,
                local_world_size=plan.size,
                master_addr=master_host,
                master_port=int(master_port),
                pipeline_parallel_size=plan.size,
                executor_backend="mp",
            ).canonical_environment()
            worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
            worker_future = worker_pool.submit_with_environment(
                worker_environment,
                _pipeline_worker_entry,
                str(listener_host),
                str(listener_port),
                listener_authkey.hex(),
                detach_request_state=True,
            )
            workers.append((worker_pool, worker_future))

        try:
            for _ in worker_load_specs:
                connection = listener.accept()
                rank = connection.recv()
                ready_connections[rank - 1] = connection
                connection.send((
                    plan.size,
                    init_method,
                    worker_load_specs[rank - 1],
                    device_identities[rank],
                ))
            listener.close()
            device_provider.select(plan.stages[0].device)
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=0,
                world_size=plan.size,
                timeout=timedelta(minutes=5),
            )
            device_group = create_device_process_group(
                range(plan.size),
                plan.stages[0].device,
            )
            coordinator = TorchDistributedProcessGroupCoordinator(0, plan.stages[0].device, device_group)
            worker_geometries: list[PipelineStageMemoryGeometry] = []
            for rank, connection in enumerate(ready_connections, start=1):
                if not connection.poll(300):
                    raise TimeoutError(f"Pipeline rank {rank} did not finish loading within five minutes")
                response = connection.recv()
                if response.get("kind") != "ready":
                    raise RuntimeError(f"Pipeline rank {rank} failed to start:\n{response['traceback']}")
                worker_geometries.append(response["geometry"])
            return TorchDistributedPipelineExecutor(
                plan,
                stages[0],
                coordinator,
                workers,
                ready_connections,
                device_group,
                worker_geometries,
            )
        except Exception:
            for worker_pool, _worker_future in workers:
                worker_pool.shutdown(wait=False)
            if dist.is_initialized():
                dist.destroy_process_group()
            raise
        finally:
            listener.close()
