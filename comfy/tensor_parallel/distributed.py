from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import gc
import logging
import multiprocessing.connection
import os
import secrets
import socket
import traceback

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration, resolve_distributed_configuration
from ..distributed.device import accelerator_device_provider
from ..distributed.executors import ContextVarExecutor, ContextVarProcessPoolExecutor
from ..distributed.process_group import (
    create_independent_process_groups,
    destroy_independent_process_groups,
)
from ..distributed.tracing import distributed_command_span, inject_trace_context
from ..model_management_types import ModelManageableStub
from ..interruption import defer_interruption
from ..pipeline_parallel.types import (
    TensorDescriptor,
    pack_pipeline_value,
    prepare_model_parallel_value,
    unpack_pipeline_value,
)
from .runtime import TorchDistributedTensorParallelOperations


def _free_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        return f"tcp://127.0.0.1:{port}"


def _abort_device_group(process_group):
    try:
        abort = getattr(dist.distributed_c10d, "_abort_process_group", None)
        if abort is not None:
            abort(process_group)
        else:
            process_group.abort()
    except Exception:
        logging.exception("Could not abort failed tensor-parallel device group")


def _without_execution_caches(value):
    return prepare_model_parallel_value(value)


def _broadcast_tensors(operations, tensors=None, descriptors=None):
    if descriptors is None:
        prepared = {
            name: tensor.to(
                device=tensor.device if tensor.device.type == "cpu" else operations.device,
                non_blocking=True,
            ).contiguous()
            for name, tensor in tensors.items()
        }
    else:
        prepared = {
            name: torch.empty(
                descriptor.shape,
                dtype=descriptor.dtype,
                device="cpu" if descriptor.device_type == "cpu" else operations.device,
            )
            for name, descriptor in descriptors.items()
        }
    completions = [
        (tensor, dist.broadcast(
            tensor,
            src=0,
            group=(
                operations.control_process_group
                if tensor.device.type == "cpu"
                else operations.process_group
            ),
            async_op=True,
        ))
        for tensor in prepared.values()
    ]
    for tensor, completion in completions:
        if tensor.device.type == "cpu":
            completion.wait()
        else:
            completion.block_current_stream()
    return prepared


class TorchDistributedModelParallelExecutor:
    def __init__(self, root_model, operations, coordinator, workers, connections, devices, rank_sizes, root_patcher=None, process_groups=None):
        self.root_model = root_model
        self.root_patcher = root_patcher
        self.operations = operations
        self.coordinator = coordinator
        self.workers = tuple(workers)
        self.connections = tuple(connections)
        self.devices = tuple(devices)
        self.rank_sizes = tuple(rank_sizes)
        self.process_groups = process_groups
        self._closed = False
        self._device_group_active = True
        self._responses = ContextVarExecutor(max_workers=max(1, len(self.devices) - 1))

    def configure_torch_compile(self, compile_kwargs, root_patcher=None):
        root_patcher = root_patcher or self.root_patcher
        if root_patcher is None:
            raise RuntimeError("Model-parallel compiler requires its root model patcher")
        for rank in range(1, len(self.devices)):
            response = self.remote_rank_command(
                rank,
                {"kind": "configure_torch_compile", "compile_kwargs": compile_kwargs},
            )
            if response["kind"] == "error":
                raise RuntimeError(
                    f"Model-parallel rank {rank} failed to configure torch.compile:\n"
                    f"{response['traceback']}"
                )
        from comfy_api.torch_helpers.torch_compile import compile_model_for_execution

        self.root_model, root_strategy = compile_model_for_execution(
            root_patcher,
            self.root_model,
            compile_kwargs,
        )
        return {"diffusion_model": root_strategy}

    def execute(self, *args, **kwargs):
        return self.execute_method("forward", *args, **kwargs)

    @contextmanager
    def model_context(self, method):
        """Keep call-local model state on each rank until this scope exits."""
        original = self.root_model
        with getattr(original, method)() as model:
            self.coordinator.broadcast_command({"kind": "model_context", "method": method})
            try:
                responses = [self.coordinator.receive_object(rank) for rank in range(1, len(self.devices))]
                errors = [response["traceback"] for response in responses if response["kind"] == "error"]
                if errors:
                    raise RuntimeError("Model-parallel context failed:\n" + "\n".join(errors))
                self.root_model = model
                yield
            finally:
                self.root_model = original
                self.coordinator.broadcast_command({"kind": "end_model_context"})
                responses = [self.coordinator.receive_object(rank) for rank in range(1, len(self.devices))]
                errors = [response["traceback"] for response in responses if response["kind"] == "error"]
                if errors:
                    raise RuntimeError("Model-parallel context cleanup failed:\n" + "\n".join(errors))

    @defer_interruption()
    def execute_method(self, method, *args, **kwargs):
        tensors = {}
        structure = pack_pipeline_value(
            _without_execution_caches((args, kwargs)), tensors, "tensor_parallel_input"
        )
        descriptors = {
            name: TensorDescriptor(
                tuple(tensor.shape), tensor.dtype, tensor.nbytes, tensor.device.type
            )
            for name, tensor in tensors.items()
        }
        self.coordinator.broadcast_command({
            "kind": "execute",
            "method": method,
            "structure": structure,
            "descriptors": descriptors,
        })
        tensors = _broadcast_tensors(self.operations, tensors=tensors)
        args, kwargs = unpack_pipeline_value(structure, tensors)
        responses = [self._responses.submit(self._receive_execution_result, rank)
                     for rank in range(1, len(self.devices))]
        try:
            output = getattr(self.root_model, method)(*args, **kwargs)
        except BaseException:
            # A peer can already be blocked in a model collective when the
            # root rank fails (for example, during lazy torch compilation).
            # Abort the NCCL group so that peer leaves that collective and can
            # return to the control loop instead of hanging the workflow.
            self._abort_device_group()
            concurrent.futures.wait(responses)
            raise
        for response in responses:
            response.result()
        return output

    def _receive_execution_result(self, rank):
        response = self.coordinator.receive_object(rank)
        if response["kind"] == "error":
            # Watch the control channel while the root computes: its next CUDA
            # synchronization may otherwise wait forever for this failed peer.
            self._abort_device_group()
            raise RuntimeError(f"Tensor-parallel rank {rank} failed:\n{response['traceback']}")

    def _abort_device_group(self):
        if not self._device_group_active:
            return
        self._device_group_active = False
        _abort_device_group(self.operations.process_group)

    def remote_rank_command(self, rank, command):
        self.coordinator.broadcast_command({"kind": "rank_model", "rank": rank, "command": command})
        return self.coordinator.receive_object(rank)

    def finish_execution(self):
        self.coordinator.broadcast_command({"kind": "finish_execution"})
        from .. import model_prefetch

        model_prefetch.finish_model_execution()
        for rank in range(1, len(self.devices)):
            response = self.coordinator.receive_object(rank)
            if response["kind"] == "error":
                raise RuntimeError(
                    f"Tensor-parallel rank {rank} failed to finish execution:\n"
                    f"{response['traceback']}"
                )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._responses.shutdown(wait=False, cancel_futures=True)
        try:
            self.coordinator.broadcast_command({"kind": "close"})
        except Exception:
            pass

        # NCCL teardown must happen concurrently on every rank.  Waiting for
        # worker futures first deadlocks because each worker is already inside
        # ncclCommDestroy waiting for rank zero.
        if self.process_groups is not None:
            try:
                destroy_independent_process_groups(self.process_groups)
            except Exception:
                pass
            self._device_group_active = False
        for worker_pool, worker_future in self.workers:
            try:
                worker_future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                worker_pool.shutdown(wait=False)
            except Exception:
                worker_pool.shutdown(wait=True)
            else:
                worker_pool.shutdown(wait=True)
        if self.process_groups is None and dist.is_initialized():
            if self._device_group_active:
                try:
                    dist.destroy_process_group(self.operations.process_group)
                finally:
                    self._device_group_active = False
                    dist.destroy_process_group()
            else:
                dist.destroy_process_group()

    def __del__(self):
        self.close()


class RemoteModelParallelRankModel(ModelManageableStub):
    def __init__(self, executor, rank, device, size, dtype, dynamic, ckpt_name=None):
        self.executor = executor
        self.rank = rank
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.model = torch.nn.Module()
        self.size = size
        self.dtype = dtype
        self.dynamic = dynamic
        self.ckpt_name = ckpt_name
        self._loaded_size = 0
        self._memory_required = 0
        self._minimum_memory_required = None
        self._force_full_load = False

    def model_size(self):
        return self.size

    def model_dtype(self):
        return self.dtype

    def model_mmap_residency(self, free=False):
        del free
        return self.size, self.size

    def current_loaded_device(self):
        return self.load_device if self._loaded_size else self.offload_device

    @property
    def current_device(self):
        return self.current_loaded_device()

    def loaded_size(self):
        return self._loaded_size

    def reclaimable_non_vbar_memory(self):
        return self._loaded_size

    def loaded_ram_size(self):
        return self.size

    def is_dynamic(self):
        return self.dynamic

    def manages_own_device_memory(self):
        return True

    def set_inference_memory_requirements(
        self,
        memory_required,
        minimum_memory_required,
        force_full_load=False,
    ):
        self._memory_required = memory_required
        self._minimum_memory_required = minimum_memory_required
        self._force_full_load = force_full_load

    def get_additional_models(self):
        return []

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        del patches, strength_patch, strength_model
        raise NotImplementedError("Tensor-parallel LoRA loading is not implemented")

    def model_state_dict(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_key_patches(self, filter_prefix=None):
        del filter_prefix
        return {}

    def reset_dynamic_buffers(self):
        self.executor.remote_rank_command(self.rank, {"kind": "reset_dynamic_buffers"})

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        del load_weights
        self.partially_load(device_to or self.load_device, lowvram_model_memory, force_patch_weights)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        del unpatch_weights
        self.partially_unload(device_to or self.offload_device, self._loaded_size)
        return self.model

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        response = self.executor.remote_rank_command(self.rank, {
            "kind": "load",
            "extra_memory": extra_memory,
            "force_patch_weights": force_patch_weights,
            "manage_device_memory": True,
            "memory_required": self._memory_required,
            "minimum_memory_required": self._minimum_memory_required,
            "force_full_load": self._force_full_load,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        return self._loaded_size - previous

    def partially_unload(self, device_to, memory_to_free=0):
        response = self.executor.remote_rank_command(self.rank, {
            "kind": "unload",
            "memory_to_free": memory_to_free,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        return previous - self._loaded_size


class _ObjectCoordinator:
    def __init__(self, control_process_group=None):
        self.control_process_group = control_process_group

    def _group_kwargs(self):
        if self.control_process_group is None:
            return {}
        return {"group": self.control_process_group}

    def broadcast_command(self, command=None):
        if command is not None:
            command = inject_trace_context(command)
        values = [command]
        dist.broadcast_object_list(values, src=0, **self._group_kwargs())
        return values[0]

    def send_object(self, value, destination_rank):
        dist.send_object_list(
            [value],
            dst=destination_rank,
            **self._group_kwargs(),
        )

    def receive_object(self, source_rank):
        values = [None]
        dist.recv_object_list(
            values,
            src=source_rank,
            **self._group_kwargs(),
        )
        return values[0]


def _send_worker_error(coordinator, rank):
    error = traceback.format_exc()
    logging.error("Model-parallel rank %s failed:\n%s", rank, error)
    coordinator.send_object({"kind": "error", "traceback": error}, 0)


def _run_worker(operations, coordinator, patcher, span_name, execution_model=None):
    if execution_model is None:
        execution_model = patcher.model.diffusion_model
    while True:
        command = coordinator.broadcast_command()
        with distributed_command_span(
            command,
            span_name,
            {
                "distributed.rank": operations.rank,
                "distributed.world_size": operations.world_size,
                "comfy.command.kind": command["kind"],
            },
        ):
                if command["kind"] == "close":
                    return True
                if command["kind"] == "end_model_context":
                    return False
                if command["kind"] == "model_context":
                    started = False
                    try:
                        with getattr(execution_model, command["method"])() as model:
                            started = True
                            coordinator.send_object({"kind": "done"}, 0)
                            closed = _run_worker(operations, coordinator, patcher, span_name, model)
                        if closed:
                            return True
                        coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        _send_worker_error(coordinator, operations.rank)
                        if not started:
                            # Rank zero closes every rank's scope if any entry
                            # failed. Acknowledge that close without exiting the
                            # worker or leaving a response for its next request.
                            ending = coordinator.broadcast_command()
                            if ending["kind"] == "close":
                                return True
                            if ending["kind"] != "end_model_context":
                                raise RuntimeError("Expected model-context cleanup after entry failure")
                            coordinator.send_object({"kind": "done"}, 0)
                    continue
                if command["kind"] == "rank_model":
                    if command["rank"] != operations.rank:
                        continue
                    try:
                        model_command = command["command"]
                        if model_command["kind"] == "load":
                            if model_command.get("manage_device_memory", False):
                                from .. import model_management

                                model_management.load_models_gpu(
                                    [patcher],
                                    memory_required=model_command.get(
                                        "memory_required",
                                        0,
                                    ),
                                    minimum_memory_required=model_command.get(
                                        "minimum_memory_required",
                                    ),
                                    force_full_load=model_command.get(
                                        "force_full_load",
                                        False,
                                    ),
                                    force_patch_weights=model_command[
                                        "force_patch_weights"
                                    ],
                                )
                            else:
                                patcher.partially_load(
                                    operations.device,
                                    model_command["extra_memory"],
                                    force_patch_weights=model_command[
                                        "force_patch_weights"
                                    ],
                                )
                        elif model_command["kind"] == "unload":
                            patcher.partially_unload(patcher.offload_device, model_command["memory_to_free"])
                        elif model_command["kind"] == "reset_dynamic_buffers":
                            patcher.reset_dynamic_buffers()
                        elif model_command["kind"] == "configure_torch_compile":
                            from comfy_api.torch_helpers.torch_compile import (
                                compile_model_for_execution,
                            )

                            execution_model, strategy = compile_model_for_execution(
                                patcher,
                                execution_model,
                                model_command["compile_kwargs"],
                            )
                            coordinator.send_object(
                                {"kind": "rank_model", "strategy": strategy},
                                0,
                            )
                            continue
                        coordinator.send_object({"kind": "rank_model", "loaded_size": patcher.loaded_size()}, 0)
                    except Exception:
                        _send_worker_error(coordinator, operations.rank)
                    continue
                if command["kind"] == "finish_execution":
                    try:
                        from .. import model_prefetch

                        model_prefetch.finish_model_execution()
                        coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        _send_worker_error(coordinator, operations.rank)
                    continue
                try:
                    tensors = _broadcast_tensors(operations, descriptors=command["descriptors"])
                    args, kwargs = unpack_pipeline_value(command["structure"], tensors)
                    getattr(execution_model, command["method"])(*args, **kwargs)
                    coordinator.send_object({"kind": "done"}, 0)
                except Exception:
                    # The root may already be inside the next device collective.
                    # Sending an error on the control group alone cannot wake it.
                    _abort_device_group(operations.process_group)
                    _send_worker_error(coordinator, operations.rank)
                finally:
                    # Do not keep the last invocation's inputs or custom-node
                    # cache tensors alive while waiting for another command.
                    tensors = args = kwargs = None


def worker_main(host, port, authkey, parallel_kind="tensor"):
    connection = multiprocessing.connection.Client((host, int(port)), authkey=bytes.fromhex(authkey))
    distributed = resolve_distributed_configuration(environment=os.environ)
    rank = distributed.rank
    connection.send(rank)
    world_size, init_method, group_name, load_spec, device_identity = connection.recv()
    device_provider = accelerator_device_provider(device_identity.device_type)
    device = device_provider.resolve(device_identity)
    device_provider.select(device)
    process_groups = create_independent_process_groups(
        init_method,
        rank,
        world_size,
        device,
        group_name,
    )
    control_group = process_groups.control_process_group
    device_group = process_groups.device_process_group
    if parallel_kind == "tensor":
        operations = TorchDistributedTensorParallelOperations(
            rank, world_size, device, device_group, control_group
        )
        span_name = "Tensor Parallel Rank Command"
    elif parallel_kind in ("xdit_ulysses", "xdit_ring"):
        from ..xdit.runtime import TorchDistributedXDiTSequenceParallelOperations

        operations = TorchDistributedXDiTSequenceParallelOperations(
            rank, world_size, device, device_group, control_group
        )
        span_name = f"xDiT {parallel_kind.removeprefix('xdit_').title()} Rank Command"
    else:
        raise ValueError(f"Unknown model-parallel worker kind {parallel_kind!r}")
    from .. import model_management
    model_management.set_torch_device(device)
    from .. import aimdo_integration  # noqa: F401
    model_management.share_pinned_memory_budget(world_size)
    try:
        if parallel_kind == "tensor":
            from .loader import load_tensor_parallel_rank

            patcher = load_tensor_parallel_rank(load_spec, rank, device, operations)
        else:
            from ..xdit.loader import load_xdit_sequence_parallel_rank

            patcher = load_xdit_sequence_parallel_rank(
                load_spec,
                rank,
                device,
                operations,
            )
        connection.send({"kind": "ready", "size": patcher.model_size()})
        _run_worker(
            operations,
            _ObjectCoordinator(control_group),
            patcher,
            span_name,
        )
        del patcher
        gc.collect()
    except Exception:
        connection.send({"kind": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        connection.close()
        destroy_independent_process_groups(process_groups)


def launch_model_parallel(load_spec, devices, load_root, parallel_kind):
    # Cached models and their executors form cycles. Reap discarded workers
    # before allocating another checkpoint's rank-local weights.
    gc.collect()
    size = len(devices)
    device_provider = accelerator_device_provider(devices[0].type)
    device_identities = tuple(device_provider.identify(device) for device in devices)
    init_method = _free_tcp_init_method()
    group_name = f"comfy-model-parallel-{secrets.token_hex(8)}"
    authkey = secrets.token_bytes(32)
    listener = multiprocessing.connection.Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    master_host, master_port = init_method.removeprefix("tcp://").rsplit(":", 1)
    from .. import model_management

    model_management.share_pinned_memory_budget(size)
    workers = []
    connections = [None] * (size - 1)
    for rank in range(1, size):
        environment = DistributedConfiguration(
            rank=rank,
            world_size=size,
            local_rank=rank,
            local_world_size=size,
            master_addr=master_host,
            master_port=int(master_port),
            pipeline_parallel_size=1,
            tensor_parallel_size=size if parallel_kind == "tensor" else 1,
            ulysses_degree=size if parallel_kind == "xdit_ulysses" else 1,
            ring_degree=size if parallel_kind == "xdit_ring" else 1,
            executor_backend="mp",
        ).canonical_environment()
        worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
        worker_future = worker_pool.submit_with_environment(
            environment,
            worker_main,
            str(host),
            str(port),
            authkey.hex(),
            parallel_kind,
            detach_request_state=True,
        )
        workers.append((worker_pool, worker_future))
    try:
        for _ in range(size - 1):
            connection = listener.accept()
            rank = connection.recv()
            connections[rank - 1] = connection
            connection.send(
                (
                    size,
                    init_method,
                    group_name,
                    load_spec,
                    device_identities[rank],
                )
            )
        device_provider.select(devices[0])
        process_groups = create_independent_process_groups(
            init_method,
            0,
            size,
            devices[0],
            group_name,
        )
        control_group = process_groups.control_process_group
        device_group = process_groups.device_process_group
        if parallel_kind == "tensor":
            operations = TorchDistributedTensorParallelOperations(
                0, size, devices[0], device_group, control_group
            )
        else:
            from ..xdit.runtime import TorchDistributedXDiTSequenceParallelOperations

            operations = TorchDistributedXDiTSequenceParallelOperations(
                0, size, devices[0], device_group, control_group
            )
        root = load_root(operations)
        rank_sizes = [root.model_size()]
        for rank, connection in enumerate(connections, start=1):
            if not connection.poll(300):
                raise TimeoutError(
                    f"Model-parallel rank {rank} did not load within five minutes"
                )
            response = connection.recv()
            if response["kind"] != "ready":
                raise RuntimeError(
                    f"Model-parallel rank {rank} failed to load:\n"
                    f"{response['traceback']}"
                )
            rank_sizes.append(response["size"])
        return root, TorchDistributedModelParallelExecutor(
            root.model.diffusion_model,
            operations,
            _ObjectCoordinator(control_group),
            workers,
            connections,
            devices,
            rank_sizes,
            root_patcher=root,
            process_groups=process_groups,
        )
    except Exception:
        for worker_pool, _worker_future in workers:
            worker_pool.shutdown(wait=False)
        if "process_groups" in locals():
            destroy_independent_process_groups(process_groups)
        raise
    finally:
        listener.close()


def launch_tensor_parallel(load_spec, devices, load_root):
    return launch_model_parallel(load_spec, devices, load_root, "tensor")


# Compatibility names for attachments and third-party imports created before
# the executor became the shared replicated-rank runtime.
TorchDistributedTensorParallelExecutor = TorchDistributedModelParallelExecutor
RemoteTensorParallelRankModel = RemoteModelParallelRankModel
