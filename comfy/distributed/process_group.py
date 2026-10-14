from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
import threading

import torch
import torch.distributed as dist


NCCL_PROTOCOLS = ("auto", "simple", "ll", "ll128")
_NCCL_ENVIRONMENT_LOCK = threading.Lock()


@contextmanager
def _nccl_protocol(nccl_proto: str):
    normalized = nccl_proto.lower()
    if normalized not in NCCL_PROTOCOLS:
        raise ValueError(
            f"Unknown NCCL protocol {nccl_proto!r}; expected one of "
            f"{', '.join(NCCL_PROTOCOLS)}"
        )

    with _NCCL_ENVIRONMENT_LOCK:
        previous = os.environ.get("NCCL_PROTO")
        try:
            if normalized == "auto":
                os.environ.pop("NCCL_PROTO", None)
            else:
                os.environ["NCCL_PROTO"] = normalized.upper()
            yield
        finally:
            if previous is None:
                os.environ.pop("NCCL_PROTO", None)
            else:
                os.environ["NCCL_PROTO"] = previous


class AbstractBaseDeviceProcessGroupFactory(ABC):
    @abstractmethod
    def create(self, ranks, device: torch.device, nccl_proto: str):
        raise NotImplementedError


class TorchDistributedCudaProcessGroupFactory(
    AbstractBaseDeviceProcessGroupFactory
):
    """Create an eagerly initialized NCCL group with injected tuning."""

    def create(self, ranks, device: torch.device, nccl_proto: str):
        if device.type != "cuda":
            raise ValueError(f"NCCL process groups require a CUDA device, got {device}")

        # NCCL protocol selection is not exposed by ncclConfig_t or PyTorch's
        # NCCLConfig. device_id makes new_group initialize the communicator
        # eagerly, so the process environment can remain an implementation
        # detail rather than application configuration.
        with _nccl_protocol(nccl_proto):
            return dist.new_group(
                ranks=list(ranks),
                backend="nccl",
                device_id=device,
            )


@dataclass(frozen=True)
class IndependentProcessGroups:
    """A model executor's transport state, independent of ``group.WORLD``."""

    store: object
    control_process_group: object
    device_process_group: object


DEFAULT_PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)


class AbstractBaseIndependentProcessGroupFactory(ABC):
    @abstractmethod
    def create(
        self,
        init_method: str,
        rank: int,
        world_size: int,
        device: torch.device,
        group_name: str,
        nccl_proto: str,
        timeout: timedelta = DEFAULT_PROCESS_GROUP_TIMEOUT,
    ) -> IndependentProcessGroups:
        raise NotImplementedError


def _create_independent_process_group(
    store,
    rank: int,
    world_size: int,
    backend: str,
    group_name: str,
    timeout: timedelta,
    device_id: torch.device | None = None,
):
    """Create and register a non-default group from an executor-owned store.

    PyTorch's public ``init_process_group`` exclusively owns ``group.WORLD``.
    Model executors are independently cached by Comfy, so using that singleton
    lets a second model loader destroy the first model's transports.  The
    helper below is the same constructor used by ``init_process_group`` and
    ``new_group``, but an empty global-rank list avoids consulting or replacing
    the default world.
    """

    c10d = dist.distributed_c10d
    group, _prefix_store = c10d._new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name,
        None,
        timeout,
        None,
        device_id,
    )
    c10d._world.pg_group_ranks[group] = {
        group_rank: group_rank for group_rank in range(world_size)
    }
    return group


class TorchDistributedCudaIndependentProcessGroupFactory(
    AbstractBaseIndependentProcessGroupFactory
):
    """Create executor-owned Gloo control and NCCL device groups."""

    def create(
        self,
        init_method: str,
        rank: int,
        world_size: int,
        device: torch.device,
        group_name: str,
        nccl_proto: str,
        timeout: timedelta = DEFAULT_PROCESS_GROUP_TIMEOUT,
    ) -> IndependentProcessGroups:
        if device.type != "cuda":
            raise ValueError(f"NCCL process groups require a CUDA device, got {device}")
        if not init_method.startswith("tcp://"):
            raise ValueError(
                f"Independent process groups require a TCP init method, got {init_method!r}"
            )
        host, port = init_method.removeprefix("tcp://").rsplit(":", 1)
        store = dist.TCPStore(
            host,
            int(port),
            world_size,
            rank == 0,
            timeout,
        )
        control_process_group = _create_independent_process_group(
            store,
            rank,
            world_size,
            "gloo",
            f"{group_name}-control",
            timeout,
        )
        try:
            with _nccl_protocol(nccl_proto):
                device_process_group = _create_independent_process_group(
                    store,
                    rank,
                    world_size,
                    "nccl",
                    f"{group_name}-device",
                    timeout,
                    device,
                )
        except BaseException:
            dist.destroy_process_group(control_process_group)
            raise
        return IndependentProcessGroups(
            store,
            control_process_group,
            device_process_group,
        )


def create_independent_process_groups(
    init_method: str,
    rank: int,
    world_size: int,
    device: torch.device,
    group_name: str,
    configuration=None,
    factory: AbstractBaseIndependentProcessGroupFactory | None = None,
    timeout: timedelta = DEFAULT_PROCESS_GROUP_TIMEOUT,
) -> IndependentProcessGroups:
    if configuration is None:
        from ..execution_context import current_execution_context

        configuration = current_execution_context().configuration
    return (
        factory or TorchDistributedCudaIndependentProcessGroupFactory()
    ).create(
        init_method,
        rank,
        world_size,
        device,
        group_name,
        configuration.nccl_proto,
        timeout,
    )


def destroy_independent_process_groups(groups: IndependentProcessGroups):
    """Destroy one executor's groups without affecting any other executor."""

    errors = []
    for group in (
        groups.device_process_group,
        groups.control_process_group,
    ):
        try:
            dist.destroy_process_group(group)
        except Exception as error:  # Preserve cleanup of the remaining group.
            errors.append(error)
    if errors:
        raise errors[0]


def create_device_process_group(
    ranks,
    device: torch.device,
    configuration=None,
    factory: AbstractBaseDeviceProcessGroupFactory | None = None,
):
    if configuration is None:
        from ..execution_context import current_execution_context

        configuration = current_execution_context().configuration
    return (factory or TorchDistributedCudaProcessGroupFactory()).create(
        ranks,
        device,
        configuration.nccl_proto,
    )
