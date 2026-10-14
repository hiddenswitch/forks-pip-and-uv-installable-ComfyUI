from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AbstractBaseTensorParallelOperations(ABC):
    @property
    @abstractmethod
    def rank(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate equal rank-local outputs along their feature dimension."""
        raise NotImplementedError

class TorchDistributedTensorParallelOperations(AbstractBaseTensorParallelOperations):
    """Tensor collectives backed by an injected torch.distributed group."""

    def __init__(self, rank: int, world_size: int, device: torch.device, process_group,
                 control_process_group=None):
        self._rank = rank
        self._world_size = world_size
        self.device = device
        self.process_group = process_group
        self.control_process_group = control_process_group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        completion = dist.all_reduce(tensor, group=self.process_group, async_op=True)
        completion.block_current_stream()
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        completion = dist.all_gather(parts, tensor.contiguous(), group=self.process_group, async_op=True)
        completion.block_current_stream()
        return torch.cat(parts, dim=-1)
