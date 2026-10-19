from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import torch

from .types import PipelineIntermediateSchema, PipelineIntermediateTensors, PipelinePartitionPlan, pack_pipeline_value, unpack_pipeline_value


class AbstractBaseDeviceRuntime(ABC):
    @abstractmethod
    def supports(self, device: torch.device) -> bool:
        raise NotImplementedError

    @abstractmethod
    def can_transfer(self, source: torch.device, destination: torch.device) -> bool:
        raise NotImplementedError

    @abstractmethod
    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def copy(self, source: torch.Tensor, destination: torch.Tensor) -> object:
        raise NotImplementedError

    @abstractmethod
    def wait(self, completion: object, device: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_available(self, device: torch.device) -> int:
        raise NotImplementedError


class AbstractBasePipelineBufferPool(ABC):
    @abstractmethod
    def acquire(self, schema: PipelineIntermediateSchema, device: torch.device) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class AbstractBasePendingPipelineIntermediate(ABC):
    """An intermediate whose device transfer has been enqueued, not synchronized."""

    @abstractmethod
    def wait(self) -> PipelineIntermediateTensors:
        raise NotImplementedError


class ImmediatePipelineIntermediate(AbstractBasePendingPipelineIntermediate):
    def __init__(self, intermediate: PipelineIntermediateTensors):
        self.intermediate = intermediate

    def wait(self) -> PipelineIntermediateTensors:
        return self.intermediate


class AbstractBasePipelineTransport(ABC):
    def begin_transfer(
        self,
        intermediate: PipelineIntermediateTensors,
        destination: torch.device,
    ) -> AbstractBasePendingPipelineIntermediate:
        """Enqueue a transfer and return its stage-consumption boundary.

        Backends with asynchronous transports override this method. The default
        keeps existing synchronous transports usable through the same runner.
        """
        schema = intermediate.schema()
        return ImmediatePipelineIntermediate(PipelineIntermediateTensors(
            dict(self.transfer(intermediate.tensors, schema, destination)),
            self.transfer_metadata(intermediate.metadata, destination),
        ))

    @abstractmethod
    def transfer(
        self,
        tensors: Mapping[str, torch.Tensor],
        schema: PipelineIntermediateSchema,
        destination: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def transfer_metadata(self, metadata: Mapping[str, object], destination: torch.device) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class AbstractBasePipelineExecutor(ABC):
    @property
    @abstractmethod
    def plan(self) -> PipelinePartitionPlan:
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def finish_execution(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class AbstractBasePipelineStageRunner(ABC):
    @abstractmethod
    def run(self, stage, device: torch.device, *args, **kwargs):
        raise NotImplementedError


class AbstractBasePipelineOperations(ABC):
    @property
    @abstractmethod
    def uses_worker_processes(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports(self, devices: Sequence[torch.device]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_transport(self) -> AbstractBasePipelineTransport:
        raise NotImplementedError

    @abstractmethod
    def create_stage_runner(self) -> AbstractBasePipelineStageRunner:
        raise NotImplementedError

    @abstractmethod
    def create_executor(self, plan, stages, worker_load_specs=()) -> AbstractBasePipelineExecutor:
        raise NotImplementedError


class DirectPipelineStageRunner(AbstractBasePipelineStageRunner):
    def run(self, stage, device: torch.device, *args, **kwargs):
        return stage(*args, **kwargs)


class LatestShapeBuffers:
    """Transfer buffers reused while a stream of transfers keeps its shape, and dropped when it changes.

    A stream is a destination and a set of tensor names (a stage's hidden states, the pipeline's output).
    Steps of one denoising run or one text decode share a shape and reuse their buffers. Every prompt
    length is a new shape, though: kept per shape, each prompt's buffers (a prompt-squared attention mask
    among them) stayed allocated for the life of the process.
    """

    def __init__(self, per_stream: bool = True):
        # per_stream=False: one buffer set in all, for a receiver of a single peer's transfers, where
        # a prompt's denoising buffers would otherwise outlive it under the next prompt's text prefill
        self.per_stream = per_stream
        self._streams: dict[tuple, tuple[tuple, dict[str, torch.Tensor]]] = {}

    def get(self, descriptors, allocate, device=None) -> dict[str, torch.Tensor]:
        stream = (device, tuple(descriptors)) if self.per_stream else None
        shape = tuple((d.shape, d.dtype) for d in descriptors.values())
        current = self._streams.get(stream)
        if current is not None and current[0] == shape:
            return current[1]
        # the old shape's buffers go before the new ones are allocated; a caller still holding them keeps them
        self._streams.pop(stream, None)
        if not self.per_stream:
            self._streams.clear()
        buffers = allocate(descriptors)
        self._streams[stream] = (shape, buffers)
        return buffers

    def retained_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for _, buffers in self._streams.values() for t in buffers.values())

    def clear(self) -> None:
        self._streams.clear()


class DevicePipelineBufferPool(AbstractBasePipelineBufferPool):
    def __init__(self, runtime: AbstractBaseDeviceRuntime):
        self.runtime = runtime
        self._buffers = LatestShapeBuffers()

    def acquire(self, schema: PipelineIntermediateSchema, device: torch.device) -> Mapping[str, torch.Tensor]:
        return self._buffers.get(schema.tensors, lambda descriptors: {
            name: self.runtime.allocate(descriptor.shape, descriptor.dtype, device)
            for name, descriptor in descriptors.items()
        }, device)

    def retained_bytes(self) -> int:
        return self._buffers.retained_bytes()

    def clear(self) -> None:
        self._buffers.clear()


class SingleProcessPipelineExecutor(AbstractBasePipelineExecutor):
    def __init__(
        self,
        plan: PipelinePartitionPlan,
        stages: Sequence[object],
        transport: AbstractBasePipelineTransport,
        stage_runner: AbstractBasePipelineStageRunner | None = None,
    ):
        if len(stages) != plan.size:
            raise ValueError("Pipeline executor requires one stage implementation per stage plan")
        self._plan = plan
        self.stages = tuple(stages)
        self.transport = transport
        self.stage_runner = stage_runner or DirectPipelineStageRunner()

    @property
    def plan(self) -> PipelinePartitionPlan:
        return self._plan

    def execute(self, *args, **kwargs):
        output = self.stage_runner.run(self.stages[0], self.plan.stages[0].device, *args, **kwargs)
        for stage_plan, stage in zip(self.plan.stages[1:], self.stages[1:], strict=True):
            if not isinstance(output, PipelineIntermediateTensors):
                raise TypeError("A non-final pipeline stage must return PipelineIntermediateTensors")
            schema = output.schema()
            schema.validate(output.tensors, output.metadata)
            pending = self.transport.begin_transfer(output, stage_plan.device)
            output = self.stage_runner.run(
                stage,
                stage_plan.device,
                pending.wait(),
            )
        if isinstance(output, PipelineIntermediateTensors):
            raise TypeError("The final pipeline stage returned intermediate tensors")
        tensors = {}
        structure = pack_pipeline_value(output, tensors, "pipeline_output")
        if not tensors:
            return output
        pending = self.transport.begin_transfer(
            PipelineIntermediateTensors(tensors, {"structure": structure}),
            self.plan.stages[0].device,
        )
        transferred = pending.wait()
        return unpack_pipeline_value(
            transferred.metadata["structure"],
            transferred.tensors,
        )

    def close(self) -> None:
        self.transport.close()

    def finish_execution(self) -> None:
        from .. import model_prefetch

        model_prefetch.finish_model_execution()
