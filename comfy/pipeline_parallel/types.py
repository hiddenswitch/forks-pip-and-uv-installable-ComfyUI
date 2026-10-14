from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence
import uuid

import torch


class PipelineMissingLayer(torch.nn.Identity):
    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))


@dataclass(frozen=True)
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    device_type: str = "cuda"


@dataclass(frozen=True)
class PipelineStagePlan:
    index: int
    device: torch.device
    start_layer: int
    end_layer: int
    checkpoint_bytes: int
    owned_keys: frozenset[str]


@dataclass(frozen=True)
class PipelineDeviceMemoryBudget:
    device: torch.device
    available_weight_bytes: int


@dataclass(frozen=True)
class PipelineModelMemoryGeometry:
    block_bytes: tuple[int, ...]
    non_block_bytes: tuple[int, ...]

    def stage_bytes(self, stage: PipelineStagePlan) -> int:
        return self.non_block_bytes[stage.index] + sum(
            self.block_bytes[stage.start_layer:stage.end_layer]
        )


@dataclass(frozen=True)
class PipelineStageMemoryGeometry:
    block_bytes: Mapping[int, int]
    non_block_bytes: int


@dataclass(frozen=True)
class PipelineStageConfig:
    index: int
    count: int
    start_layer: int
    end_layer: int

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.count - 1


@dataclass(frozen=True)
class PipelinePartitionPlan:
    model_family: str
    stages: tuple[PipelineStagePlan, ...]

    @property
    def size(self) -> int:
        return len(self.stages)


@dataclass(frozen=True)
class PipelineParallelConfig:
    devices: tuple[torch.device, ...]
    partition: tuple[int, ...] | None = None

    def __init__(self, devices: Sequence[torch.device | str], partition: Sequence[int] | None = None):
        resolved_devices = tuple(torch.device(device) for device in devices)
        if len(resolved_devices) < 2:
            raise ValueError("Pipeline parallelism requires at least two devices")
        if len(set(resolved_devices)) != len(resolved_devices):
            raise ValueError("Pipeline parallel devices must be unique")
        if partition is not None:
            partition = tuple(int(count) for count in partition)
            if len(partition) != len(resolved_devices) or any(count <= 0 for count in partition):
                raise ValueError("Pipeline partition must contain one positive layer count per device")
        object.__setattr__(self, "devices", resolved_devices)
        object.__setattr__(self, "partition", partition)


@dataclass(frozen=True)
class PipelineWorkerLoadSpec:
    checkpoint_path: str
    plan: PipelinePartitionPlan
    stage_index: int
    model_options: dict
    disable_dynamic: bool
    dtype: torch.dtype
    # a model family built by a custom node: the node's package directory, which a worker rank
    # imports itself because launcher ranks other than zero do not load custom nodes
    custom_node_path: str | None = None


@dataclass(frozen=True)
class PipelineIntermediateSchema:
    tensors: Mapping[str, TensorDescriptor]
    metadata_keys: tuple[str, ...] = ()

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, object] | None = None) -> None:
        if tensors.keys() != self.tensors.keys():
            raise ValueError(f"Pipeline tensor keys do not match schema: {tuple(tensors)} != {tuple(self.tensors)}")
        for name, descriptor in self.tensors.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != descriptor.shape or tensor.dtype != descriptor.dtype:
                raise ValueError(
                    f"Pipeline tensor {name} does not match schema: "
                    f"{tuple(tensor.shape)} {tensor.dtype} != {descriptor.shape} {descriptor.dtype}"
                )
        metadata = metadata or {}
        if tuple(metadata.keys()) != self.metadata_keys:
            raise ValueError(f"Pipeline metadata keys do not match schema: {tuple(metadata)} != {self.metadata_keys}")


@dataclass
class PipelineIntermediateTensors:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]

    def schema(self) -> PipelineIntermediateSchema:
        return PipelineIntermediateSchema(
            tensors={
                name: TensorDescriptor(tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size())
                for name, tensor in self.tensors.items()
            },
            metadata_keys=tuple(self.metadata.keys()),
        )


# Layout objects are process-local caches rebuilt by the stage that owns the
# inputs; MiniMax H3 publishes its packed layout under minimax_h3_layout for
# attention patches, which pipeline stages do not run.
_PROCESS_LOCAL_TRANSFORMER_OPTIONS = frozenset(("layout", "minimax_h3_layout"))


def prepare_model_parallel_value(value):
    """Drop process-local execution state before transporting model inputs.

    Sampler, guider, and apply-model wrappers have already run by the time a
    diffusion-model executor is entered.  Only diffusion-model wrappers belong
    on every model-parallel rank.  Layout objects and the other wrapper
    callables are process-local caches/control flow and cannot be reconstructed
    from tensor transport metadata.
    """
    if isinstance(value, dict):
        cleaned = {
            key: prepare_model_parallel_value(item)
            for key, item in value.items()
            if key not in _PROCESS_LOCAL_TRANSFORMER_OPTIONS
        }
        wrappers = cleaned.get("wrappers")
        if isinstance(wrappers, dict):
            diffusion_model = wrappers.get("diffusion_model")
            if diffusion_model:
                cleaned["wrappers"] = {"diffusion_model": diffusion_model}
            else:
                cleaned.pop("wrappers", None)
        return cleaned
    if isinstance(value, list):
        return [prepare_model_parallel_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(prepare_model_parallel_value(item) for item in value)
    return value


def pack_pipeline_value(value, tensors: dict[str, torch.Tensor], prefix: str):
    if torch.is_tensor(value):
        if prefix in tensors:
            raise ValueError(f"Duplicate pipeline tensor key: {prefix}")
        tensors[prefix] = value
        return ("tensor", prefix)
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value).removeprefix("torch."))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, uuid.UUID):
        return ("uuid", str(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(pack_pipeline_value(item, tensors, f"{prefix}.{index}") for index, item in enumerate(value)))
    if isinstance(value, list):
        return ("list", tuple(pack_pipeline_value(item, tensors, f"{prefix}.{index}") for index, item in enumerate(value)))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (
                    pack_pipeline_value(key, tensors, f"{prefix}.key{index}"),
                    pack_pipeline_value(item, tensors, f"{prefix}.value{index}"),
                )
                for index, (key, item) in enumerate(value.items())
            ),
        )
    raise TypeError(f"Pipeline value at {prefix} is not transportable: {type(value).__name__}")


def unpack_pipeline_value(value, tensors: Mapping[str, torch.Tensor]):
    kind, payload = value
    if kind == "tensor":
        return tensors[payload]
    if kind == "value":
        return payload
    if kind == "dtype":
        return getattr(torch, payload)
    if kind == "device":
        return torch.device(payload)
    if kind == "uuid":
        return uuid.UUID(payload)
    if kind == "tuple":
        return tuple(unpack_pipeline_value(item, tensors) for item in payload)
    if kind == "list":
        return [unpack_pipeline_value(item, tensors) for item in payload]
    if kind == "dict":
        return {
            unpack_pipeline_value(key, tensors): unpack_pipeline_value(item, tensors)
            for key, item in payload
        }
    raise ValueError(f"Unknown pipeline value encoding: {kind}")


def serialize_pipeline_metadata(metadata: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(f"Pipeline metadata is not wire-serializable: {error}") from error


def deserialize_pipeline_metadata(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Pipeline metadata payload must decode to a dictionary")
    return value
