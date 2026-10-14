from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from functools import lru_cache
import re
from typing import Mapping, Sequence

from .types import PipelineDeviceMemoryBudget, PipelineModelMemoryGeometry, PipelineParallelConfig, PipelinePartitionPlan, PipelineStagePlan, TensorDescriptor


class AbstractBasePipelineStageSpec(ABC):
    model_family: str
    block_count: int

    @abstractmethod
    def block_index(self, key: str) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def non_block_stage(self, key: str, stage_count: int) -> int:
        raise NotImplementedError

    def plan(
        self,
        tensors: Mapping[str, TensorDescriptor],
        config: PipelineParallelConfig,
        memory_budgets: Sequence[PipelineDeviceMemoryBudget] | None = None,
        model_memory_geometry: PipelineModelMemoryGeometry | None = None,
    ) -> PipelinePartitionPlan:
        if len(config.devices) > self.block_count:
            raise ValueError(f"{self.model_family} has fewer blocks than requested pipeline stages")
        if model_memory_geometry is None:
            checkpoint_geometry = self.checkpoint_memory_geometry(tensors, len(config.devices))
            block_bytes = list(checkpoint_geometry.block_bytes)
            non_block_bytes = list(checkpoint_geometry.non_block_bytes)
        else:
            if len(model_memory_geometry.block_bytes) != self.block_count:
                raise ValueError("Pipeline model memory geometry must contain one size per transformer block")
            if len(model_memory_geometry.non_block_bytes) != len(config.devices):
                raise ValueError("Pipeline model memory geometry must contain one non-block size per stage")
            block_bytes = list(model_memory_geometry.block_bytes)
            non_block_bytes = list(model_memory_geometry.non_block_bytes)

        if memory_budgets is not None:
            if tuple(budget.device for budget in memory_budgets) != config.devices:
                raise ValueError("Pipeline memory budgets must match the configured devices in order")
            usable_budgets = tuple(budget.available_weight_bytes for budget in memory_budgets)
        else:
            usable_budgets = (1,) * len(config.devices)
        counts = config.partition or self._pressure_aware_partition(block_bytes, non_block_bytes, usable_budgets)
        if sum(counts) != self.block_count:
            raise ValueError(f"Pipeline partition covers {sum(counts)} blocks, expected {self.block_count}")

        stages = []
        start = 0
        for stage_index, (device, count) in enumerate(zip(config.devices, counts, strict=True)):
            end = start + count
            owned_keys = frozenset(
                key for key in tensors
                if self._owns_key(key, stage_index, start, end, len(config.devices))
            )
            checkpoint_bytes = sum(tensors[key].nbytes for key in owned_keys)
            stages.append(PipelineStagePlan(stage_index, device, start, end, checkpoint_bytes, owned_keys))
            start = end
        return PipelinePartitionPlan(self.model_family, tuple(stages))

    def checkpoint_memory_geometry(self, tensors, stage_count: int) -> PipelineModelMemoryGeometry:
        block_bytes = [0] * self.block_count
        non_block_bytes = [0] * stage_count
        for key, descriptor in tensors.items():
            index = self.block_index(key)
            if index is None:
                non_block_bytes[self.non_block_stage(key, stage_count)] += descriptor.nbytes
            elif index < self.block_count:
                block_bytes[index] += descriptor.nbytes
        return PipelineModelMemoryGeometry(tuple(block_bytes), tuple(non_block_bytes))

    def _owns_key(self, key: str, stage_index: int, start: int, end: int, stage_count: int) -> bool:
        block_index = self.block_index(key)
        if block_index is None:
            return self.non_block_stage(key, stage_count) == stage_index
        return start <= block_index < end

    def owns_key(self, key: str, stage: PipelineStagePlan, stage_count: int) -> bool:
        return self._owns_key(key, stage.index, stage.start_layer, stage.end_layer, stage_count)

    def _pressure_aware_partition(
        self,
        block_bytes: Sequence[int],
        non_block_bytes: Sequence[int],
        usable_budgets: Sequence[int],
    ) -> tuple[int, ...]:
        stage_count = len(non_block_bytes)
        prefix = [0]
        for size in block_bytes:
            prefix.append(prefix[-1] + size)

        costs = [[None] * (self.block_count + 1) for _ in range(stage_count)]
        paths = [[None] * (self.block_count + 1) for _ in range(stage_count)]
        for end in range(1, self.block_count - stage_count + 2):
            size = non_block_bytes[0] + prefix[end]
            pressure = Fraction(size, max(1, usable_budgets[0]))
            overflow = max(Fraction(0), pressure - 1)
            costs[0][end] = (pressure, overflow, end)

        for stage in range(1, stage_count):
            min_end = stage + 1
            max_end = self.block_count - (stage_count - stage - 1)
            for end in range(min_end, max_end + 1):
                best = None
                best_start = None
                for start in range(stage, end):
                    previous = costs[stage - 1][start]
                    if previous is None:
                        continue
                    size = non_block_bytes[stage] + prefix[end] - prefix[start]
                    pressure = Fraction(size, max(1, usable_budgets[stage]))
                    overflow = max(Fraction(0), pressure - 1)
                    candidate = (
                        max(previous[0], pressure),
                        previous[1] + overflow,
                        max(previous[2], end - start),
                    )
                    if best is None or candidate < best:
                        best = candidate
                        best_start = start
                costs[stage][end] = best
                paths[stage][end] = best_start

        boundaries = [self.block_count]
        end = self.block_count
        for stage in range(stage_count - 1, 0, -1):
            end = paths[stage][end]
            boundaries.append(end)
        boundaries.append(0)
        boundaries.reverse()
        return tuple(boundaries[index + 1] - boundaries[index] for index in range(stage_count))


class _ContiguousBlockStageSpec(AbstractBasePipelineStageSpec):
    block_prefix: str
    exit_prefixes: tuple[str, ...]

    def __init__(self):
        self._block_pattern = re.compile(rf"^{re.escape(self.block_prefix)}(\d+)\.")

    def block_index(self, key: str) -> int | None:
        match = self._block_pattern.match(key)
        return int(match.group(1)) if match else None

    def non_block_stage(self, key: str, stage_count: int) -> int:
        return stage_count - 1 if key.startswith(self.exit_prefixes) else 0


class QwenImagePipelineStageSpec(_ContiguousBlockStageSpec):
    model_family = "qwen_image"
    block_count = 60
    block_prefix = "transformer_blocks."
    exit_prefixes = ("norm_out.", "proj_out.")


class MiniMaxH3PipelineStageSpec(_ContiguousBlockStageSpec):
    model_family = "minimax_h3"
    block_count = 50
    block_prefix = "blocks."
    exit_prefixes = ("final_layer.",)


class HunyuanImage3PipelineStageSpec(_ContiguousBlockStageSpec):
    """The HunyuanImage3 custom node's transformer, in its ComfyUI checkpoint layout.

    The first stage builds each step's input sequence: the token embeddings, the latent patch
    embedding (`patch_embed`, conditioned by `time_embed`), the timestep, guidance and meanflow
    embedders, and the vision aligner for conditioning images. The last stage turns the image
    rows back into a velocity: `final_layer`, conditioned by `time_embed_2`. The 32 MoE decoder
    layers between them are partitioned by bytes.
    """

    model_family = "hunyuan_image3"
    block_count = 32
    block_prefix = "model.layers."
    exit_prefixes = ("final_layer.", "time_embed_2.")


class Flux2PipelineStageSpec(AbstractBasePipelineStageSpec):
    model_family = "flux2"
    double_block_count = 8
    single_block_count = 48
    block_count = double_block_count + single_block_count
    _double_pattern = re.compile(r"^double_blocks\.(\d+)\.")
    _single_pattern = re.compile(r"^single_blocks\.(\d+)\.")

    def plan(
        self,
        tensors: Mapping[str, TensorDescriptor],
        config: PipelineParallelConfig,
        memory_budgets: Sequence[PipelineDeviceMemoryBudget] | None = None,
        model_memory_geometry: PipelineModelMemoryGeometry | None = None,
    ) -> PipelinePartitionPlan:
        double_indices = {
            int(match.group(1))
            for key in tensors
            if (match := self._double_pattern.match(key))
        }
        single_indices = {
            int(match.group(1))
            for key in tensors
            if (match := self._single_pattern.match(key))
        }
        if (
            double_indices != set(range(self.double_block_count))
            or single_indices != set(range(self.single_block_count))
        ):
            raise ValueError(
                "Flux2 pipeline parallelism currently supports Flux2 Dev's "
                "8 double-stream and 48 single-stream blocks"
            )
        return super().plan(
            tensors,
            config,
            memory_budgets,
            model_memory_geometry,
        )

    def block_index(self, key: str) -> int | None:
        match = self._double_pattern.match(key)
        if match:
            return int(match.group(1))
        match = self._single_pattern.match(key)
        if match:
            return self.double_block_count + int(match.group(1))
        return None

    def non_block_stage(self, key: str, stage_count: int) -> int:
        return stage_count - 1 if key.startswith("final_layer.") else 0


@lru_cache(maxsize=None)
def get_pipeline_stage_spec(image_model: str) -> AbstractBasePipelineStageSpec:
    if image_model == "qwen_image":
        return QwenImagePipelineStageSpec()
    if image_model == "minimax_h3":
        return MiniMaxH3PipelineStageSpec()
    if image_model == "flux2":
        return Flux2PipelineStageSpec()
    if image_model == "hunyuan_image3":
        return HunyuanImage3PipelineStageSpec()
    raise ValueError(f"Pipeline parallel loading is not supported for {image_model}")
