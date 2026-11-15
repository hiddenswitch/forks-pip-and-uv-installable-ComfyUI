import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.model import Magi2Model
from comfy.ldm.magi2.refiner import Magi2RefinerModel
from comfy.pipeline_parallel import Magi2PipelineStageSpec, SingleProcessPipelineExecutor
from comfy.pipeline_parallel.types import PipelinePartitionPlan, PipelineStageConfig, PipelineStagePlan
from comfy.tensor_parallel import TensorParallelConfig, full_domain_tensor_parallel_operations, shard_tensor_parallel_state_dict, tensor_parallel_operations

from .test_pipeline_parallel import FakeDeviceRuntime, FakePipelineTransport, descriptor
from .test_tensor_parallel import ThreadCollective, ThreadTensorParallelOperations, _run_parallel

PREVIEW = dict(num_layers=4, hidden_size=128, head_dim=32, num_query_groups=4, video_in_channels=8, audio_in_channels=6, text_in_channels=24,
               intermediate_size=64, multimodal_layers=(0, 3), moe_layers=(1, 2), moe_heads=2, num_experts=4, top_k=2,
               expert_intermediate_size=16, shared_expert_intermediate_size=32, modality_shared_expert_intermediate_size=32)
REFINER = dict(num_layers=3, hidden_size=128, head_dim=32, num_query_groups=2, video_in_channels=8, audio_in_channels=6, text_in_channels=24,
               intermediate_size=34, multimodal_layers=(0, 2), window_block_size=(2, 2, 2), window_radius=(1, 1, 1))


def _randomized(model):
    generator = torch.Generator().manual_seed(1)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if name.endswith("rope.bands"):
            value.data.copy_(1.0 / 10000 ** (torch.arange(value.numel()) / value.numel()))
        elif value.ndim >= 2:
            value.data.copy_(torch.empty(value.shape).uniform_(-1, 1, generator=generator) * (3.0 / value.shape[-1]) ** 0.5)
        else:
            value.data.copy_(torch.empty(value.shape).uniform_(-0.3, 0.3, generator=generator))
    return model.requires_grad_(False)


def _preview_inputs():
    generator = torch.Generator().manual_seed(2)
    x = [torch.randn(1, 8, 3, 4, 6, generator=generator), torch.randn(1, 6, 7, generator=generator)]
    refs = [{"latent": torch.randn(1, 8, 1, 2, 3, generator=generator), "token": torch.randn(24, generator=generator)}]
    return x, torch.tensor([612.7]), torch.randn(1, 5, 24, generator=generator), refs


def test_preview_two_stage_forward_matches_unpartitioned():
    full = _randomized(Magi2Model(**PREVIEW, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init))
    first = Magi2Model(**PREVIEW, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init, pipeline_stage=PipelineStageConfig(0, 2, 0, 2))
    second = Magi2Model(**PREVIEW, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init, pipeline_stage=PipelineStageConfig(1, 2, 2, 4))
    first.load_state_dict(full.state_dict(), strict=False)
    second.load_state_dict(full.state_dict(), strict=False)
    plans = (PipelineStagePlan(0, torch.device("cpu"), 0, 2, 1, frozenset()), PipelineStagePlan(1, torch.device("cpu"), 2, 4, 1, frozenset()))
    executor = SingleProcessPipelineExecutor(PipelinePartitionPlan("magi2", plans), (first.requires_grad_(False), second.requires_grad_(False).forward_pipeline_stage),
                                             FakePipelineTransport(FakeDeviceRuntime()))
    x, timestep, context, refs = _preview_inputs()
    expected = full(x, timestep, context, magi2_refs=refs)
    actual = executor.execute(x, timestep, context, magi2_refs=refs)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_refiner_two_stage_forward_matches_unpartitioned():
    full = _randomized(Magi2RefinerModel(**REFINER, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init))
    stages = [Magi2RefinerModel(**REFINER, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init,
                                pipeline_stage=PipelineStageConfig(index, 2, *bounds)).requires_grad_(False) for index, bounds in enumerate(((0, 1), (1, 3)))]
    for stage in stages:
        stage.load_state_dict(full.state_dict(), strict=False)
    plans = (PipelineStagePlan(0, torch.device("cpu"), 0, 1, 1, frozenset()), PipelineStagePlan(1, torch.device("cpu"), 1, 3, 1, frozenset()))
    executor = SingleProcessPipelineExecutor(PipelinePartitionPlan("magi2_refiner", plans), (stages[0], stages[1].forward_pipeline_stage),
                                             FakePipelineTransport(FakeDeviceRuntime()))
    latent = torch.randn(1, 8, 5, 4, 6, generator=torch.Generator().manual_seed(3))
    context = torch.randn(1, 4, 24, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(executor.execute(latent, torch.tensor([400.0]), context), full(latent, torch.tensor([400.0]), context))


def test_pipeline_partition_gives_adapters_to_the_end_stages():
    tensors = {"pre_adapter.video_embedder.weight": descriptor(10), "post_adapter.final_linear_video.weight": descriptor(10)}
    tensors.update({f"block.layers.{index}.attention.sinks": descriptor(5) for index in range(40)})
    from comfy.pipeline_parallel import PipelineParallelConfig
    plan = Magi2PipelineStageSpec().plan(tensors, PipelineParallelConfig(["cuda:0", "cuda:1"]))
    assert "pre_adapter.video_embedder.weight" in plan.stages[0].owned_keys
    assert "post_adapter.final_linear_video.weight" in plan.stages[1].owned_keys
    assert plan.stages[0].end_layer == plan.stages[1].start_layer


@pytest.mark.parametrize("decorator", [tensor_parallel_operations, full_domain_tensor_parallel_operations])
@pytest.mark.parametrize("family", ["magi2", "magi2_refiner"])
def test_tensor_parallel_matches_full_model(decorator, family):
    cls, config = (Magi2Model, PREVIEW) if family == "magi2" else (Magi2RefinerModel, REFINER)
    full = _randomized(cls(**config, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init))
    state = full.state_dict()
    collective = ThreadCollective(2)
    ranks = []
    for rank in range(2):
        operations = decorator(comfy_ops.disable_weight_init, TensorParallelConfig(ThreadTensorParallelOperations(collective, rank)))
        model = cls(**config, dtype=torch.float32, device="cpu", operations=operations)
        model.load_state_dict(shard_tensor_parallel_state_dict(state, family, rank, 2, full_domain_rows=decorator is full_domain_tensor_parallel_operations), strict=True)
        ranks.append(model.requires_grad_(False))
    if family == "magi2":
        x, timestep, context, refs = _preview_inputs()
        run = lambda model: model(x, timestep, context, magi2_refs=refs)
    else:
        x = torch.randn(1, 8, 5, 4, 6, generator=torch.Generator().manual_seed(3))
        context = torch.randn(1, 4, 24, generator=torch.Generator().manual_seed(4))
        run = lambda model: model(x, torch.tensor([400.0]), context)
    expected = run(full)
    for output in _run_parallel([lambda _, model=model: run(model) for model in ranks], None):
        for value, reference in zip(output if isinstance(output, list) else [output], expected if isinstance(expected, list) else [expected]):
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-4)
