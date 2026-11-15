import pytest
import torch

import comfy.model_detection
import comfy.model_base
import comfy.ops as comfy_ops
import comfy.supported_models
from comfy.ldm.pan2.model import PAN2
from comfy.pipeline_parallel import PAN2PipelineStageSpec, PipelineParallelConfig, SingleProcessPipelineExecutor
from comfy.pipeline_parallel.types import PipelinePartitionPlan, PipelineStageConfig, PipelineStagePlan
from comfy.tensor_parallel import (
    TensorParallelConfig,
    full_domain_tensor_parallel_operations,
    shard_tensor_parallel_state_dict,
    tensor_parallel_operations,
)

from .test_pipeline_parallel import FakeDeviceRuntime, FakePipelineTransport, descriptor
from .test_tensor_parallel import ThreadCollective, ThreadTensorParallelOperations, _run_parallel

PAN2_SMALL = dict(
    in_channels=97,
    out_channels=48,
    dim=128,
    num_heads=4,
    num_layers=4,
    mlp_ratio=2.0,
    context_mlp_ratio=2.0,
    text_dim=64,
    num_refiner_layers=2,
    patch_size=(1, 2, 2),
    rope_axes_dim=(8, 12, 12),
    adaln_curve_grid=16,
    adaln_curve_dim=32,
)


def _randomized(model: torch.nn.Module) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(7)
    for _, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.ndim >= 2:
            value.data.copy_(
                torch.empty(value.shape).uniform_(-1, 1, generator=generator) * (3.0 / value.shape[-1]) ** 0.5
            )
        else:
            value.data.copy_(torch.empty(value.shape).uniform_(-0.3, 0.3, generator=generator))
    return model.requires_grad_(False)


def _pan2_inputs():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(1, 97, 2, 4, 4, generator=generator)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 5, 64, generator=generator)
    return x, timestep, context


def test_pan2_two_stage_forward_matches_unpartitioned():
    full = _randomized(PAN2(**PAN2_SMALL, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init))
    first = PAN2(
        **PAN2_SMALL,
        dtype=torch.float32,
        device="cpu",
        operations=comfy_ops.disable_weight_init,
        pipeline_stage=PipelineStageConfig(0, 2, 0, 2),
    ).requires_grad_(False)
    second = PAN2(
        **PAN2_SMALL,
        dtype=torch.float32,
        device="cpu",
        operations=comfy_ops.disable_weight_init,
        pipeline_stage=PipelineStageConfig(1, 2, 2, 4),
    ).requires_grad_(False)
    first.load_state_dict(full.state_dict(), strict=False)
    second.load_state_dict(full.state_dict(), strict=False)
    plans = (
        PipelineStagePlan(0, torch.device("cpu"), 0, 2, 1, frozenset()),
        PipelineStagePlan(1, torch.device("cpu"), 2, 4, 1, frozenset()),
    )
    executor = SingleProcessPipelineExecutor(
        PipelinePartitionPlan("pan2", plans),
        (first, second.forward_pipeline_stage),
        FakePipelineTransport(FakeDeviceRuntime()),
    )
    x, timestep, context = _pan2_inputs()
    expected = full(x, timestep, context)
    actual = executor.execute(x, timestep, context)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_pan2_pipeline_partition_plan():
    tensors = {
        "x_embedder.weight": descriptor(10),
        "context_refiner.0.norm1.weight": descriptor(10),
        "adaln_t_table": descriptor(10),
        "norm_out.linear.weight": descriptor(10),
        "proj_out.weight": descriptor(10),
    }
    tensors.update({f"transformer_blocks.{index}.attn.to_q.weight": descriptor(5) for index in range(44)})
    plan = PAN2PipelineStageSpec().plan(tensors, PipelineParallelConfig(["cuda:0", "cuda:1"]))
    assert "x_embedder.weight" in plan.stages[0].owned_keys
    assert "context_refiner.0.norm1.weight" in plan.stages[0].owned_keys
    assert "adaln_t_table" in plan.stages[0].owned_keys
    assert "norm_out.linear.weight" in plan.stages[1].owned_keys
    assert "proj_out.weight" in plan.stages[1].owned_keys
    assert plan.stages[0].end_layer == plan.stages[1].start_layer


@pytest.mark.parametrize("decorator", [tensor_parallel_operations, full_domain_tensor_parallel_operations])
def test_pan2_tensor_parallel_matches_full_model(decorator):
    full = _randomized(PAN2(**PAN2_SMALL, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init))
    state = full.state_dict()
    collective = ThreadCollective(2)
    ranks = []
    for rank in range(2):
        operations = decorator(
            comfy_ops.disable_weight_init,
            TensorParallelConfig(ThreadTensorParallelOperations(collective, rank)),
        )
        model = PAN2(**PAN2_SMALL, dtype=torch.float32, device="cpu", operations=operations)
        sharded = shard_tensor_parallel_state_dict(
            state,
            "pan2",
            rank,
            2,
            full_domain_rows=decorator is full_domain_tensor_parallel_operations,
        )
        model.load_state_dict(sharded, strict=True)
        ranks.append(model.requires_grad_(False))

    x, timestep, context = _pan2_inputs()
    expected = full(x, timestep, context)
    for output in _run_parallel(
        [lambda _, m=m: m(x, timestep, context) for m in ranks],
        None,
    ):
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


def test_pan2_shard_quantized_checkpoint_weights():
    comfy_quant = torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
    qweight = torch.randint(-127, 127, (256, 128), dtype=torch.int8)
    scale = torch.full((256,), 0.05, dtype=torch.float32)
    state = {
        "transformer_blocks.0.ff.net.0.proj.weight": qweight,
        "transformer_blocks.0.ff.net.0.proj.weight_scale": scale,
        "transformer_blocks.0.ff.net.0.proj.comfy_quant": comfy_quant,
        "transformer_blocks.0.attn.to_out.0.weight": qweight,
        "transformer_blocks.0.attn.to_out.0.weight_scale": scale,
        "transformer_blocks.0.attn.to_out.0.comfy_quant": comfy_quant,
        "context_refiner.0.ff.net.0.proj.weight": qweight,
        "context_refiner.0.ff.net.0.proj.weight_scale": scale,
        "context_refiner.0.ff.net.0.proj.comfy_quant": comfy_quant,
    }
    sharded = shard_tensor_parallel_state_dict(state, "pan2", 0, 2, full_domain_rows=True)
    assert sharded["transformer_blocks.0.ff.net.0.proj.weight"].shape == (128, 128)
    assert sharded["transformer_blocks.0.ff.net.0.proj.weight_scale"].shape == (128,)
    assert sharded["transformer_blocks.0.attn.to_out.0.weight"].shape == (128, 128)
    assert sharded["context_refiner.0.ff.net.0.proj.weight"].shape == (128, 128)
    assert sharded["context_refiner.0.ff.net.0.proj.weight_scale"].shape == (128,)


def test_pan2_detection_and_concat_cond():
    fake_sd = {
        "x_embedder.weight": torch.empty(4096, 97, 1, 2, 2),
        "context_refiner.0.norm1.weight": torch.empty(4096),
        "context_refiner.1.norm1.weight": torch.empty(4096),
        "adaln_t_table": torch.empty(257, 256),
        "transformer_blocks.0.scale_shift_table_context": torch.empty(24576),
        "transformer_blocks.0.ff.net.0.proj.weight": torch.empty(32768, 4096),
        "transformer_blocks.0.ff_context.net.0.proj.weight": torch.empty(16384, 4096),
    }
    unet_config = comfy.model_detection.detect_unet_config(fake_sd, "")
    assert unet_config["image_model"] == "pan2"
    assert unet_config["in_channels"] == 97
    assert unet_config["num_refiner_layers"] == 2
    assert unet_config["num_layers"] == 1
    assert unet_config["adaln_curve_grid"] == 257
    assert unet_config["adaln_curve_dim"] == 256

    model_config = comfy.model_detection.model_config_from_unet_config(unet_config, fake_sd)
    assert isinstance(model_config, comfy.supported_models.PAN2)

    latent_image = torch.ones(1, 48, 2, 4, 4)
    cond_t2v = comfy.model_base.PAN2.concat_cond(None, latent_image=latent_image, noise=latent_image, device="cpu")
    assert cond_t2v.shape == (1, 49, 2, 4, 4)
    assert torch.count_nonzero(cond_t2v) == 0

    denoise_mask = torch.zeros(1, 1, 2, 4, 4)
    denoise_mask[:, :, 1:, :, :] = 1.0
    cond_i2v = comfy.model_base.PAN2.concat_cond(
        None,
        latent_image=latent_image * 2.0,
        noise=latent_image,
        device="cpu",
        denoise_mask=denoise_mask,
    )
    assert cond_i2v.shape == (1, 49, 2, 4, 4)
    torch.testing.assert_close(cond_i2v[:, :48, 0], torch.full((1, 48, 4, 4), 2.0))
    torch.testing.assert_close(cond_i2v[:, :48, 1], torch.zeros(1, 48, 4, 4))
    torch.testing.assert_close(cond_i2v[:, 48:, 0], torch.ones(1, 1, 4, 4))
    torch.testing.assert_close(cond_i2v[:, 48:, 1], torch.zeros(1, 1, 4, 4))
