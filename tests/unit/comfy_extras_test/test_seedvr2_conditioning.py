"""SeedVR2 conditioning node regression tests."""

import pytest
import torch
import torch.nn as nn

from comfy.ldm.seedvr.constants import SEEDVR2_LATENT_CHANNELS



from comfy_extras.nodes import nodes_seedvr


class _Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(4))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = _Rope()


class _DiffusionModel(nn.Module):
    def __init__(self, n_blocks=3, conditioning_dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.register_buffer("positive_conditioning", torch.ones((2, 4), dtype=conditioning_dtype))
        self.register_buffer("negative_conditioning", torch.zeros((3, 4), dtype=conditioning_dtype))


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_seedvr2_conditioning_schema_exposes_conditioning_outputs():
    schema = nodes_seedvr.SeedVR2Conditioning.define_schema()
    assert [input_item.id for input_item in schema.inputs] == [
        "model",
        "vae_conditioning",
    ]
    assert schema.inputs[1].display_name == "latent"
    assert [output.display_name for output in schema.outputs] == [
        "positive",
        "negative",
    ]


def test_seedvr2_conditioning_rejects_wrong_latent_channels():
    patcher = _ModelPatcher(_DiffusionModel())
    vae_conditioning = {"samples": torch.zeros(1, 8, 2, 2, 2)}

    with pytest.raises(ValueError, match=f"{SEEDVR2_LATENT_CHANNELS} channels"):
        nodes_seedvr.SeedVR2Conditioning.execute(patcher, vae_conditioning)


def test_seedvr2_conditioning_returns_conditioning_deterministically():
    diffusion_model = _DiffusionModel()
    patcher = _ModelPatcher(diffusion_model)
    samples = torch.arange(
        1,
        1 + SEEDVR2_LATENT_CHANNELS * 3 * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, SEEDVR2_LATENT_CHANNELS, 3, 2, 2)
    vae_conditioning = {"samples": samples}

    first_positive, first_negative = (
        nodes_seedvr.SeedVR2Conditioning.execute(
            patcher,
            vae_conditioning,
        )
    )
    second_positive, second_negative = (
        nodes_seedvr.SeedVR2Conditioning.execute(
            patcher,
            vae_conditioning,
        )
    )

    channel_last = samples.movedim(1, -1).contiguous()
    expected_condition = torch.cat(
        [
            channel_last,
            torch.ones((*channel_last.shape[:-1], 1)),
        ],
        dim=-1,
    ).movedim(-1, 1)

    assert torch.equal(
        first_positive[0][1]["condition"],
        expected_condition,
    )
    assert torch.equal(
        second_positive[0][1]["condition"],
        expected_condition,
    )
    assert torch.equal(
        first_negative[0][1]["condition"],
        expected_condition,
    )
    assert torch.equal(
        second_negative[0][1]["condition"],
        expected_condition,
    )
