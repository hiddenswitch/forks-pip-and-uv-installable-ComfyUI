"""SeedVR2 native conditioning and temporal patch contracts on CPU."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def seedvr2():
    config = dict(supported_models.SeedVR2.unet_config)
    config.update(
        norm_eps=1e-5, num_layers=32, mlp_type="swiglu",
        vid_dim=24, heads=2, head_dim=12, rope_dim=12,
        mm_layers=1, expand_ratio=2, patch_size=(2, 2, 2),
        vid_out_norm=True, dtype=torch.float32,
    )
    descriptor = supported_models.SeedVR2(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    native = model.diffusion_model
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in native.state_dict().items()
    }
    assert supported_models.SeedVR2.matches(
        model_detection.detect_unet_config(state, ""), state_dict=state
    )
    latent = torch.randn(1, 16, 3, 4, 6)
    context = torch.randn(1, 3, 5120)
    condition = torch.randn(1, 17, 3, 4, 6)
    sigma = torch.tensor([0.5])
    outputs = []
    for changed in (False, True):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent, device=torch.device("cpu"), cross_attn=context,
                condition=condition + int(changed),
            ).items()
        }
        with torch.no_grad():
            result = model.apply_model(
                latent, sigma, transformer_options={"cond_or_uncond": [0]}, **conditions
            )
            repeat = model.apply_model(
                latent, sigma, transformer_options={"cond_or_uncond": [0]}, **conditions
            )
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning latents were ignored"
    with torch.no_grad():
        changed_text = model.apply_model(
            latent, sigma, c_crossattn=context.flip(-1), condition=condition,
            transformer_options={"cond_or_uncond": [1]},
        )
        batched = model.apply_model(
            latent.repeat(2, 1, 1, 1, 1), sigma.expand(2),
            c_crossattn=torch.cat([context.flip(-1), context]),
            condition=condition.repeat(2, 1, 1, 1, 1),
            transformer_options={"cond_or_uncond": [1, 0]},
        )
    assert torch.isfinite(changed_text).all() and not torch.equal(outputs[0], changed_text)
    torch.testing.assert_close(batched[:1], changed_text)
    torch.testing.assert_close(batched[1:], outputs[0])
    with pytest.raises(ValueError, match="requires conditioning latents"):
        model.apply_model(latent, sigma, c_crossattn=context)
    with pytest.raises(ValueError, match="must match latent"):
        model.apply_model(latent, sigma, c_crossattn=context, condition=condition[:, :, :1])
