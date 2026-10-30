"""Pixel transformers and latent-conditioned restoration with real CPU math."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def pixel_transformer(model_name):
    model_class = getattr(supported_models, model_name)
    restoration = model_name == "PiD"
    config = dict(model_class.unet_config)
    config.update(
        in_channels=3,
        num_groups=2,
        hidden_size=32,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=32,
        pixel_num_groups=2,
        patch_depth=2,
        pixel_depth=2,
        patch_size=2,
        txt_embed_dim=12,
        txt_max_length=3,
        dtype=torch.float32,
    )
    if restoration:
        config.update(
            lq_hidden_dim=16,
            lq_num_res_blocks=2,
            lq_interval=1,
            lq_gate_per_token=True,
            pit_lq_inject=True,
        )
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    for module in model.diffusion_model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.weight.fill_(1)
    prefix = "" if restoration else "core."
    state = {
        prefix + key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    generator = torch.Generator().manual_seed(42)
    latent = torch.randn(1, 3, 3, 5, generator=generator)
    context = torch.randn(1, 3, 12, generator=generator)
    outputs = []
    for conditioned in (False, True):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context)
        if restoration:
            options.update(
                lq_latent=torch.randn(1, 16, 2, 3, generator=generator)
                if conditioned
                else torch.zeros(1, 16, 2, 3),
                degrade_sigma=torch.tensor([0.2 if conditioned else 0.1]),
            )
        elif conditioned:
            options["cross_attn"] = context.flip(-1)
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"
    if restoration:
        with pytest.raises(ValueError, match="requires lq_latent"):
            model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context)
