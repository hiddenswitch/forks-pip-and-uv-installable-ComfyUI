"""Stable Cascade stages with real prior and CLIP conditioning."""

from types import SimpleNamespace

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def stable_cascade(model_name):
    model_class = getattr(supported_models, model_name)
    stage_c = model_name == "Stable_Cascade_C"
    config = dict(model_class.unet_config)
    config.update(
        c_cond=16,
        c_hidden=[16, 32],
        c_r=8,
        nhead=[2, 2],
        blocks=[[1, 1], [1, 1]],
        block_repeat=[[2, 1], [1, 2]],
        level_config=["CTA", "CTA"],
        dropout=[0.0, 0.0],
        c_clip_seq=2,
        dtype=torch.float32,
    )
    if stage_c:
        config.update(c_clip_text=12, c_clip_text_pooled=12, switch_level=[True])
    else:
        config.update(c_clip=12)
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn((1, 16, 4, 6) if stage_c else (1, 4, 12, 16))
    pooled = torch.randn(1, 12)
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            pooled_output=pooled,
            cross_attn=torch.randn(1, 3, 12),
        )
        if conditioned:
            options["pooled_output"] = pooled + 1
            if stage_c:
                options["unclip_conditioning"] = [
                    {
                        "strength": weight,
                        "clip_vision_output": SimpleNamespace(
                            image_embeds=torch.randn(1, 768)
                        ),
                    }
                    for weight in (0.25, 0.75)
                ]
            else:
                options["stable_cascade_prior"] = torch.randn(1, 16, 2, 3)
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        if stage_c:
            assert conditions["clip_img"].shape == (1, 2 if conditioned else 1, 768)
        elif conditioned:
            conditions["pixels"] = torch.randn(1, 3, 2, 3)
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"
