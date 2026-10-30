"""Tiny Cosmos forwards with native positional and inpainting conditioning."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def cosmos(model_name):
    model_class = getattr(supported_models, model_name)
    predict2 = model_name.endswith("Predict2")
    config = dict(model_class.unet_config)
    config.update(
        max_img_h=8,
        max_img_w=8,
        max_frames=4,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=32,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2,
        crossattn_emb_channels=12,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=8,
        extra_per_block_abs_pos_emb=not predict2,
        dtype=torch.float32,
    )
    if not predict2:
        config.update(
            block_x_format="THWBD",
            affline_emb_norm=True,
            extra_per_block_abs_pos_emb_type="learnable",
        )
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    detected = model_detection.detect_unet_config(state, "")
    assert model_class.matches(detected)
    assert detected["model_channels"] == 32
    for key in model_class.unet_config:
        incomplete = dict(model_class.unet_config)
        del incomplete[key]
        assert not model_class.matches(incomplete)

    latent = torch.randn(1, 16, 2, 3 if predict2 else 4, 5 if predict2 else 6)
    context = torch.randn(1, 3, 12)
    sigma = torch.tensor([0.5])
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            cross_attn=context + int(conditioned),
            concat_latent_image=torch.zeros_like(latent),
        )
        if conditioned:
            mask = torch.ones_like(latent[:, :1])
            mask[:, :, 0] = 0
            options.update(concat_mask=mask, frame_rate=12)
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        if model.image_to_video:
            assert conditions["c_concat"].shape == latent[:, :1].shape
            if conditioned:
                torch.testing.assert_close(conditions["c_concat"], 1 - mask)
        if predict2 and conditioned:
            timesteps = model.process_timestep(sigma, latent, **conditions)
            torch.testing.assert_close(timesteps, torch.tensor([[0.0, 0.5]]))
        with torch.no_grad():
            result = model.apply_model(latent, sigma, **conditions)
            repeat = model.apply_model(latent, sigma, **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"

    scaled = model.scale_latent_inpaint(sigma, torch.zeros_like(latent), latent)
    assert scaled.shape == latent.shape and torch.isfinite(scaled).all()


def anima():
    config = dict(supported_models.Anima.unet_config)
    config.update(max_img_h=8, max_img_w=8, max_frames=4, in_channels=16,
                  out_channels=16, patch_spatial=2, patch_temporal=1,
                  model_channels=32, num_blocks=2, num_heads=2, mlp_ratio=2,
                  crossattn_emb_channels=1024, pos_emb_cls="rope3d",
                  use_adaln_lora=True, adaln_lora_dim=8,
                  extra_per_block_abs_pos_emb=False, dtype=torch.float32)
    descriptor = supported_models.Anima(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.Anima.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 16, 1, 3, 5)
    context = torch.randn(1, 3, 1024)
    outputs = []
    for precompute, weighted in ((False, False), (False, True), (True, False), (True, True)):
        with torch.inference_mode(precompute):
            options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context,
                           t5xxl_ids=torch.tensor([1, 2, 3]))
            if weighted:
                options["t5xxl_weights"] = torch.ones(3)
            conditions = {key: value.process_cond(batch_size=1, area=None).cond
                          for key, value in model.extra_conds(**options).items()}
            assert ("t5xxl_ids" in conditions) != precompute
            with torch.no_grad():
                result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            assert result.shape == latent.shape and result.dtype == latent.dtype
            assert torch.isfinite(result).all()
            outputs.append(result.clone())
    for result in outputs[1:]:
        torch.testing.assert_close(outputs[0], result)
    with torch.no_grad():
        changed = model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context)
    assert not torch.equal(changed, outputs[0]), "text adapter was ignored"
