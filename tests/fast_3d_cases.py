"""Small point-latent contracts for the registered Hunyuan 3D models."""

import torch

from comfy import model_detection, ops, supported_models, utils
from tests.fast_model_cases import _initialize


def hunyuan_3d(model_name):
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    config.update(in_channels=8, hidden_size=32, num_heads=2, dtype=torch.float32)
    if model_name == "Hunyuan3Dv2_1":
        config.update(
            depth=3,
            context_dim=12,
            num_moe_layers=1,
            num_experts=3,
            moe_top_k=2,
            guidance_cond_proj_dim=None,
        )
    else:
        config.update(
            depth=8 if model_name == "Hunyuan3Dv2mini" else 2,
            context_in_dim=12,
            depth_single_blocks=2,
            mlp_ratio=2,
            guidance_embed=True,
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
    assert detected["in_channels"] == 8
    assert detected["depth"] == config["depth"]
    latent = torch.randn(1, 8, 5)
    context = torch.randn(2, 3, 12)
    outputs = []
    for batch in (1, 2):
        inputs = latent.repeat(batch, 1, 1)
        conditions = {
            key: value.process_cond(batch_size=batch, area=None).cond
            for key, value in model.extra_conds(
                noise=inputs,
                device=torch.device("cpu"),
                cross_attn=context[:batch],
                guidance=3.0,
            ).items()
        }
        with torch.no_grad():
            result = model.apply_model(inputs, torch.full((batch,), 0.5), **conditions)
            repeat = model.apply_model(inputs, torch.full((batch,), 0.5), **conditions)
        assert result.shape == inputs.shape and result.dtype == inputs.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    assert not torch.equal(outputs[1][0], outputs[1][1]), (
        "image conditioning was ignored"
    )


def triposplat():
    descriptor = supported_models.TripoSplat(
        dict(
            image_model="triposplat",
            q_token_length=5,
            in_channels=16,
            model_channels=32,
            cond_channels=12,
            out_channels=16,
            num_blocks=2,
            num_refiner_blocks=2,
            num_heads=2,
            num_head_channels=16,
            cam_channels=5,
            cond2_channels=128,
            mlp_ratio=2,
            dtype=torch.float32,
        )
    )
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert supported_models.TripoSplat.matches(
        model_detection.detect_unet_config(state, "")
    )
    points = torch.randn(1, 5, 16)
    camera = torch.randn(1, 1, 5)
    latent, shapes = utils.pack_latents([points, camera])
    context = torch.randn(1, 5, 12)
    outputs = []
    for reference in (None, [torch.randn(1, 128, 1, 3)]):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=context,
                reference_latents=reference,
                latent_shapes=shapes,
            ).items()
        }
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        unpacked = utils.unpack_latents(result, shapes)
        assert unpacked[0].shape == points.shape and unpacked[1].shape == camera.shape
        outputs.append(unpacked)
    assert not torch.equal(outputs[0][0], outputs[1][0]), (
        "point conditioning was ignored"
    )
    assert not torch.equal(outputs[0][1], outputs[1][1]), (
        "camera conditioning was ignored"
    )
