"""Reduced Hunyuan video/image wrappers with their native conditioning."""

from types import SimpleNamespace

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def hunyuan_video(model_name):
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    image = model_name == "HunyuanImage21"
    refiner = model_name == "HunyuanImage21Refiner"
    video15 = model_name.startswith("HunyuanVideo15")
    sr = model_name == "HunyuanVideo15_SR_Distilled"
    channels = 64 if image or refiner else 32 if video15 else 16
    config.setdefault("in_channels", 128 if refiner else 65 if video15 else channels)
    config.setdefault("vec_in_dim", 4)
    config.setdefault("patch_size", [2, 2] if image else [1, 2, 2])
    config.setdefault("vision_in_dim", None)
    config.update(
        out_channels=channels,
        context_in_dim=12,
        hidden_size=16,
        mlp_ratio=2,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 4] if image else [2, 2, 4],
        theta=10000,
        qkv_bias=True,
        guidance_embed=True,
        byt5=image or refiner or video15,
        meanflow=refiner or sr,
        meanflow_sum=sr,
        use_cond_type_embedding=video15,
        dtype=torch.float32,
    )
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn((1, channels, 4, 6) if image else (1, channels, 2, 4, 6))
    context = torch.randn(1, 3, 12)
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            seed=42,
            cross_attn=context,
            pooled_output=torch.randn(1, 4),
        )
        if model_name.endswith("I2V"):
            options["concat_latent_image"] = torch.zeros_like(latent)
        if conditioned:
            options["cross_attn"] = context + 1
            if refiner:
                options["concat_latent_image"] = torch.randn(1, 32, 3, 2, 3)
                options["noise_augmentation"] = 0.1
            elif sr:
                options["concat_latent_image"] = torch.randn(1, 66, 2, 2, 3)
                options["noise_augmentation"] = 0.1
            elif video15 or model_name.endswith("I2V"):
                options["concat_latent_image"] = torch.randn(1, channels, 2, 2, 3)
                options["concat_mask"] = torch.ones(1, 1, 2, 4, 6)
            if config["byt5"]:
                options["conditioning_byt5small"] = torch.randn(1, 2, 1472)
            else:
                options["guiding_frame_index"] = 0
            if video15:
                options["clip_vision_output"] = SimpleNamespace(
                    last_hidden_state=torch.randn(1, 2, 1152)
                )
        else:
            options["attention_mask"] = torch.tensor([[1, 1, 0]])
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        schedule = {
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([0.5, 0.25, 0.0]),
        }
        with torch.no_grad():
            result = model.apply_model(
                latent, torch.tensor([0.5]), transformer_options=schedule, **conditions
            )
            repeat = model.apply_model(
                latent, torch.tensor([0.5]), transformer_options=schedule, **conditions
            )
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"
