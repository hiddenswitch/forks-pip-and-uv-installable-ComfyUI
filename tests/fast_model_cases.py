"""Small real model forwards with native attention and deterministic weights."""

import torch
import comfy.ops as comfy_ops


def _initialize(model):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(42)
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model.eval().requires_grad_(False)


def _check(model, latent, *args, **kwargs):
    with torch.no_grad():
        result = model(latent, *args, **kwargs)
        repeat = model(latent, *args, **kwargs)
    values = result if isinstance(result, list) else [result]
    inputs = latent if isinstance(latent, list) else [latent]
    assert len(values) == len(inputs)
    for value, original in zip(values, inputs):
        assert value.shape == original.shape
        assert value.dtype == original.dtype
        assert torch.isfinite(value).all()
    torch.testing.assert_close(result, repeat)
    return result


def qwen_image():
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel

    model = _initialize(
        QwenImageTransformer2DModel(
            **{
                "patch_size": 2,
                "in_channels": 4,
                "out_channels": 1,
                "num_layers": 2,
                "attention_head_dim": 6,
                "num_attention_heads": 2,
                "joint_attention_dim": 12,
                "pooled_projection_dim": 12,
                "axes_dims_rope": (2, 2, 2),
                "dtype": torch.float32,
                "device": "cpu",
                "operations": comfy_ops.manual_cast,
            }
        )
    )
    latent = torch.randn(1, 1, 1, 4, 4)
    timestep, context = torch.tensor([0.5]), torch.randn(1, 3, 12)
    result = _check(model, latent, timestep, context)
    changed = _check(model, latent + 1, timestep, context)
    assert not torch.allclose(result, changed)


def _flux(guidance_embed=False, global_modulation=True, txt_norm=True, patch_size=1):
    from comfy.ldm.flux.model import Flux

    model = _initialize(
        Flux(
            **{
                "in_channels": 4,
                "out_channels": 4,
                "vec_in_dim": 4,
                "context_in_dim": 8,
                "hidden_size": 12,
                "mlp_ratio": 2.0,
                "num_heads": 2,
                "depth": 2,
                "depth_single_blocks": 2,
                "axes_dim": [2, 2, 2],
                "theta": 10000,
                "patch_size": patch_size,
                "qkv_bias": True,
                "guidance_embed": guidance_embed,
                "txt_ids_dims": [],
                "global_modulation": global_modulation,
                "txt_norm": txt_norm,
                "dtype": torch.float32,
                "device": "cpu",
                "operations": comfy_ops.manual_cast,
            }
        )
    )
    latent = torch.randn(1, 4, 4, 4)
    timestep, context, pooled = (
        torch.tensor([0.5]),
        torch.randn(1, 3, 8),
        torch.randn(1, 4),
    )
    result = _check(model, latent, timestep, context, pooled, guidance=torch.ones(1))
    changed = _check(
        model, latent + 1, timestep, context, pooled, guidance=torch.ones(1)
    )
    assert not torch.allclose(result, changed)


def minimax_h3():
    from comfy.ldm.minimax.model import MiniMaxH3Model

    model = _initialize(
        MiniMaxH3Model(
            **{
                "hidden_size": 12,
                "num_layers": 2,
                "token_refiner_num_layers": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 6,
                "ffn_hidden_size": 16,
                "latents_dim": 2,
                "audio_latents_dim": 4,
                "patch_size": (1, 2, 2),
                "text_dim": 12,
                "timestep_input_dim": 8,
                "time_embed_hidden_size": 12,
                "time_embed_dim": 6,
                "rope_inv_freq_len": 1,
                "dtype": torch.float32,
                "device": "cpu",
                "operations": comfy_ops.manual_cast,
            }
        )
    )
    video, audio = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 4, 2, 2)
    _check(
        model,
        [video, audio],
        torch.tensor([500.0]),
        torch.randn(1, 3, 12),
        transformer_options={"sample_sigmas": torch.tensor([0.5, 0.25, 0.0])},
    )


def flux2():
    _flux()


def flux():
    _flux(guidance_embed=True, global_modulation=False, txt_norm=False, patch_size=2)


def flux_schnell():
    _flux(guidance_embed=False, global_modulation=False, txt_norm=False, patch_size=2)


def stable_diffusion(model_name):
    """Instantiate each registered wrapper with small spatial/hidden dimensions."""
    from comfy import supported_models

    model_class = getattr(supported_models, model_name)
    signature = model_class.unet_config.copy()
    assert model_class.matches(signature)
    for key in signature:
        incomplete = signature.copy()
        del incomplete[key]
        assert not model_class.matches(incomplete), key
    config = dict(signature)
    config.update(
        image_size=8,
        model_channels=64,
        out_channels=4,
        num_res_blocks=1,
        channel_mult=(1, 2),
        use_spatial_transformer=True,
        transformer_depth=[1, 1],
        transformer_depth_output=[1, 1, 1, 1],
        transformer_depth_middle=1,
        dtype=torch.float32,
    )
    config.setdefault("in_channels", 4)
    config.setdefault("context_dim", 8)
    if config.get("adm_in_channels") is not None:
        config["num_classes"] = "sequential"
    temporal = config.get("use_temporal_attention", False)
    if temporal:
        config.update(
            merge_strategy="learned_with_images",
            merge_factor=0.5,
            video_kernel_size=[3, 1, 1],
            use_spatial_context=True,
            extra_ff_mix_layer=True,
        )
    descriptor = model_class(config)
    if "disable_self_attentions" in descriptor.unet_config:
        descriptor.unet_config["disable_self_attentions"] = [True, False]
    descriptor.custom_operations = comfy_ops.manual_cast
    weights = {}
    if model_name == "Stable_Zero123":
        weights = {
            "cc_projection.weight": torch.randn(768, 772) * 0.01,
            "cc_projection.bias": torch.zeros(768),
        }
        assert model_class.matches(signature, weights)
        for key in weights:
            assert not model_class.matches(
                signature, {k: v for k, v in weights.items() if k != key}
            )
    model = _initialize(descriptor.get_model(weights, device=torch.device("cpu")))
    assert model.model_config is descriptor
    assert type(model.model_config) is model_class
    from comfy import model_detection

    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    detected = model_detection.detect_unet_config(state, "")
    for key in ("in_channels", "out_channels", "model_channels", "context_dim"):
        assert detected[key] == config[key], (model_name, key)
    latent = torch.randn(2 if temporal else 1, 4, 4, 6)
    context_width = 772 if model_name == "Stable_Zero123" else config["context_dim"]
    arguments = dict(
        noise=latent,
        device=torch.device("cpu"),
        seed=42,
        width=48,
        height=32,
        pooled_output=torch.randn(1, 1280),
        cross_attn=torch.randn(1, 3, context_width),
    )
    for conditioned in (False, True):
        options = arguments.copy()
        if conditioned:
            options.update(
                concat_latent_image=torch.randn_like(latent),
                concat_image=torch.randn(1, 3, 2, 3),
                noise_augmentation=0.1,
            )
        conditions = {
            key: value.process_cond(batch_size=latent.shape[0], area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        if temporal:
            conditions["image_only_indicator"] = torch.full((1, 2), conditioned)
        sigma = torch.full((latent.shape[0],), 0.5)
        with torch.no_grad():
            result = model.apply_model(latent, sigma, **conditions)
            repeated = model.apply_model(latent, sigma, **conditions)
        assert result.shape == latent.shape
        assert result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeated)


def image_dit(model_name):
    from comfy import supported_models, model_detection

    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    config.update(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=24,
        depth=4,
        num_heads=2,
        mlp_ratio=2,
        dtype=torch.float32,
    )
    pixart = model_name.startswith("PixArt")
    if pixart:
        config.update(
            caption_channels=16,
            model_max_length=4,
            micro_condition=model_name == "PixArtAlpha",
            pe_interpolation=1,
        )
    else:
        config.update(
            hidden_size=32,
            text_states_dim=16,
            text_states_dim_t5=16,
            text_len=3,
            text_len_t5=4,
            size_cond=model_name == "HunyuanDiT1",
            use_style_cond=model_name == "HunyuanDiT1",
        )
    descriptor = model_class(config)
    descriptor.custom_operations = comfy_ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    if model_name == "HunyuanDiT1":
        # Detection identifies v1 by the checkpoint's full conditioning width.
        state["extra_embedder.0.weight"] = torch.empty(32, 3968, device="meta")
    assert (
        model_detection.detect_unet_config(state, "")["image_model"]
        == config["image_model"]
    )
    latent = torch.randn(1, 4, 34, 36) if pixart else torch.randn(1, 4, 4, 6)
    options = dict(
        noise=latent,
        device=torch.device("cpu"),
        width=48,
        height=32,
        cross_attn=torch.randn(1, 4 if pixart else 3, 16),
    )
    if not pixart:
        options.update(
            conditioning_mt5xl=torch.randn(1, 4, 16),
            attention_mask=torch.tensor([[True, True, False]]),
            attention_mask_mt5xl=torch.tensor([[True, True, True, False]]),
        )
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
