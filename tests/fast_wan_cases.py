"""Tiny video forwards through the registered Wan wrappers."""

from types import SimpleNamespace

import torch
import pytest

from comfy import model_detection, model_management, ops, supported_models
from comfy.ldm.wan.model_animate2 import PoseBranchCache
from tests.fast_model_cases import _initialize


def wan_video(model_name):
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    config.update(
        dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=12,
        text_len=3,
        num_heads=2,
        num_layers=2,
        dtype=torch.float32,
    )
    config.setdefault("out_dim", 16)
    if model_name == "WAN21_FlowRVS":
        config["in_dim"] = 36
    if model_name in ("WAN21_SCAIL", "WAN21_SCAIL2"):
        config["in_dim"] = 20
    config.setdefault("in_dim", config["out_dim"])
    if model_name == "WAN21_Vace":
        config.update(vace_layers=1, vace_in_dim=96)
    if model_name == "WAN22_WanDancer":
        config.update(
            music_dim=24, music_heads=2, music_latent_dim=24,
            audio_inject_layers=[0, 1],
        )
    if model_name == "WAN22_Animate":
        config.update(num_layers=5, in_dim=36)
    if model_name == "WAN_Animate2":
        config["in_dim"] = 36
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    detected = model_detection.detect_unet_config(state, "")
    if model_name == "WAN_Animate2":
        assert supported_models.WAN21_I2V.matches(detected), detected
        detected = model_detection.detect_unet_config(
            state, "", metadata={"config": '{"transformer":{"model_type":"animate2"}}'}
        )
    if model_name == "WAN21_FlowRVS":
        # FlowRVS changes sampling; its transformer uses the T2V checkpoint layout.
        assert supported_models.WAN21_T2V.matches(detected), detected
    else:
        assert model_class.matches(detected), detected
    width = 15 if model_name == "WAN22_WanDancer" else 5
    latent = torch.randn(1, config["out_dim"], 2, 3, width)
    context = torch.randn(1, 3, 12)
    outputs = []
    for conditioned in (False, True):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context)
        if model.diffusion_model.img_emb is not None:
            options["clip_vision_output"] = SimpleNamespace(
                penultimate_hidden_states=torch.randn(1, 2, 1280)
            )
        if conditioned:
            options["cross_attn"] = context + 1
            extra_channels = config["in_dim"] - config["out_dim"]
            if extra_channels > 4:
                image_channels = (
                    extra_channels - 4 if extra_channels % 16 else extra_channels
                )
                options["concat_latent_image"] = torch.randn(1, image_channels, 2, 2, 3)
                options["concat_mask"] = torch.ones(1, 1, 1, 2, 3)
                if model_name == "WAN22_WanDancer":
                    options["concat_latent_image"] = torch.randn(1, image_channels, 2, 3, width)
                    options["concat_mask"] = torch.ones(1, 1, 1, 3, width)
            if "Camera" in model_name:
                options["camera_conditions"] = torch.randn(1, 24, 2, 32, 48)
            if model_name == "WAN22_T2V":
                options["denoise_mask"] = (
                    torch.tensor([0.0, 1.0])
                    .reshape(1, 1, 2, 1, 1)
                    .expand(1, 1, 2, 3, 5)
                )
            if model_name == "WAN21_Vace":
                options.update(
                    vace_frames=[torch.randn(1, 32, 2, 3, 5) for _ in range(2)],
                    vace_mask=[torch.ones(1, 64, 2, 3, 5) for _ in range(2)],
                    vace_strength=[0.25, 0.75],
                )
            if model_name == "WAN22_WanDancer":
                options.update(
                    audio_embed=torch.randn(1, 5, 35),
                    clip_vision_output_ref=SimpleNamespace(
                        penultimate_hidden_states=torch.randn(1, 2, 1280)
                    ),
                    fps=24,
                )
            if model_name == "WAN22_Animate":
                options.update(
                    pose_video_latent=torch.randn(1, 16, 1, 4, 6),
                    face_video_pixels=torch.randn(1, 3, 1, 512, 512),
                )
            if model_name == "WAN_Animate2":
                options.update(
                    pose_video_latent=torch.randn(1, 16, 1, 3, 5),
                    cross_attn_pose=torch.randn(1, 3, 12),
                    clip_vision_output_pose=SimpleNamespace(
                        penultimate_hidden_states=torch.randn(1, 2, 1280)
                    ),
                )
            if model_name in ("WAN21_SCAIL", "WAN21_SCAIL2"):
                options.update(
                    reference_latents=[torch.randn(1, 16, 1, 3, 5) for _ in range(2)],
                    pose_video_latent=torch.randn(1, 16, 2, 2, 3),
                    concat_mask=torch.zeros(1, 1, 1, 2, 3),
                )
            if model_name == "WAN21_SCAIL2":
                options.update(
                    driving_mask_28ch=torch.ones(1, 2, 28, 2, 3),
                    ref_mask_28ch=torch.ones(1, 4, 28, 3, 5),
                    ref_mask_flag=False,
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
        outputs.append(result)
        if conditioned and model_name == "WAN_Animate2":
            with torch.no_grad():
                for key in ("pose_strength", "reference_strength"):
                    changed = model.apply_model(
                        latent, torch.tensor([0.5]), **{**conditions, key: 0.0}
                    )
                    assert changed.shape == latent.shape and torch.isfinite(changed).all()
                    assert not torch.equal(result, changed), f"{key} was ignored"
                cache = PoseBranchCache(store_device="cpu")
                try:
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(model_management, "MAX_PINNED_MEMORY", 0)
                        for _ in range(2):
                            cached = model.apply_model(
                                latent, torch.tensor([0.5]),
                                transformer_options={"animate2_cache": cache}, **conditions,
                            )
                            assert cache.filled(2) and cache.memory_bytes() > 0
                            torch.testing.assert_close(result, cached)
                finally:
                    cache.free()
                assert cache.memory_bytes() == 0
                with pytest.raises(ValueError, match="generation frames minus"):
                    model.apply_model(
                        latent, torch.tensor([0.5]),
                        **{**conditions, "pose_latents": torch.randn(1, 16, 2, 3, 5)},
                    )
        if conditioned and model_name == "WAN22_Animate":
            with torch.no_grad():
                no_face = model.apply_model(
                    latent, torch.tensor([0.5]), **{**conditions, "face_pixel_values": None}
                )
                no_pose = model.apply_model(
                    latent, torch.tensor([0.5]), **{**conditions, "pose_latents": None}
                )
            for changed in (no_face, no_pose):
                assert changed.shape == latent.shape and torch.isfinite(changed).all()
                assert not torch.equal(result, changed)
        if conditioned and model_name == "WAN22_WanDancer":
            with torch.no_grad():
                muted = model.apply_model(
                    latent, torch.tensor([0.5]), audio_inject_scale=0.0, **conditions
                )
                normal_fps = model.apply_model(
                    latent, torch.tensor([0.5]), **{**conditions, "fps": 30}
                )
            for changed in (muted, normal_fps):
                assert changed.shape == latent.shape and torch.isfinite(changed).all()
                assert not torch.equal(result, changed)
        if conditioned and model_name == "WAN21_SCAIL2":
            conditions["ref_mask_flag"] = True
            with torch.no_grad():
                animation = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            assert animation.shape == latent.shape and torch.isfinite(animation).all()
            assert not torch.equal(result, animation)
    assert not torch.equal(*outputs), "conditioning was ignored"


def wan_humo():
    for in_dim in (16, 36):
        config = dict(supported_models.WAN21_HuMo.unet_config)
        config.update(
            dim=24, ffn_dim=48, freq_dim=8, text_dim=12, text_len=3,
            num_heads=2, num_layers=2, in_dim=in_dim, out_dim=16,
            dtype=torch.float32,
        )
        descriptor = supported_models.WAN21_HuMo(config)
        descriptor.custom_operations = ops.manual_cast
        model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
        state = {
            key: torch.empty(value.shape, device="meta")
            for key, value in model.diffusion_model.state_dict().items()
        }
        assert supported_models.WAN21_HuMo.matches(
            model_detection.detect_unet_config(state, "")
        )
        latent = torch.randn(1, 16, 2, 3, 5)
        options = dict(
            noise=latent, device=torch.device("cpu"),
            cross_attn=torch.randn(1, 3, 12),
        )
        outputs = []
        for conditioning in ("text", "audio", "reference"):
            if conditioning == "audio":
                options["audio_embed"] = torch.randn(1, 2, 8, 5, 1280)
            if conditioning == "reference":
                options["reference_latents"] = [torch.randn(1, 16, 1, 4, 6)]
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
        assert not torch.equal(outputs[0], outputs[1]), "audio was ignored"
        assert not torch.equal(outputs[1], outputs[2]), "reference was ignored"


def wan_s2v():
    config = dict(supported_models.WAN22_S2V.unet_config)
    config.update(
        dim=24, ffn_dim=48, freq_dim=8, text_dim=12, text_len=3,
        num_heads=2, num_layers=2, in_dim=16, out_dim=16,
        audio_dim=12, audio_inject_layers=[0, 1], dtype=torch.float32,
    )
    descriptor = supported_models.WAN22_S2V(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert supported_models.WAN22_S2V.matches(
        model_detection.detect_unet_config(state, "")
    )
    latent = torch.randn(1, 16, 2, 3, 5)
    options = dict(
        noise=latent, device=torch.device("cpu"),
        cross_attn=torch.randn(1, 3, 12),
    )
    outputs = []
    for conditioning in ("text", "audio", "reference", "control"):
        if conditioning == "audio":
            options["audio_embed"] = torch.randn(1, 25, 12, 8)
        if conditioning == "reference":
            options["reference_latents"] = [torch.randn(1, 16, 1, 4, 6)]
        if conditioning == "control":
            options["control_video"] = torch.randn(1, 16, 2, 4, 6)
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
        if outputs:
            assert not torch.equal(outputs[-1], result), f"{conditioning} was ignored"
        outputs.append(result)


def wan_causal():
    config = dict(supported_models.WAN21_CausalAR_T2V.unet_config)
    config.update(
        dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=12,
        text_len=3,
        num_heads=2,
        num_layers=2,
        in_dim=16,
        out_dim=16,
        dtype=torch.float32,
    )
    descriptor = supported_models.WAN21_CausalAR_T2V(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    native = model.diffusion_model
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in native.state_dict().items()
    }
    # Causal execution uses the same checkpoint layout as the regular T2V model.
    assert supported_models.WAN21_T2V.matches(
        model_detection.detect_unet_config(state, "")
    )
    caches = native.init_kv_caches(1, 12, torch.device("cpu"), torch.float32)
    cross_caches = native.init_crossattn_caches(1, torch.device("cpu"), torch.float32)
    latent = torch.randn(1, 16, 1, 3, 5)
    context = torch.randn(1, 3, 12)
    sigma = torch.tensor([0.5])

    def forward(start):
        return model.apply_model(
            latent,
            sigma,
            c_crossattn=context,
            transformer_options={
                "ar_state": {
                    "start_frame": start,
                    "kv_caches": caches,
                    "crossattn_caches": cross_caches,
                }
            },
        )

    with torch.no_grad():
        ordinary = model.apply_model(latent, sigma, c_crossattn=context)
        first = forward(0)
        assert all(cache["end"] == 6 for cache in caches)
        assert all(cache["is_init"] for cache in cross_caches)
        second = forward(1)
        assert all(cache["end"] == 12 for cache in caches)
        native.reset_kv_caches(caches)
        native.reset_crossattn_caches(cross_caches)
        assert all(cache["end"] == 0 for cache in caches)
        assert all(not cache["is_init"] for cache in cross_caches)
        repeat = forward(0)
    for result in (first, second, repeat):
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
    torch.testing.assert_close(ordinary, first)
    torch.testing.assert_close(first, repeat)
    assert not torch.equal(first, second), "cached history was ignored"
