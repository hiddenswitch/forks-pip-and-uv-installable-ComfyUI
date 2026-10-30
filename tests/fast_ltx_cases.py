"""Native LTX video forwards with small grids and guide conditioning."""

import pytest
import torch

from comfy import model_detection, ops, supported_models, utils
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from tests.fast_model_cases import _initialize


def ltx_video():
    config = dict(supported_models.LTXV.unet_config)
    config.update(in_channels=128, cross_attention_dim=32, attention_head_dim=16,
                  num_attention_heads=2, caption_channels=12, num_layers=2,
                  use_keyframes_abs_pos_embedding=True, dtype=torch.float32)
    descriptor = supported_models.LTXV(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.LTXV.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 128, 2, 3, 5)
    context = torch.randn(1, 3, 12)
    mask = torch.ones(1, 1, 2, 3, 5)
    mask[:, :, -1] = 0
    coords = model.diffusion_model.patchifier.patchify(latent[:, :, :1])[1]
    keyframes = latent_to_pixel_coords(coords, model.diffusion_model.vae_scale_factors, causal_fix=False)
    outputs = []
    for guide_strength in (0., 1.):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context,
                       attention_mask=torch.tensor([[1, 1, 0]]), frame_rate=12,
                       denoise_mask=mask, keyframe_idxs=keyframes,
                       guide_attention_entries=[dict(pre_filter_count=15, strength=guide_strength)])
        conditions = {key: value.process_cond(batch_size=1, area=None).cond
                      for key, value in model.extra_conds(**options).items()}
        timestep = model.process_timestep(torch.tensor([0.5]), latent, **conditions)
        torch.testing.assert_close(timestep.flatten(), mask.flatten() * .5)
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "guide attention strength was ignored"
    with torch.no_grad():
        plain = model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context, attention_mask=None)
        changed = model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context.flip(-1), attention_mask=None)
    assert torch.isfinite(plain).all() and torch.isfinite(changed).all()
    assert not torch.equal(plain, changed), "text conditioning was ignored"
    with pytest.raises(ValueError, match="whole number"):
        model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context,
                          attention_mask=None, denoise_mask=mask, keyframe_idxs=keyframes[:, :, :-1])


def ltx_audio_video():
    config = dict(supported_models.LTXAV.unet_config)
    config.update(in_channels=128, audio_in_channels=128, cross_attention_dim=32,
                  audio_cross_attention_dim=32, attention_head_dim=16,
                  audio_attention_head_dim=16, num_attention_heads=2,
                  audio_num_attention_heads=2, caption_channels=32, num_layers=2,
                  connector_attention_head_dim=16, connector_num_attention_heads=2,
                  connector_num_layers=2, rope_type="split", dtype=torch.float32)
    descriptor = supported_models.LTXAV(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.LTXAV.matches(model_detection.detect_unet_config(state, ""))
    video = torch.randn(1, 128, 2, 3, 5)
    audio = torch.randn(1, 8, 3, 16)
    latent, shapes = utils.pack_latents([video, audio])
    video_mask = torch.ones_like(video)
    video_mask[:, :, 0] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[:, :, 0] = 0
    masks, _ = utils.pack_latents([video_mask, audio_mask])
    context = torch.randn(1, 3, 32)
    outputs = []
    for reference in (False, True):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context,
                       latent_shapes=shapes, denoise_mask=masks, frame_rate=12)
        if reference:
            options["ref_audio"] = {"tokens": torch.randn(1, 2, 128)}
        conditions = {key: value.process_cond(batch_size=1, area=None).cond
                      for key, value in model.extra_conds(**options).items()}
        assert conditions["c_crossattn"].shape == (1, 1024, 64)
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        unpacked = utils.unpack_latents(result, shapes)
        assert [tuple(x.shape) for x in unpacked] == [tuple(video.shape), tuple(audio.shape)]
        outputs.append(unpacked)
    for plain, referenced in zip(*outputs):
        assert not torch.equal(plain, referenced), "reference audio was ignored"
