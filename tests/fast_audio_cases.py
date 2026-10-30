"""Stable Audio contracts with native duration and inpainting conditioning."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from comfy.ldm.audio.embedders import NumberConditioner
from comfy.ldm.minimax_music import dit as music_dit
from tests.fast_model_cases import _initialize


def stable_audio(model_name):
    version3 = model_name == "StableAudio3"
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    config.update(
        io_channels=8,
        embed_dim=64,
        depth=2,
        num_heads=2,
        cond_token_dim=768,
        project_cond_tokens=True,
        global_cond_dim=768 if version3 else 1536,
        global_cond_type="adaLN" if version3 else "prepend",
        timestep_features_type="expo" if version3 else "learned",
        dtype=torch.float32,
    )
    if version3:
        config.update(local_add_cond_dim=9, num_memory_tokens=2)
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    weights = {}
    names = ["seconds_total"] if version3 else ["seconds_start", "seconds_total"]
    for name in names:
        conditioner = _initialize(
            NumberConditioner(
                768,
                min_val=0,
                max_val=384 if version3 else 512,
                fourier_features_type=config["timestep_features_type"],
            )
        )
        weights.update(
            {
                f"conditioner.conditioners.{name}.{key}": value
                for key, value in conditioner.state_dict().items()
            }
        )
    if version3:
        weights["conditioner.conditioners.prompt.padding_embedding"] = torch.randn(768)
    model = _initialize(descriptor.get_model(weights, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    detected = model_detection.detect_unet_config(state, "")
    assert model_class.matches(detected)
    assert detected["io_channels"] == 8
    latent = torch.randn(1, 8, 7)
    context = torch.randn(1, 2, 768)
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            cross_attn=context,
            seconds_start=0,
            seconds_total=10,
        )
        if conditioned:
            options.update(
                cross_attn=context + 1,
                seconds_start=2,
                seconds_total=20,
                concat_latent_image=torch.randn_like(latent),
                concat_mask=torch.ones(1, 1, 7),
            )
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        assert conditions["c_crossattn"].shape[1] == (257 if version3 else 4)
        assert conditions["global_embed"].shape == (1, 768 if version3 else 1536)
        if version3:
            assert conditions["local_add_cond"].shape == (1, 9, 7)
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"


def ace_step():
    descriptor = supported_models.ACEStep(
        dict(
            audio_model="ace",
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            mlp_ratio=2,
            text_embedding_dim=12,
            lyric_encoder_vocab_size=32,
            lyric_hidden_size=1024,
            ssl_latent_dims=[8, 8],
            max_width=8,
            dtype=torch.float32,
        )
    )
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert supported_models.ACEStep.matches(
        model_detection.detect_unet_config(state, "")
    )
    latent = torch.randn(1, 8, 16, 3)
    context = torch.randn(1, 3, 12)
    outputs = []
    for strength in (0.0, 1.0):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=context,
                conditioning_lyrics=torch.tensor([[1, 2, 3]]),
                lyrics_strength=strength,
            ).items()
        }
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "lyrics strength was ignored"


def minimax_music3():
    descriptor = supported_models.MiniMaxMusic3(
        dict(audio_model="minimax_music3", dtype=torch.float32))
    descriptor.custom_operations = ops.manual_cast
    # This architecture fixes its widths and depth in the constructor. Allocate
    # only two native blocks, retaining the real conditioning and transformer math.
    model = descriptor.get_model({}, device=torch.device("meta"))
    transformer = model.diffusion_model.diffusion_transformer.transformer
    transformer.layers = torch.nn.ModuleList(list(transformer.layers[:2]))
    model.diffusion_model.to_empty(device=torch.device("cpu"))
    _initialize(model)
    for module in model.diffusion_model.modules():
        if isinstance(module, music_dit.LayerNorm):
            module.beta.zero_()
            module.gamma.fill_(1)
    transformer.rotary_pos_emb.inv_freq.copy_(
        1.0 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32)))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.MiniMaxMusic3.matches(model_detection.detect_unet_config(state, ""))
    context = torch.randn(1, 2, 8 * 4096)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(music_dit, "MAX_CONDITION_FRAMES", 2)
        patch.setattr(music_dit, "CONDITION_HOP_FRAMES", 1)
        for length in (3, 9):
            latent = torch.randn(1, 128, length)
            outputs = []
            for strength in (0., 1.):
                conditions = {key: value.process_cond(batch_size=1, area=None).cond
                              for key, value in model.extra_conds(
                                  noise=latent, device=torch.device("cpu"), cross_attn=context,
                                  conditioning_scale=torch.full((1, 1, 1), strength)).items()}
                with torch.no_grad():
                    result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
                    repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
                assert result.shape == latent.shape and result.dtype == latent.dtype
                assert torch.isfinite(result).all()
                torch.testing.assert_close(result, repeat)
                if length == 9:
                    with torch.no_grad():
                        condition = model.diffusion_model.aligned_condition(context) * strength
                        condition = torch.nn.functional.pad(condition, (0, 9 - condition.shape[-1]))
                        first = -model.diffusion_model.diffusion_transformer(
                            latent[..., :6], torch.tensor([0.5]), condition[..., :6])
                        second = -model.diffusion_model.diffusion_transformer(
                            latent[..., 3:], torch.tensor([0.5]), condition[..., 3:])
                        velocity = torch.cat((first[..., :3], (first[..., 3:] + second[..., :3]) / 2,
                                              second[..., 3:]), dim=-1)
                        expected = model.model_sampling.calculate_denoised(torch.tensor([0.5]), velocity, latent)
                    torch.testing.assert_close(result, expected)
                outputs.append(result)
            assert not torch.equal(*outputs), "audio conditioning was ignored"


def ace_step15():
    config = dict(audio_model="ace1.5", hidden_size=32, text_hidden_dim=12,
                  num_dit_layers=2, num_lyric_layers=2, num_timbre_layers=2,
                  num_tokenizer_layers=2, num_heads=2, num_kv_heads=1, head_dim=16,
                  intermediate_size=64, encoder_hidden_size=32,
                  encoder_intermediate_size=64, encoder_num_heads=2,
                  fsq_dim=32, sliding_window=1, dtype=torch.float32)
    descriptor = supported_models.ACEStep15(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.ACEStep15.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 64, 7)
    context = torch.randn(1, 3, 12)
    lyrics = torch.randn(1, 2, 12)
    reference = torch.randn_like(latent)
    outputs = []
    for conditional in (False, True):
        conditions = {key: value.process_cond(batch_size=1, area=None).cond
                      for key, value in model.extra_conds(
                          noise=latent, device=torch.device("cpu"),
                          cross_attn=context if conditional else torch.zeros_like(context),
                          conditioning_lyrics=lyrics,
                          reference_audio_timbre_latents=[reference]).items()}
        assert conditions["is_covers"]
        assert conditions.get("replace_with_null_embeds", False) != conditional
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "null text conditioning was ignored"
    with torch.no_grad():
        coded = model.apply_model(latent, torch.tensor([0.5]), **conditions,
                                  audio_codes=torch.zeros(1, 2, dtype=torch.long))
    assert coded.shape == latent.shape and torch.isfinite(coded).all()
    assert not torch.equal(coded, outputs[1]), "audio codes were ignored"
