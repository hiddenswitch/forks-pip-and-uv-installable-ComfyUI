"""Pixel-space generation with native language and vision conditioning."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from comfy.ldm.sensenova import model as sensenova_model
from tests.fast_model_cases import _initialize


def hidream_o1():
    descriptor = supported_models.HiDreamO1(dict(
        image_model="hidream_o1", dtype=torch.float32,
        text_config_overrides=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            rope_dims=[3, 3, 2],
        ),
        vision_config_overrides=dict(
            hidden_size=32, num_heads=2, intermediate_size=64,
            depth=2, deepstack_visual_indexes=(),
        ),
    ))
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    native = model.diffusion_model
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in native.state_dict().items()
    }
    assert supported_models.HiDreamO1.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 3, 64, 96)
    sigma = torch.tensor([0.5])
    options = dict(
        noise=latent, device=torch.device("cpu"),
        text_input_ids=torch.tensor([[1, 2, 3, native.tms_token_id]]),
    )
    outputs = []
    for conditioning in ("text", "changed_text", "reference"):
        if conditioning == "changed_text":
            options["text_input_ids"] = torch.tensor([[4, 5, 6, native.tms_token_id]])
        if conditioning == "reference":
            options["reference_latents"] = [torch.rand(1, 32, 32, 3)]
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        with torch.no_grad():
            result = model.apply_model(latent, sigma, **conditions)
            repeat = model.apply_model(latent, sigma, **conditions)
            assert native._kv_cache_entries
            native.clear_kv_cache()
            fresh = model.apply_model(latent, sigma, **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        torch.testing.assert_close(result, fresh)
        if outputs:
            assert not torch.equal(outputs[-1], result), f"{conditioning} was ignored"
        outputs.append(result)
    native.clear_kv_cache()
    assert not native._kv_cache_entries and native._visual_cache is None
    with pytest.raises(ValueError, match="requires input_ids and position_ids"):
        model.apply_model(latent, sigma)


def sensenova():
    with pytest.MonkeyPatch.context() as patch:
        for name, value in dict(
            HIDDEN_SIZE=32, INTERMEDIATE_SIZE=64, NUM_LAYERS=2,
            NUM_HEADS=2, NUM_KV_HEADS=1, HEAD_DIM=16,
        ).items():
            patch.setattr(sensenova_model, name, value)
        descriptor = supported_models.SenseNovaU15(dict(
            image_model="sensenova_u15", dtype=torch.float32,
        ))
        descriptor.custom_operations = ops.manual_cast
        model = descriptor.get_model({}, device=torch.device("cpu"))
        # Preserve the native pixel-shuffle decoder with widths matching the
        # reduced language model; its constructor fixes released-model widths.
        head = model.diffusion_model.fm_modules["fm_head"]
        head.conv1 = ops.manual_cast.Conv2d(8, 32, 3, padding=1)
        head.conv2 = ops.manual_cast.Conv2d(8, 192, 3, padding=1)
        _initialize(model)
        state = {
            key: torch.empty(value.shape, device="meta")
            for key, value in model.diffusion_model.state_dict().items()
        }
        # Detection recognizes the released query width, independent of packing.
        key = "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"
        state[key] = torch.empty(4096, 32, device="meta")
        assert supported_models.SenseNovaU15.matches(model_detection.detect_unet_config(state, ""))
        latent = torch.randn(1, 3, 35, 49)
        sigma = torch.tensor([0.5])
        outputs = []
        for references in (None, [torch.rand(1, 19, 35, 3)]):
            variants = []
            for precompute in (False, True):
                options = dict(
                    noise=latent, device=torch.device("cpu"),
                    text_input_ids=torch.tensor([[1, 2, 3]]),
                    reference_latents=references,
                )
                if not precompute:
                    options["hooks"] = object()
                with torch.no_grad():
                    conditions = {
                        key: value.process_cond(batch_size=1, area=None).cond
                        for key, value in model.extra_conds(**options).items()
                    }
                    assert ("prefix_keys" in conditions) == precompute
                    result = model.apply_model(latent, sigma, **conditions)
                    repeat = model.apply_model(latent, sigma, **conditions)
                assert result.shape == latent.shape and result.dtype == latent.dtype
                assert torch.isfinite(result).all()
                torch.testing.assert_close(result, repeat)
                variants.append(result)
            torch.testing.assert_close(*variants)
            outputs.append(variants[0])
        assert not torch.equal(*outputs), "reference was ignored"
        with pytest.raises(ValueError, match="requires text conditioning"):
            model.apply_model(latent, sigma)
