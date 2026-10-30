"""CogVideoX spatial/temporal patching and image-conditioning contracts."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def cogvideo(model_name):
    model_class = getattr(supported_models, model_name)
    for temporal_patch in (None, 2):
        config = dict(model_class.unet_config)
        config.setdefault("in_channels", 16)
        config.update(
            out_channels=16,
            num_attention_heads=3,
            attention_head_dim=64,
            time_embed_dim=32,
            text_embed_dim=12,
            num_layers=2,
            sample_width=6,
            sample_height=4,
            sample_frames=9,
            max_text_seq_length=4,
            patch_size=2,
            patch_size_t=temporal_patch,
            use_rotary_positional_embeddings=temporal_patch is not None,
            ofs_embed_dim=32 if temporal_patch is not None else None,
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
        detected = model_detection.detect_unet_config(state, "")
        assert model_class.matches(detected)
        assert detected.get("patch_size_t") == temporal_patch
        latent = torch.randn(1, 16, 3, 3, 5)
        context = torch.randn(1, 4, 12)
        outputs = []
        for conditioned in (False, True):
            options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context)
            if conditioned:
                options.update(
                    cross_attn=context + 1,
                    concat_latent_image=torch.randn(
                        1, 16, 1 if temporal_patch else 5, 2, 3
                    ),
                    ofs=torch.tensor([3.0]),
                )
            conditions = {
                key: value.process_cond(batch_size=1, area=None).cond
                for key, value in model.extra_conds(**options).items()
            }
            extra_channels = config["in_channels"] - 16
            if extra_channels:
                assert conditions["c_concat"].shape == (1, extra_channels, 3, 3, 5)
                if not conditioned:
                    assert torch.count_nonzero(conditions["c_concat"]) == 0
            if temporal_patch:
                torch.testing.assert_close(
                    conditions["ofs"], torch.tensor([3.0 if conditioned else 2.0])
                )
            with torch.no_grad():
                result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
                repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            assert result.shape == latent.shape and result.dtype == latent.dtype
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, repeat)
            outputs.append(result)
        assert not torch.equal(*outputs), "conditioning was ignored"
