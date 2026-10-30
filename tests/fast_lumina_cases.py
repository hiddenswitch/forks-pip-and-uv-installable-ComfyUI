"""Native Lumina and Z Image attention, padding and pixel decoder contracts."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def lumina(model_name):
    model_class = getattr(supported_models, model_name)
    z_image = model_name != "Lumina2"
    pixel = model_name == "ZImagePixelSpace"
    ming = model_name == "MingImage"
    config = dict(model_class.unet_config)
    config.update(
        dim=256 if z_image else 32,
        in_channels=3 if pixel else 16,
        n_layers=2,
        n_refiner_layers=2,
        n_heads=2,
        n_kv_heads=2,
        multiple_of=8,
        ffn_dim_multiplier=2,
        cap_feat_dim=12,
        axes_dims=[32, 48, 48] if z_image else [4, 6, 6],
        axes_lens=[32, 32, 32],
        qk_norm=True,
        z_image_modulation=z_image,
        time_scale=1000 if z_image else 1,
        pad_tokens_multiple=4 if z_image and not ming else None,
        masked_pad_multiple=4 if ming else None,
        dtype=torch.float32,
    )
    if pixel:
        config.update(decoder_hidden_size=16, decoder_num_res_blocks=2, decoder_max_freqs=2)
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    # Detection identifies these releases by checkpoint width; forward uses a tiny width.
    state["cap_embedder.1.weight"] = torch.empty(3840 if z_image else 2304, 12, device="meta")
    detected = model_detection.detect_unet_config(state, "")
    assert model_class.matches(detected)
    assert detected["n_layers"] == 2
    if pixel:
        assert detected["decoder_hidden_size"] == 16
        assert detected["decoder_max_freqs"] == 2
    for key in model_class.unet_config:
        incomplete = dict(model_class.unet_config)
        del incomplete[key]
        assert not model_class.matches(incomplete)

    latent = torch.randn(1, config["in_channels"], 3, 5)
    if ming:
        latent = latent.unsqueeze(2).repeat(1, 1, 2, 1, 1)
    context = torch.randn(1, 3, 12)
    outputs = []
    for conditioned in (False, True):
        options = {}
        if conditioned and ming:
            options.update(
                reference_latents=[torch.randn(1, 16, 2, 3, 5)],
                direct_context=torch.randn(1, 2, config["dim"]),
            )
        elif conditioned and model_name == "ZImage":
            options.update(
                reference_latents=[torch.randn(1, 16, 4, 6)],
                reference_latents_text_embeds=[context],
            )
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=context + int(conditioned),
                attention_mask=torch.tensor([[1, 1, 0 if conditioned else 1]]),
                **options,
            ).items()
        }
        assert conditions["num_tokens"] == (2 if conditioned else 3)
        assert ("attention_mask" in conditions) == conditioned
        if ming and conditioned:
            assert len(conditions["ref_frames"]) == 2
            assert conditions["direct_context"].shape == (1, 2, config["dim"])
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"
