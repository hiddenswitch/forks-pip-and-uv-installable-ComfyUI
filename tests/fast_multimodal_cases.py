"""Small native transformers with pooled, multi-layer and image conditioning."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def hidream():
    descriptor = supported_models.HiDream(
        dict(
            image_model="hidream",
            patch_size=2,
            in_channels=16,
            out_channels=16,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            caption_channels=[12, 12],
            text_emb_dim=8,
            num_routed_experts=3,
            num_activated_experts=2,
            axes_dims_rope=[4, 6, 6],
            max_resolution=[8, 8],
            llama_layers=[0, 1, 2, 3],
            dtype=torch.float32,
        )
    )
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert supported_models.HiDream.matches(
        model_detection.detect_unet_config(state, "")
    )
    latent = torch.randn(1, 16, 3, 5)
    context = torch.randn(1, 3, 12)
    llama = torch.randn(1, 4, 2, 12)
    pooled = torch.randn(1, 8)
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            cross_attn=context,
            conditioning_llama3=llama + int(conditioned),
            pooled_output=pooled,
        )
        if conditioned:
            options["concat_latent_image"] = torch.randn_like(latent)
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
    assert not torch.equal(*outputs), "conditioning was ignored"


def kandinsky(model_name):
    image = model_name == "Kandinsky5Image"
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    config.update(
        in_visual_dim=16,
        out_visual_dim=16,
        in_text_dim=12,
        in_text_dim2=8,
        time_dim=16,
        model_dim=32,
        ff_dim=64,
        visual_embed_dim=64 if image else 132,
        patch_size=(1, 2, 2),
        num_text_blocks=2,
        num_visual_blocks=2,
        axes_dims=(4, 6, 6),
        dtype=torch.float32,
    )
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    if image:
        # Image checkpoints are distinguished from video by their native width.
        state["visual_embeddings.in_layer.bias"] = torch.empty(2560, device="meta")
    detected = model_detection.detect_unet_config(state, "")
    assert model_class.matches(detected)
    assert detected["num_visual_blocks"] == 2
    assert detected["visual_embed_dim"] == config["visual_embed_dim"]
    latent = torch.randn((1, 16, 3, 5) if image else (1, 16, 2, 3, 5))
    context = torch.randn(1, 3, 12)
    pooled = torch.randn(1, 8)
    outputs = []
    for conditioned in (False, True):
        options = dict(
            noise=latent,
            device=torch.device("cpu"),
            cross_attn=context + int(conditioned),
            pooled_output=pooled,
        )
        if conditioned and not image:
            options["concat_mask"] = torch.zeros(1, 1, 1, 2, 3)
            options["time_dim_replace"] = torch.randn(1, 16, 1, 3, 5)
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**options).items()
        }
        if not image:
            assert conditions["c_concat"].shape == (1, 17, 2, 3, 5)
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning was ignored"


def omnigen(model_name="Omnigen2"):
    model_class = getattr(supported_models, model_name)
    config = dict(
        image_model="omnigen2",
        patch_size=2,
        in_channels=16,
        out_channels=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=2,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=8,
        ffn_dim_multiplier=2,
        axes_dim_rope=(4, 6, 6),
        axes_lens=(32, 32, 32),
        text_feat_dim=12,
        dtype=torch.float32,
    )
    if model_name == "Boogu":
        config.update(image_model="boogu", num_double_stream_layers=2)
        config["instruction_feat_dim"] = config.pop("text_feat_dim")
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 16, 3, 5)
    context = torch.randn(1, 3, 12)
    outputs = []
    for references in (None, [], [torch.randn(1, 16, 3, 5), torch.randn(1, 16, 2, 3)]):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=context,
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                reference_latents=references,
            ).items()
        }
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    torch.testing.assert_close(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2]), "reference images were ignored"


def mochi():
    # The native image MLP requires width 12,288 even with a reduced token width.
    descriptor = supported_models.GenmoMochi(
        dict(
            image_model="mochi_preview",
            patch_size=2,
            in_channels=12,
            hidden_size_x=32,
            hidden_size_y=16,
            depth=2,
            num_heads=2,
            mlp_ratio_x=384,
            mlp_ratio_y=2,
            use_t5=True,
            t5_feat_dim=16,
            t5_token_length=3,
            learn_sigma=False,
            qk_norm=True,
            dtype=torch.float32,
        )
    )
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert supported_models.GenmoMochi.matches(
        model_detection.detect_unet_config(state, "")
    )
    latent = torch.randn(1, 12, 2, 4, 6)
    context = torch.randn(1, 3, 16)
    outputs = []
    for mask in (torch.tensor([[1, 1, 1]]), torch.tensor([[1, 1, 0]])):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=context,
                attention_mask=mask,
            ).items()
        }
        assert conditions["num_tokens"] == mask.sum().item()
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "text mask was ignored"
