"""Registered DiT wrappers with reduced depth and real conditioning paths."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def forward_contract(model_name):
    model_class = getattr(supported_models, model_name)
    signature = model_class.unet_config.copy()
    assert model_class.matches(signature)
    for key in signature:
        incomplete = signature.copy()
        del incomplete[key]
        assert not model_class.matches(incomplete)
    config = dict(signature, dtype=torch.float32)
    if model_name == "AuraFlow":
        config.update(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            dim=32,
            n_layers=4,
            n_double_layers=2,
            n_heads=2,
            global_conddim=32,
            max_seq=16,
        )
    elif model_name == "SD3":
        config.update(
            input_size=8,
            patch_size=2,
            depth=2,
            mlp_ratio=2,
            out_channels=16,
            adm_in_channels=4,
            pos_embed_max_size=8,
            context_embedder_config={
                "target": "torch.nn.Linear",
                "params": {"in_features": 8, "out_features": 128},
            },
        )
    else:
        config.update(
            context_in_dim=8,
            hidden_size=12,
            mlp_ratio=2,
            num_heads=2,
            depth=2,
            depth_single_blocks=2,
            axes_dim=[2, 2, 2],
            theta=10000,
            patch_size=2,
            qkv_bias=True,
            txt_ids_dims=[],
            vec_in_dim=4,
        )
        if model_name in {"Chroma", "ChromaRadiance"}:
            config.update(
                in_channels=64,
                out_channels=64,
                in_dim=64,
                out_dim=12,
                hidden_dim=24,
                n_layers=2,
            )
            if model_name == "ChromaRadiance":
                config.update(
                    in_channels=3,
                    out_channels=3,
                    nerf_hidden_size=8,
                    nerf_mlp_ratio=2,
                    nerf_depth=2,
                    nerf_max_freqs=2,
                    nerf_tile_size=2,
                    nerf_final_head_type="conv",
                    nerf_embedder_dtype=torch.float32,
                    use_x0=False,
                    use_sequential_txt_ids=True,
                )
        else:
            config.update(out_channels=16)
            if model_name == "LongCatImage":
                config.update(signature)
                config["in_channels"] = 16
    descriptor = model_class(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    assert type(model.model_config) is model_class
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, model.latent_format.latent_channels, 3, 5)
    options = dict(
        noise=latent,
        device=torch.device("cpu"),
        cross_attn=torch.randn(
            1, 3, config.get("cond_seq_dim", config.get("context_in_dim", 8))
        ),
        pooled_output=torch.randn(1, 4),
    )
    outputs = []
    for conditioned in (False, True):
        arguments = options.copy()
        if conditioned:
            arguments.update(
                cross_attn=options["cross_attn"] + 1,
                concat_latent_image=torch.randn(1, 16, 2, 3),
                concat_mask=torch.zeros(1, 1, 2, 3),
            )
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(**arguments).items()
        }
        if model_name == "FluxInpaint":
            assert conditions["c_concat"].shape == (1, 80, 3, 5)
            torch.testing.assert_close(
                conditions["c_concat"][:, -64:],
                torch.full((1, 64, 3, 5), 0.0 if conditioned else 1.0),
            )
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(*outputs), "conditioning did not affect the output"
