"""Native image transformers with reduced dimensions and real conditioning."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def image_transformer(model_name):
    model_class = getattr(supported_models, model_name)
    config = dict(model_class.unet_config)
    if model_name == "MageFlow":
        config.update(
            in_channels=128,
            out_channels=128,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=2560,
            axes_dims_rope=(4, 6, 6),
        )
    elif model_name == "JoyImage":
        config.update(
            in_channels=16,
            out_channels=16,
            hidden_size=32,
            num_attention_heads=2,
            text_dim=12,
            mlp_width_ratio=2,
            num_layers=2,
        )
    elif model_name == "ErnieImage":
        config.update(
            hidden_size=32,
            num_attention_heads=2,
            num_layers=2,
            ffn_hidden_size=64,
            in_channels=128,
            out_channels=128,
            text_in_dim=12,
            rope_axes_dim=(4, 6, 6),
        )
    config["dtype"] = torch.float32
    descriptor = model_class(config)
    if model_name == "JoyImage":
        descriptor.unet_config["rope_dim_list"] = [4, 6, 6]
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(
        (1, 16, 1, 3, 5) if model_name == "JoyImage" else (1, 128, 3, 5)
    )
    context = torch.randn(1, 3, 2560 if model_name == "MageFlow" else 12)
    outputs = []
    for conditioned in (False, True):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context)
        if conditioned:
            options["cross_attn"] = context + 1
            if model_name != "ErnieImage":
                options["reference_latents"] = [torch.randn_like(latent)]
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


def krea2():
    config = dict(supported_models.Krea2.unet_config)
    config.update(features=256, heads=2, kvheads=1, layers=2, tdim=16,
                  txtdim=32, txtlayers=2, txtheads=2, txtkvheads=1,
                  multiplier=2, dtype=torch.float32)
    descriptor = supported_models.Krea2(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    detected = model_detection.detect_unet_config(state, "")
    assert supported_models.Krea2.matches(detected)
    for key in ("features", "heads", "kvheads", "layers", "txtdim", "txtlayers"):
        assert detected[key] == config[key]
    latent = torch.randn(1, 16, 3, 5)
    context = torch.randn(1, 3, 64)
    references = [torch.randn(1, 16, 3, 3), torch.randn(1, 16, 2, 5)]
    outputs = []
    for method in (None, "index", "index_timestep_zero"):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context)
        if method is not None:
            options.update(reference_latents=references, reference_latents_method=method)
        conditions = {key: value.process_cond(batch_size=1, area=None).cond
                      for key, value in model.extra_conds(**options).items()}
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    assert not torch.equal(outputs[0], outputs[1]), "references were ignored"
    assert not torch.equal(outputs[1], outputs[2]), "reference timestep was ignored"
    with pytest.raises(ValueError, match="Krea2 expects conditioning"):
        model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context[..., :-1])


def moss_image():
    config = dict(supported_models.MOSSImage.unet_config)
    config.update(dim=32, ffn_dim=64, num_heads=2, num_layers=2,
                  text_dim=12, latent_channels=128, dtype=torch.float32)
    descriptor = supported_models.MOSSImage(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.MOSSImage.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 128, 3, 5)
    context = torch.randn(1, 3, 12)
    references = [torch.randn(1, 128, 2, 3), torch.randn(1, 128, 3, 2)]
    options = dict(noise=latent, device=torch.device("cpu"), cross_attn=context,
                   reference_latents=references)
    outputs = []
    for precompute in (False, True):
        with torch.inference_mode(precompute):
            conditions = {key: value.process_cond(batch_size=1, area=None).cond
                          for key, value in model.extra_conds(**options).items()}
            assert ("ref_kv" in conditions) == precompute
            with torch.no_grad():
                result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
                repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            assert result.shape == latent.shape and result.dtype == latent.dtype
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, repeat)
            outputs.append(result.clone())
    torch.testing.assert_close(*outputs)
    with torch.no_grad():
        plain = model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context)
        weighted = model.apply_model(latent, torch.tensor([0.5]), c_crossattn=context,
                                     context_attention_bias=torch.tensor([[0., -10., 0.]]))
    assert not torch.equal(plain, outputs[0]), "references were ignored"
    assert not torch.equal(plain, weighted), "text attention bias was ignored"


def flux3():
    latent = torch.randn(1, 128, 3, 5)
    context = torch.randn(1, 3, 20480)
    references = [torch.randn(1, 128, 2, 3)]
    for flow_map in (False, True):
        for fused_qkv in (False, True):
            config = dict(supported_models.Flux3.unet_config)
            config.update(hidden_size=128, num_heads=1, depth=2, depth_single_blocks=2,
                          fused_qkv=fused_qkv, flow_map=flow_map, dtype=torch.float32)
            descriptor = supported_models.Flux3(config)
            descriptor.custom_operations = ops.manual_cast
            model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
            state = {key: torch.empty(value.shape, device="meta")
                     for key, value in model.diffusion_model.state_dict().items()}
            detected = model_detection.detect_unet_config(state, "")
            assert supported_models.Flux3.matches(detected)
            assert detected["fused_qkv"] == fused_qkv
            assert detected["flow_map"] == flow_map
            outputs = []
            for precompute in (False, True):
                with torch.inference_mode(precompute):
                    options = dict(noise=latent, device=torch.device("cpu"),
                                   cross_attn=context, reference_latents=references)
                    conditions = {key: value.process_cond(batch_size=1, area=None).cond
                                  for key, value in model.extra_conds(**options).items()}
                    assert ("txt_states" in conditions) == (precompute and not flow_map)
                    with torch.no_grad():
                        result = model.apply_model(
                            latent, torch.tensor([0.5]), **conditions,
                            transformer_options={"sample_sigmas": torch.tensor([1., .5, 0.])})
                    assert result.shape == latent.shape and result.dtype == latent.dtype
                    assert torch.isfinite(result).all()
                    outputs.append(result.clone())
            torch.testing.assert_close(*outputs)
            with torch.no_grad():
                changed = model.apply_model(
                    latent, torch.tensor([0.5]), c_crossattn=context + 1,
                    transformer_options={"sample_sigmas": torch.tensor([1., .5, .25, 0.])})
            assert not torch.equal(changed, outputs[0]), "conditioning was ignored"
            if flow_map:
                with torch.no_grad():
                    changed_target = model.apply_model(
                        latent, torch.tensor([0.5]), **conditions,
                        transformer_options={"sample_sigmas": torch.tensor([1., .5, .25, 0.])})
                assert not torch.equal(changed_target, outputs[0]), "target timestep was ignored"


def ideogram4():
    descriptor = supported_models.Ideogram4(dict(supported_models.Ideogram4.unet_config))
    descriptor.unet_config.update(
        num_layers=2, num_attention_heads=2, attention_head_dim=32,
        intermediate_size=128, adaln_dim=16, llm_features_dim=12,
        mrope_section=[4, 2, 2], dtype=torch.float32)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.diffusion_model.state_dict().items()}
    assert supported_models.Ideogram4.matches(model_detection.detect_unet_config(state, ""))
    latent = torch.randn(1, 128, 3, 5)
    context = torch.randn(1, 3, 12)
    changed = context.clone()
    changed[:, -1] += 100
    outputs = []
    for text in (None, torch.zeros_like(context), context, changed):
        options = dict(noise=latent, device=torch.device("cpu"), cross_attn=text,
                       attention_mask=torch.tensor([[1, 1, 0]]))
        conditions = {key: value.process_cond(batch_size=1, area=None).cond
                      for key, value in model.extra_conds(**options).items()}
        with torch.no_grad():
            result = model.apply_model(latent, torch.tensor([0.5]), **conditions)
            repeat = model.apply_model(latent, torch.tensor([0.5]), **conditions)
        assert result.shape == latent.shape and result.dtype == latent.dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat)
        outputs.append(result)
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[2], outputs[3])
    assert not torch.equal(outputs[0], outputs[2]), "text conditioning was ignored"
