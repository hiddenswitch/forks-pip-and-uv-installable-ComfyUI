"""Native image perception models use prediction dictionaries, not denoising."""

import pytest
import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def depth_anything3():
    descriptor = supported_models.DepthAnything3(dict(
        image_model="DepthAnything3", backbone_name="vits", out_layers=[0, 3, 7, 11],
        head_dim_in=384, head_features=16, head_out_channels=[16, 16, 16, 16],
        head_output_dim=2, dtype=torch.float32))
    descriptor.custom_operations = ops.manual_cast
    wrapper = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    model = wrapper.diffusion_model
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.state_dict().items()}
    assert supported_models.DepthAnything3.matches(model_detection.detect_unet_config(state, ""))
    pixels = torch.randn(1, 3, 28, 42)
    with torch.no_grad():
        result = model(pixels, export_feat_layers=[0])
        repeat = model(pixels.unsqueeze(1), export_feat_layers=[0])
        changed = model(pixels.flip(-1))
    for key in ("depth", "depth_conf", "sky"):
        assert result[key].shape == (1, 28, 42)
        assert result[key].dtype == pixels.dtype
        assert torch.isfinite(result[key]).all()
        torch.testing.assert_close(result[key], repeat[key])
    assert not torch.equal(result["depth"], changed["depth"]), "image was ignored"
    assert result["aux_features"][0].shape == (1, 1, 2, 3, 384)
    with pytest.raises(AssertionError, match="multiples of 14"):
        model(pixels[..., :-1])


def rtdetr_v4():
    descriptor = supported_models.RT_DETR_v4(dict(
        image_model="RT_DETR_v4", num_classes=4, num_queries=5,
        enc_h=32, dec_h=32, enc_ff=64, dec_ff=64, dtype=torch.float32))
    descriptor.custom_operations = ops.manual_cast
    wrapper = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    model = wrapper.diffusion_model
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)) and module.weight is not None:
            module.weight.fill_(1)
            if module.bias is not None:
                module.bias.zero_()
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    model.decoder.up.fill_(0.5)
    model.decoder.reg_scale.fill_(8)
    model.load_device = torch.device("cpu")
    # The released model fixes its evaluation grid at 640. Rebuild the same
    # native positions and anchors for a small test image.
    model.encoder.eval_spatial_size = model.decoder.eval_spatial_size = (64, 96)
    model.encoder.pos_embed2 = model.encoder._build_pe(3, 2, 32)
    model.decoder.anchors, model.decoder.valid_mask = model.decoder._gen_anchors()
    state = {key: torch.empty(value.shape, device="meta")
             for key, value in model.state_dict().items()}
    assert supported_models.RT_DETR_v4.matches(model_detection.detect_unet_config(state, ""))
    pixels = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        result = model(pixels, orig_size=(96, 64))
        repeat = model(pixels, orig_size=(96, 64))
        scaled = model(pixels, orig_size=(192, 128))
    assert len(result) == 2
    assert not torch.equal(result[0]["scores"], result[1]["scores"]), "image was ignored"
    for sample, again, enlarged in zip(result, repeat, scaled):
        assert sample["labels"].shape == sample["scores"].shape == (5,)
        assert sample["boxes"].shape == (5, 4)
        assert sample["boxes"].device.type == "cpu"
        assert sample["labels"].dtype == torch.long
        assert ((sample["labels"] >= 0) & (sample["labels"] < 4)).all()
        assert torch.isfinite(sample["boxes"]).all()
        assert ((sample["scores"] >= 0) & (sample["scores"] <= 1)).all()
        for key in sample:
            torch.testing.assert_close(sample[key], again[key])
        torch.testing.assert_close(enlarged["boxes"], sample["boxes"] * 2)
        torch.testing.assert_close(enlarged["scores"], sample["scores"])


def sam3(model_name):
    model_class = getattr(supported_models, model_name)
    descriptor = model_class(dict(
        image_model=model_name, embed_dim=64, depth=2, mlp_ratio=2,
        img_size=56, pretrain_img_size=28, window_size=2,
        global_att_blocks=(1,), num_queries=3, dtype=torch.float32,
    ))
    descriptor.custom_operations = ops.manual_cast
    wrapper = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    model = wrapper.diffusion_model
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.state_dict().items()
    }
    assert model_class.matches(model_detection.detect_unet_config(state, ""))
    pixels = torch.randn(1, 3, 56, 56)
    text = torch.randn(1, 3, 64)
    text_mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        result = model(pixels, text_embeddings=text, text_mask=text_mask, orig_size=(35, 49))
        repeat = model(pixels, text_embeddings=text, text_mask=text_mask, orig_size=(35, 49))
        changed = model(pixels, text_embeddings=text.flip(-1), text_mask=text_mask, orig_size=(35, 49))
        no_prompt = model(pixels, orig_size=(35, 49))
    assert result["boxes"].shape == (1, 3, 4)
    assert result["scores"].shape == (1, 3)
    assert result["masks"].shape == (1, 3, 35, 49)
    for key in ("boxes", "scores", "masks", "presence"):
        assert result[key].dtype == pixels.dtype and torch.isfinite(result[key]).all()
        torch.testing.assert_close(result[key], repeat[key])
    assert not torch.equal(result["scores"], changed["scores"]), "text was ignored"
    assert not torch.equal(result["masks"], no_prompt["masks"]), "prompt was ignored"
    assert torch.count_nonzero(no_prompt["scores"]) == 0
