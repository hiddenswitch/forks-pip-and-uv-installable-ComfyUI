"""Lens multi-layer text conditioning with real dual-stream attention."""

import torch

from comfy import model_detection, ops, supported_models
from tests.fast_model_cases import _initialize


def lens():
    config = dict(supported_models.Lens.unet_config)
    config.update(
        in_channels=128,
        out_channels=32,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        enc_hidden_dim=12,
        axes_dims_rope=(4, 6, 6),
        dtype=torch.float32,
    )
    descriptor = supported_models.Lens(config)
    descriptor.custom_operations = ops.manual_cast
    model = _initialize(descriptor.get_model({}, device=torch.device("cpu")))
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.diffusion_model.state_dict().items()
    }
    detected = model_detection.detect_unet_config(state, "")
    assert supported_models.Lens.matches(detected)
    assert detected["num_layers"] == 2
    latent = torch.randn(1, 128, 3, 5)
    context = torch.randn(1, 3, 48)
    mask = torch.tensor([[1, 1, 0]])

    def forward(text, attention_mask):
        conditions = {
            key: value.process_cond(batch_size=1, area=None).cond
            for key, value in model.extra_conds(
                noise=latent,
                device=torch.device("cpu"),
                cross_attn=text,
                attention_mask=attention_mask,
            ).items()
        }
        with torch.no_grad():
            return model.apply_model(latent, torch.tensor([0.5]), **conditions)

    result = forward(context, mask)
    assert result.shape == latent.shape and result.dtype == latent.dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, forward(context, mask))
    assert not torch.equal(result, forward(context + 1, mask))
    masked_context = context.clone()
    masked_context[:, 2] += 100
    torch.testing.assert_close(result, forward(masked_context, mask))
    assert not torch.equal(result, forward(context, torch.ones_like(mask)))
