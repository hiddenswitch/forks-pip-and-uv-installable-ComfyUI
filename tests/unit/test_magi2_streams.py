import pytest
import torch

from comfy.ldm.magi2 import model as magi


@pytest.mark.parametrize("modalities", [1, 3])
def test_stream_projection_bounds_fp32_workspace(monkeypatch, modalities):
    layer = magi.TransformerLayer.__new__(magi.TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.num_streams = 4
    layer.mhc_norm = magi.MultiModalityRMSNorm(32, modalities, out_dtype=torch.float32)
    layer.mhc_phi_fused_attn = torch.nn.Parameter(torch.randn(32, 24))
    groups = (11, 0, 6)
    hidden = torch.randn(2, sum(groups), 32).bfloat16()
    with torch.no_grad():
        layer.mhc_norm.weight.uniform_(-0.5, 0.5)
        expected = layer.mhc_norm(hidden, groups) @ layer.mhc_phi_fused_attn
        limit = 2 * 3 * 32 * 4
        monkeypatch.setattr(magi, "MHC_FP32_CHUNK_BYTES", limit, raising=False)
        original = torch.nn.functional.rms_norm

        def checked(x, *args, **kwargs):
            assert x.numel() * x.element_size() <= limit
            return original(x, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "rms_norm", checked)
        pre, post, res = layer._logits(hidden, "attn", groups)
        torch.testing.assert_close(torch.cat((pre, post, res.flatten(-2)), dim=-1), expected)
