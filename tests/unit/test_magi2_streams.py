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


@pytest.mark.parametrize("modalities", [1, 3])
def test_dense_mlp_bounds_gate_workspace(monkeypatch, modalities):
    torch.manual_seed(123)
    mlp = magi.MLP(8, 16, modalities, operations=torch.nn)
    groups = (11, 0, 6)
    x = torch.randn(2, sum(groups), 8)
    with torch.no_grad():
        mlp.pre_norm.weight.uniform_(-0.5, 0.5)
        normalized = mlp.pre_norm(x, groups)
        gate = magi.modality_linear(mlp.up_gate_proj, normalized, groups)
        expected = magi.modality_linear(mlp.down_proj, magi.swiglu7(gate), groups)
        limit = 2 * 3 * 32 * x.element_size()
        monkeypatch.setattr(magi, "MLP_GATE_CHUNK_BYTES", limit, raising=False)

        def check_projection(module, args, output):
            assert output.numel() * output.element_size() <= limit

        layers = mlp.up_gate_proj if modalities > 1 else [mlp.up_gate_proj]
        for layer in layers:
            layer.register_forward_hook(check_projection)
        torch.testing.assert_close(mlp(x, groups), expected)
