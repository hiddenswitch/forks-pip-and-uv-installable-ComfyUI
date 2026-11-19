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


@pytest.mark.parametrize("modalities", [1, 3])
@torch.no_grad()
def test_qk_rope_bounds_fp32_workspace(monkeypatch, modalities):
    groups = (11, 0, 6)
    q, k = (torch.randn(2, sum(groups), heads, 16).bfloat16() for heads in (4, 2))
    q_norm, k_norm = (magi.MultiModalityRMSNorm(16, modalities) for _ in range(2))
    q_norm.weight.uniform_(-0.5, 0.5)
    k_norm.weight.uniform_(-0.5, 0.5)
    freqs = torch.randn(1, sum(groups), 1, 4, 2, 2)
    original = magi.comfy_kitchen.rms_rope_split_half
    expected_q, expected_k = [], []
    offset = 0
    for index, length in enumerate(groups if modalities > 1 else (sum(groups),)):
        if length:
            span = slice(offset, offset + length)
            qo, ko = original(q[:, span].float(), k[:, span].float(), freqs[:, span],
                              q_norm.scales(q.device)[index], k_norm.scales(k.device)[index], epsilon=q_norm.eps, rot_dim=8)
            expected_q.append(qo.bfloat16())
            expected_k.append(ko.bfloat16())
        offset += length
    limit = 2 * 3 * (4 + 2) * 16 * 4
    monkeypatch.setattr(magi, "QK_ROPE_FP32_CHUNK_BYTES", limit, raising=False)

    def checked(qp, kp, *args, **kwargs):
        assert (qp.numel() + kp.numel()) * 4 <= limit
        return original(qp, kp, *args, **kwargs)

    monkeypatch.setattr(magi.comfy_kitchen, "rms_rope_split_half", checked)
    qo, ko = magi.qk_norm_rope(q, k, q_norm, k_norm, freqs, groups)
    assert torch.equal(qo, torch.cat(expected_q, dim=1))
    assert torch.equal(ko, torch.cat(expected_k, dim=1))


@pytest.mark.parametrize("modalities", [1, 3])
@torch.no_grad()
def test_refiner_mlp_bounds_fp32_gate_and_preserves_residual(monkeypatch, modalities):
    from comfy.ldm.magi2 import refiner

    torch.manual_seed(55)
    mlp = refiner.RefinerMLP(16, 32, modalities, dtype=torch.bfloat16, operations=torch.nn)
    mlp.pre_norm.weight.uniform_(-0.5, 0.5)
    groups = (11, 0, 6)
    x = torch.randn(2, sum(groups), 16)
    normalized = mlp.pre_norm(x, groups).bfloat16()
    gate = magi.modality_linear(mlp.up_gate_proj, normalized, groups).float()
    expected = magi.modality_linear(mlp.down_proj, magi.swiglu7(gate).bfloat16(), groups).float()
    limit = 2 * 3 * 64 * 4
    monkeypatch.setattr(refiner, "MLP_GATE_CHUNK_BYTES", limit, raising=False)
    original = refiner.swiglu7

    def bounded(gate):
        assert gate.dtype == torch.float32
        assert gate.nbytes <= limit
        return original(gate)

    monkeypatch.setattr(refiner, "swiglu7", bounded)
    actual = mlp(x, groups, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("audio_tokens,with_timestep", [(0, True), (7, False)])
@torch.no_grad()
def test_input_adapter_bounds_fp32_workspace(monkeypatch, audio_tokens, with_timestep):
    torch.manual_seed(3)
    batch, width = 2, 16
    adapter = magi.PreAdapter(width, 8, 4, 6, 8)
    projections = (adapter.video_embedder, adapter.audio_embedder, adapter.text_embedder)
    tokens = [torch.randn(batch, n, c).bfloat16() for n, c in [(13, 4), (audio_tokens, 6), (5, 8)]]
    for layer in projections:
        layer.weight.uniform_(-0.5, 0.5)
        layer.bias.uniform_(-0.1, 0.1)
    expected = torch.cat([layer(part.float()) for layer, part in zip(projections, tokens)], dim=1)
    time = torch.randn(batch, expected.shape[1], 4) if with_timestep else None
    if time is not None:
        expected[..., :4] = time

    limit = batch * 3 * width * 4
    monkeypatch.setattr(magi, "ADAPTER_FP32_CHUNK_BYTES", limit)

    def check_projection(module, args, output):
        assert output.dtype == torch.float32
        assert output.numel() * output.element_size() <= limit

    for layer in projections:
        layer.register_forward_hook(check_projection)
    actual = adapter(*tokens, time, torch.bfloat16)
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
