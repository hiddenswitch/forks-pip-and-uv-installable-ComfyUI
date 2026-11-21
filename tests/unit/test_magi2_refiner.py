"""MAGI-2 refiner DiT against SandAI's reference (bf16, CUDA, tiny random weights, block-window attention)."""

import types

import math

import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.convert import comfy_tensors
from comfy.ldm.magi2.refiner import Magi2RefinerModel, block_window_attention
from comfy.ldm.modules.attention import ComfyAttention

from . import magi2_reference
from .test_magi2_model import TOLERANCE, _device, _randomize, _relative_error

REFINER = dict(num_layers=3, hidden_size=128, head_dim=32, num_query_groups=2, video_in_channels=8, audio_in_channels=6,
               text_in_channels=24, intermediate_size=340, multimodal_layers=(0, 2), window_block_size=(2, 2, 2), window_radius=(1, 1, 1))


def _reference_refiner(ref, device):
    config = ref.config.Magi2RefinerModelConfig(
        num_layers=3, hidden_size=128, head_dim=32, num_query_groups=2, video_in_channels=8, audio_in_channels=6,
        text_in_channels=24, mm_layers=[0, 2], local_attn_layers=[0, 1, 2], num_heads_q=4, num_heads_kv=2)
    torch.manual_seed(0)
    model = ref.refiner.Transformer(config)
    _randomize(model)
    return model.to(device).eval()


def _reference_proxy(ref):
    return ref.refiner_proxy.Magi2RefinerDataProxy(ref.config.Magi2RefinerDataProxyConfig(
        t_patch_size=1, patch_size=1, frame_receptive_field=11, attn_config={
            "mode": "window", "block_t_size": 2, "block_size": 2,
            "window": {"level": "block", "block_mode": "grid", "block_t_radius": 1, "block_h_radius": 1, "block_w_radius": 1}}))


@pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or not torch.cuda.is_available(),
                    reason="needs MAGI2_REFERENCE_SRC and CUDA")
@pytest.mark.parametrize("audio_len", [0, 4])
def test_refiner_matches_sandai_reference(audio_len, monkeypatch, tmp_path):
    ref = magi2_reference.load()
    device = _device()
    generator = torch.Generator().manual_seed(3)
    latent = torch.randn(1, 8, 5, 6, 5, generator=generator).to(device)
    audio = torch.randn(1, audio_len, 6, generator=generator).to(device)
    text = torch.randn(1, 5, 24, generator=generator).to(torch.bfloat16).to(device)
    with ref.eager(), torch.no_grad():
        reference = _reference_refiner(ref, device)
        proxy = _reference_proxy(ref)
        data = types.SimpleNamespace(
            x_t=latent, audio_x_t=audio, audio_feat_len=torch.tensor([audio_len]), txt_feat=text.float(),
            txt_feat_len=torch.tensor([5]), ref_audio_feat=audio.new_zeros(1, 0, 6), ref_audio_feat_len=torch.tensor([0]),
            ref_video_feat=torch.empty_like(latent), ref_video_feat_len=torch.tensor([0]))
        expected_video, expected_audio = proxy.process_output(reference(*proxy.process_input(data)))

    state = {}
    for name, tensor in reference.state_dict().items():
        state.update(comfy_tensors(name, tensor, REFINER["multimodal_layers"]))
    model = Magi2RefinerModel(**REFINER, dtype=torch.bfloat16, device=device, operations=comfy_ops.disable_weight_init)
    model.load_state_dict({key: value.contiguous() for key, value in state.items()}, strict=True)
    model.requires_grad_(False)
    x = [latent, audio.transpose(1, 2)] if audio_len else latent
    out = model(x, torch.tensor([750.0], device=device), text.float())
    video = out[0] if audio_len else out
    assert video.shape == expected_video.shape
    assert _relative_error(video, expected_video) < TOLERANCE
    if audio_len:
        assert _relative_error(out[1].transpose(1, 2), expected_audio) < TOLERANCE

    from comfy.ldm.magi2 import refiner

    # Force chunk transfers and disk-backed residuals on the same weights and sample.
    monkeypatch.setattr(refiner, "QKV_PROJECTION_CHUNK_BYTES", 4096)
    monkeypatch.setattr(refiner, "MLP_GATE_CHUNK_BYTES", 4096)
    monkeypatch.setattr(refiner.model_management, "get_free_memory", lambda dev, *args: 0)
    monkeypatch.setattr(refiner.model_management, "is_unified_memory_device", lambda dev: False)
    monkeypatch.setattr(refiner.model_prefetch, "malloc_graph_enabled", lambda dev: False)
    monkeypatch.setattr("comfy.cmd.folder_paths.get_temp_directory", lambda: str(tmp_path))
    inputs = []
    hook = model.block.layers[0].register_forward_pre_hook(lambda _, args: inputs.append(args[0].device.type))
    with torch.inference_mode():
        streamed = model(x, torch.tensor([750.0], device=device), text.float())
    hook.remove()
    assert inputs == ["cpu"]
    for before, after in zip(out if audio_len else [out], streamed if audio_len else [streamed]):
        assert after.device == before.device
        assert _relative_error(after, before) < TOLERANCE
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("shape", "tail"), [((5, 6, 7), 3), ((4, 6, 4), 0)])
def test_block_window_attention_matches_dense_masked_attention(monkeypatch, shape, tail):
    from comfy.ldm.magi2 import refiner

    # Force multiple chunks, including partial edge blocks in the first case.
    monkeypatch.setattr(refiner, "WINDOW_KV_CHUNK_BYTES", 4096)
    torch.manual_seed(4)
    block, radius = (2, 3, 2), (1, 1, 2)
    video = shape[0] * shape[1] * shape[2]
    q = torch.randn(2, video + tail, 4, 8)
    k = torch.randn(2, video + tail, 2, 8)
    v = torch.randn(2, video + tail, 2, 8)
    coords = torch.stack(torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"), dim=-1).view(-1, 3)
    block_of = coords // torch.tensor(block)
    near = ((block_of[:, None] - block_of[None, :]).abs() <= torch.tensor(radius)).all(-1)
    mask = torch.ones(video + tail, video + tail, dtype=torch.bool)
    mask[:video, :video] = near
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.repeat_interleave(2, dim=2).transpose(1, 2), v.repeat_interleave(2, dim=2).transpose(1, 2), attn_mask=mask)
    actual = block_window_attention(q, k, v, shape, block, radius, ComfyAttention(), {})
    torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-5, atol=1e-5)

    padded = math.prod(math.ceil(n / b) * b for n, b in zip(shape, block))
    buffer = q.new_empty((q.shape[0], padded + tail, *q.shape[2:]))
    queries = buffer[:, :video + tail]
    queries.copy_(q)
    reused = block_window_attention(queries, k, v, shape, block, radius, ComfyAttention(), {}, out=buffer)
    assert reused.data_ptr() == queries.data_ptr()
    torch.testing.assert_close(reused, actual, rtol=0, atol=0)

    if torch.cuda.is_available():
        keys, values = k.cuda(), v.cuda()
        expected = block_window_attention(q.cuda(), keys, values, shape, block, radius, ComfyAttention(), {})
        queries.copy_(q)
        streamed = block_window_attention(queries, keys, values, shape, block, radius, ComfyAttention(), {}, out=buffer)
        torch.testing.assert_close(streamed, expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("masked", [False, True])
def test_low_memory_rocm_attention_matches_full_attention(monkeypatch, masked):
    from comfy.ldm.magi2 import refiner

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(7)
    device = _device()
    q = torch.randn(2, 4, 11, 16, device=device)
    k, v = (torch.randn(2, 2, 19, 16, device=device) for _ in range(2))
    mask = torch.ones(2, 1, 1, 19, dtype=torch.bool, device=device)
    mask[0, ..., 5:] = False
    mask[1, ..., :7] = False
    if not masked:
        mask = None
    attention = ComfyAttention()
    sub_quad = refiner.attention_sub_quad
    calls = []

    def record_fallback(*args, **kwargs):
        calls.append(kwargs["mask"])
        return sub_quad(*args, **kwargs)

    with torch.inference_mode():
        monkeypatch.setattr(refiner.model_management, "is_amd", lambda: False)
        expected = refiner._attend(q, k, v, attention, {}, mask=mask)
        monkeypatch.setattr(refiner.model_management, "is_amd", lambda: True)
        monkeypatch.setattr(refiner.model_management, "get_free_memory", lambda device, split=False: (65536, 0) if split else 65536)
        monkeypatch.setattr(refiner, "attention_sub_quad", record_fallback)
        actual = refiner._attend(q, k, v, attention, {}, mask=mask)

    assert len(calls) == 1 and calls[0] is mask
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_refiner_projects_into_host_buffer_and_runs_mlp_on_device(monkeypatch):
    from comfy.ldm.magi2 import refiner

    device = _device()
    monkeypatch.setattr(refiner, "QKV_PROJECTION_CHUNK_BYTES", 64)
    attention = refiner.RefinerAttention(8, 2, 1, 4, 1, (1, 1, 1), (0, 0, 0),
                                         dtype=torch.float32, device=device, operations=comfy_ops.manual_cast)
    mlp = refiner.RefinerMLP(8, 12, 1, dtype=torch.float32, device=device, operations=comfy_ops.manual_cast)
    with torch.inference_mode():
        _randomize(attention)
        _randomize(mlp)
        inputs = torch.randn(1, 7, 8, device=device)
        attended = torch.randn(1, 7, 2, 4, device=device)
        gates = torch.randn(1, 7, 2, device=device)
        expected = attention._project_output(attended, gates, (7,), torch.float32)
        host = attended.cpu()
        actual = attention._project_output(host, gates, (7,), torch.float32, reuse_attended=True)
        assert actual.data_ptr() == host.data_ptr()
        torch.testing.assert_close(actual, expected.cpu())

        expected = mlp(inputs, (7,), torch.float32, attention_output=expected)
        devices = []
        handle = mlp.pre_norm.register_forward_pre_hook(lambda _, args: devices.append(args[0].device))
        actual = mlp(inputs.cpu(), (7,), torch.float32, attention_output=actual, compute_device=device)
        handle.remove()
        assert devices and all(value == device for value in devices)
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected.cpu())
