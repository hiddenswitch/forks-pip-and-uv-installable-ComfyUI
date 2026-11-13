"""MAGI-2 refiner DiT against SandAI's reference (bf16, CUDA, tiny random weights, block-window attention)."""

import types

import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.convert import comfy_tensors
from comfy.ldm.magi2.refiner import Magi2RefinerModel, block_window_attention
from comfy.ldm.modules.attention import ComfyAttention

from . import magi2_reference
from .test_magi2_model import TOLERANCE, _device, _randomize, _relative_error

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or not torch.cuda.is_available(),
                                reason="needs MAGI2_REFERENCE_SRC and CUDA")

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


@pytest.mark.parametrize("audio_len", [0, 4])
def test_refiner_matches_sandai_reference(audio_len):
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


def test_block_window_attention_matches_dense_masked_attention():
    torch.manual_seed(4)
    shape, block, radius = (5, 6, 7), (2, 3, 2), (1, 1, 2)
    video = shape[0] * shape[1] * shape[2]
    q = torch.randn(2, video + 3, 4, 8)
    k = torch.randn(2, video + 3, 2, 8)
    v = torch.randn(2, video + 3, 2, 8)
    coords = torch.stack(torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"), dim=-1).view(-1, 3)
    block_of = coords // torch.tensor(block)
    near = ((block_of[:, None] - block_of[None, :]).abs() <= torch.tensor(radius)).all(-1)
    mask = torch.ones(video + 3, video + 3, dtype=torch.bool)
    mask[:video, :video] = near
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.repeat_interleave(2, dim=2).transpose(1, 2), v.repeat_interleave(2, dim=2).transpose(1, 2), attn_mask=mask)
    actual = block_window_attention(q, k, v, shape, block, radius, ComfyAttention(), {})
    torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-5, atol=1e-5)
