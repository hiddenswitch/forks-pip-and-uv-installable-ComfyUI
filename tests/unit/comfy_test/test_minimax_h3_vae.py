from types import SimpleNamespace

import pytest
import torch



import comfy.ops  # noqa: E402
from comfy import quant_ops
from comfy.ldm.minimax.vae import Attention, MiniMaxH3VideoVAE


class _OffloadedScale:
    """Stands in for a buffer that comfy's dynamic-VRAM offload left on another
    device: it only tracks its device label and can be moved with .to(), like
    the real qk_norm_scale tensor the bug report is about."""

    def __init__(self, device):
        self.device = torch.device(device)

    def to(self, device):
        return _OffloadedScale(device)


class _FakeCK:
    def rms_rope_split_half(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    def rms_rope_split_half_(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    @staticmethod
    def _check(query, key, scale):
        if scale.device != query.device:
            # mirrors comfy_kitchen's eager backend, which does no device
            # alignment and refuses to run the fused kernel on mixed devices
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got weight is on "
                f"{scale.device}, different from other tensors on {query.device}"
            )
        return query, key


def test_attention_moves_offloaded_qk_norm_scale_to_input_device(monkeypatch):
    monkeypatch.setattr(quant_ops, "ck", _FakeCK(), raising=False)

    heads, dim_head = 2, 4
    attn = Attention(heads=heads, dim_head=dim_head)
    # simulate a partially-offloaded VAE: qk_norm_scale sits on a different
    # device than the tensors flowing through the fused rope kernel
    attn._buffers["qk_norm_scale"] = _OffloadedScale("meta")

    batch_size, seq_len = 1, 3
    x = torch.arange(1, 25, dtype=torch.float32).reshape(batch_size, seq_len, heads * dim_head)
    rotary_pos_emb = torch.randn(batch_size, 1, seq_len, dim_head // 2, 2, 2)

    pre_norm = comfy.ops.manual_cast.RMSNorm(heads * dim_head, elementwise_affine=False, eps=1e-5)
    qkv_outputs = []
    hook = attn.to_qkv.register_forward_hook(lambda module, inputs, output: qkv_outputs.append(output))
    with torch.no_grad():
        attn.to_qkv.weight.copy_(torch.eye(heads * dim_head).repeat(3, 1))
        attn.to_qkv.bias.zero_()
        attn.to_out.weight.copy_(torch.eye(heads * dim_head))
        attn.to_out.bias.zero_()
        out = attn.forward(
            x, rotary_pos_emb, pre_norm=pre_norm, residual=None, residual_scale=None
        )
    hook.remove()

    assert out.device == x.device
    normalized = torch.nn.functional.rms_norm(x, (heads * dim_head,), eps=pre_norm.eps)
    expected = torch.nn.functional.linear(normalized, attn.to_qkv.weight, attn.to_qkv.bias)
    unnormalized = torch.nn.functional.linear(x, attn.to_qkv.weight, attn.to_qkv.bias)
    assert len(qkv_outputs) == 1
    torch.testing.assert_close(qkv_outputs[0], expected)
    assert not torch.allclose(qkv_outputs[0], unnormalized)



def test_attention_retries_in_chunks_when_sdpa_runs_out_of_memory(monkeypatch):
    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("attention workspace exhausted")

    monkeypatch.setattr("comfy.ldm.minimax.vae.optimized_attention", out_of_memory)
    attention = Attention(heads=2, dim_head=4)
    q, k, v = (torch.randn(1, 2, length, 4) for length in (3, 5, 5))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected = expected.transpose(1, 2).reshape(1, 3, 8)
    torch.testing.assert_close(attention.attention(q, k, v), expected)


@pytest.mark.parametrize(
    ("guessed", "architecture", "flash", "expected"),
    [
        (True, "gfx1102", True, torch.float16),
        (False, "gfx1102", True, torch.float32),
        (True, "gfx1100", True, torch.float32),
        (True, "gfx1102", False, torch.float32),
    ],
)
def test_h3_video_vae_precision_on_rocm(monkeypatch, guessed, architecture, flash, expected):
    from comfy import sd

    class VideoVAE(torch.nn.Module):
        tokens_chunk_size = 5
        token_overlap = 2
        vae_ratio_t = 4

        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(sd, "MiniMaxH3VideoVAE", VideoVAE)
    monkeypatch.setattr(sd.args, "fp32_vae", True)
    monkeypatch.setattr(sd.args, "_fp32_vae_guessed", guessed)
    monkeypatch.setattr(sd.model_management, "AMD_FLASH_SDP_SUPPORTED", flash)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=architecture))
    checkpoint = {"decoder.transformer_blocks.0.scale1": torch.zeros(1)}
    vae = sd.VAE(sd=checkpoint, device=torch.device("cuda:0"))
    assert vae.vae_dtype == expected


def test_still_decodes_as_first_latent_of_a_full_clip():
    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    seen = []

    def decode_pixels(z):  # every output frame holds its own index / 100
        seen.append(z.shape[2])
        frames = torch.arange(z.shape[2] * vae.vae_ratio_t, dtype=torch.float32) / 100
        return frames.view(1, 1, -1, 1, 1).expand(z.shape[0], 3, -1, z.shape[-2] * vae.vae_ratio, z.shape[-1] * vae.vae_ratio)

    vae._decode_pixels = decode_pixels
    out = vae.decode(torch.zeros(1, 24, 1, 40, 24))  # 640x384, several tiles
    assert out.shape == (1, 3, 1, 640, 384)
    assert set(seen) == {vae.tokens_chunk_size}
    torch.testing.assert_close(out, torch.full_like(out, vae.frame_pre_padding / 100))
