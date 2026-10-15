import pytest
import torch
import torch.nn.functional as F

from comfy.ldm.modules import attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not attention.SAGE_ATTENTION_IS_AVAILABLE,
    reason="SageAttention needs CUDA and the sageattention package",
)


def _leave_non_finite_on_chip():
    # Ampere's L1 and shared memory are one SRAM: a kernel that touched NaNs leaves them where the next
    # kernel's shared memory tiles land, as the sampler's earlier kernels do in a real run.
    torch.full((1 << 20,), float("nan"), device="cuda").sum()


# Krea 2's text fusion attends over its 12 tapped Qwen3-VL layers (20 heads of 128), and over the prompt's
# tokens in the refiner; both are shorter than one SageAttention KV tile.
@pytest.mark.parametrize("q_len,kv_len", [(12, 12), (28, 28), (63, 63), (1052, 12)])
def test_sage_attention_with_fewer_keys_than_a_tile_matches_sdpa_hnd(q_len, kv_len):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(2, 20, q_len, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(2, 20, kv_len, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(2, 20, kv_len, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    expected = F.scaled_dot_product_attention(q, k, v)

    _leave_non_finite_on_chip()
    out = attention.attention_sage(q, k, v, 20, skip_reshape=True, skip_output_reshape=True)

    assert out.isfinite().all()
    torch.testing.assert_close(out, expected, atol=4e-3, rtol=1.6e-2)


def test_sage_attention_with_fewer_keys_than_a_tile_matches_sdpa_nhd():
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 12, 20 * 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3))
    heads = lambda t: t.view(2, 12, 20, 128).transpose(1, 2)
    expected = F.scaled_dot_product_attention(heads(q), heads(k), heads(v)).transpose(1, 2).reshape(2, 12, 20 * 128)

    _leave_non_finite_on_chip()
    out = attention.attention_sage(q, k, v, 20)

    assert out.isfinite().all()
    torch.testing.assert_close(out, expected, atol=4e-3, rtol=1.6e-2)
