import pytest
import torch

from comfy.ldm.modules.sub_quadratic_attention import efficient_dot_product_attention


@pytest.mark.parametrize("mask_rows", [0, 1, 13])
@torch.inference_mode()
def test_chunked_inference_matches_dense_attention(mask_rows):
    generator = torch.Generator().manual_seed(103)
    q = torch.randn(2, 13, 8, generator=generator)
    k = torch.randn(2, 17, 8, generator=generator)
    v = torch.randn(2, 17, 6, generator=generator)
    mask = torch.randn(2, mask_rows, 17, generator=generator) if mask_rows else None
    scores = (q @ k.transpose(1, 2)) * (8 ** -0.5)
    if mask is not None:
        scores += mask
    expected = scores.softmax(dim=-1) @ v
    actual = efficient_dot_product_attention(
        q, k.transpose(1, 2), v, query_chunk_size=5, kv_chunk_size=4,
        use_checkpoint=False, mask=mask,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kv_chunk_size", [3, 8, 16])
@pytest.mark.parametrize("additive", [False, True])
@torch.inference_mode()
def test_masked_tiles_and_empty_query_rows(kv_chunk_size, additive):
    generator = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(2, n, 8, generator=generator) for n in (5, 16, 16))
    mask = torch.ones(2, 5, 16, dtype=torch.bool)
    mask[:, 0] = False
    mask[:, 1:, :8] = False
    if additive:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(~mask, -torch.inf)
        mask[:, 1:, 8:] -= 1000
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = efficient_dot_product_attention(
        q, k.transpose(1, 2), v, query_chunk_size=3, kv_chunk_size=kv_chunk_size,
        use_checkpoint=False, mask=mask,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
