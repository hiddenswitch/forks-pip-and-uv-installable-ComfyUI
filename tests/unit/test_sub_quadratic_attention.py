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


@pytest.mark.parametrize("layout", ["packed", "heads"])
@pytest.mark.parametrize("mask_rows", [0, 1, 5])
@torch.inference_mode()
def test_grouped_query_layout_and_mask(layout, mask_rows):
    from comfy.ldm.modules.attention import attention_sub_quad

    generator = torch.Generator().manual_seed(35)
    q = torch.randn(2, 6, 5, 8, generator=generator)
    k, v = (torch.randn(2, 2, 15, 8, generator=generator) for _ in range(2))
    mask = torch.rand(2, 1, mask_rows, 15, generator=generator) > .3 if mask_rows else None
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=mask,
    )
    packed = layout == "packed"
    if packed:
        q, k, v, ref = (x.transpose(1, 2).flatten(2) for x in (q, k, v, ref))
    actual = attention_sub_quad(q, k, v, 6, mask=mask, skip_reshape=not packed, skip_output_reshape=not packed, enable_gqa=True)
    torch.testing.assert_close(actual, ref)
