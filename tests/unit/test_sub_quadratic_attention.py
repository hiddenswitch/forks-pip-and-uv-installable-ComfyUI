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
