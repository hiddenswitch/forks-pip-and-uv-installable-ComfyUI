import pytest
import torch

from comfy import ops
from comfy.ldm.flux import layers


@pytest.mark.parametrize("block_type", [layers.DoubleStreamBlock, layers.SingleStreamBlock])
@torch.inference_mode()
def test_chunked_flux_block_matches_full_projection(monkeypatch, block_type):
    monkeypatch.setattr(layers, "CHUNK_TOKENS", 3)
    block = block_type(32, 4, 2.0, operations=ops.manual_cast, device="cpu", dtype=torch.float32)
    block.requires_grad_(False)
    rng = torch.Generator().manual_seed(1)
    for param in block.parameters():
        param.uniform_(-0.5, 0.5, generator=rng)
    img = torch.randn(1, 8, 32, generator=rng)
    txt = torch.randn(1, 4, 32, generator=rng)
    vec = torch.randn(1, 32, generator=rng)
    pe = layers.rope(torch.arange(12).unsqueeze(0), 8, 10000).unsqueeze(1)

    def run(options):
        if block_type is layers.DoubleStreamBlock:
            return block(img.clone(), txt.clone(), vec, pe, transformer_options=options)
        return block(torch.cat((txt, img), dim=1), vec, pe, transformer_options=options)

    torch.testing.assert_close(run({"chunk_tokens": True}), run({}), atol=1e-5, rtol=1e-4)
