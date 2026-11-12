import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.turbo_vae import TurboVAEDecoder, decoder_config

from . import magi2_reference

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or not torch.cuda.is_available(),
                                reason="needs MAGI2_REFERENCE_SRC and CUDA (the reference keeps its latent statistics on CUDA)")


@pytest.mark.parametrize("frames", [2, 3, 7, 8])
def test_turbo_decoder_matches_sandai(frames):
    ref = magi2_reference.load()
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    config = dict(latent_channels=48, decoder_block_out_channels=(8, 16, 16, 32), decoder_layers_per_block=(1, 2, 1, 1, 2),
                  decoder_spatio_temporal_scaling=(False, True, True, True), decoder_spatio_only=(False, True, False, False),
                  patch_size=2, first_chunk_size=3, step_size=2, use_unpatchify=True)
    torch.manual_seed(0)
    with ref.eager(), torch.cuda.device(device):
        reference = ref.turbo.TurboVAED(**config)
    generator = torch.Generator().manual_seed(1)
    for parameter in reference.parameters():
        parameter.data.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    reference = reference.to(device, torch.float32).eval()
    state = {key: value for key, value in reference.state_dict().items() if key.startswith("decoder.")}
    model = TurboVAEDecoder(**decoder_config(state), first_chunk_size=3, step_size=2, device=device, operations=comfy_ops.disable_weight_init)
    model.load_state_dict(state, strict=True)

    z = torch.randn(1, 48, frames, 3, 4, generator=generator).to(device)
    with torch.no_grad(), ref.eager():
        expected = reference.decode(z)
    mean, inverse_std = reference.scale
    raw = z / inverse_std.view(1, 48, 1, 1, 1) + mean.view(1, 48, 1, 1, 1)
    with torch.no_grad():
        actual = model.decode(raw)
    assert actual.shape == expected.shape == (1, 3, (frames - 1) * 4 + 1, 48, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
