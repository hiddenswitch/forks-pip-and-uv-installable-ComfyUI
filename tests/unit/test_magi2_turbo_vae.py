import pytest
import torch

import comfy.ops as comfy_ops
from comfy import sd
from comfy.ldm.magi2.latents import LATENT_FORMAT
from comfy.ldm.magi2.turbo_vae import TurboVAEDecoder, decoder_config
from comfy.nested_tensor import NestedTensor
from comfy.nodes.base_nodes import VAEDecode

from . import magi2_reference

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or not torch.cuda.is_available(),
                                reason="needs MAGI2_REFERENCE_SRC and CUDA (the reference keeps its latent statistics on CUDA)")


def _models(ref, device):
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
    return reference, model


def _latent(frames, device):
    return torch.randn(1, 48, frames, 3, 4, generator=torch.Generator().manual_seed(2)).to(device)


@pytest.mark.parametrize("frames", [2, 3, 7, 8])
def test_turbo_decoder_matches_sandai(frames):
    """TurboVAEDecoder decodes Wan 2.2-basis latents; SandAI's decode un-normalizes first."""
    ref = magi2_reference.load()
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    reference, model = _models(ref, device)
    z = _latent(frames, device)
    with torch.no_grad(), ref.eager():
        expected = reference.decode(z)
    mean, inverse_std = reference.scale
    with torch.no_grad():
        actual = model.decode(z / inverse_std.view(1, 48, 1, 1, 1) + mean.view(1, 48, 1, 1, 1))
    assert actual.shape == expected.shape == (1, 3, (frames - 1) * 4 + 1, 48, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_stock_vae_decode_matches_sandai_bit_for_bit():
    """Stock VAE Decode on a MAGI-2 LATENT: the bf16 decoder input and the frames equal SandAI's exactly."""
    ref = magi2_reference.load()
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    reference, _ = _models(ref, device)
    vae = sd.VAE(sd={key: value.cpu() for key, value in reference.state_dict().items() if key.startswith("decoder.")})
    assert isinstance(vae.first_stage_model, TurboVAEDecoder) and vae.vae_dtype == torch.bfloat16
    # the chunk schedule is the release config's (7, 7), not part of the weights; this tiny reference uses (3, 2)
    vae.first_stage_model.first_chunk_size, vae.first_stage_model.step_size = 3, 2
    reference = reference.to(torch.bfloat16)
    inputs = {"reference": [], "comfy": []}
    reference.decoder.register_forward_pre_hook(lambda module, args, kwargs: inputs["reference"].append(args[0]), with_kwargs=True)
    vae.first_stage_model.decoder.register_forward_pre_hook(lambda module, args: inputs["comfy"].append(args[0]))
    z = _latent(7, device) * 3.0
    audio = torch.randn(1, 64, 20, generator=torch.Generator().manual_seed(5))
    latent = {"samples": NestedTensor((z.cpu(), audio)), "latent_format": LATENT_FORMAT}
    with torch.no_grad(), ref.eager():
        expected = reference.decode(z.to(torch.bfloat16)).float().mul(0.5).add(0.5).clamp(0, 1)
    with torch.no_grad():
        frames, = VAEDecode().decode(vae, latent)
    assert len(inputs["comfy"]) == len(inputs["reference"]) > 1
    for comfy_input, reference_input in zip(inputs["comfy"], inputs["reference"]):
        assert comfy_input.dtype == reference_input.dtype == torch.bfloat16
        assert torch.equal(comfy_input.to(reference_input.device), reference_input)
    assert torch.equal(frames, expected[0].permute(1, 2, 3, 0).cpu())
    # un-normalizing the fp32 latent before the bf16 cast, as a plain latent format would, does not give the same input
    mean, inverse_std = reference.scale
    unnormalized_first = (z / inverse_std.view(1, 48, 1, 1, 1) + mean.view(1, 48, 1, 1, 1)).to(torch.bfloat16)
    assert not torch.equal(unnormalized_first, torch.cat([inputs["reference"][0][:, :, :3], inputs["reference"][1][:, :, 1:3], inputs["reference"][2][:, :, 1:3]], dim=2))
