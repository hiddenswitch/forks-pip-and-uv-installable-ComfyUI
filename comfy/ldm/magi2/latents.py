"""MAGI-2 latents through the stock VAE Decode and VAE Decode Audio nodes.

MAGI-2 keeps its latents in the model's normalized basis, as SandAI's pipeline does
(latent_formats.Magi2), and its LATENT dicts say so with latent_format "magi2". The VAE maps them
to its own input exactly as SandAI decodes:

- video (the turbo decoder or the Wan 2.2 VAE): the normalized latent rounded to the VAE dtype,
  divided by (1 / std) plus the mean in fp32, rounded again (latent_formats.Magi2.denormalize);
- audio (the Stable Audio Open VAE): decoded with fp32 weights, then resampled from MAGI-2's 25
  latents per second (51.2 kHz at 2048 samples per latent) to 44.1 kHz with scipy.signal.resample.

Any other VAE refuses a MAGI-2 latent instead of decoding it in the wrong basis.
"""

import numpy as np
import scipy.signal
import torch

from ...latent_formats import Magi2
from ..audio.autoencoder import AudioOobleckVAE
from ..wan.vae2_2 import WanVAE as Wan22VAE
from .turbo_vae import TurboVAEDecoder

LATENT_FORMAT = "magi2"
# SandAI's time stretch: 44.1 kHz playback of audio laid out at 51.2 kHz
AUDIO_TIME_STRETCH = 441 / 512


def check_latent_format(latent_format):
    if latent_format != LATENT_FORMAT:
        raise ValueError(f"Unknown latent format {latent_format!r}")


def video_decoder_input(vae, samples):
    """The VAE-basis decoder input for a MAGI-2 video latent, in the reference's operation order."""
    if not isinstance(vae.first_stage_model, (TurboVAEDecoder, Wan22VAE)):
        raise ValueError("MAGI-2 video latents decode with the MAGI-2 turbo VAE or the Wan 2.2 VAE")
    return Magi2().denormalize(samples, vae.vae_dtype)


def decode_audio(vae, samples):
    """A MAGI-2 audio latent [B, 64, L] decoded as SandAI does: [B, samples, channels] at 44.1 kHz."""
    if not (isinstance(vae.first_stage_model, AudioOobleckVAE) and vae.latent_channels == 64 and vae.audio_sample_rate == 44100):
        raise ValueError("MAGI-2 audio latents decode with the Stable Audio Open VAE")
    waveform = vae.float32_copy().decode(samples)
    stretched = [scipy.signal.resample(item.detach().float().cpu().numpy(), int(item.shape[0] * AUDIO_TIME_STRETCH)) for item in waveform]
    return torch.from_numpy(np.stack(stretched)).to(waveform.device)
