"""MAGI-2 noise levels and starting states, as set up by SandAI's FlowUniPC pipeline."""

import numpy as np
import torch

from ... import model_sampling


def flow_unipc_training_sigmas(shift):
    """FlowUniPCMultistepScheduler's training table, ascending: shift(1 - linspace(1, 1/1000, 1000))."""
    sigmas = 1.0 - np.linspace(1, 1 / 1000, 1000)
    return torch.from_numpy(shift * sigmas / (1 + (shift - 1) * sigmas)).to(torch.float32)


def zero_snr_signal_scale(index):
    """sqrt(alpha_bar) of SandAI's zero-terminal-SNR DDPM schedule at a training step (the refiner's re-noise level)."""
    betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float64) ** 2
    alphas_cumprod = torch.tensor(np.cumprod(1.0 - betas.numpy(), axis=0), dtype=torch.float32)
    scale = alphas_cumprod.sqrt()
    first, last = scale[0].clone(), scale[-1].clone()
    scale -= last
    scale *= first / (first - last)
    return scale[index]


class Magi2ModelSampling(model_sampling.ModelSamplingDiscreteFlow, model_sampling.CONST):
    """Flow sampling on FlowUniPC's table: the top level is shift(0.999) and the preview starts from pure noise there."""

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        self.register_buffer("sigmas", flow_unipc_training_sigmas(shift))

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            return noise * self.noise_scale
        return super().noise_scaling(sigma, noise, latent_image, max_denoise)


class Magi2RefinerModelSampling(Magi2ModelSampling):
    """The refiner starts from the upsampled preview latent re-noised the variance-preserving way."""

    def __init__(self, model_config=None):
        super().__init__(model_config)
        settings = {} if model_config is None else model_config.sampling_settings
        self.register_buffer("signal_scale", zero_snr_signal_scale(settings.get("renoise_index", 220)))

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            scale = self.signal_scale.to(latent_image.device)
            return latent_image * scale + noise * (1 - scale ** 2) ** 0.5
        return super().noise_scaling(sigma, noise, latent_image, max_denoise)
