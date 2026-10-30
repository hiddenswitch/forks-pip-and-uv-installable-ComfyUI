"""Exercise real integrators with a cheap, deterministic denoiser."""
import pytest
import torch
from comfy.k_diffusion import sampling


@pytest.mark.parametrize("name", ["sample_euler", "sample_heun", "sample_dpm_2", "sample_dpmpp_2m"])
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_tiny_sampler(name, steps):
    calls, callbacks = [], []
    def denoise(x, sigma, **kwargs):
        calls.append(sigma.clone())
        return torch.full_like(x, 0.25)
    x = torch.ones(1, 2, 3, 5)
    sigmas = torch.cat((torch.linspace(4, 1, steps), torch.zeros(1)))
    result = getattr(sampling, name)(denoise, x.clone(), sigmas, callback=callbacks.append, disable=True)
    assert len(callbacks) == steps
    assert bool(calls) == bool(steps)
    assert result.shape == x.shape and result.dtype == x.dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full_like(x, 0.25) if steps else x)
