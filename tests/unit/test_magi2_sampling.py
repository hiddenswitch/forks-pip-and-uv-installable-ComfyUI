import importlib

import pytest
import torch

from comfy.k_diffusion.sampling import sample_uni_pc_flow
from comfy.ldm.magi2.sampling import zero_snr_signal_scale
from comfy.samplers import flow_unipc_scheduler

from . import magi2_reference

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None, reason="needs MAGI2_REFERENCE_SRC")


class _Shift:
    def __init__(self, shift):
        self.shift = shift


def _velocity(x):
    return torch.tanh(1.3 * x) + 0.2 * x - 0.1


@pytest.mark.parametrize("steps,shift", [(100, 7.0), (5, 5.0), (12, 7.0)])
def test_flow_unipc_matches_sandai_scheduler(steps, shift):
    ref = magi2_reference.load()
    scheduler = ref.sampler.FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(steps, device="cpu", shift=shift)
    sigmas = flow_unipc_scheduler(_Shift(shift), steps)
    assert torch.equal(sigmas, scheduler.sigmas)

    noise = torch.randn(1, 3, 4, 5, generator=torch.Generator().manual_seed(0))
    expected = noise.clone()
    for t in scheduler.timesteps:
        expected = scheduler.step(_velocity(expected), t, expected, return_dict=False)[0]
    actual = sample_uni_pc_flow(lambda x, sigma: x - sigma.view(-1, 1, 1, 1) * _velocity(x), noise.clone(), sigmas, disable=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_refiner_renoise_level_matches_sandai():
    magi2_reference.load()
    engine = importlib.import_module("inference.pipeline.inference_engine")
    sigmas = engine.ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
    assert torch.equal(zero_snr_signal_scale(220), sigmas[220])
