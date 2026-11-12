import importlib
import json

import numpy as np
import pytest
import scipy.signal
import torch
from PIL import Image

from comfy_extras.nodes.nodes_magi2 import AUDIO_STRETCH, EmptyMagi2Latent, Magi2RefinerLatent, fft_resample, reference_hint, reference_pixels, video_latent_frames

from . import magi2_reference


def test_fft_resample_matches_scipy():
    generator = torch.Generator().manual_seed(0)
    for length in (512000, 2048 * 7, 4096 + 2):
        waveform = torch.randn(1, 2, length, generator=generator)
        num = length * AUDIO_STRETCH[0] // AUDIO_STRETCH[1]
        expected = scipy.signal.resample(waveform[0].T.numpy(), int(length * (441 / 512)))
        actual = fft_resample(waveform, num)[0].T.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_empty_latent_matches_reference_lengths():
    latent = EmptyMagi2Latent.execute(896, 512, 10.0, 1).result[0]["samples"]
    video, audio = latent.unbind()
    assert video.shape == (1, 48, 32, 32, 56)
    assert audio.shape == (1, 64, 250)
    assert video_latent_frames(10.0) == 32
    refined = Magi2RefinerLatent.execute({"samples": latent}, 1920, 1088).result[0]["samples"]
    assert refined.shape == (1, 48, 63, 68, 120)


reference_required = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None, reason="needs MAGI2_REFERENCE_SRC")


@reference_required
@pytest.mark.parametrize("prompt", ["A dog runs.", json.dumps({"global_layer": {"context": "x"}})])
def test_reference_hint_matches_sandai(prompt):
    magi2_reference.load()
    engine = importlib.import_module("inference.pipeline.inference_engine")
    assert reference_hint(prompt) == engine.Magi2InferenceEngine._ensure_figure_tokens(prompt, [1])


@reference_required
@pytest.mark.parametrize("size,target", [((640, 480), (896, 512)), ((300, 700), (896, 512)), ((1000, 1000), (448, 256)), ((517, 233), (512, 896))])
def test_reference_pixels_match_sandai_preprocessing(size, target):
    magi2_reference.load()
    engine = importlib.import_module("inference.pipeline.inference_engine")
    from diffusers.video_processor import VideoProcessor

    rng = np.random.default_rng(0)
    pil = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    width, height = target
    length = max(height, width)
    if pil.width > pil.height:
        target_w, target_h = length, int(pil.height * length / pil.width)
    else:
        target_h, target_w = length, int(pil.width * length / pil.height)
    expected = VideoProcessor(vae_scale_factor=32).preprocess(engine.resizepad(pil, target_h, target_w), height=target_h, width=target_w)
    image = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)
    actual = reference_pixels(image, width, height) * 2.0 - 1.0
    torch.testing.assert_close(actual.movedim(-1, 1), expected, rtol=0, atol=1e-6)
