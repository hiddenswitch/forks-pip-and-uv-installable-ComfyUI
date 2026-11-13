import importlib
import json

import numpy as np
import pytest
import scipy.signal
import torch
from PIL import Image

from comfy.latent_formats import Magi2, Wan22
from comfy_extras.nodes import nodes_magi2
from comfy_extras.nodes.nodes_magi2 import (AUDIO_STRETCH, EmptyMagi2Latent, Magi2RefinerLatent, encode_reference, fft_resample, reference_hint, reference_pixels,
                                           video_latent_frames)

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
    torch.testing.assert_close(actual.movedim(-1, 1), expected, rtol=0, atol=0)


def test_vae_input_reproduces_bf16_rounded_reference_pixels():
    """SandAI rounds the [-1, 1] pixels to bf16 before its fp32 encode; VAE.encode's x * 2 - 1 recovers them exactly."""
    pixels = torch.arange(256, dtype=torch.float32) / 255.0
    signed = (pixels * 2.0 - 1.0).to(torch.bfloat16).float()
    assert torch.equal(((signed + 1.0) / 2.0) * 2.0 - 1.0, signed)


class _RecordingVae:
    vae_dtype = torch.float32
    ckpt_name = "wan.safetensors"

    def __init__(self, latent):
        self.latent, self.inputs = latent, []

    def encode(self, pixels):
        self.inputs.append(pixels)
        return self.latent


def test_encode_reference_normalizes_like_sandai():
    generator = torch.Generator().manual_seed(0)
    mu = torch.randn(1, 48, 2, 3, 4, generator=generator)
    vae = _RecordingVae(mu)
    pixels = torch.rand(1, 48, 64, 3, generator=generator)
    latent = encode_reference(vae, pixels)
    assert torch.equal(vae.inputs[0] * 2.0 - 1.0, (pixels * 2.0 - 1.0).to(torch.bfloat16).float())
    stats = Wan22()
    expected = (mu[:, :, :1] - stats.latents_mean) * (1.0 / stats.latents_std)
    assert latent.dtype == torch.float32 and torch.equal(latent, expected)
    assert torch.equal(Magi2().denormalize(latent, torch.float32), (latent / (1.0 / stats.latents_std) + stats.latents_mean))


def test_float32_vae_reloads_the_checkpoint_in_fp32(monkeypatch, caplog):
    built = []
    monkeypatch.setattr(nodes_magi2.folder_paths, "get_full_path_or_raise", lambda folder, name: f"/{folder}/{name}")
    monkeypatch.setattr(nodes_magi2.utils, "load_torch_file", lambda path, return_metadata: ({"w": torch.zeros(1, dtype=torch.float16)}, None))
    monkeypatch.setattr(nodes_magi2.sd, "VAE", lambda **kwargs: built.append(kwargs) or "fp32 vae")
    fp32 = _RecordingVae(None)
    assert nodes_magi2.float32_vae(fp32) is fp32
    bf16 = _RecordingVae(None)
    bf16.vae_dtype = torch.bfloat16
    assert nodes_magi2.float32_vae(bf16) == "fp32 vae"
    assert built[0]["dtype"] == torch.float32 and built[0]["ckpt_name"] == "wan.safetensors"
    assert "not an fp32 checkpoint" in caplog.text
    bf16.ckpt_name = ""
    with pytest.raises(ValueError):
        nodes_magi2.float32_vae(bf16)
