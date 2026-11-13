import importlib
import json

import numpy as np
import pytest
import scipy.signal
import torch
from PIL import Image

from comfy import sd
from comfy.latent_formats import Magi2, Wan22
from comfy.ldm.audio.autoencoder import AudioOobleckVAE
from comfy.ldm.magi2 import latents as magi2_latents
from comfy.ldm.magi2.latents import LATENT_FORMAT
from comfy.nested_tensor import NestedTensor
from comfy.nodes.base_nodes import VAEDecode
from comfy_extras.nodes.nodes_audio import vae_decode_audio
from comfy_extras.nodes.nodes_magi2 import EmptyMagi2Latent, Magi2RefinerLatent, encode_reference, reference_hint, reference_pixels, video_latent_frames

from . import magi2_reference


def test_empty_latent_matches_reference_lengths():
    empty = EmptyMagi2Latent.execute(896, 512, 10.0, 1).result[0]
    assert empty["latent_format"] == LATENT_FORMAT
    latent = empty["samples"]
    video, audio = latent.unbind()
    assert video.shape == (1, 48, 32, 32, 56)
    assert audio.shape == (1, 64, 250)
    assert video_latent_frames(10.0) == 32
    refined = Magi2RefinerLatent.execute(empty, 1920, 1088).result[0]
    assert refined["samples"].shape == (1, 48, 63, 68, 120) and refined["latent_format"] == LATENT_FORMAT


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

    def float32_copy(self):
        return self

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


def test_float32_copy_reloads_the_source_in_fp32(monkeypatch, caplog):
    built = []

    class Built:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(sd, "VAE", Built)
    stub = type("Stub", (), {})()
    stub.vae_dtype, stub.ckpt_name, stub.device = torch.bfloat16, "wan.safetensors", "cpu"
    stub.state_dict_source = lambda: ({"w": torch.zeros(1, dtype=torch.float16)}, None)
    copy = _FLOAT32_COPY(stub)
    assert built[0]["dtype"] == torch.float32 and built[0]["ckpt_name"] == "wan.safetensors"
    assert "does not store fp32 weights" in caplog.text
    assert _FLOAT32_COPY(stub) is copy and len(built) == 1
    stub.vae_dtype = torch.float32
    del stub._float32_copy
    assert _FLOAT32_COPY(stub) is stub
    stub.vae_dtype, stub.state_dict_source = torch.bfloat16, None
    with pytest.raises(ValueError):
        _FLOAT32_COPY(stub)


_FLOAT32_COPY = sd.VAE.float32_copy


class _StubVae:
    """Enough of a VAE for the MAGI-2 latent mapping: the decoder class, dtype and an fp32 copy."""

    def __init__(self, model, latent_dim, latent_channels, vae_dtype=torch.bfloat16, waveform=None):
        self.first_stage_model, self.latent_dim, self.latent_channels, self.vae_dtype = model, latent_dim, latent_channels, vae_dtype
        self.audio_sample_rate, self.waveform, self.decoded = 44100, waveform, []

    def float32_copy(self):
        copy = _StubVae(self.first_stage_model, self.latent_dim, self.latent_channels, torch.float32, self.waveform)
        copy.decoded = self.decoded
        return copy

    def decode(self, samples, latent_format=None):
        if latent_format is not None:
            return sd.VAE.decode(self, samples, latent_format=latent_format)
        self.decoded.append((samples, self.vae_dtype))
        return self.waveform

    def throw_exception_if_invalid(self):
        pass


def test_magi2_audio_decodes_like_sandai_through_vae_decode_audio():
    """Stock VAE Decode Audio: fp32 decode, scipy.signal.resample by 441/512, no loudness cap, 44.1 kHz."""
    generator = torch.Generator().manual_seed(6)
    waveform = torch.randn(1, 2048 * 10, 2, generator=generator) * 3.0
    vae = _StubVae(AudioOobleckVAE.__new__(AudioOobleckVAE), 1, 64, waveform=waveform)
    audio_latent = torch.randn(1, 64, 10, generator=generator)
    latent = {"samples": NestedTensor((torch.zeros(1, 48, 2, 2, 2), audio_latent)), "latent_format": LATENT_FORMAT}
    audio = vae_decode_audio(vae, latent)
    assert vae.decoded[0][1] == torch.float32 and torch.equal(vae.decoded[0][0], audio_latent)
    expected = scipy.signal.resample(waveform[0].numpy(), int(waveform.shape[1] * 441 / 512))
    assert audio["sample_rate"] == 44100
    assert torch.equal(audio["waveform"][0], torch.from_numpy(expected).T)


def test_magi2_latents_refuse_other_vaes():
    latent = {"samples": torch.zeros(1, 48, 2, 2, 2), "latent_format": LATENT_FORMAT}
    with pytest.raises(ValueError, match="turbo VAE or the Wan 2.2 VAE"):
        VAEDecode().decode(_StubVae(torch.nn.Identity(), 3, 48), latent)
    with pytest.raises(ValueError, match="Stable Audio Open"):
        magi2_latents.decode_audio(_StubVae(torch.nn.Identity(), 1, 64), torch.zeros(1, 64, 4))
    with pytest.raises(ValueError, match="Unknown latent format"):
        VAEDecode().decode(_StubVae(torch.nn.Identity(), 3, 48), {"samples": torch.zeros(1, 48, 2, 2, 2), "latent_format": "other"})
