"""SandAI MAGI-2 nodes: the packed video+audio latent, image references, the 1080p refiner input and decoding.

Sampling uses stock nodes: LTXV Dual CFG Guider (video 5, audio 7), KSamplerSelect "uni_pc_flow"
and BasicScheduler "flow_unipc" (100 steps) for the preview; CFGGuider (2.0) with the same sampler
and scheduler (5 steps) for the refiner.

MAGI-2 video latents stay in the model's normalized basis, as in SandAI's pipeline (latent_formats.Magi2):
decode them with MAGI-2 Decode rather than VAE Decode.
"""

import json
import logging
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from comfy import latent_formats, model_management, nested_tensor, node_helpers, sd, utils
from comfy.cmd import folder_paths
from comfy.ldm.magi2.model import ieee_fp32_matmul
from comfy.ldm.magi2.turbo_vae import TurboVAEDecoder
from comfy.nodes.common import MAX_RESOLUTION
from comfy_api.latest import ComfyExtension, InputImpl, Types, io

logger = logging.getLogger(__name__)

VIDEO_FPS = 12.5
AUDIO_LATENT_FPS = 25.0
LATENT_FRAME_STRIDE = 8
SPATIAL_STRIDE = 16
REFERENCE = "<Figure 1>"
REFERENCE_HINT = "The first frame refers to <Figure 1>"
# the audio VAE runs at 2048 samples per latent; MAGI-2 places 25 latents per second, i.e. 51.2 kHz, played at 44.1 kHz
AUDIO_STRETCH = (441, 512)


def video_latent_frames(seconds):
    """The reference counts frames at twice the 12.5 fps preview rate on an 8-frame latent stride."""
    return (int(round(seconds * VIDEO_FPS * 2)) - 1) // LATENT_FRAME_STRIDE + 1


def _video_stream(samples):
    return samples.unbind()[0] if samples.is_nested else samples


def reference_hint(prompt):
    """Point the prompt at <Figure 1>, as the reference does before encoding an image-conditioned prompt."""
    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        parsed["reference_layer"] = [REFERENCE_HINT]
        return json.dumps(parsed, ensure_ascii=False)
    return prompt + "reference_layer:" + REFERENCE_HINT


def reference_pixels(image, width, height):
    """Letterbox onto white with the long edge at max(width, height), then floor to multiples of 32, both with Lanczos."""
    pil = Image.fromarray((image[0].float() * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()).convert("RGB")
    length = max(width, height)
    if pil.width > pil.height:
        target_w, target_h = length, int(pil.height * length / pil.width)
    else:
        target_w, target_h = int(pil.width * length / pil.height), length
    scale = min(target_w / pil.width, target_h / pil.height)
    fit = (max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale))))
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    canvas.paste(pil.resize(fit, resample=Image.Resampling.LANCZOS), ((target_w - fit[0]) // 2, (target_h - fit[1]) // 2))
    canvas = canvas.resize((target_w - target_w % 32, target_h - target_h % 32), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0).unsqueeze(0)


def float32_vae(vae):
    """The VAE with fp32 weights: SandAI encodes reference images with its fp32 Wan 2.2 VAE whatever the global precision."""
    if vae.vae_dtype == torch.float32:
        return vae
    if not vae.ckpt_name:
        raise ValueError("MAGI-2 encodes reference images with the Wan 2.2 VAE in fp32: load it with Load VAE so it can be reloaded in fp32")
    state_dict, metadata = utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae.ckpt_name), return_metadata=True)
    if any(value.dtype != torch.float32 for value in state_dict.values() if value.is_floating_point()):
        logger.warning("%s is not an fp32 checkpoint; MAGI-2's reference images are encoded with the fp32 Wan 2.2 VAE (vae/magi2_wan2.2_vae_fp32.safetensors)", vae.ckpt_name)
    return sd.VAE(sd=state_dict, metadata=metadata, dtype=torch.float32, ckpt_name=vae.ckpt_name)


def encode_reference(vae, pixels):
    """SandAI's reference image latent: [-1, 1] pixels rounded to bf16, the fp32 Wan 2.2 encoder mean, normalized in fp32.

    VAE.encode maps [0, 1] to [-1, 1] by x * 2 - 1, which is exact on (b + 1) / 2 for every bf16 value b in [-1, 1].
    """
    signed = (pixels * 2.0 - 1.0).to(torch.bfloat16).float()
    vae = float32_vae(vae)
    with ieee_fp32_matmul():
        latent = vae.encode((signed + 1.0) / 2.0)
    return latent_formats.Magi2().normalize(latent[:, :, :1])


def fft_resample(waveform, num):
    """scipy.signal.resample along the last axis of a real signal: keep the low band of the spectrum, rescale."""
    length = waveform.shape[-1]
    kept = min(num, length)
    spectrum = torch.fft.rfft(waveform.float(), dim=-1)[..., :kept // 2 + 1]
    if kept % 2 == 0 and num != length:
        spectrum[..., kept // 2] *= 2 if num < length else 0.5
    return torch.fft.irfft(spectrum / (length / num), n=num, dim=-1)


class EmptyMagi2Latent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMagi2Latent",
            display_name="Empty MAGI-2 Latent",
            category="model/latent/magi2",
            description="Joint video and audio latent for MAGI-2 Preview: 125 frames at 12.5 fps for 10 seconds. Presets: 448x256 (272p) and 896x512 (540p).",
            inputs=[
                io.Int.Input("width", default=896, min=16, max=MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=512, min=16, max=MAX_RESOLUTION, step=16),
                io.Float.Input("seconds", default=10.0, min=0.04, max=60.0, step=0.04, tooltip="The model is trained on 10 second clips."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, seconds, batch_size) -> io.NodeOutput:
        device = model_management.intermediate_device()
        video = torch.zeros(batch_size, 48, video_latent_frames(seconds), height // SPATIAL_STRIDE, width // SPATIAL_STRIDE, device=device)
        audio = torch.zeros(batch_size, 64, int(round(seconds * AUDIO_LATENT_FPS)), device=device)
        return io.NodeOutput({"samples": nested_tensor.NestedTensor((video, audio))})


class Magi2ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magi2ImageToVideo",
            display_name="MAGI-2 Image to Video",
            category="model/conditioning/magi2",
            description="Encodes the prompt with a reference to the image and attaches the image to both conditionings, as the reference image-to-video pipeline does.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", tooltip="The fp32 Wan 2.2 VAE (magi2_wan2.2_vae_fp32); it is reloaded in fp32 if it was loaded in another precision."),
                io.Image.Input("image"),
                io.Int.Input("width", default=896, min=16, max=MAX_RESOLUTION, step=16, tooltip="The generated video's width."),
                io.Int.Input("height", default=512, min=16, max=MAX_RESOLUTION, step=16, tooltip="The generated video's height."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Conditioning.Output(display_name="negative")],
        )

    @classmethod
    def execute(cls, clip, prompt, negative, vae, image, width, height) -> io.NodeOutput:
        prompt = reference_hint(prompt)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        indices = clip.tokenizer.token_indices(prompt, REFERENCE)
        features = positive[0][0]
        token = features[0, indices].float().mean(dim=0) if indices else features.new_zeros(features.shape[-1], dtype=torch.float32)
        width, height = width - width % SPATIAL_STRIDE, height - height % SPATIAL_STRIDE
        references = [{"latent": encode_reference(vae, reference_pixels(image, width, height)), "token": token.to(torch.bfloat16)}]
        positive = node_helpers.conditioning_set_values(positive, {"magi2_reference_images": references})
        negative = node_helpers.conditioning_set_values(negative, {"magi2_reference_images": references})
        return io.NodeOutput(positive, negative)


class Magi2RefinerLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magi2RefinerLatent",
            display_name="MAGI-2 Refiner Latent",
            category="model/latent/magi2",
            description="Upsamples a MAGI-2 preview latent to the refiner's size and twice the frame rate (2T-1 latent frames). Sample it with the refiner from full noise: the refiner re-noises it itself.",
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("width", default=1920, min=16, max=MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1088, min=16, max=MAX_RESOLUTION, step=16),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, width, height) -> io.NodeOutput:
        video = _video_stream(samples["samples"])
        video = F.interpolate(video.float(), size=(video.shape[2] * 2 - 1, height // SPATIAL_STRIDE, width // SPATIAL_STRIDE),
                              mode="trilinear", align_corners=True)
        return io.NodeOutput({"samples": video})


class Magi2Decode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magi2Decode",
            display_name="MAGI-2 Decode",
            category="model/latent/magi2",
            description="Decodes the video with the turbo (or Wan 2.2) VAE and the audio with the Stable Audio Open VAE, time-stretched to 44.1 kHz.",
            inputs=[
                io.Latent.Input("samples", tooltip="A preview AV latent, or a refined video latent."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Latent.Input("audio", optional=True, tooltip="The preview AV latent whose audio to use with a refined video latent."),
                io.Float.Input("frame_rate", default=VIDEO_FPS, min=1.0, max=120.0, step=0.5, tooltip="12.5 for the preview, 25 after the refiner."),
            ],
            outputs=[io.Video.Output(), io.Image.Output(), io.Audio.Output()],
        )

    @classmethod
    def execute(cls, samples, video_vae, audio_vae, frame_rate, audio=None) -> io.NodeOutput:
        latent = samples["samples"]
        audio_latent = (audio or samples)["samples"]
        if not audio_latent.is_nested:
            raise ValueError("MAGI-2 Decode needs an AV latent for the audio")
        video = _video_stream(latent)
        if not isinstance(video_vae.first_stage_model, TurboVAEDecoder):
            # the turbo decoder maps the model's normalized latents to the Wan 2.2 basis itself
            video = latent_formats.Magi2().denormalize(video, video_vae.vae_dtype)
        images = video_vae.decode(video)
        images = images.reshape(-1, *images.shape[-3:])
        waveform = audio_vae.decode(audio_latent.unbind()[1]).movedim(-1, 1)
        waveform = fft_resample(waveform, waveform.shape[-1] * AUDIO_STRETCH[0] // AUDIO_STRETCH[1])
        sound = {"waveform": waveform[:1].cpu(), "sample_rate": 44100}
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=sound, frame_rate=Fraction(frame_rate)))
        return io.NodeOutput(video, images, sound)


class Magi2Extension(ComfyExtension):
    async def get_node_list(self):
        return [EmptyMagi2Latent, Magi2ImageToVideo, Magi2RefinerLatent, Magi2Decode]


async def comfy_entrypoint() -> Magi2Extension:
    return Magi2Extension()
