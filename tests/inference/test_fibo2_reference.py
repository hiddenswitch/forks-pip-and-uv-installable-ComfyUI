"""Fibo 2.0 on its real weights against diffusers' BriaFibo2Pipeline (huggingface/diffusers de988b7e), one model at
a time on the GPU: the text bundles, one transformer forward, and a whole CFG sample from the same noise.

Needs briaai/Fibo-2.0-dev in the Hugging Face cache (gated: accept Bria's terms) and the repackaged transformer
(appmana/Fibo-2.0-ComfyUI diffusion_models/fibo2_dev_bf16.safetensors, or FIBO2_UNET), a CUDA GPU with 20 GB free.
"""
import gc
import json
import os

import pytest
import torch

pytestmark = [
    pytest.mark.requires_weights,
    pytest.mark.requires_accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

REPO = "briaai/Fibo-2.0-dev"
PROMPT = json.dumps({
    "short_description": "A pixel art sprite of a small cook holding a wooden spoon, standing on a flat teal background.",
    "objects": [{"description": "a small cook in a white apron and a tall chef's hat, holding a wooden spoon",
                 "location": "center", "relative_size": "medium"}],
    "background_setting": "an empty flat teal background",
    "lighting": {"conditions": "flat, even", "direction": "from the top left"},
    "aesthetics": {"composition": "centered", "color_scheme": "limited palette", "mood_atmosphere": "cheerful"},
    "style_medium": "pixel art",
    "artistic_style": "16-bit game sprite, hard pixel edges, no anti-aliasing",
})
WIDTH = HEIGHT = 512
STEPS = 12
CFG = 5.0


def _snapshot():
    import huggingface_hub
    try:
        return huggingface_hub.snapshot_download(REPO, local_files_only=True)
    except Exception:
        pytest.skip(f"{REPO} is not in the local Hugging Face cache")


def _unet_file():
    path = os.environ.get("FIBO2_UNET")
    if path:
        return path
    import huggingface_hub
    try:
        return huggingface_hub.hf_hub_download("appmana/Fibo-2.0-ComfyUI", "diffusion_models/fibo2_dev_bf16.safetensors",
                                               local_files_only=True)
    except Exception:
        pytest.skip("the repackaged transformer is not in the local Hugging Face cache (set FIBO2_UNET)")


def _free():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def reference():
    """The pipeline's embeddings, one transformer forward and a whole sample, each model loaded alone."""
    from diffusers import BriaFibo2Pipeline, BriaFibo2Transformer2DModel, FlowMatchEulerDiscreteScheduler, AutoencoderKLFlux2
    from diffusers.pipelines.bria_fibo2 import pipeline_bria_fibo2 as module
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from comfy.text_encoders.fibo2 import NEGATIVE_T2I
    snap = _snapshot()
    out = {}
    processor = Qwen3VLProcessor.from_pretrained(snap, subfolder="processor")
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(snap, subfolder="text_encoder", dtype=torch.bfloat16).to("cuda").eval()
    with torch.no_grad():
        # __call__ puts the prompt in its trained form and encodes the negative prompt as it is
        positive = json.dumps(BriaFibo2Pipeline._parse_prompt(PROMPT)[0], separators=(",", ":"), ensure_ascii=False)
        for name, text in (("positive", positive), ("negative", NEGATIVE_T2I)):
            embeds, mask = BriaFibo2Pipeline._get_qwen_prompt_embeds(encoder, processor, text, device="cuda")
            out[name] = embeds.cpu()
            out[f"{name}_ids"] = processor(text=[module.PROMPT_TEMPLATE.format(text)], return_tensors="pt").input_ids[0].tolist()
    del encoder
    _free()

    transformer = BriaFibo2Transformer2DModel.from_pretrained(snap, subfolder="transformer", dtype=torch.bfloat16).to("cuda").eval()
    g = torch.Generator("cpu").manual_seed(0)
    out["noise"] = torch.randn(1, 32, HEIGHT // 8, WIDTH // 8, generator=g)
    out["t"] = torch.tensor([0.7])
    with torch.no_grad():
        out["forward"] = transformer(out["noise"].to("cuda", torch.bfloat16), out["t"].cuda(), out["positive"].cuda(),
                                     return_dict=False)[0].float().cpu()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snap, subfolder="scheduler")
    vae = AutoencoderKLFlux2.from_pretrained(snap, subfolder="vae", dtype=torch.bfloat16).to("cuda")
    pipe = BriaFibo2Pipeline(transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=None, processor=None)
    with torch.no_grad():
        out["latents"] = pipe(prompt_embeds=out["positive"].cuda(), negative_prompt_embeds=out["negative"].cuda(), width=WIDTH,
                              height=HEIGHT, num_inference_steps=STEPS, guidance_scale=CFG, latents=out["noise"].cuda(),
                              output_type="latent").images.float().cpu()
    del pipe, transformer, vae
    _free()
    return out


def patchify(latents):
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def comfy_context(embeds):
    b, n, length, d = embeds.shape
    return embeds.permute(0, 2, 1, 3).reshape(b, length, n * d)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


@pytest.fixture(scope="module")
def comfy_conditioning(reference):
    """ComfyUI's encoding of the prompt and the negative with the stock Qwen3-VL-4B file."""
    import huggingface_hub
    from comfy import sd
    from comfy.text_encoders.fibo2 import NEGATIVE_T2I, generation_prompt
    path = huggingface_hub.hf_hub_download(REPO, "text_encoder/model.safetensors", local_files_only=True)
    clip = sd.load_clip(ckpt_paths=[path], clip_type=sd.CLIPType.FIBO2)
    out = {}
    for name, text in (("positive", generation_prompt(PROMPT)), ("negative", NEGATIVE_T2I)):
        tokens = clip.tokenize(text)
        out[f"{name}_ids"] = [t[0] for t in tokens["qwen3vl_4b"][0]]
        out[name] = clip.encode_from_tokens_scheduled(tokens)
    del clip
    _free()
    return out


def test_token_ids_are_the_processors(reference, comfy_conditioning):
    for name in ("positive", "negative"):
        assert comfy_conditioning[f"{name}_ids"] == reference[f"{name}_ids"]


def test_text_bundles_match_the_pipeline(reference, comfy_conditioning):
    for name in ("positive", "negative"):
        ours = comfy_conditioning[name][0][0]
        expected = comfy_context(reference[name]).float()
        assert ours.shape == expected.shape
        # ComfyUI runs Qwen3-VL's activations in float32 over the bf16 weights, the reference in bf16
        assert _cosine(ours, expected) > 0.999
        per_token = torch.nn.functional.cosine_similarity(ours[0].double(), expected[0].double(), dim=-1)
        assert per_token.min().item() > 0.99, per_token.min().item()


@pytest.fixture(scope="module")
def comfy_model():
    from comfy import sd
    model = sd.load_diffusion_model(_unet_file())
    yield model
    del model
    _free()


@pytest.fixture
def reference_attention_kernel(monkeypatch):
    """PyTorch's own SDPA kernel choice, as diffusers makes it: ComfyUI prefers cuDNN attention on NVIDIA."""
    import comfy.ldm.modules.attention
    monkeypatch.setattr(comfy.ldm.modules.attention, "scaled_dot_product_attention", torch.nn.functional.scaled_dot_product_attention)


def _forward(comfy_model, reference):
    from comfy import model_management
    model_management.load_models_gpu([comfy_model])
    diffusion = comfy_model.model.diffusion_model
    with torch.no_grad():
        return diffusion(patchify(reference["noise"]).to("cuda", torch.bfloat16), reference["t"].cuda(),
                         comfy_context(reference["positive"]).to("cuda", torch.bfloat16)).float().cpu()


def test_transformer_forward_is_the_reference_bit_for_bit(reference, comfy_model, reference_attention_kernel):
    assert torch.equal(_forward(comfy_model, reference), patchify(reference["forward"]))


def test_transformer_forward_on_comfyui_attention_is_within_kernel_noise(reference, comfy_model):
    # cuDNN attention against the reference's flash kernel: the reference's own flash-against-math difference is
    # cos 0.99996, a mean relative difference of 0.8%
    out, expected = _forward(comfy_model, reference), patchify(reference["forward"])
    assert _cosine(out, expected) > 0.9999
    assert ((out - expected).abs().mean() / expected.abs().mean()).item() < 0.01


def _sample(model, positive, negative, noise):
    from comfy import samplers
    from comfy_extras.nodes.nodes_fibo2 import fibo2_sigmas
    guider = samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(CFG)
    latent = torch.zeros_like(noise)
    return guider.sample(noise, latent, samplers.sampler_object("euler"), fibo2_sigmas(STEPS, WIDTH, HEIGHT), seed=0,
                         disable_pbar=True).float().cpu()


def test_a_whole_sample_matches_the_pipeline(reference, comfy_model, reference_attention_kernel):
    positive = [[comfy_context(reference["positive"]).float(), {}]]
    negative = [[comfy_context(reference["negative"]).float(), {}]]
    out = _sample(comfy_model, positive, negative, patchify(reference["noise"]))
    expected = patchify(reference["latents"])
    assert _cosine(out, expected) > 0.999
    assert ((out - expected).abs().mean() / expected.abs().mean()).item() < 0.03


def test_a_whole_sample_on_comfyui_conditioning_matches_the_pipeline(reference, comfy_model, comfy_conditioning):
    out = _sample(comfy_model, comfy_conditioning["positive"], comfy_conditioning["negative"], patchify(reference["noise"]))
    expected = patchify(reference["latents"])
    assert _cosine(out, expected) > 0.99
