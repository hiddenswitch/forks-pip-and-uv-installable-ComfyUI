"""Fibo 2.0 against diffusers' BriaFibo2 reference (huggingface/diffusers de988b7e): the transformer on a tiny
config, the prompt form, the text bundles and the schedule."""
import json

import pytest
import torch

import comfy.ops
from comfy import model_detection, supported_models
from comfy.ldm.fibo2.model import Fibo2
from comfy.model_downloader import KNOWN_CLIP_MODELS, KNOWN_UNET_MODELS, KNOWN_VAES
from comfy.nodes.base_nodes import CLIPLoader
from comfy.text_encoders import fibo2 as fibo2_te
from comfy_extras.nodes.nodes_fibo2 import fibo2_sigmas

reference = pytest.importorskip("diffusers.models.transformers.transformer_bria_fibo2")
pipeline_module = pytest.importorskip("diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = dict(in_channels=4, patch_size=2, dim=64, n_layers=4, n_refiner_layers=2, n_heads=2, norm_eps=1e-5, cap_feat_dim=24,
            injection_layer_ids=(1, 3), perceiver_num_layers=2, min_num_gist_tokens=4, max_num_gist_tokens=12, gist_step=4,
            gist_min_text_len=4, gist_max_text_len=20, rope_theta=256.0, t_scale=1000.0, axes_dims=(8, 12, 12),
            axes_lens=(64, 32, 32))


@pytest.fixture
def pytorch_attention(monkeypatch):
    import comfy.model_management
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: False)
    monkeypatch.setattr(comfy.model_management, "sage_attention_enabled", lambda: False)
    monkeypatch.setattr(comfy.model_management, "flash_attention_enabled", lambda: False)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_enabled", lambda: True)


def _models(seed=0):
    torch.manual_seed(seed)
    ref = reference.BriaFibo2Transformer2DModel(**TINY)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.2)
    ref = ref.to(DEVICE).eval()
    ours = Fibo2(**TINY, operations=comfy.ops.disable_weight_init)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True)
    assert not missing and not unexpected
    return ref, ours.to(DEVICE).eval()


def patchify(latents):
    """The reference's normalized (B, C, H/8, W/8) latent as ComfyUI's FLUX.2 latent: (B, 4C, H/16, W/16), (c, dy, dx)."""
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def comfy_context(encoder_hidden_states):
    """(B, bundles, L, D) -> (B, L, bundles * D)."""
    b, n, length, d = encoder_hidden_states.shape
    return encoder_hidden_states.permute(0, 2, 1, 3).reshape(b, length, n * d)


def _inputs(batch=1, text_len=9, h=8, w=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(batch, TINY["in_channels"], h, w, generator=g)
    t = torch.rand(batch, generator=g)
    bundles = 1 + len(TINY["injection_layer_ids"])
    text = torch.randn(batch, bundles, text_len, TINY["cap_feat_dim"], generator=g)
    return latents.to(DEVICE), t.to(DEVICE), text.to(DEVICE)


@pytest.mark.parametrize("text_len", [3, 9, 25])
def test_forward_matches_the_reference(text_len, pytorch_attention):
    ref, ours = _models()
    latents, t, text = _inputs(text_len=text_len)
    with torch.no_grad():
        expected = ref(latents, t, text, return_dict=False)[0]
        actual = ours(patchify(latents), t, comfy_context(text))
    torch.testing.assert_close(actual, patchify(expected), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 on CUDA")
@pytest.mark.parametrize("text_len", [3, 25])
def test_bf16_forward_is_the_reference_bit_for_bit(text_len, pytorch_attention):
    ref, ours = _models()
    ref, ours = ref.to(torch.bfloat16), ours.to(torch.bfloat16)
    latents, t, text = _inputs(text_len=text_len)
    with torch.no_grad():
        expected = ref(latents.to(torch.bfloat16), t, text.to(torch.bfloat16), return_dict=False)[0]
        actual = ours(patchify(latents).to(torch.bfloat16), t, comfy_context(text).to(torch.bfloat16))
    assert torch.equal(actual, patchify(expected))


def test_padded_prompts_with_different_gist_counts_match_the_reference(pytorch_attention):
    # lengths 5 and 17 give 4 and 12 gist tokens: the gist padding goes last, masked, as in the reference
    ref, ours = _models()
    latents, t, text = _inputs(batch=2, text_len=17)
    mask = torch.tensor([[i < 5 for i in range(17)], [True] * 17], device=DEVICE)
    with torch.no_grad():
        expected = ref(latents, t, text, encoder_attention_mask=mask, return_dict=False)[0]
        actual = ours(patchify(latents), t, comfy_context(text), attention_mask=mask)
        alone = ours(patchify(latents[:1]), t[:1], comfy_context(text[:1, :, :5]))
    torch.testing.assert_close(actual, patchify(expected), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual[:1], alone, rtol=1e-4, atol=1e-4)


def test_context_images_match_the_reference(pytorch_attention):
    ref, ours = _models()
    latents, t, text = _inputs()
    g = torch.Generator().manual_seed(3)
    contexts = [torch.randn(1, TINY["in_channels"], 6, 10, generator=g).to(DEVICE), torch.randn(1, TINY["in_channels"], 8, 8, generator=g).to(DEVICE)]
    with torch.no_grad():
        expected = ref(latents, t, text, context_latents=contexts, return_dict=False)[0]
        actual = ours(patchify(latents), t, comfy_context(text), ref_latents=[patchify(c) for c in contexts])
    torch.testing.assert_close(actual, patchify(expected), rtol=1e-4, atol=1e-4)


def test_backpropagates_through_every_parameter(pytorch_attention, monkeypatch):
    # training runs the differentiable rope (comfy.model_management.in_training), as trainers set it
    import comfy.model_management
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    _, ours = _models()
    ours.train()
    latents, t, text = _inputs()
    ours(patchify(latents), t, comfy_context(text)).square().mean().backward()
    for name, p in ours.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_accumulated_gradients_match_reference_and_full_batch(pytorch_attention, monkeypatch):
    import comfy.model_management
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    ref, ours = _models()
    ref.train()
    ours.train()
    latents, t, text = _inputs(batch=2)
    ref(latents, t, text, return_dict=False)[0].square().mean().backward()
    ours(patchify(latents), t, comfy_context(text)).square().mean().backward()
    full_batch = {name: p.grad.clone() for name, p in ours.named_parameters()}
    ours.zero_grad(set_to_none=True)
    for i in range(2):
        loss = ours(patchify(latents[i:i + 1]), t[i:i + 1], comfy_context(text[i:i + 1])).square().mean()
        (loss / 2).backward()
    reference_params = dict(ref.named_parameters())
    for name, p in ours.named_parameters():
        assert torch.isfinite(p.grad).all(), name
        for expected in (full_batch[name], reference_params[name].grad):
            # Reduction order changes with batch size; measure each parameter's full gradient, including near-zero entries.
            relative_error = (p.grad - expected).norm() / expected.norm().clamp_min(1e-12)
            assert relative_error < 1e-4, (name, relative_error.item())


def _state(**shapes):
    return {k: torch.empty(*v, device="meta") for k, v in shapes.items()}


def test_detection_reads_the_config_from_the_weights():
    ref = reference.BriaFibo2Transformer2DModel(**TINY)
    sd = {k: torch.empty(v.shape, device="meta") for k, v in ref.state_dict().items()}
    config = model_detection.detect_unet_config(sd, "")
    assert config == {"image_model": "fibo2", "dim": 64, "patch_size": 2, "in_channels": 4, "n_heads": 2, "n_layers": 4,
                      "n_refiner_layers": 2, "perceiver_num_layers": 2, "cap_feat_dim": 24, "max_num_gist_tokens": 12,
                      "injection_layer_ids": [1, 3]}
    model = supported_models.Fibo2(config)
    assert model.clip_target().tokenizer is fibo2_te.Fibo2Tokenizer
    assert "fibo2" in CLIPLoader.INPUT_TYPES()["required"]["type"][0]


def test_release_detection_is_the_release_config():
    import safetensors
    import huggingface_hub
    try:
        index = huggingface_hub.hf_hub_download("briaai/Fibo-2.0-dev", "transformer/diffusion_pytorch_model.safetensors.index.json",
                                                local_files_only=True)
        config = json.loads(open(huggingface_hub.hf_hub_download("briaai/Fibo-2.0-dev", "transformer/config.json", local_files_only=True)).read())
    except Exception:
        pytest.skip("briaai/Fibo-2.0-dev is not in the local Hugging Face cache")
    shapes = {}
    for shard in sorted(set(json.load(open(index))["weight_map"].values())):
        path = huggingface_hub.hf_hub_download("briaai/Fibo-2.0-dev", f"transformer/{shard}", local_files_only=True)
        with safetensors.safe_open(path, "pt") as f:
            for k in f.keys():
                shapes[k] = torch.empty(f.get_slice(k).get_shape(), device="meta")
    detected = model_detection.detect_unet_config(shapes, "")
    for key in ("dim", "n_layers", "n_refiner_layers", "n_heads", "cap_feat_dim", "perceiver_num_layers", "max_num_gist_tokens",
                "in_channels", "patch_size"):
        assert detected[key] == config[key], key
    assert detected["injection_layer_ids"] == config["injection_layer_ids"]
    defaults = Fibo2.__init__.__defaults__
    names = Fibo2.__init__.__code__.co_varnames[1:1 + len(defaults)]
    for key, value in zip(names, defaults):
        if key in config and key not in detected:
            assert tuple(config[key]) == tuple(value) if isinstance(value, tuple) else config[key] == value, key


PROMPTS = [
    '{"short_description":"A red bicycle.","objects":[{"description":"a bicycle","location":"center","pose":""}],'
    '"lighting":{},"aesthetics":{"aesthetic_score":"high","composition":"centered"},"preference_score":"high","style_medium":"photograph"}',
    json.dumps({"short_description": "x", "text_render": []}, indent=2),
    json.dumps({"short_description": "", "text_render": [{"bbox": [1]}]}),
    '{\n  "short_description": "pixel art",\n  "objects": [],\n  "context": "",\n  "artistic_style": "pixel art, 16x16 cells"\n}',
    "a short prose description, (not JSON)",
    "[1, 2]",
    fibo2_te.DEFAULT_NEGATIVE_PROMPT,
    fibo2_te.NEGATIVE_T2I,
]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_generation_prompt_is_the_pipelines(prompt):
    pipe = pipeline_module.BriaFibo2Pipeline
    caption, _ = pipe._parse_prompt(prompt)
    expected = prompt if isinstance(caption, str) else json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)
    assert fibo2_te.generation_prompt(prompt) == expected


def test_the_card_negative_is_the_model_cards():
    assert json.loads(fibo2_te.NEGATIVE_T2I)["aesthetics"]["artifacts"][-1] == "haze"
    assert fibo2_te.DEFAULT_NEGATIVE_PROMPT == pipeline_module.DEFAULT_NEGATIVE_PROMPT


@pytest.mark.parametrize("prompt", ['{"short_description":"a sign","text_render":[{"text":"OPEN"}]}',
                                    '{"edit_instruction":"make it red"}'])
def test_unported_paths_are_refused(prompt):
    with pytest.raises(ValueError):
        fibo2_te.generation_prompt(prompt)


def test_bundles_are_the_pipelines():
    assert fibo2_te.TEXT_ENCODER_OUT_LAYERS == pipeline_module.TEXT_ENCODER_OUT_LAYERS
    assert fibo2_te.PROMPT_TEMPLATE == pipeline_module.PROMPT_TEMPLATE
    g = torch.Generator().manual_seed(0)
    hidden_states = [torch.randn(2, 7, 2560, generator=g).to(torch.bfloat16) * (i + 1) for i in range(37)]
    normalized = {}
    for layer in sorted({layer for layers in pipeline_module.TEXT_ENCODER_OUT_LAYERS for layer in layers}):
        states = hidden_states[layer].float()
        normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    expected = torch.stack([torch.cat([normalized[layer] for layer in layers], dim=-1)
                            for layers in pipeline_module.TEXT_ENCODER_OUT_LAYERS], dim=1)
    taps = torch.stack([hidden_states[layer].float() for layer in fibo2_te.TAP_LAYERS], dim=1)
    actual = fibo2_te.bundles(taps, torch.bfloat16)
    assert torch.equal(actual, comfy_context(expected))


@pytest.mark.parametrize("steps", [20, 30, 50, 60])
@pytest.mark.parametrize("size", [(1024, 1024), (512, 512), (1424, 800)])
def test_schedule_is_the_pipelines(steps, size):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    width, height = size
    scheduler = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15,
                                                num_train_timesteps=1000, shift=3.0, time_shift_type="exponential",
                                                use_dynamic_shifting=True)
    mu = pipeline_module.calculate_shift((height // 16) * (width // 16), 256, 4096, 0.5, 1.15)
    pipeline_module.retrieve_timesteps(scheduler, steps, "cpu", sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    assert torch.equal(fibo2_sigmas(steps, width, height), scheduler.sigmas)


def test_the_weights_are_registered():
    unets = {str(m): m for m in KNOWN_UNET_MODELS.data}
    assert unets["fibo2_dev_bf16.safetensors"].repo_id == "appmana/Fibo-2.0-ComfyUI"
    clips = {str(m): m for m in KNOWN_CLIP_MODELS.data}
    assert clips["qwen3vl_4b_bf16.safetensors"].repo_id in ("Comfy-Org/Qwen3-VL", "Comfy-Org/Mage-Flow")
    assert "flux2-vae.safetensors" in {str(m) for m in KNOWN_VAES.data}


def test_the_tokenizer_encodes_the_prompt_as_written():
    # encode_prompt's form: the user turn around the text, no cleaning and no think block
    from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer
    tokenizer = fibo2_te.Fibo2Tokenizer()
    spaced = json.dumps({"short_description": "a cat", "lighting": {}}, indent=2)
    ids = [t[0] for t in tokenizer.tokenize_with_weights(spaced)["qwen3vl_4b"][0]]
    reference = Qwen3VLTokenizer(model_type="qwen3vl_4b").tokenize_with_weights(
        fibo2_te.PROMPT_TEMPLATE.format(spaced), skip_template=True)
    assert ids == [t[0] for t in reference["qwen3vl_4b"][0]]


def test_the_caption_node_is_the_pipelines_prompt_form():
    from comfy_extras.nodes.nodes_fibo2 import Fibo2Caption
    assert Fibo2Caption.execute(PROMPTS[0]).result == (fibo2_te.generation_prompt(PROMPTS[0]),)
