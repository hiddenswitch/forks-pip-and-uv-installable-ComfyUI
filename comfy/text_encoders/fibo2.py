"""Fibo 2.0 (briaai/Fibo-2.0-dev) text encoder: Qwen3-VL-4B-Instruct, six three-layer bundles.

diffusers BriaFibo2Pipeline (huggingface/diffusers de988b7e) wraps the prompt in the Qwen3-VL user turn
(no system turn, no think block), runs Qwen3-VL and takes hidden_states[k] for the layers of
TEXT_ENCODER_OUT_LAYERS: each layer scaled to unit RMS per token in float32 and cast back to the encoder's
dtype, each bundle its three layers concatenated (7680 features). The transformer reads the first bundle
through its Perceiver and one bundle in each injection block. ComfyUI carries conditioning as a 3D tensor,
so the six bundles are concatenated on the feature axis here, (B, text_len, 6 * 7680), and split in the
model. Fibo-2.0-dev's text_encoder is stock Qwen3-VL-4B-Instruct, tensor for tensor.

A generation prompt that is a JSON object is put in the form Fibo 2.0 was trained with, as the pipeline
does (_parse_prompt): empty values and the aesthetic_score and preference_score fields dropped, serialized
compactly. Any other prompt is encoded as it is. The pipeline also shows Qwen3-VL images of a caption's
text_render texts; that path is not ported, so such a caption is refused rather than encoded without them.
"""

import json
import math

import torch

from .. import sd1_clip
from . import qwen3vl

# Qwen3-VL hidden_states indices behind the transformer's six text inputs: the Perceiver's, then one per injection block
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = sorted({layer for bundle in TEXT_ENCODER_OUT_LAYERS for layer in bundle})
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
# the null caption Fibo 2.0 was trained with for classifier-free guidance, the pipeline's default negative prompt
DEFAULT_NEGATIVE_PROMPT = ('{"short_description":"","objects":[],"background_setting":"","lighting":{},'
                           '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
                           '"context":"","artistic_style":""}')
# the model card's text-to-image negative prompt (briaai/Fibo-2.0-dev README, Setup: NEGATIVE_T2I)
NEGATIVE_T2I = ('{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
                'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, '
                'no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":'
                '"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture",'
                '"lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights",'
                '"lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, '
                'heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},'
                '"style_medium":"","context":"","artistic_style":""}')
RMS_EPS = 1e-6


def _empty(value):
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean(value):
    if isinstance(value, dict):
        value = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean, value) if not _empty(v)]
    return value


def parse_prompt(prompt: str):
    """BriaFibo2Pipeline._parse_prompt: (the caption in its trained form, or the prompt itself when it is not
    a JSON object; the edit instruction or None)."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction
    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    caption = {
        key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
        for key, value in _clean(caption).items()
        if key not in scores
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    # a caption of exactly a short_description and a text_render list keeps both, the list even when empty
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "", "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def text_render_texts(caption) -> list[str]:
    """The texts the pipeline would draw as images for Qwen3-VL: text_render entries with text."""
    if not isinstance(caption, dict):
        return []
    return [entry["text"] for entry in caption.get("text_render") or []
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]]


def generation_prompt(prompt: str) -> str:
    """A text-to-image prompt as BriaFibo2Pipeline.__call__ encodes it: a JSON caption in its trained form,
    serialized compactly; anything else unchanged."""
    caption, instruction = parse_prompt(prompt)
    if instruction is not None:
        raise ValueError("a Fibo 2.0 edit prompt (edit_instruction) needs the images it edits; image editing is not ported")
    if texts := text_render_texts(caption):
        raise ValueError(f"the caption asks for rendered text {texts}: the pipeline shows Qwen3-VL images of those texts, "
                         "which is not ported")
    if isinstance(caption, str):
        return prompt
    return json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)


class Fibo2Tokenizer(qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], thinking=True, **kwargs):
        if llama_template is None and not text.startswith("<|im_start|>"):
            text, llama_template = generation_prompt(text), PROMPT_TEMPLATE
        # thinking=True: the Qwen3-VL tokenizer adds no empty think block, as the pipeline's template has none
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images,
                                             thinking=thinking, **kwargs)


class Fibo2Qwen3VLClipModel(qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}, textmodel_json_config=None):
        super().__init__(device=device, layer=TAP_LAYERS, layer_idx=None, dtype=dtype, attention_mask=attention_mask,
                         model_options=model_options, model_type="qwen3vl_4b", textmodel_json_config=textmodel_json_config)


def bundles(taps: torch.Tensor, dtype) -> torch.Tensor:
    """(B, len(TAP_LAYERS), L, 2560) hidden states -> (B, L, 6 * 7680): each layer at unit RMS per token in
    float32, cast to `dtype`, three layers per bundle, the bundles concatenated."""
    normalized = {}
    for i, layer in enumerate(TAP_LAYERS):
        states = taps[:, i].float()
        normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)).to(dtype)
    return torch.cat([normalized[layer] for bundle in TEXT_ENCODER_OUT_LAYERS for layer in bundle], dim=-1)


class Fibo2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, textmodel_json_config=None):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=Fibo2Qwen3VLClipModel, model_options=model_options,
                         textmodel_json_config=textmodel_json_config)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        out = bundles(out, self.qwen3vl_4b.transformer.dtype).to(out.dtype)
        if "attention_mask" in extra and extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Fibo2TEModel_(Fibo2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}, textmodel_json_config=None):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options, textmodel_json_config=textmodel_json_config)
    return Fibo2TEModel_
