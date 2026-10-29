# Qwen Image 2.1 Edit prompting

Reviewed: 2026-10-29. The authoritative editing reference is the complete
[official Edit enhancer system prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt),
not the text-to-image enhancer prompt. This page summarizes that contract;
read the linked source for its full decision rules.

Lead with the requested editing operation. Preserve unspecified content and
ground descriptions in the supplied images. Distinguish a local modification
from a new scene using an identity reference. For multiple images, use literal
`<image1>`, `<image2>` references and state each image's role, including which
is the canvas. Single-image prose uses natural references instead.

Chinese requests retain Chinese prose; English requests retain English; other
languages use English prose. Quoted rendered text follows explicit user wording
or language first, existing image text second, then the request language.

The enhancer returns JSON with `rewritten_prompt`, `wh_ratio`, and `ratio_follow`.
The prompt is one paragraph of editing instructions with quoted lettering and
no resolution or ratio instructions. Exactly one ratio field is populated;
the other is an empty string. Explicit dimensions determine the ratio; ordinary
edits inherit the canvas ratio through `ratio_follow` (for example `<image1>`).
New scenes, outpainting and panel layouts have additional rules in the source.

## Enhancer checkpoint and workflow

The [official enhancer README](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/README.md)
specifies separate fine-tuned PE-T2I and PE-I2I checkpoints and system prompts.
Use PE-I2I with ordered input images and thinking enabled. Its documented
`presence_penalty` is 0 and `max_new_tokens` is 24000, versus 1.5 and 16256 for
PE-T2I. Stock Qwen3.5-VL weights are not the task-specific enhancer.

Pass `rewritten_prompt` as the positive diffusion prompt and apply the returned
ratio metadata to the canvas; do not pass the JSON envelope as the caption.
The reference pipeline uses an empty negative prompt.

See the [generation and shared model guide](qwen_image_2_1_prompting.md) for
the retained author README and task-specific sampling notes.
