# FLUX 3 Image prompting

Reviewed: 2026-10-29. Maintained summary with links to BFL's current guides.

Describe the subject, action, setting, framing and lighting in natural language.
Quote lettering exactly and specify its location and appearance. For editing,
identify the target, the change, and the details to preserve. With references,
name each image by its position and explain what it contributes.

For explicit composition in bounding-box mode, follow BFL's caption-and-element layout format:
each element has an `id`, `bbox`, and `desc`; the caption references its `<id>`.
Boxes use `[top, left, bottom, right]` on a 0–1000 grid. Match the aspect ratio
to the intended layout. This is distinct from Ideogram's JSON caption schema.

Read the full [prompt reference](https://docs.bfl.ml/guides/prompting_unified_reference.md),
[basics](https://docs.bfl.ml/guides/prompting_unified_basics.md),
[style and text guide](https://docs.bfl.ml/guides/prompting_unified_style.md), and
[editing guide](https://docs.bfl.ml/guides/prompting_editing_overview.md).

## Hosted API and local workflows

BFL's hosted full FLUX 3 Image API accepts up to ten reference images. Its request
schema does not expose `seed`, `guidance`, `negative_prompt`, or
`prompt_upsampling`; unsupported fields are rejected. Those restrictions describe
that endpoint, not Klein's API or a local ComfyUI sampler.
See [generation](https://docs.bfl.ml/flux_3/flux3_image_generate.md) and
[Klein deployment](https://docs.bfl.ml/flux_3/flux3_image_klein.md) for the exact
variant. The Klein Hugging Face raw card required access during this review;
the public BFL documentation remains the linked reference.

## Klein: expand prompts before generation

Klein was trained on detailed scene captions. BFL recommends its
[dedicated upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
for short prompts and editing instructions. Preserve reference-image order
through enhancement and generation. The reference CLI applies enhancement by
default; the Diffusers pipeline requires a separate enhancement call. A local
workflow should include the matching enhancer or receive an already complete
caption. Stock Qwen weights are not the fine-tuned upsampler.

The public [upsampler system_prompts.json](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json)
ships four variants: `t2i`, `i2i`, `t2i_bbox`, and `i2i_bbox`. Plain modes output
prose without element IDs or boxes. Bounding-box T2I uses the `id`/`bbox`/`desc`
layout above; bounding-box I2I instead uses `id`, `from`, `src_bbox`, `tgt_bbox`,
and `desc`, with zero-based `ref_image_0` sources. Select the format expected by
the workflow. See the [enhancer audit](prompt_upsampling.md#flux) for details.

The [Klein guide](https://docs.bfl.ml/flux_3/flux3_image_klein.md) specifies
16 steps/CFG 1 for Distilled and 50 steps/CFG 3.5 for Base. Klein's hosted API
adds `seed`, `steps` and `output_format`; do not apply the full model's field
restrictions to it. Train LoRAs on Base and use the appropriate inference recipe.

Earlier families retain their own instructions: [FLUX.2](bfl_flux2_prompting.md)
and [FLUX.1 Kontext](bfl_flux_prompting_kontext_i2i.md). FLUX 3 Video and Action
are separate models; this page documents Image.
