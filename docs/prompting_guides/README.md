# Prompting Guides

Initial snapshot set fetched: 2026-06-03. Individual files may carry a newer
fetch date in their header.

Start with the [model prompting reference](model_coverage.md), which maps every
native model registration to a vendored guide, author instructions, model card,
or non-text input contract. It also links additional workflow families and
hosted providers. A model card is explicitly distinguished from a handbook.

Full audit: 2026-10-29. All existing guide sources were checked, including the
multiple sources in composite guides. Changed guidance appears above retained
historical snapshots or as dated corrections. Unchanged snapshots keep their
original fetch dates. Links to moving vendor pages are not immutable citations:
BFL's general pages now describe FLUX 3, and Alibaba's pages cover newer hosted
models than their older local snapshots.

Access limitations: Tencent's QQ handbook could not be retrieved; its author
repository and local base guide remain linked. Some author cards require access,
including Lens, Mage-Flow, Wan Dancer, FLUX 3 Klein and LTX-2.5; available public
documentation or Comfy-Org cards are linked instead. SD 2.x author card URLs also
returned 401. These are not claimed as freshly read snapshots.

The snapshots retain upstream instructions for provenance. Do not copy their
environment-install commands into a ComfyUI environment: use the repository's
`uv` setup and the native ComfyUI workflow instead.

Some snapshots describe hosted APIs rather than local models. Their presence here
does not establish local node support, available weights, or identical parameters.
Check the selected workflow and model before applying a provider-specific recipe.

## Black Forest Labs
- [FLUX 3 Image prompting](flux3_prompting.md) (current linked guidance; local/hosted distinctions)
- [FLUX Prompting Guide](bfl_flux_prompting_summary.md) ([source](https://docs.bfl.ml/guides/prompting_summary))
- [FLUX Prompting Basics](bfl_flux_prompting_t2i_fundamentals.md) ([source](https://docs.bfl.ml/guides/prompting_unified_basics); replaces retired `prompting_guide_t2i_fundamentals`)
- [FLUX T2I Essentials](bfl_flux_prompting_t2i_essentials.md) ([source](https://docs.bfl.ml/guides/prompting_unified_style); replaces retired `prompting_guide_t2i_essentials`)
- [FLUX T2I Advanced](bfl_flux_prompting_t2i_advanced.md) ([source](https://docs.bfl.ml/guides/prompting_unified_reference); replaces retired `prompting_guide_t2i_advanced`)
- [FLUX Negative Prompts](bfl_flux_prompting_t2i_negative.md) ([source](https://docs.bfl.ml/guides/prompting_guide_t2i_negative))
- [FLUX Kontext I2I](bfl_flux_prompting_kontext_i2i.md) ([source](https://docs.bfl.ml/guides/prompting_editing_overview); replaces retired `prompting_guide_kontext_i2i`)
- [FLUX.2 Prompting Guide](bfl_flux2_prompting.md) ([source](https://docs.bfl.ml/guides/prompting_guide_flux2))
- [FLUX.2 family API and Klein references](bfl_flux2_klein_prompting.md) ([source](https://docs.bfl.ml/flux_2/flux2_text_to_image); not a dedicated Klein handbook)

## Ideogram
- [Ideogram 4 model card](ideogram4_comfy_org_model_card.md) ([source](https://huggingface.co/Comfy-Org/Ideogram-4))
- [Ideogram 4 Comfy workflow template](ideogram4_comfy_workflow_template.json) ([source](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ideogram4_t2i.json))
- [Ideogram 4 prompting guide](ideogram4_prompting.md) ([source](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md))
- [Ideogram 4.0 JSON prompting, Ideogram docs](ideogram4_docs_json_prompting.md) ([source](https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/4.-json-prompting-ideogram-4.0))
- [How to JSON prompt for Ideogram 4.0, Ideogram blog](ideogram4_blog_json_prompting.md) ([source](https://ideogram.ai/blog/ideogram-4-json-prompting/))
- [Ideogram 4 magic-prompt system prompt v1](ideogram4_magic_prompt_system_prompt_v1.md) ([source](https://github.com/ideogram-oss/ideogram4/blob/main/src/ideogram4/magic_prompt_system_prompts/v1.txt))
- [Ideogram 4.5 API](ideogram4_5_api.md) ([source](https://developer.ideogram.ai/api-reference/images/generate/ideogram-4-5); hosted API reference)
- [Ideogram 4.5 Precise Edit API](ideogram4_5_precise_edit_api.md) ([source](https://developer.ideogram.ai/api-reference/images/precise-edit/ideogram-4-5))
- [Ideogram 4.5 prompting guide](ideogram4_5_prompting.md) ([source](https://docs.ideogram.ai/prompting/prompting); the docs' Prompting section, which covers 4.5, 4.0 and 3.0)

## Alibaba
- [Qwen Image 2.1 Edit enhancer instructions](qwen_image_2_1_edit_prompting.md) ([official system prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt); task-specific contract)
- [Wan text-to-video/image-to-video prompt guide](alibaba_wan_video_prompting.md) ([source](https://www.alibabacloud.com/help/en/model-studio/text-to-video-prompt))
- [Qwen Image Edit guide](alibaba_qwen_image_edit_prompting.md) ([source](https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-guide))
- [Qwen Image 2.1 prompting guide](qwen_image_2_1_prompting.md) ([source](https://github.com/QwenLM/Qwen-Image-2.1); README, prompt_rewrite README and the official t2i rewriting system prompt, complete)

## MiniMax
- [MiniMax H3 video-generation guide](minimax_h3_video_generation.md) ([source](https://platform.minimax.io/docs/guides/video-generation))

## Tongyi-MAI
- [Z-Image Turbo model card](z_image_turbo_model_card.md) ([source](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo))

## Kandinsky Lab
- [Kandinsky 6.0 Video prompting](kandinsky6_prompting.md) ([official text-to-video instructions](https://github.com/kandinskylab/kandinsky-6/blob/main/comfyui/kandinsky6/beautifier_prompts/t2av_system.txt), [image-to-video instructions](https://github.com/kandinskylab/kandinsky-6/blob/main/comfyui/kandinsky6/beautifier_prompts/i2av_system.txt); source summary)

## Tencent
- [Hunyuan Image 3.0 Instruct prompting references](hunyuan_image3_instruct_prompting.md) ([official model card](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct), [Prompt Handbook](https://docs.qq.com/doc/DUVVadmhCdG9qRXBU); source summary)
- [Hunyuan Image 3.0 prompting](hunyuan_image3_prompting.md) ([source](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/Hunyuan-Image3.md))
- [Hunyuan Video 1.5 prompt handbook](hunyuan_video_1_5_prompt_handbook.md) ([source](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md))

## Lightricks
- [LTX-Video README prompt section](ltx_video_readme.md) ([source](https://github.com/Lightricks/LTX-Video/blob/main/README.md))
- [LTX-2 family: current 2.5 guidance and 2.3 snapshot](ltx_2_readme.md) ([source](https://github.com/Lightricks/LTX-2/blob/main/README.md))

## NVIDIA Toronto AI Lab
- [ChronoEdit prompt guidance](chronoedit_prompt_guidance.md) ([source](https://github.com/nv-tlabs/ChronoEdit/blob/main/docs/PROMPT_GUIDANCE.md))

## NVIDIA
- [PixelDiT prompting](nvidia_pixeldit_prompting.md) ([pinned source](https://github.com/NVlabs/PixelDiT/tree/f7d7c35/t2i); the CHI system prompt, the 300-row window, the training captions and the sampling recipes)

## VectorSpaceLab
- [OmniGen2 README](omnigen2_readme.md) ([source](https://github.com/VectorSpaceLab/OmniGen2))

## lodestones
- [Chroma1-HD model card](chroma1_hd_model_card.md) ([source](https://huggingface.co/lodestones/Chroma1-HD))

## circlestone-labs
- [Anima model card](anima_model_card.md) ([source](https://huggingface.co/circlestone-labs/Anima))

## NewBie-AI
- [NewBie Image Exp0.1 model card](newbie_image_exp01_model_card.md) ([source](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1))

## Phantom-video
- [HuMo README](humo_readme.md) ([source](https://github.com/Phantom-video/HuMo))

## ACE-Step
- [ACE-Step 1.5 tutorial](ace_step_1_5_tutorial.md) ([source](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md))
