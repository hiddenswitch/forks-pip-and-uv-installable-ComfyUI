# Official prompt enhancement references

Audited: 2026-10-29. This supplements the [model coverage index](model_coverage.md)
with instructions shipped by model authors for rewriting, expanding, or
structuring prompts. These can be the most precise available prompting guide,
even when embedded in a Python module, demo Space, or authoring skill.

The tables account for all 107 native registrations, followed by the additional
families in the coverage index. A public instruction is linked at the reviewed
revision where available. Read the entire linked prompt, including examples and
the calling code: the short descriptions here do not replace it. Upstream skills
are references for their models, not instructions to install or execute here.

## How to interpret the audit

- **Published**: an author-provided expansion/rewrite instruction was located.
- **Skill**: the author publishes an authoring workflow and reference documents,
  rather than a single system-message string.
- **Related**: a specialized enhancer, encoder prefix, captioner, evaluator, or
  agent instruction exists, with its actual role stated explicitly.
- **Not located**: no public model-specific enhancer instruction was located in
  the checked author cards, guides, repositories and linked demos. This is a
  bounded audit finding, not proof that an internal or unlinked prompt does not
  exist. Use the linked model's ordinary prompting instructions.
- **Restricted**: author material could not be read; public workflow references
  do not establish what the author uses internally.
- **Input task**: the model consumes images, geometry, masks, latents, or task
  queries rather than an expanded generative scene caption.

These distinctions matter when using an enhancer. Some return plain text; others
return a JSON envelope whose caption must be extracted, or structured JSON that
is itself the model input. An encoder's system prefix does not mean the encoder
generates a rewritten caption. A stock language model with the same architecture
does not reproduce a fine-tuned enhancer's behavior. Published word budgets are
enhancer instructions, not diffusion tokenizer limits or sampling parameters.

## Native image models

| Family / registrations | Finding and source |
|---|---|
| SD 1.x — `SD15` | Not located: [checkpoint card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). |
| SD 2.x — `SD20`, `SD21UnclipL`, `SD21UnclipH` | Restricted: [author card](https://huggingface.co/stabilityai/stable-diffusion-2-1); see [access notes](model_coverage.md). |
| SDXL — `SDXL`, `SDXLRefiner` | Not located: [base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0). |
| SDXL derivatives — `SSD1B`, `Segmind_Vega`, `KOALA_700M`, `KOALA_1B` | Not located: [SSD-1B](https://huggingface.co/segmind/SSD-1B), [Vega](https://huggingface.co/segmind/Segmind-Vega), [KOALA](https://github.com/youngwanLEE/KOALA). |
| Instruction editing — `SD15_instructpix2pix`, `SDXL_instructpix2pix` | Related: [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) uses generated instruction data; no inference enhancer located for it or [CosXL](https://huggingface.co/stabilityai/cosxl). |
| SD3 / SD3.5 — `SD3` | Not located: [author card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large). |
| Stable Cascade — `Stable_Cascade_C`, `Stable_Cascade_B` | Not located: [author repository](https://github.com/Stability-AI/StableCascade); Stage B is a decoder stage. |
| AuraFlow — `AuraFlow` | Not located: [author card](https://huggingface.co/fal/AuraFlow-v0.3). |
| PixArt — `PixArtAlpha`, `PixArtSigma` | Not located: [Alpha](https://github.com/PixArt-alpha/PixArt-alpha), [Sigma](https://github.com/PixArt-alpha/PixArt-sigma); caption training is not an inference system prompt. |
| HunyuanDiT — `HunyuanDiT`, `HunyuanDiT1` | Related: [DialogGen](#hunyuan-image-and-dit) is a fine-tuned conversational enhancer with a short task wrapper. |
| FLUX.1 — `Flux`, `FluxSchnell`, `FluxInpaint` | Not located: [author repository](https://github.com/black-forest-labs/flux). Do not infer the hosted upsampler's private instructions from an API option. |
| FLUX.2 — `Flux2` | Published: [separate T2I and I2I instructions](#flux). |
| FLUX 3 — `Flux3` | Published: [four Klein enhancer variants](#flux), with and without bounding boxes. |
| Lens — `Lens` | Restricted: [author card](https://huggingface.co/microsoft/Lens); [public workflow card](https://huggingface.co/Comfy-Org/Lens). |
| Lumina — `Lumina2` | Not located: [author repository](https://github.com/Alpha-VLLM/Lumina-Image-2.0). |
| Z-Image — `ZImage`, `ZImagePixelSpace` | Published: [official demo Space prompts](#other-image-enhancers). |
| Ming — `MingImage` | Published: [design-layout JSON rewriter](#other-image-enhancers). |
| PixelDiT — `PixelDiTT2I` | Related: [CHI encoder prefix](nvidia_pixeldit_prompting.md#the-system-prompt-chi-and-the-300-row-window), not an autoregressive expansion call. |
| HiDream-I1 — `HiDream` | Not located: [I1 repository](https://github.com/HiDream-ai/HiDream-I1). O1's prompt agent is a separate release. |
| HiDream O1 — `HiDreamO1` | Published: [prompt agent](#other-image-enhancers). |
| SenseNova — `SenseNovaU15` | Published: [infographic enhancement](#other-image-enhancers), with a separate builder prompt. |
| Chroma — `Chroma`, `ChromaRadiance` | Not located: [HD](chroma1_hd_model_card.md), [Radiance](https://huggingface.co/lodestones/Chroma1-Radiance). |
| Anima — `Anima` | Not located: [checkpoint instructions](anima_model_card.md); preserve variant-specific tags and triggers. |
| OmniGen2 — `Omnigen2` | Related: [prompt_generator.py](https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/omnicontext/prompt_generator.py) generates evaluation instructions, not enhanced generation prompts. No inference rewriter located. |
| Boogu — `Boogu` | Published: [T2I, editing, and two-stage slide prompts](#boogu). |
| MOSS — `MOSSImage` | Skill: [caption/layout enhancer](#other-image-enhancers), including compilation into model input. |
| Fibo 2 — `Fibo2` | Published: [Bria VGL task prompts](#fibo-2); checkpoint style-preset files are restricted. |
| Mage-Flow — `MageFlow` | Restricted: [author card](https://huggingface.co/microsoft/Mage-Flow); [public workflow card](https://huggingface.co/Comfy-Org/Mage-Flow). |
| Qwen Image / Edit / Layered — `QwenImage` | Published: [original and 2512 rewriters](#qwen-image). No distinct Layered enhancer located in its [card](https://huggingface.co/Qwen/Qwen-Image-Layered). |
| Qwen Image 2.1 — `QwenImage21` | Published: [separate PE-T2I and PE-I2I instructions](#qwen-image). |
| JoyAI — `JoyImage` | Not located: [author edit card](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-ComfyUI). |
| Ideogram 4 — `Ideogram4` | Published: [complete vendored Magic Prompt v1](ideogram4_magic_prompt_system_prompt_v1.md), producing the [native structured caption](ideogram4_prompting.md). |
| Krea 2 — `Krea2` | Published: [official expansion.txt](#other-image-enhancers). |
| HunyuanImage 2.1 — `HunyuanImage21`, `HunyuanImage21Refiner` | Published: [reprompt and 32B variants](#hunyuan-image-and-dit). The refiner does not introduce a separate enhancer contract. |
| Kandinsky 5 Image — `Kandinsky5Image` | Related: [image/edit encoder prefixes](#kandinsky); the published generative expansion method is I2V-specific. |
| LongCat Image — `LongCatImage` | Published: [bilingual T2I rewrite instructions](#other-image-enhancers). Do not automatically apply them to the Edit checkpoint. |
| ERNIE — `ErnieImage` | Published: [PE JSON contract](#other-image-enhancers). |

## Native video models

| Family / registrations | Finding and source |
|---|---|
| Mochi — `GenmoMochi` | Not located: [author inference repository](https://github.com/genmoai/models). |
| LTX — `LTXV`, `LTXAV` | Published: [LTX-Video and four encoder/task-specific LTX-2 prompts](#ltx). |
| MiniMax — `MiniMaxH3` | Skill: [official H3 base/keyframe and full-reference rewrite guides](#minimax). |
| HunyuanVideo — `HunyuanVideo`, `HunyuanVideoI2V`, `HunyuanVideoSkyreelsI2V` | Published for original T2V: [normal/master rewrite](#hunyuan-video). No separate I2V or SkyReels-specific enhancer located in [I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) / [SkyReels](https://github.com/SkyworkAI/SkyReels-V1) sources. |
| HunyuanVideo 1.5 — `HunyuanVideo15`, `HunyuanVideo15_SR_Distilled` | Published: [distinct T2V and I2V rewrites](#hunyuan-video); no independent SR enhancer. |
| Cosmos — `CosmosT2V`, `CosmosI2V`, `CosmosT2IPredict2`, `CosmosI2VPredict2` | Published/related: [Cosmos 1 upsampler and Predict2 visual refiner](#cosmos). Predict2's image encoder prefix is not the refiner. |
| Wan — `WAN21_T2V`, `WAN21_I2V`, `WAN22_T2V` | Published: [versioned T2V/I2V/first-last-frame instructions](#wan-and-derivatives). |
| Self Forcing — `WAN21_CausalAR_T2V` | Published: [bundled Wan prompt-extension module](#wan-and-derivatives); its presence does not establish default invocation. |
| VideoX-Fun — `WAN21_FunControl2V`, `WAN21_Camera`, `WAN22_Camera` | Published: [beautification and caption-rewrite utility prompts](#wan-and-derivatives); control/camera inputs remain separate. |
| VACE — `WAN21_Vace` | Published: [wrapper around Wan's expander](#wan-and-derivatives). |
| HuMo — `WAN21_HuMo` | Not located: [author README](humo_readme.md); text, reference image and audio have distinct roles. |
| Wan speech / animation — `WAN22_S2V`, `WAN22_Animate`, `WAN_Animate2` | Related: [Wan 2.2 repository](https://github.com/Wan-Video/Wan2.2) includes the T2V/I2V expander; no dedicated speech/animation rewriting contract located. [Animate 2 card](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B). |
| SCAIL — `WAN21_SCAIL`, `WAN21_SCAIL2` | Published for SCAIL evaluation: [English expansion helper](#wan-and-derivatives). No separate SCAIL-2 enhancer located in its [card](https://huggingface.co/zai-org/SCAIL-2). |
| Wan Dancer — `WAN22_WanDancer` | Restricted: [author card](https://huggingface.co/Wan-AI/Wan2.2-Dancer-14B); [public workflow card](https://huggingface.co/Comfy-Org/Wan-Dancer). |
| Kandinsky 5 — `Kandinsky5` | Published: [I2V expansion instruction](#kandinsky), distinct from encoder prefixes. |
| CogVideoX — `CogVideoX_T2V`, `CogVideoX_I2V`, `CogVideoX_Inpaint` | Published: [T2V/I2V converter](#other-video-enhancers). No separate inpainting enhancer located. |

## Native audio and other input tasks

| Family / registrations | Finding and source |
|---|---|
| Stable Audio — `StableAudio`, `StableAudio3` | Not located: [Open](https://huggingface.co/stabilityai/stable-audio-open-1.0), [Audio 3](https://huggingface.co/stabilityai/stable-audio-3-medium). |
| ACE-Step — `ACEStep`, `ACEStep15` | Related/published for 1.5: [specialized LM instructions](#audio). No separate original-version enhancer located. |
| YuE2 — `YuE2` | Related: [native symbolic planning and score editing](https://huggingface.co/m-a-p/YuE2-3B), not a general scene-caption expander. No separate style-rewrite system prompt located. |
| MiniMax Music — `MiniMaxMusic3` | Skill: [official music-caption rewriter](#minimax). |
| SVD — `SVD_img2vid` | Input task: [image-to-video](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). |
| Stable View / Zero123 — `SV3D_u`, `SV3D_p`, `Stable_Zero123` | Input task: [SV3D](https://huggingface.co/stabilityai/sv3d), [Zero123](https://huggingface.co/stabilityai/stable-zero123). |
| SD x4 — `SD_X4Upscaler` | Not located: [text-conditioned upscaler card](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). |
| Lotus — `LotusD` | Input task: [depth/normal estimation](https://github.com/EnVision-Research/Lotus). |
| PiD — `PiD` | Input task: [decoder](https://github.com/nv-tlabs/PiD). |
| SeedVR2 — `SeedVR2` | Input task: [restoration](https://github.com/ByteDance-Seed/SeedVR). |
| TRELLIS.2 — `Trellis2` | Input task: [image-to-3D](https://github.com/microsoft/TRELLIS.2). |
| Hunyuan3D — `Hunyuan3Dv2`, `Hunyuan3Dv2_1`, `Hunyuan3Dv2mini` | Input task: [2.0](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1). |
| TripoSplat — `TripoSplat` | Input task: [reference images](https://huggingface.co/VAST-AI/TripoSplat). |
| RT-DETR — `RT_DETR_v4` | Input task: [detection](https://github.com/lyuwenyu/RT-DETR). |
| Depth Anything — `DepthAnything3` | Input task: [image/video geometry](https://github.com/ByteDance-Seed/Depth-Anything-3). |
| SAM — `SAM3`, `SAM31` | Related: [SAM3 agent instructions](https://github.com/facebookresearch/sam3/tree/main/sam3/agent/system_prompts) orchestrate segmentation tools and checking; they are not image/video caption enhancement. |
| FlowRVS — `WAN21_FlowRVS` | Input task: [referring-video segmentation](https://github.com/xmz111/FlowRVS); retain the referring query. |

## Source details

### FLUX

FLUX.2 publishes [system_messages.py](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/system_messages.py).
Use `SYSTEM_MESSAGE_UPSAMPLING_T2I` for expanded visual descriptions (retain
structured inputs), or `SYSTEM_MESSAGE_UPSAMPLING_I2I` for a concise edit that
specifies changes and preserved content. The latter targets 50–80 words, shorter
for brief requests. The ordinary encoder `SYSTEM_MESSAGE` and content-filter
instructions in that file serve different purposes. The author's
[text encoder](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/text_encoder.py)
implements Mistral upsampling; its Qwen3 Klein embedder does not implement that
method. Do not describe it as universally enabled for all FLUX.2 variants.

FLUX 3 Klein's public [system_prompts.json](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json)
contains four instructions: `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`. Plain T2I requests
100–350 English words; plain I2I requests 50–150 words and ordinal image references.
Bounding-box T2I adds an element array with `id`, `bbox`, `desc`, referenced by
`<id>` in the caption. Bounding-box I2I instead uses `id`, `from`, `src_bbox`,
`tgt_bbox`, `desc`, with zero-based `ref_image_0` source names and nullable boxes
for additions/removals. Boxes are `[ymin, xmin, ymax, xmax]` on a 0–1000 grid.
These are distinct modes, not a universal JSON envelope. The public prompt file
was readable even though the generation checkpoint's raw card required access.
See the [Klein workflow notes](flux3_prompting.md).

### Qwen Image

The original [prompt_utils.py](https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/src/examples/tools/prompt_utils.py)
contains Chinese/English T2I instructions and a separate editing instruction.
The editing result uses `Rewritten`; it is not the 2.1 result schema.
[prompt_utils_2512.py](https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/src/examples/tools/prompt_utils_2512.py)
adds portrait/text/general generation rules and returns prose. Some constants
named “system prompt” are placed in the user message by the calling code.

Qwen Image 2.1 publishes distinct
[T2I](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite/prompts/system_prompt_t2i.txt)
and [Edit](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite/prompts/system_prompt_edit.txt)
instructions and [enhancer setup](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite/README.md).
Use the matching fine-tuned PE checkpoint. The [generation guide](qwen_image_2_1_prompting.md)
contains the full T2I instructions; the [Edit guide](qwen_image_2_1_edit_prompting.md)
explains image order, literal `<image1>` references, `rewritten_prompt`,
`wh_ratio`, and `ratio_follow`. Extract the caption and apply ratio metadata
separately. Do not feed the returned envelope to the diffusion text encoder.

### Hunyuan Image and DiT

HunyuanDiT's [MLLM guide](https://github.com/Tencent-Hunyuan/HunyuanDiT/blob/bf37693476a86e281006173ddd364debbb3c3624/mllm/README.md)
distinguishes training-data captioning from inference-time DialogGen.
[dialoggen_demo.py](https://github.com/Tencent-Hunyuan/HunyuanDiT/blob/bf37693476a86e281006173ddd364debbb3c3624/mllm/dialoggen_demo.py)
wraps the user request in a short Chinese intent-classification instruction,
uses the fine-tuned model's conversation template, and removes the `<画图>` prefix.
It is not a long universal rewrite system prompt.

HunyuanImage 2.1 has [reprompt.py](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/blob/307df8801d176740dafb67b2872c831cb9362cf9/hyimage/models/reprompt/reprompt.py)
and [reprompt_32b.py](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/blob/307df8801d176740dafb67b2872c831cb9362cf9/hyimage/models/reprompt/reprompt_32b.py).
The first preserves intent, organizes an objective overview/details/summary,
and extracts the answer; the second uses a reasoning-and-rewrite prefix and
removes the thinking portion. The standalone author enhancer also publishes
[standard](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer/blob/add1f59fd9883e02df5aba41c11f873e01729184/inference/prompt_enhancer.py),
[v2](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer/blob/add1f59fd9883e02df5aba41c11f873e01729184/inference/prompt_enhancer_v2.py),
and [image-edit](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer/blob/add1f59fd9883e02df5aba41c11f873e01729184/inference/prompt_enhancer_img2img.py)
wrappers. Match checkpoint and output extraction, rather than interchanging them.

HunyuanImage 3 publishes external [PE/system_prompt.py](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/6e9113a692a27a0751d82aba3b2015a876646c03/PE/system_prompt.py)
for general imagery and text/UI rendering. Separately,
[hunyuan_image_3/system_prompt.py](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/6e9113a692a27a0751d82aba3b2015a876646c03/hunyuan_image_3/system_prompt.py)
defines vanilla, recaption, thinking/recaption and unified generation/edit modes.
The latter use `<recaption>` and, where selected, `<think>` wrappers. These are
different from the external PE's plain description output.

### Boogu

The [external T2I rewriter](https://github.com/boogu-project/Boogu-Image/blob/9b327526b1053e1f27f1bd0ae36b20ce2c39cb9e/utils/t2i_external_prompt_rewriter.py)
and [built-in rewrite skills](https://github.com/boogu-project/Boogu-Image/blob/9b327526b1053e1f27f1bd0ae36b20ce2c39cb9e/boogu/pipelines/boogu/instruct_reasoner_static_skills.py)
contain bilingual generation instructions; the latter also contains bilingual
image-edit instructions. T2I emphasizes minimal rewriting for already clear
requests, with detailed layout/text exceptions. Editing is grounded in the
input image and preserves unspecified content. The external wrapper places
the rewrite instruction in a user message.

The [PPT static skills](https://github.com/boogu-project/Boogu-Image/blob/9b327526b1053e1f27f1bd0ae36b20ce2c39cb9e/boogu/pipelines/boogu/static_skills.py)
are a separate two-stage contract: first a slide-design JSON with explicit
`text_blocks`, then JSON containing `caption_PE`, a single Chinese paragraph.
The English and edit lists alias these Chinese templates at this revision;
their names do not imply translated or independently specialized instructions.

### Fibo 2

The [Fibo 2 card](https://huggingface.co/briaai/Fibo-2.0-dev) points to Bria's
[VGL library](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385),
whose Thales dialect combines a task's `system_prompt.txt`, `final_prompt.txt`
and schema. The full task pairs are published under:

- [generate](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/generate)
  and [inspire](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/inspire): new scene from text or a visual reference.
- [edit_image](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_image)
  and [edit_image_input_vgl](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_image_input_vgl): preserve unaffected appearance; the second also receives baseline JSON.
- [edit_image_mask](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_image_mask)
  and [edit_image_mask_input_vgl](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_image_mask_input_vgl): complete masked regions while retaining visible context.
- [edit_multi_images](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_multi_images)
  and [edit_multi_images_input_vgl](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/edit_multi_images_input_vgl): bind ordered inputs to one-based `<image_1>` tokens in `edit_instruction`.
- [caption](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/caption)
  and [refine](https://github.com/Bria-AI/vgl/tree/676b993f882c5aab4156542a87b0cb286ce00385/src/bria_vgl/dialects/thales/prompts/refine): describe an image or update an existing VGL scene.

Output is structured scene JSON. In edit tasks, descriptive fields describe the
finished image; only `edit_instruction` narrates the change. Control maps supply
structure, not the desired surface appearance. Keep the dialect's schema and
serialization pipeline; this is not Qwen's extract-one-caption envelope.
The checkpoint tree also lists style-specific prompt presets under
[styles](https://huggingface.co/briaai/Fibo-2.0-dev/tree/main/styles), but the
tested `boring_reality/system_prompt.txt` fetch returned 401. Those preset
contents were not verified. FIBO 1's enhancer is not a substitute for Fibo 2 VGL.

### Other image enhancers

| Family | Complete author source and contract |
|---|---|
| Z-Image | Official Space [base pe.py](https://huggingface.co/spaces/Tongyi-MAI/Z-Image/blob/main/pe.py) and [Turbo pe.py](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py). The Chinese rewrite template preserves requested entities/counts/colors, makes abstract requests concrete, quotes lettering, and avoids generic quality-tag padding. These live in the demo Spaces, not the model repository. |
| Ming | [t2i_rewriter_system_prompt.txt](https://github.com/inclusionAI/Ming-Image/blob/3515cd7b1e3049fc517eed7d35020251cd06519a/assets/t2i_rewriter_system_prompt.txt). JSON `canvas_settings` plus ordered `layers`; each layer includes description, coordinate string, hierarchy/relation and color specifications. Do not substitute another model's box format. |
| HiDream O1 | [prompt_agent.py](https://github.com/HiDream-ai/HiDream-O1-Image/blob/2c2d29ff729e48f33e41f49edfdbd81d5ac103b4/prompt_agent.py), `REWRITE_SYSTEM_PROMPT`. JSON `prompt`, `reasoning`, `resolved_knowledge`; the renderable `prompt` is an English paragraph, with an 80–220-word soft target. Resolve requested factual/relational content before expansion; preserve literal lettering. |
| SenseNova U1 / U1.5 | [infographic_system.md](https://github.com/OpenSenseNova/SenseNova-U1/blob/11ab73b698dbe0f07f05b5615b6c1c026a168d43/src/sensenova_u1/prompt_enhance/templates/infographic_system.md), [builder_prompt.txt](https://github.com/OpenSenseNova/SenseNova-U1/blob/11ab73b698dbe0f07f05b5615b6c1c026a168d43/apps/comfyui/prompts/builder_prompt.txt), and [enhancement guide](https://github.com/OpenSenseNova/SenseNova-U1/blob/11ab73b698dbe0f07f05b5615b6c1c026a168d43/docs/prompt_enhancement.md). Infographic expansion targets 400–600 words in the request's Chinese/English language, preserving facts and specifying layout/icons/text. Skip expansion for an already complete production prompt. |
| MOSS | [enhancer skill](https://github.com/OpenMOSS/MOSS-Image/blob/2b46b1e0e13b5aaec253879e0d006a42564b8b27/skills/moss-image-prompt-enhancer/SKILL.md) and [layout reference](https://github.com/OpenMOSS/MOSS-Image/blob/2b46b1e0e13b5aaec253879e0d006a42564b8b27/skills/moss-image-prompt-enhancer/references/layout.md). Caption drafts compile to prose; Layout drafts compile to model JSON. Draft pixel `xyxy` boxes and compiled centered `yxyx` coordinates are different representations. Preserve the compiler contract. |
| Krea 2 | [expansion.txt](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/docs/expansion.txt). Faithful single-paragraph expansion, no JSON or commentary; retain medium and exact quoted text, avoid inventing objects, and lightly polish already detailed prompts. |
| LongCat Image | [prompt_rewrite_api.py](https://github.com/meituan-longcat/LongCat-Image/blob/f0e4c43c5ef74b011ff71570fbfc2bdffbc9ab06/misc/prompt_rewrite_api.py). Chinese/English T2I instructions with few-shot examples; same-language prose, at most 512 generated tokens in the wrapper. The instruction is concatenated into the user message. |
| ERNIE | [pe_prompt.txt](https://github.com/baidu/ERNIE-Image/blob/1f30f5bb499726c606da0a24cc2a4a6c4d6ad775/src/pe_prompt.txt). Input includes `prompt`, `width`, `height`; output JSON contains `rewritten_prompt`, an objective Chinese description with specified lettering in single quotes. Use the matching PE workflow and extract its caption. |

### LTX

Original LTX-Video publishes [prompt_enhance_utils.py](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/utils/prompt_enhance_utils.py):
`T2V_CINEMATIC_PROMPT` and `I2V_CINEMATIC_PROMPT`, one paragraph of up to 150 words.
Its I2V path first captions the image, then incorporates the requested motion.

Current LTX-2 publishes four distinct files:

| Encoder | Text-to-video | Image-to-video |
|---|---|---|
| Gemma 3 | [gemma3_t2v_system_prompt.txt](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma3_t2v_system_prompt.txt) | [gemma3_i2v_system_prompt.txt](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma3_i2v_system_prompt.txt) |
| Gemma 4 | [gemma4_t2v_system_prompt.txt](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_t2v_system_prompt.txt) | [gemma4_i2v_system_prompt.txt](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt) |

Both families produce English audiovisual captions. Gemma 3 explicitly avoids
inventing camera motion. Gemma 4 calls for shot/motion/viewpoint to be specified,
even when absent, and targets 150–220 words. I2V also differs: Gemma 3 focuses on
changes and prioritizes requested changes on conflict; Gemma 4 anchors the exact
first frame and a continuous shot. Match the actual encoder and release; do not
silently replace one set with the other or use removed unversioned filenames.

### Hunyuan Video

Original [prompt_rewrite.py](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e/hyvideo/prompt_rewrite.py)
ships normal and master cinematic recaption instructions. HunyuanVideo 1.5 instead
publishes [t2v_prompt.py](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/utils/rewrite/t2v_prompt.py)
and [i2v_prompt.py](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/utils/rewrite/i2v_prompt.py).
T2V requests English visual detail; I2V requests Chinese motion-focused output
and prohibits inventing camera/lighting changes absent the request. The
[dispatcher](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/utils/rewrite/rewrite_utils.py)
selects by task. These are not interchangeable with the image-family PE prompts.

### Wan and derivatives

Wan 2.1's [prompt_extend.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/prompt_extend.py)
contains English/Chinese text-only, image-conditioned and first/last-frame
instructions. Wan 2.2 separates them into
[system_prompt.py](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/utils/system_prompt.py),
including T2V-A14B, I2V-A14B and empty-user-prompt I2V variants. Its
[dispatcher](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/utils/prompt_extend.py)
also reuses appropriate branches for TI2V-5B. I2V expansion emphasizes motion
rather than re-describing all static image details.

The older 2.1 English visual prompt contains conflicting language rules
(Chinese in one rule, English in another), and length wording differs between
language branches. Preserve this provenance; do not present a corrected local
instruction as an exact upstream copy.

[VACE](https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/annotators/prompt_extend.py)
wraps the Wan expander with optional custom instructions.
[Self Forcing](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/wan/utils/prompt_extend.py)
and [Wan-Move](https://github.com/ali-vilab/Wan-Move/blob/80c58a7d2ad175fa82a4d57f79f2a1415317dcfa/wan/utils/prompt_extend.py)
bundle Wan expansion code. Follow each derivative's actual invocation and
conditioning; sharing a backbone does not establish a task-specific enhancer.

VideoX-Fun publishes [beautiful_prompt.txt](https://github.com/aigc-apps/VideoX-Fun/blob/e8c314fbfa494d0429bbef605c9f6112b79a7ead/videox_fun/video_caption/prompt/beautiful_prompt.txt)
(JSON `detailed description`, at most 200 words) and
[rewrite.txt](https://github.com/aigc-apps/VideoX-Fun/blob/e8c314fbfa494d0429bbef605c9f6112b79a7ead/videox_fun/video_caption/prompt/rewrite.txt)
(JSON `rewritten description`, clean a video caption without inventing camera
or style details). They are utilities in the video-caption tooling, not evidence
that every camera/control workflow runs enhancement automatically.

SCAIL's [eval/prompt_utils.py](https://github.com/zai-org/SCAIL/blob/8f94dcd6bed710854232a56304fcfeb04e053752/eval/prompt_utils.py)
contains `upsample_prompt_glm`, with an English-description instruction and
few-shot conversation. It is evaluation tooling with a deployment-specific
endpoint, not a portable configured service or a SCAIL-2-specific system prompt.

### Cosmos

Cosmos 1's [text-to-world upsampler](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/prompt_upsampler/text2world_prompt_upsampler_inference.py)
uses a specialized upsampler checkpoint and short user instruction. Its
[video-to-world upsampler](https://github.com/nvidia-cosmos/cosmos-predict1/blob/724daa1b2df5ec96bdf111bb947479d2216b3b08/cosmos_predict1/diffusion/prompt_upsampler/video2world_prompt_upsampler_inference.py)
uses visual input and an instruction to describe content within 150 words.

Predict2 publishes `SYSTEM_PROMPT_REFINE` in
[cosmos_reason1.py](https://github.com/nvidia-cosmos/cosmos-predict2/blob/661da4774b0ca41d082a0ecbeb47550bcf07e03f/cosmos_predict2/auxiliary/cosmos_reason1.py):
combine image/video and user intent into approximately 80–100 English words.
The video critic in the same file and the short
[text-encoder prefix](https://github.com/nvidia-cosmos/cosmos-predict2/blob/661da4774b0ca41d082a0ecbeb47550bcf07e03f/imaginaire/auxiliary/text_encoder.py)
are separate mechanisms. The refiner's implementation also restricts generated
tokens to ASCII, which limits its instruction to preserve non-English lettering.

### Kandinsky

Kandinsky 5 embeds `INSTRUCTION_I2V_EXPAND` in
[text_embedders.py](https://github.com/kandinskylab/kandinsky-5/blob/3a74261b957df0446c1969c5cedf8bbb07969ac2/kandinsky/models/text_embedders.py).
It requests an English dynamic caption grounded in the image, focusing on
object changes, with training-style examples. `expand_text_prompt` supplies it
as a user instruction and generates a caption. In contrast, the same file's
`PROMPT_TEMPLATE` entries for video, image-to-video, image and image-edit are
prefixes used to obtain embeddings; do not count those as four extra upsamplers.

Kandinsky 6 publishes separate
[T2AV](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/comfyui/kandinsky6/beautifier_prompts/t2av_system.txt)
and [I2AV](https://github.com/kandinskylab/kandinsky-6/blob/7ceeefad7fb625011f7db602c43893a989f19c15/comfyui/kandinsky6/beautifier_prompts/i2av_system.txt)
beautifier instructions. See the [task-specific guide](kandinsky6_prompting.md)
for dialogue/audio tags and first-frame preservation. Do not apply the version 6
audio format to version 5.

### Other video enhancers

[CogVideoX convert_demo.py](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/convert_demo.py)
contains T2V and I2V conversion instructions. Preserve the T2V few-shot message
sequence, not just its initial system string. I2V describes the initial frame
and a continuous video without invented cuts or perspective changes.

[LongCat Video prompt_enhancer.py](https://github.com/meituan-longcat/LongCat-Video/blob/6b3f4b8582a8bc3f20f795735f5383716c4ba794/longcat_video/utils/prompt_enhancer.py)
publishes English/Chinese T2V and I2V variants. English T2V targets 100–150 words,
I2V 50–80; Chinese branches use their own character budgets. I2V focuses on
movement grounded in the reference image.

[ChronoEdit prompt_enhancer.py](https://github.com/nv-tlabs/ChronoEdit/blob/37b0da51d8a42928b23e84cc998bba200c246e57/scripts/prompt_enhancer.py)
uses the original image and editing instruction to generate an approximately
80–100-word task analysis beginning with “The user wants to”. It distinguishes
ordinary edits from moving an identity into a new scene. This is the author's
specialized edit-conditioning format, not a generic final-image caption.

### MiniMax

MiniMax H3 publishes an official
[h3-prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/199b7d0e1fd95e7b3eafa8c15eca692534142427/skills/h3-prompt-writing/SKILL.md),
[base/keyframe guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/199b7d0e1fd95e7b3eafa8c15eca692534142427/skills/h3-prompt-writing/references/base-en.txt),
and [full-reference guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/199b7d0e1fd95e7b3eafa8c15eca692534142427/skills/h3-prompt-writing/references/ref-en.txt).
Base modes use `integrated_multimodal_description`, `overall_soundscape`,
`non_diegetic_music`, with task-specific keyframe alignment. Full-reference
rewrites instead use six ordered sections: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Preserve the reference labels, shot timing, speaker IDs
and dialogue markup from the selected guide. This public authoring contract
does not establish access to an internal hosted enhancer's system prompt.

MiniMax Music 3 publishes the
[music-caption-rewriter skill](https://github.com/MiniMax-AI/MiniMax-Music3/blob/efecc2167c68dd6b1ec75fa49f97f23b475513dd/skills/music-caption-rewriter/SKILL.md)
with a [genre router and reference indexes](https://github.com/MiniMax-AI/MiniMax-Music3/tree/efecc2167c68dd6b1ec75fa49f97f23b475513dd/skills/music-caption-rewriter/references)
and a template library. Its output has exactly `Global Metadata`, `Vocal Details`,
and `Arrangement`, normally 250–450 English words. Preserve section-tag musical
directives, keep lyrics in their separate input, and do not invent precise BPM,
key or vocal properties unnecessarily. The skill selects a few relevant templates;
the template library is not itself one giant system prompt.

### Audio

ACE-Step 1.5 publishes `DEFAULT_LM_INSPIRED_INSTRUCTION` and
`DEFAULT_LM_REWRITE_INSTRUCTION` in
[constants.py](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/constants.py).
These short instructions expand or format a musical description for the
specialized LM. Separate instructions handle semantic-token generation,
understanding and diffusion tasks. The [tutorial](ace_step_1_5_tutorial.md)
and [LM handler](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/llm_inference.py)
define the surrounding metadata/lyrics workflow; these are not generic image
prompting instructions.

## Additional families and limits

| Family | Audit finding |
|---|---|
| HunyuanImage 3 / Instruct | Published: [external PE and native recaption modes](#hunyuan-image-and-dit). |
| Kandinsky 6 | Published: [T2AV/I2AV beautifiers](#kandinsky). |
| NewBie | Not located: [Exp0.1 card](newbie_image_exp01_model_card.md). |
| ChronoEdit | Published: [image-grounded edit enhancer](#other-video-enhancers). |
| Ovis-Image | Not located: [author card](https://huggingface.co/ATH-MaaS/Ovis-Image-7B). |
| USO / OneReward | Not located: [USO](https://huggingface.co/bytedance-research/USO), [OneReward](https://huggingface.co/yichengup/flux.1-fill-dev-OneReward); do not assume a new enhancer from a fine-tune. |
| FireRed Image Edit | Not located: [author card](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1). |
| Bernini-R | Not located: [author edit card](https://huggingface.co/ByteDance/Bernini-R). |
| LongCat Video | Published: [bilingual T2V/I2V enhancer](#other-video-enhancers). |
| Pixal3D | Input task: [image/3D instructions](https://huggingface.co/TencentARC/Pixal3D). |
| AnimateDiff | Not located: [author repository](https://github.com/guoyww/AnimateDiff); use the base model's caption contract. |
| VOID | Input task: [video and quadmask](https://github.com/Netflix/void-model). |
| Wan-Move | Published: [bundled Wan expansion](#wan-and-derivatives). |
| PhotoMaker | Not located: [author repository](https://github.com/TencentARC/PhotoMaker); preserve identity references and trigger syntax. |
| Florence-2 / PaliGemma | Related: [Florence task tokens](https://huggingface.co/microsoft/Florence-2-large) and [PaliGemma task instructions](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions) request captions/grounding; they are not a downstream generator's official expansion guide. |
| Marigold / MoGe | Input task: [Marigold](https://github.com/prs-eth/Marigold), [MoGe](https://github.com/microsoft/MoGe). |
| SAM 3D Body / SDPose | Input task: [body reconstruction](https://github.com/facebookresearch/sam-3d-body), [pose estimation](https://github.com/MartyrPenink/SDPose). |
| BiRefNet | Input task: [foreground extraction](https://github.com/ZhengPeng7/BiRefNet). |
| RAFT / restoration | Input task: [optical flow](https://docs.pytorch.org/vision/stable/models/raft.html), [restoration models](https://github.com/chaiNNer-org/spandrel). |
| ControlNet / IP-Adapter | Related: [ControlNet](https://github.com/lllyasviel/ControlNet), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter); base-model prompting plus control/reference inputs, no universal adapter enhancer located. |
| Fine-tunes / LoRAs / quantizations | Follow the exact publisher and base-family guide in the [coverage index](model_coverage.md). This audit does not enumerate every community derivative. |

Language checkpoints (Qwen, Gemma, Llama, gpt-oss and others) supply chat templates
and sometimes power the enhancers above; that does not give each a model-specific
image/video rewrite contract. Hosted OpenAI, Gemini, Veo, Runway, Luma, BFL,
Ideogram, MiniMax and Alibaba services can perform undisclosed prompt rewriting.
Their [provider guides](model_coverage.md#language-and-hosted-models) remain useful,
but a public enhancement switch or returned revised prompt does not reveal the
provider's system prompt. No exhaustive claim is made about private services or
arbitrary third-party custom nodes.
