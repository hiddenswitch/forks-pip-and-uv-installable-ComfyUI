# Model prompting reference

This index maps the native registrations in `comfy/supported_models.py` to
prompting instructions, author examples, or model cards. Multiple registrations
can share a guide. A model card is identified as such when there is no separate
handbook; it is not presented as a dedicated prompting guide.

Use the instructions for the exact checkpoint and task. Distilled, base, editing,
and hosted variants can have different conditioning and sampling contracts.
Reference-image, segmentation, depth, restoration, and decoding models need input
guidance even when they do not take a generative text prompt. Sources were reviewed
on 2026-10-29; restricted sources are called out explicitly.

Access notes: SD 2.x, Lens, Mage-Flow and Wan Dancer author URLs returned 401.
For the latter three, the public Comfy-Org cards provide the available workflow
reference; this review does not claim to have read the restricted author cards.

## Image generation and editing

| Family / native registrations | Prompting or input reference |
|---|---|
| SD 1.x — `SD15` | [SD 1.5 model card and example](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5); fine-tunes can add their own trigger words |
| SD 2.x — `SD20`, `SD21UnclipL`, `SD21UnclipH` | [Author prompt examples](https://stability.ai/news-updates/stablediffusion2-1-release7-dec-2022), [SD 2.1 card (access required)](https://huggingface.co/stabilityai/stable-diffusion-2-1), [unCLIP card (access required)](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip) |
| SDXL — `SDXL`, `SDXLRefiner` | [Base prompt examples](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [refiner model card](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) |
| SDXL derivatives — `SSD1B`, `Segmind_Vega`, `KOALA_700M`, `KOALA_1B` | [SSD-1B](https://huggingface.co/segmind/SSD-1B), [Vega](https://huggingface.co/segmind/Segmind-Vega), [KOALA authors' examples](https://github.com/youngwanLEE/KOALA); use each checkpoint's settings |
| InstructPix2Pix / CosXL Edit — `SD15_instructpix2pix`, `SDXL_instructpix2pix` | [Instruction-edit examples](https://github.com/timothybrooks/instruct-pix2pix), [CosXL model card](https://huggingface.co/stabilityai/cosxl) |
| SD3 / SD3.5 — `SD3` | [SD3.5 model card and prompt examples](https://huggingface.co/stabilityai/stable-diffusion-3.5-large); Turbo has a separate sampling recipe |
| Stable Cascade — `Stable_Cascade_C`, `Stable_Cascade_B` | [Official inference examples](https://github.com/Stability-AI/StableCascade); Stage B decodes Stage C conditioning |
| AuraFlow — `AuraFlow` | [Author model card](https://huggingface.co/fal/AuraFlow-v0.3) |
| PixArt — `PixArtAlpha`, `PixArtSigma` | [Alpha examples](https://github.com/PixArt-alpha/PixArt-alpha), [Sigma examples](https://github.com/PixArt-alpha/PixArt-sigma) |
| HunyuanDiT — `HunyuanDiT`, `HunyuanDiT1` | [Official bilingual generation and prompt enhancement](https://github.com/Tencent-Hunyuan/HunyuanDiT) |
| FLUX.1 — `Flux`, `FluxSchnell`, `FluxInpaint` | [Family guide](bfl_flux_prompting_summary.md), [Kontext editing](bfl_flux_prompting_kontext_i2i.md), [Fill model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) |
| FLUX.2 — `Flux2` | [FLUX.2 prompting](bfl_flux2_prompting.md), [Klein variants](bfl_flux2_klein_prompting.md) |
| FLUX 3 Image — `Flux3` | [FLUX 3 Image guide](flux3_prompting.md); hosted API fields are not local sampler controls |
| Lens — `Lens` | [Microsoft model card](https://huggingface.co/microsoft/Lens), [ComfyUI model/workflow reference](https://huggingface.co/Comfy-Org/Lens) |
| Lumina — `Lumina2` | [Lumina-Image 2.0 inference and prompts](https://github.com/Alpha-VLLM/Lumina-Image-2.0) |
| Z-Image — `ZImage`, `ZImagePixelSpace` | [Turbo snapshot](z_image_turbo_model_card.md), [base and editing variants](https://github.com/Tongyi-MAI/Z-Image); pixel-space decoding does not define a new prompt language |
| Ming — `MingImage` | [Design model card](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), [layer variant](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) |
| PixelDiT — `PixelDiTT2I` | [Encoder, caption and sampling reference](nvidia_pixeldit_prompting.md) |
| HiDream — `HiDream` | [HiDream-I1 author examples](https://github.com/HiDream-ai/HiDream-I1) |
| HiDream O1 — `HiDreamO1` | [Author prompt agent and task examples](https://github.com/HiDream-ai/HiDream-O1-Image); use the matching generation/edit model |
| SenseNova — `SenseNovaU15` | [Official U1.5 documentation](https://github.com/OpenSenseNova/SenseNova-U1); prompt enhancement and reference reconstruction are separate workflows |
| Chroma — `Chroma`, `ChromaRadiance` | [Chroma1-HD snapshot](chroma1_hd_model_card.md), [Radiance model card](https://huggingface.co/lodestones/Chroma1-Radiance) |
| Anima — `Anima` | [Base, Aesthetic and Turbo prompting](anima_model_card.md) |
| OmniGen2 — `Omnigen2` | [Author instructions and reference-image examples](omnigen2_readme.md) |
| Boogu — `Boogu` | [Base model card](https://huggingface.co/Boogu/Boogu-Image-0.1-Base), [edit model card](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit); Turbo recipes differ |
| MOSS — `MOSSImage` | [SFT model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [base model](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base); layout and edit adapters need their own workflows |
| Fibo 2 — `Fibo2` | [Dev model card](https://huggingface.co/briaai/Fibo-2.0-dev), [edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo); preserve the model's structured-caption pipeline |
| Mage-Flow — `MageFlow` | [Microsoft model card](https://huggingface.co/microsoft/Mage-Flow), [task-specific ComfyUI examples](https://huggingface.co/Comfy-Org/Mage-Flow) |
| Qwen Image / Edit / Layered — `QwenImage` | [Official generation repository](https://github.com/QwenLM/Qwen-Image), [editing guide](alibaba_qwen_image_edit_prompting.md), [Layered input contract](https://huggingface.co/Qwen/Qwen-Image-Layered) |
| Qwen Image 2.1 — `QwenImage21` | [Generation and shared model instructions](qwen_image_2_1_prompting.md), [official Edit enhancer contract](qwen_image_2_1_edit_prompting.md) |
| JoyAI — `JoyImage` | [Official Image-Edit-Plus card and examples](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-ComfyUI) |
| Ideogram 4 — `Ideogram4` | [Native structured prompt schema](ideogram4_prompting.md), [Magic Prompt](ideogram4_magic_prompt_system_prompt_v1.md) |
| Krea 2 — `Krea2` | [Raw model card](https://huggingface.co/krea/Krea-2-Raw), [Turbo model card](https://huggingface.co/krea/Krea-2-Turbo), [style-LoRA trigger words](https://huggingface.co/Comfy-Org/Krea-2); not FLUX.1 Krea |
| HunyuanImage 2.1 — `HunyuanImage21`, `HunyuanImage21Refiner` | [Author inference and prompt enhancement](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) |
| Kandinsky 5 Image — `Kandinsky5Image` | [Official Kandinsky 5 examples](https://github.com/kandinskylab/kandinsky-5) |
| LongCat Image — `LongCatImage` | [Generation card](https://huggingface.co/meituan-longcat/LongCat-Image), [editing card](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) |
| ERNIE — `ErnieImage` | [Base card](https://huggingface.co/baidu/ERNIE-Image), [Turbo card](https://huggingface.co/baidu/ERNIE-Image-Turbo); use the matching prompt enhancer |

## Video

| Family / native registrations | Prompting or input reference |
|---|---|
| Mochi — `GenmoMochi` | [Official inference examples](https://github.com/genmoai/models) |
| LTX — `LTXV`, `LTXAV` | [LTX-Video snapshot](ltx_video_readme.md), [current LTX-2 family prompting](ltx_2_readme.md) |
| MiniMax — `MiniMaxH3` | [H3 prompting and hosted/local distinctions](minimax_h3_video_generation.md) |
| HunyuanVideo — `HunyuanVideo`, `HunyuanVideoI2V`, `HunyuanVideoSkyreelsI2V` | [Original video guide](https://github.com/Tencent-Hunyuan/HunyuanVideo), [I2V guide](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V), [SkyReels variant](https://github.com/SkyworkAI/SkyReels-V1) |
| HunyuanVideo 1.5 — `HunyuanVideo15`, `HunyuanVideo15_SR_Distilled` | [1.5 prompt handbook](hunyuan_video_1_5_prompt_handbook.md); super-resolution reuses the source video's conditioning |
| Cosmos — `CosmosT2V`, `CosmosI2V`, `CosmosT2IPredict2`, `CosmosI2VPredict2` | [Cosmos 1](https://github.com/NVIDIA/Cosmos), [Predict2 generation and prompt upsampling](https://github.com/nvidia-cosmos/cosmos-predict2) |
| Wan 2.1 / 2.2 — `WAN21_T2V`, `WAN21_I2V`, `WAN22_T2V` | [Prompt guide](alibaba_wan_video_prompting.md), [2.1 native recipes](https://github.com/Wan-Video/Wan2.1), [2.2 native recipes](https://github.com/Wan-Video/Wan2.2) |
| Wan autoregressive — `WAN21_CausalAR_T2V` | [Self Forcing inference examples](https://github.com/guandeh17/Self-Forcing); retain the checkpoint's causal workflow |
| Wan Fun / camera — `WAN21_FunControl2V`, `WAN21_Camera`, `WAN22_Camera` | [VideoX-Fun control and camera examples](https://github.com/aigc-apps/VideoX-Fun); camera/control inputs accompany the caption |
| VACE — `WAN21_Vace` | [Official task and prompt examples](https://github.com/ali-vilab/VACE) |
| HuMo — `WAN21_HuMo` | [Text, image and audio conditioning](humo_readme.md) |
| Wan speech / animation — `WAN22_S2V`, `WAN22_Animate`, `WAN_Animate2` | [Wan 2.2 speech/animation tasks](https://github.com/Wan-Video/Wan2.2), [Animate 2 card](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B) |
| SCAIL — `WAN21_SCAIL`, `WAN21_SCAIL2` | [SCAIL reference and motion instructions](https://github.com/zai-org/SCAIL), [SCAIL-2 card](https://huggingface.co/zai-org/SCAIL-2) |
| Wan Dancer — `WAN22_WanDancer` | [Author card (access required)](https://huggingface.co/Wan-AI/Wan2.2-Dancer-14B), [public workflow reference](https://huggingface.co/Comfy-Org/Wan-Dancer) |
| Kandinsky 5 Video — `Kandinsky5` | [Official task examples](https://github.com/kandinskylab/kandinsky-5) |
| CogVideoX — `CogVideoX_T2V`, `CogVideoX_I2V`, `CogVideoX_Inpaint` | [Official caption and prompt-optimization instructions](https://github.com/THUDM/CogVideo); inpainting also requires its mask/video inputs |

## Audio and music

| Family / native registrations | Prompting or input reference |
|---|---|
| Stable Audio — `StableAudio`, `StableAudio3` | [Open 1.0 model examples](https://huggingface.co/stabilityai/stable-audio-open-1.0), [Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium), [music/SFX variant index](https://huggingface.co/Comfy-Org/stable-audio-3) |
| ACE-Step — `ACEStep`, `ACEStep15` | [Original ACE-Step lyrics and tags](https://github.com/ace-step/ACE-Step), [1.5 tutorial](ace_step_1_5_tutorial.md) |
| YuE2 — `YuE2` | [Official 3B model card](https://huggingface.co/m-a-p/YuE2-3B), [text-to-music and cover workflows](https://huggingface.co/Comfy-Org/YuE2) |
| MiniMax Music — `MiniMaxMusic3` | [Music3 author model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3); use its lyric/style fields rather than video prompting syntax |

## Other conditioning tasks

| Family / native registrations | Input contract |
|---|---|
| Stable Video — `SVD_img2vid` | [Image-to-video model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt); image-conditioned, not a text-to-video prompt model |
| Stable View / Zero123 — `SV3D_u`, `SV3D_p`, `Stable_Zero123` | [SV3D](https://huggingface.co/stabilityai/sv3d), [Zero123](https://huggingface.co/stabilityai/stable-zero123); reference image and camera/view controls |
| SD x4 upscaler — `SD_X4Upscaler` | [Image and text conditioning example](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Lotus — `LotusD` | [Depth/normal task inputs](https://github.com/EnVision-Research/Lotus); not free-form image generation |
| PiD — `PiD` | [Official decoder instructions](https://github.com/nv-tlabs/PiD); match the source latent/model family, not a separate text-prompt recipe |
| SeedVR2 — `SeedVR2` | [Restoration input instructions](https://github.com/ByteDance-Seed/SeedVR); source image/video supplies content |
| TRELLIS.2 — `Trellis2` | [Official image-to-3D examples](https://github.com/microsoft/TRELLIS.2) |
| Hunyuan3D — `Hunyuan3Dv2`, `Hunyuan3Dv2_1`, `Hunyuan3Dv2mini` | [2.0 image-conditioned generation](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |
| TripoSplat — `TripoSplat` | [Author model card](https://huggingface.co/VAST-AI/TripoSplat); supply reference images through the task's image inputs |
| RT-DETR — `RT_DETR_v4` | [Object-detection inputs](https://github.com/lyuwenyu/RT-DETR); no generative text prompt |
| Depth Anything — `DepthAnything3` | [Official image/video input examples](https://github.com/ByteDance-Seed/Depth-Anything-3); no generative text prompt |
| SAM — `SAM3`, `SAM31` | [SAM3 concept/visual prompting](https://github.com/facebookresearch/sam3), [3.1 model reference](https://huggingface.co/Comfy-Org/sam3.1); segmentation prompts are not image-generation captions |
| FlowRVS — `WAN21_FlowRVS` | [Referring-video segmentation instructions](https://github.com/xmz111/FlowRVS); text identifies what to segment |

## Additional workflow and checkpoint families

These are exposed through custom nodes, adapters, or checkpoint variants rather
than separate native registration names. Quantized copies inherit the original
model's prompting contract; fine-tunes and LoRAs can introduce trigger words.

| Family | Reference |
|---|---|
| HunyuanImage 3 / Instruct | [Base guide](hunyuan_image3_prompting.md), [Instruct tasks](hunyuan_image3_instruct_prompting.md) |
| Kandinsky 6 | [Audio/video prompt expansion](kandinsky6_prompting.md) |
| NewBie | [Exp0.1 model card](newbie_image_exp01_model_card.md) |
| ChronoEdit | [Official edit prompting](chronoedit_prompt_guidance.md) |
| Ovis-Image | [Author model card](https://huggingface.co/ATH-MaaS/Ovis-Image-7B) |
| USO / OneReward | [USO reference conditioning](https://huggingface.co/bytedance-research/USO), [OneReward Fill card](https://huggingface.co/yichengup/flux.1-fill-dev-OneReward) |
| FireRed Image Edit | [Author model card](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) |
| Bernini-R | [Author image/video editing card](https://huggingface.co/ByteDance/Bernini-R) |
| LongCat Video | [Author video repository](https://github.com/meituan-longcat/LongCat-Video) |
| Pixal3D | [Author input instructions](https://huggingface.co/TencentARC/Pixal3D) |
| AnimateDiff | [Motion-module and animation examples](https://github.com/guoyww/AnimateDiff); text follows its base diffusion model |
| VOID | [Video object-removal instructions](https://github.com/Netflix/void-model); source video and quadmask conditioning |
| Wan-Move | [Author trajectory-conditioning examples](https://github.com/ali-vilab/Wan-Move); motion tracks accompany image/video conditioning |
| PhotoMaker | [Identity-reference and trigger instructions](https://github.com/TencentARC/PhotoMaker) |
| Florence-2 | [Task-token and caption/grounding examples](https://huggingface.co/microsoft/Florence-2-large) |
| PaliGemma | [Task prompting instructions](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions) |
| Marigold / MoGe | [Marigold image-input examples](https://github.com/prs-eth/Marigold), [MoGe geometry inputs](https://github.com/microsoft/MoGe); not generative image captions |
| SAM 3D Body / SDPose | [Body reconstruction inputs](https://github.com/facebookresearch/sam-3d-body), [pose estimation inputs](https://github.com/MartyrPenink/SDPose) |
| BiRefNet | [Foreground extraction examples](https://github.com/ZhengPeng7/BiRefNet); image input without a text prompt |
| RAFT / restoration upscalers | [Optical-flow input pairs](https://docs.pytorch.org/vision/stable/models/raft.html), [Spandrel model/input reference](https://github.com/chaiNNer-org/spandrel); image-conditioned tools, not prompt models |
| ControlNet / IP-Adapter | [ControlNet input examples](https://github.com/lllyasviel/ControlNet), [IP-Adapter references](https://github.com/tencent-ailab/IP-Adapter); captions still follow the base model |
| Checkpoint fine-tunes and acceleration LoRAs | Read the exact publisher card for triggers and required settings; a base-family guide does not document every community derivative |

## Language and hosted models

Native text-generation nodes accept model-specific chat templates and message
roles; image prompting guides do not replace those contracts. See
[text generation](../llm.md), [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B),
[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and
[Gemma 4](https://huggingface.co/google/gemma-4-E4B-it). Check the actual loaded
checkpoint, including its instruction/base variant.

Other language checkpoints have their own templates:
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
[Qwen3](https://huggingface.co/Qwen/Qwen3-8B),
[Gemma 3](https://huggingface.co/google/gemma-3-12b-it),
[Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and
[gpt-oss](https://huggingface.co/openai/gpt-oss-20b).
For fine-tuned prompt enhancers, use the matching task instructions:
[Qwen Image 2.1](qwen_image_2_1_prompting.md), [LTX](ltx_2_readme.md),
and [Kandinsky 6](kandinsky6_prompting.md). A stock language model does not
automatically inherit a fine-tuned enhancer's output contract.

Hosted-node catalogs change independently of this repository. Use the provider's
guide for the selected API model: [Ideogram](ideogram4_5_prompting.md),
[BFL](bfl_flux_prompting_summary.md),
[MiniMax video](minimax_h3_video_generation.md),
[Alibaba image/video](alibaba_wan_video_prompting.md),
[OpenAI image generation](https://platform.openai.com/docs/guides/image-generation),
[Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation),
[Veo](https://ai.google.dev/gemini-api/docs/video),
[Runway Gen-4 video](https://help.runwayml.com/hc/en-us/articles/39789879462419-Gen-4-Video-Prompting-Guide),
[Runway Gen-4.5 text-to-video](https://help.runwayml.com/hc/en-us/articles/42460036199443-Text-to-Video-Prompting-Guide), and
[Luma](https://docs.lumalabs.ai/docs/video-generation).
This index does not claim every hosted model or third-party custom node is locally
supported. The automated coverage check covers every native registration above.
