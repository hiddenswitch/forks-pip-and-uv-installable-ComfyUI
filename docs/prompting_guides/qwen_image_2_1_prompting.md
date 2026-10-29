# Qwen Image 2.1 prompting guide

Updated: 2026-10-29. [Qwen Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
now supports generation and editing in eight denoising steps using its supplied
sampling schedule. The 40-step advice below is for the original 2.1 checkpoint;
do not transfer that schedule to Turbo. Hosted Pro and Turbo APIs are also
linked in the [current author README](https://github.com/QwenLM/Qwen-Image-2.1).
Prompt rewriting remains a separate stage; the retained rewriting instructions
below have not changed. The README snapshot predates the Turbo release.

Editing uses a separate [official Edit enhancer system prompt and contract](qwen_image_2_1_edit_prompting.md).
The complete system-prompt snapshot below is **T2I only**; its English-caption
instructions do not replace the Edit enhancer's language and image-role rules.

Vendored snapshot, complete, with source links made absolute, of:
1. https://github.com/QwenLM/Qwen-Image-2.1/blob/main/README.md
2. https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/README.md
3. https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt

Fetched: 2026-10-02

The official t2i prompt format is what the rewriter in (3) produces: one long English paragraph
describing the finished image. Official sampling is 40 steps at guidance 1 (README, SGLang
example). The bundled `image_qwen_image_2_1_t2i` template runs the same rewriter
(`qwen3.5_9b_qwen_image_2.1_pe_t2i`) through `TextGenerate`, with this system prompt ending in a
plain-paragraph output contract instead of the JSON one.

The template's KSampler starts at 25 steps; run it with `--steps 40` for the official count. On
Qwen Image 2.1 the `simple` scheduler samples the official scheduler config
(`scheduler/scheduler_config.json`): `linspace(1, 1/N, N)` shifted by mu, which goes from 0.5 at
256 image tokens to 0.9 at 8192 (one token per 16x16 pixels, 0.6935 at 1024x1024), stretched so
the last step is at 0.02.

---

# 1. README.md

<p align="center">
    <img src="https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/assets/logo.png" width="400"/>
</p>
<p align="center">
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-2.1">ModelScope</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-2.1">HuggingFace</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;📑 <a href="https://qwen.ai/blog?id=qwen-image-2.1">Blog</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-2.1">Demo</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;🫨 <a href="https://discord.gg/BEYSk3pkSu">Discord</a>&nbsp;&nbsp;|
    &nbsp;&nbsp;💬 <a href="https://github.com/QwenLM/Qwen-Image-2.1/blob/main/assets/qr.png">WeChat</a>
</p>

## Introduction

We are excited to open-source **Qwen-Image-2.1**, a unified text-to-image generation and image editing model in the Qwen family. With just **7B parameters in its visual generation component** (32 Single-Stream DiT layers), Qwen-Image-2.1 balances generation quality, inference efficiency, and versatility.

Four key improvements define this release:

- **Compact and Efficient** — A lightweight architecture with mixed-granularity attention and prefix KV cache reuse delivers strong image quality at low computational cost.
- **Native Transparency, Unified Creation and Editing** — Generate regular or transparent (RGBA) images from text, edit transparent layers, and extract subjects from photographs—all in one model.
- **Versatile Editing** — Support up to **10 reference images**, specify local edits via circles, painted annotations, or separate masks, and preserve identity for people and products.
- **Realistic Textures and Refined Aesthetics** — Improved typography, portrait lighting, and fine details for more visually compelling results.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-01.png" width="100%"/>
</p>

## News

- 2026.09.20: We released Qwen-Image-2.1! Check our [Blog](https://qwen.ai/blog?id=qwen-image-2.1) for more details. Weights available at [HuggingFace](https://huggingface.co/Qwen/Qwen-Image-2.1) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-2.1).
- 2026.09.20: [Diffusers](https://github.com/huggingface/diffusers) supports Qwen-Image-2.1 from Day 0 via `QwenImage21Pipeline`. See [PR #14804](https://github.com/huggingface/diffusers/pull/14804).
- 2026.09.20: [ComfyUI](https://github.com/Comfy-Org/ComfyUI) natively supports Qwen-Image-2.1 from Day 0. Compatible weights at [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), with example workflows for [text-to-image](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json) and [image editing](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json).
- 2026.09.20: [vLLM-Omni](https://github.com/vllm-project/vllm-omni) supports high-performance Qwen-Image-2.1 inference from Day 0, with step-wise execution, prefix KV caching, CUDA Graph decode, FP8 quantization, and TP/Ulysses parallelism.
- 2026.09.20: [SGLang](https://github.com/sgl-project/sglang) provides Day-0 native support for Qwen-Image-2.1, including prefix caching, Cache-DiT, CUDA graphs, TP/Ulysses/Ring/CFG parallelism, and component offload. See [PR #39983](https://github.com/sgl-project/sglang/pull/39983).
- 2026.09.20: [LightX2V](https://github.com/ModelTC/LightX2V) delivers Day 0 acceleration for Qwen-Image-2.1! Check out the [usage guide](https://github.com/ModelTC/LightX2V/tree/main/scripts/qwen_image_21) for more details.

## Quick Start

### Requirements

```bash
pip install torch>=2.4.0
pip install transformers>=5.17
pip install git+https://github.com/huggingface/diffusers
pip install accelerate
pip install pillow
```

### Text-to-Image

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("t2i_example.png")
```

### Image Editing (Single Image)

```python
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

input_image = Image.open("input.png")

image = pipe(
    prompt="Change the background to a sunset beach",
    image=input_image,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("edit_example.png")
```

### Image Editing (Multiple Reference Images)

Qwen-Image-2.1 supports up to **10 reference images** for multi-subject composition:

```python
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

images = [Image.open(f"ref_{i}.png") for i in range(3)]

result = pipe(
    prompt="These three characters are sitting around a campfire in a forest",
    image=images,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

result.save("multi_ref_example.png")
```

### Transparent Image Generation (RGBA)

The model natively generates transparent images. For best results, use the recommended prompt format:

> `This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.",
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("transparent_example.png")  # Saved as RGBA when the model generates transparency
```

### Supported Aspect Ratios

Qwen-Image-2.1 natively supports 2K resolution. Recommended sizes:

```python
aspect_ratios = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt="A panoramic mountain landscape",
    width=width,
    height=height,
    num_inference_steps=40,
).images[0]
```

### Default Parameters

| Parameter | Default | Notes |
|---|---|---|
| `num_inference_steps` | 40 | Number of denoising steps |
| `width` / `height` | 2048 × 2048 | Native 2K resolution; see aspect ratio table above |

## Prompt Rewriting

For best results, we recommend using the official **prompt rewriting models** to expand short prompts into detailed, high-quality descriptions. Two fine-tuned Qwen3.5-VL 9B checkpoints are provided — one for text-to-image, one for image editing — sharing a unified codebase that auto-detects the mode from input.

The rewriting code and weights are available at:
- **T2I**: [Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
- **Edit**: [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)
- **Code**: [`prompt_rewrite/`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/) — unified codebase with `--task t2i` or `--task edit`

```
prompt_rewrite/
├── run_transformers.py       # Local inference, batch size 1
├── run_vllm.py               # vLLM offline batch (recommended at scale)
├── serve.sh + client.py      # vLLM server + client
├── pe_core.py                # Task profiles, parsing, output records
├── requirements.txt
└── data/                     # Example inputs (t2i + edit with images)
```

### Text-to-Image

```bash
cd prompt_rewrite
pip install -r requirements.txt

# vLLM batch (recommended)
python run_vllm.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl

# Or local transformers
python run_transformers.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl
```

Output:

```json
{
  "rewritten_prompt": "<long detailed English prompt>",
  "wh_ratio": "16:9"
}
```

### Image Editing

```bash
python run_vllm.py --task edit \
    --ckpt Qwen/Qwen-Image-2.1-PE-I2I \
    --input data/edit_example.jsonl --output out.jsonl
```

Input format (JSONL):

```json
{"id": "abc123", "prompt": "make the sky sunset", "input_images": ["images/photo.png"]}
```

Output:

```json
{
  "rewritten_prompt": "Replace the daytime sky with a warm sunset ...",
  "wh_ratio": "",
  "ratio_follow": "<image1>"
}
```

- `wh_ratio` — model chose a new aspect ratio (e.g. `"16:9"`)
- `ratio_follow` — output inherits the specified input image's aspect ratio (e.g. `"<image1>"`)

### vLLM Server

```bash
CKPT=Qwen/Qwen-Image-2.1-PE-T2I bash serve.sh
# then:
python client.py --task t2i --model Qwen/Qwen-Image-2.1-PE-T2I \
    "a corgi playing guitar in the rain"
```

### Integration with the Pipeline

```python
import json
import torch
from diffusers import QwenImage21Pipeline

WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# After running the rewriter, read the output
rewrite = {"rewritten_prompt": "...", "wh_ratio": "16:9"}  # from run_vllm.py output
prompt = rewrite["rewritten_prompt"]
width, height = WH_RATIO_TO_SIZE.get(rewrite["wh_ratio"], (2048, 2048))

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt=prompt,
    width=width, height=height,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("rewritten_example.png")
```

## Advanced Usage

### Memory Optimization

For GPUs with limited memory, use model offloading:

```python
pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
```

### Prefix KV Cache

The transformer automatically caches the text and condition-image prefix across denoising steps when the checkpoint has `causal_condition: true` (the default). This provides significant speedup for image editing tasks with multiple condition images — the condition context is encoded once and reused for all denoising steps.

## Inference with vLLM

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) supports high-performance serving with prefix KV caching, CUDA Graph decode, FP8 quantization, and tensor parallelism.

### Offline Inference

```bash
# Text-to-image
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --output qwen21_t2i.png \
  --num-inference-steps 40

# Image editing
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image input.png \
  --prompt "Let this mascot dance under the moon" \
  --output qwen21_edit.png \
  --num-inference-steps 40
```

### Online Serving

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 40,
    "seed": 42
  }'
```

For step-wise execution (batch-level scheduling):

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

See the [vLLM-Omni recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1) for FP8 quantization, prefix KV cache options, multi-GPU parallelism, and detailed benchmarks.

## Inference with SGLang

**SGLang-Diffusion** provides native, high-performance inference for Qwen-Image 2.1, supporting text-to-image generation, multi-image editing, and transparent RGBA output. It offers multi-GPU parallelism, memory offloading, and optimized kernels across datacenter and consumer GPUs.

Generate a 1024×1024 image:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-2.1 \
  --prompt "A capybara reading a book by candlelight" \
  --height 1024 --width 1024 \
  --num-inference-steps 40 --guidance-scale 1 \
  --seed 42 --save-output
```

For image editing, add `--image-path input.png`. See the [Qwen-Image 2.1 cookbook](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1) for installation, GPU-specific commands, image editing, and transparent-background examples.

## Inference with LightX2V

[LightX2V](https://github.com/ModelTC/LightX2V) is a framework for image and video generation models, highly optimized for inference speed and GPU memory efficiency on both data center and consumer GPUs.

LightX2V supports Qwen-Image-2.1 for both text-to-image generation and image editing. See the [usage guide](https://github.com/ModelTC/LightX2V/tree/main/scripts/qwen_image_21) to get started.

## Architecture

Qwen-Image-2.1 is a single-stream DiT with the following design:

- **Transformer**: 32 layers, 7B parameters, single-stream architecture with block-causal attention (`(q_idx >= kv_idx) or same_image_block`). Text uses token-level causal mask; images use chunk-level bidirectional mask.
- **Text Encoder**: Qwen3-VL 8B (vision-language model) — encodes both text instructions and condition images into a unified representation.
- **VAE**: 64-channel RGBA autoencoder with 16× spatial compression, supporting native transparency.
- **Scheduler**: Flow Matching with Euler discrete scheduling and dynamic shifting.

The mixed-granularity attention architecture enables efficient **prefix KV cache reuse**: input images and text instructions are computed once at the first denoising step and cached for all subsequent steps.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-03.png" width="100%"/>
</p>

## Showcase

### Native Transparency

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-04.png" width="30%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-05.png" width="30%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-06.png" width="30%"/>
</p>

### Multi-Reference Editing

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-15.png" width="100%"/>
</p>
<p align="center"><em>Group photograph generated from six individual portrait references</em></p>

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-16.png" width="100%"/>
</p>
<p align="center"><em>Complete outfit assembled from five reference images (model, clothing, shoes, bag, hat)</em></p>

### Local Editing

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-18.png" width="48%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-19.png" width="48%"/>
</p>
<p align="center"><em>Circle-guided multi-region editing: remove watch, change hair color, replace clothing</em></p>

### Portrait and Product Fidelity

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-25.png" width="48%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-26.png" width="48%"/>
</p>

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-31.png" width="48%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-32.png" width="48%"/>
</p>

### Text Rendering

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-43.png" width="48%"/>
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-44.png" width="48%"/>
</p>

### Panorama and Storyboard

<p align="center">
<img src="https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/assets/example-38.jpg" width="100%"/>
</p>
<p align="center"><em>Panorama generated from a selfie</em></p>

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/images/example-42.png" width="100%"/>
</p>
<p align="center"><em>Storyboard generated from a three-view character reference</em></p>

## Community Support

### Diffusers (Recommended)

[Diffusers](https://github.com/huggingface/diffusers) supports Qwen-Image-2.1 via `QwenImage21Pipeline`, handling both text-to-image and image-conditioned generation in a single pipeline. See [PR #14804](https://github.com/huggingface/diffusers/pull/14804).

### ComfyUI

Qwen-Image 2.1 is natively supported in [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on Day 0. The compatible model weights can be downloaded from Hugging Face [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). See example workflows for [text-to-image](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json) and [image editing](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json).

### vLLM-Omni

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) accelerates Qwen-Image 2.1 through cross-step prefix KV cache reuse and dedicated CUDA Graphs, reducing redundant computation and kernel launch overhead. Request-level and step-level continuous batching improve GPU utilization and throughput, with phase-aware prefill and decode scheduling. It also supports tensor and Ulysses sequence parallelism, distributed VAE decoding with adaptive OOM recovery, FP8 weights and prefix KV storage, and CPU offloading for varying memory budgets. See the [Qwen-Image-2.1 recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1) for details.

### SGLang

[SGLang-Diffusion](https://github.com/sgl-project/sglang) provides native, high-performance inference with multi-GPU parallelism, memory offloading, and optimized kernels. See the [Qwen-Image 2.1 cookbook](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1) and [PR #39983](https://github.com/sgl-project/sglang/pull/39983).

### Wuli.art

For users in mainland China, [wuli.art](https://wuli.art/explore) offers free access to all Qwen Image 2.1 features in both Chatbox and Canvas, including image generations with transparent background.

![Qwen Image 2.1 on wuli.art](https://img.alicdn.com/imgextra/i4/O1CN01THqz79Bcz2L5YPrs_!!6000000002303-0-tps-2736-1536.jpg)

### ModelScope

ModelScope fully supports Qwen-Image-2.1. Built on its open-source [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) framework, the platform enables seamless model download, online generation and LoRA training. Explore these capabilities at [ModelScope Civision](https://modelscope.cn/aigc).

## Hardware Support

### AMD Radeon GPU

Get ready to run Qwen-Image 2.1 on AMD Radeon GPU. With ROCm, PyTorch, and Diffusers, developers can easily explore high-quality text-to-image generation on AMD GPUs.

### Diverse AI Chips via FlagOS

[FlagOS](https://github.com/flagos-ai) is a fully open-source system software stack for heterogeneous AI chips. It unifies the model–system–chip layers to enable a "develop once, run anywhere" workflow, eliminating the fragmentation among vendor-specific software stacks and substantially lowering the cost of porting AI workloads across accelerators.

In this release, Qwen-Image-2.1 leverages the FlagOS software stack to provide direct multi-chip support. By integrating the Triton-based operator library [FlagGems](https://github.com/flagos-ai/FlagGems) via the [Torch-FL](https://github.com/flagos-ai/Torch-FL) plugin, FlagOS enables seamless adaptation of the Diffusers library across chip platforms; the usage experience remains identical to that on NVIDIA, requiring zero code modifications. **Inference accuracy across all platforms has been aligned with the official implementation.**

Prebuilt images and weights for 8 chip platforms are released under [FlagRelease](https://modelscope.cn/organization/FlagRelease) — for example, [T-Head zhenwu](https://modelscope.cn/models/FlagRelease/Qwen-Image-2.1-BF16-zhenwu-FlagOS) and [Arm](https://modelscope.cn/models/FlagRelease/Qwen-Image-2.1-W8A8-arm-FlagOS).

## License Agreement

This repository is licensed under the [Qwen Research License Agreement](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE).

## Feedback

Having issues with Qwen-Image-2.1?
Our official feedback form connects you directly with the Qwen Image research team.
Share your prompts, images, or workflows to help us investigate and improve.

[**Submit Feedback →**](https://alidocs.dingtalk.com/notable/share/form/v01WgZOZA5DaVQPeqLX_dv19yqvsgs3oebp3pcjys_1qX0QQ0?source=link)

## Contact and Join Us

If you'd like to get in touch with our research team, join our [Discord](https://discord.gg/z3GAxXZ9Ce). We welcome issues and pull requests on GitHub.

If you're passionate about fundamental research, we're hiring full-time employees and research interns. Reach out at fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image-2.1&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image-2.1&Date)

---

# 2. prompt_rewrite/README.md

# Prompt Enhancer (Qwen3.5-VL 9B) -- t2i + edit

Two prompt-enhancer models that share one codebase:

| `--task` | What it does | Input | Answer fields |
| -------- | ------------ | ----- | ------------- |
| `t2i` | **Text-to-image prompt expansion.** Turns a short, rough image request into a long English prompt that reads like a description of the finished picture. | text | `rewritten_prompt`, `wh_ratio` |
| `edit` | **Image-editing instruction rewrite.** Turns a vague edit instruction plus its source image(s) into a precise, actionable prompt a downstream editor can follow without guessing. | text + 1..N images | `rewritten_prompt`, `wh_ratio`, `ratio_follow` |

Both checkpoints are **fine-tuned Qwen3.5-VL 9B, not the official base model**:
stock architecture (`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`,
hybrid linear/full attention, thinking on by default), post-trained weights.
Their `chat_template.jinja` and `tokenizer.json` are byte-identical, which is
what makes one codebase honest rather than merely convenient.

**Each task has its own checkpoint and its own system prompt.** They are not
interchangeable and there is no merged prompt: the answer contract is part of
what each model was trained on. Point `--ckpt` at one and give it that model's
prompt (via `--system-prompt`, or ship it as `system_prompt.txt` inside the
checkpoint directory and it is picked up automatically).

Pointing `--ckpt` at the official open-source Qwen3.5-VL 9B release will load and
generate, but it was never trained against either system prompt, so it does not
reliably emit the answer JSON -- expect `parse_ok: false` on most rows.

## Files

| File | What it is |
| ---- | ---------- |
| `pe_core.py` | Task profiles, message construction, answer parsing, output records. Everything the two tasks genuinely share. |
| `run_vllm.py` | Offline batch through `LLM.chat()`. **Use this for real workloads.** |
| `run_transformers.py` | Plain HuggingFace, batch size 1. Clear and hackable; for sanity checks and for modifying. |
| `serve.sh` | Start an OpenAI-compatible vLLM server on one checkpoint. |
| `client.py` | Talk to that server: one prompt, or a JSONL batch. |

All four entry points emit the **same** output records, so online and offline
results are interchangeable and nothing downstream needs to know which produced
a file.

## Install

```bash
pip install -r requirements.txt
```

Tested with `transformers==5.4.0`, `vllm==0.19.1`, `torch==2.10.0+cu128` on CUDA
12.x. The checkpoint loads through `AutoModelForImageTextToText`, which
dispatches on `config.model_type` (`qwen3_5` here).

## Hardware

One GPU is enough for either task. Weights are ~20 GB in `bfloat16`; vLLM needs
headroom for the KV cache on top, so a 40 GB card is comfortable at the default
`--gpu-memory-utilization 0.85`, and a 24 GB card wants `--max-model-len 12000`
or a lower utilization. `--tp 2` splits the weights to ~12 GB per GPU but buys
nothing at this size. System RAM: 64 GB is plenty.

## Input format

One JSON object per line, for both tasks:

```json
{"id": "abc123", "prompt": "make the sky sunset", "input_images": ["images/abc123.png"], "task_type": "basic_edit"}
```

- `prompt` -- the user's raw request, in any language.
- `input_images` -- **`edit` only**, a list of 1..N paths resolved relative to
  the JSONL's own directory (absolute paths also work). Every image is sent in
  order, because the system prompt tells the model to address them as
  `<image1>`, `<image2>`, ... -- reorder them and every reference in the rewrite
  silently re-points. Passing images to `--task t2i` is an error, not a warning:
  dropping them quietly would look like a successful run of the wrong
  experiment.
- `task_type` -- optional, echoed to the output for your bookkeeping. Both
  models use one system prompt for every task type; there is no router.

`t2i` line:

```json
{"id": "t2i_1", "prompt": "一只在雨中弹吉他的柯基"}
```

Multi-image `edit` line:

```json
{"id": "multi_1", "prompt": "Place <image1>'s subject into <image2>'s scene, matching lighting.", "input_images": ["images/portrait.png", "images/scene.png"]}
```

## Output format

One JSON object per line, fields in a stable order:

```json
{"id": "abc123",
 "task": "edit",
 "raw_prompt": "make the sky sunset",
 "input_images": ["images/abc123.png"],
 "task_type": "basic_edit",
 "thinking": "The image shows ...",
 "positive_prompt": "Replace the daytime sky with a warm sunset ...",
 "negative_prompt": "",
 "wh_ratio": "",
 "ratio_follow": "<image1>",
 "parse_ok": true}
```

- `positive_prompt` -- the rewritten instruction; the prompt you render.
- **`wh_ratio` / `ratio_follow` decide the output canvas.** For `edit` they are
  mutually exclusive: exactly one carries a value.
  - `wh_ratio` (e.g. `"16:9"`) -- the model chose the shape, because the task
    generates a new composition rather than editing the existing frame. This is
    the only one of the two that `t2i` uses.
  - `ratio_follow` (e.g. `"<image2>"`) -- `edit` only: the output inherits that
    input image's aspect ratio, because that image is the canvas being edited.

  Pass both through to whatever renders the prompt. Ignoring them and rendering
  at the source image's ratio discards a real part of the rewrite: a prompt
  describing a wide two-subject composition, rendered onto a portrait canvas, is
  a different picture.
- `parse_ok` -- `false` when the answer did not parse as the expected JSON
  object. `positive_prompt` then holds the raw answer text (nothing is lost) and
  the ratio fields are empty. Every entry point prints a summary line; to audit
  a finished file:

  ```bash
  jq -s 'map(select(.parse_ok | not)) | length' out.jsonl
  ```
- `negative_prompt` -- always `""`; neither model emits one. The field exists
  because downstream editors expect the slot.
- `task` -- which profile produced the row, so mixed corpora stay sortable.

## Usage

Offline batch, the normal path:

```bash
# t2i
python run_vllm.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl

# edit
python run_vllm.py --task edit \
    --ckpt Qwen/Qwen-Image-2.1-PE-I2I \
    --input data/edit_example.jsonl --output out.jsonl
```

Single-sample sanity check without a serving stack:

```bash
python run_transformers.py --task edit \
    --ckpt Qwen/Qwen-Image-2.1-PE-I2I \
    --input data/edit_example.jsonl --output out.jsonl --limit 1
```

Online, when you want an endpoint:

```bash
CKPT=Qwen/Qwen-Image-2.1-PE-T2I PORT=8100 bash serve.sh
# in another shell, once `curl -sf localhost:8100/health` answers:
python client.py --task t2i --model Qwen/Qwen-Image-2.1-PE-T2I \
    --system-prompt Qwen/Qwen-Image-2.1-PE-T2I/system_prompt.txt \
    "a corgi playing guitar in the rain"
```

`client.py` also takes `--input/--output` for a batch over HTTP, and `--image`
(repeatable, in order) for a single `edit` request.

## Sampling defaults

Per task, matching each one's production inference settings. Unset flags fall
back to these; the run's first log line prints the **effective** values and
marks anything you overrode.

| | `t2i` | `edit` |
| --- | --- | --- |
| temperature | 1.0 | 1.0 |
| top_p | 0.95 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |
| **presence_penalty** | **1.5** | **0** |
| max_new_tokens | 16256 | 24000 |
| thinking | on (required) | on (required) |

`presence_penalty` is the one that matters: the two values are not
interchangeable, and a wrong penalty does not fail loudly -- it quietly changes
the distribution you sample from. That is why there is no global default and why
the effective value is logged.

`transformers` has no native `presence_penalty` (`repetition_penalty` is
different math), so `run_transformers.py` implements the vLLM semantics as a
`LogitsProcessor`. Without it the t2i path would silently run at penalty 0.

Thinking is required: both models were trained with a `<think>` block and
degrade without it. The chat template opens one by default, and every entry
point also passes `enable_thinking=True` explicitly so the intent survives a
template change.

`--seed` defaults to 42, and what that buys you depends on the path:

- **`run_vllm.py` (offline batch) is reproducible.** Same input file, same
  sampling, same `--tp`, same graph/eager mode gives byte-identical output --
  verified across separate runs, including on a different GPU.
- **`client.py` against a server is not**, even with the same seed. The server
  batches concurrent requests, and a sampler seeded per request still sees a
  different batch composition each time. Two identical client runs differ. Use
  the offline path when you need reproducibility.
- Either way, `--enforce-eager` changes the numerics enough to diverge within a
  few hundred tokens, even under greedy decoding. Two batches rendered with
  different settings there are not comparable.

## Notes

**CUDA graphs are on by default.** On vLLM 0.19.1 this architecture's
linear-attention layers capture fine, and graphs are substantially faster, so
neither `serve.sh` nor `run_vllm.py` forces eager. `--enforce-eager` /
`EAGER=1` remain as escape hatches for other vLLM versions.

**Why one codebase.** The two tasks differ in exactly four places -- whether
images are accepted, whether `ratio_follow` exists, `presence_penalty`, and
`max_new_tokens` -- and all four live in one `Profile` dataclass in
`pe_core.py`. Everything else (chat template contract, thinking split, answer
parsing, output records, image resizing) is shared, so a fix to the parser
cannot land on one task and miss the other.

**Sockets.** Clusters often preset `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` to
interfaces that do not exist on the current host, which kills vLLM on init. Both
the runner and `serve.sh` unset any that name a missing interface.

---

# 3. prompt_rewrite/prompts/system_prompt_t2i.txt

```text
# Image Prompt Rewriting Expert

You turn a user's image request into one long English paragraph that describes the
finished image as if you were looking at it, plus the aspect ratio it should be
rendered at. You are not talking to the user and not talking to a renderer: you are
an observer reporting what is in the frame.

Work through the eight steps below in order. Each step commits one decision; later
steps never revise an earlier one.

## Step 1 — Read the brief and split it in two

List what the user has fixed and what they have left open.

Fixed, and it must survive into your description unchanged: every string of text
they want shown, every named object, every count, every stated colour, every stated
position, and the aspect ratio if they gave one. Copy their text strings character
for character, in their own script, including punctuation and spacing.

A third thing they may give you is an instruction about the job rather than about the
picture — "use double quotes", "no hard-edged blocks", "4K, no noise", "make sure the
text is sharp". That is not content. Obey it silently where it applies and never echo
it: the description states what is in the frame, never what must be done.

Open, and you must decide it: everything they did not mention. A three-word request
and a three-hundred-word request both become a description of the same size, so a
short brief means you are inventing most of the frame, not writing less.

## Step 2 — Fix the frame

Decide the orientation from the subject, then pick the ratio.

If the user states a ratio, use it. Otherwise: `3:2` for anything horizontal and
`2:3` for anything vertical — these are the two defaults and cover most images.
Use `1:1` for a square badge, icon, album cover or single centred emblem, `16:9`
for a wide cinematic or presentation frame, `1:2` or `9:16` for a phone screen or a
tall standing banner. `3:4`, `2:1`, `21:9`, `4:3`, `9:21`, `4:5`, `3:1`, `5:4`,
`1:3` exist but only when the subject or the user really calls for them.

The ratio lives only in the `wh_ratio` field. Never write a ratio, a resolution, or
a pixel count into the description itself.

## Step 3 — Write the opening sentence

One sentence, around twenty words. Name the medium, the style, the subject, and the
background or palette; usually name the orientation too:

`The image is a ⟨vertical / wide / square / tall⟩ ⟨style⟩ ⟨photograph · poster · illustration · scene · portrait · infographic · close-up · graphic · page · card · sheet · logo⟩ of ⟨subject⟩, ⟨the background and its palette⟩.`

`This is a …` or a bare `A vertical realistic photograph of …` work equally well. The
medium noun is the one part that is never omitted.

The style word goes here — realistic, photorealistic, minimalist, flat-vector,
cinematic, watercolour, isometric, editorial, hand-drawn, 3D-rendered, retro. Name
it once here; you may echo it in the closing sentence.

## Step 4 — Inventory before you write

Before any more prose, settle two lists.

Every element that will appear, each with a place in the frame: upper-left,
across the top, on the far right, in the lower-third, in the centre, in front of,
behind, tucked into the corner. You will need eight to fourteen such positional
phrases, about ten typically, and they must reach the corners, the edges and the
centre — not cluster in the middle.

Every piece of text that will be legible in the image, in reading order.

## Step 5 — Walk the frame

Now describe it in order. Which order depends on how the frame is filled.

**If the frame is divided into regions** — a poster, a page, an interface, a layout, a
wide scene with several things in it — walk the regions:

1. The background and the surface it sits on — this comes immediately after the
   opening sentence, not at the end.
2. The top band: headline, header bar, sky, ceiling, whatever occupies the top edge.
3. Down and across the body of the frame: left side, then centre, then right side.
   Give each region one or two sentences.
4. The bottom band: footer, foreground, ground plane, base row.

**If one subject fills the frame** — a portrait, a close-up, a single object — walk
the subject instead: the background and how far it falls off, then the subject's pose
and where it is placed in the frame, then head and face, then body and each garment or
surface, then what is held or touching it, then whatever little is left at the edges.
Keep using positional phrases inside the subject — in the upper-left of the frame,
behind the left shoulder, along the lower edge — so the frame stays locatable.

Roughly a third of your sentences should open on the positional phrase itself —
"On the right side of the frame, …", "In the upper-left corner, …", "Across the
lower third, …" — so the reader always knows where they are looking.

Keep it to one paragraph. Break to a new paragraph only when the image is genuinely
built from stacked regions — panels, cards, sections, slides — and then one
paragraph per region, each opening on where that region sits.

## Step 6 — Set every piece of text

Skip this step if nothing in the image is meant to be read — a third of images have
no legible text at all, and inventing signage for them is a mistake.

Otherwise, for each string from your Step 4 list, in reading order, name where it sits,
what it looks like, and what it says: `a bold black headline across the top reads "…"`.

Put the string in straight double quotes, in its own script — Chinese, Russian,
Korean, Japanese and Arabic text stays in Chinese, Russian, Korean, Japanese and
Arabic. Give its weight, colour, case and relative size. Describe a line break as a
second line rather than putting a real newline inside the string. If a mark is not meant
to be read — distant signage, a label behind glass, dense body copy — call it
blurred, indistinct, or too small to read rather than inventing letters. If the image contains a chart
or a table, its axes, tick labels, legend entries, series and cell values are text
too: write them out.

## Step 7 — Give the lighting its own sentence

Every image has light in it, and the description always accounts for it: the source,
its direction, its quality, and the shadows and highlights it leaves. Soft diffused
daylight from a window on the left, hard overhead studio light, warm low sun, flat
even ambient light for a diagram.

Once the contents are placed, give it a sentence of its own — `The lighting is …` —
or, if the light is what makes a particular surface look the way it does, fold it into
that surface's sentence. Either way it is stated explicitly, not left implied.

## Step 8 — Close with the whole frame

End on a single sentence that steps back:

`The overall composition ⟨is / uses / feels⟩ …`

`The composition is …`, `The overall design …`, `The overall mood …`, `The overall
palette …` and `The image has …` are the same move. Cover balance and symmetry, the
palette, the style, and the mood in that one sentence. Write exactly one such
sentence — do not follow it with a second summary.

## Throughout

**Size.** The description runs about twenty sentences and four to five hundred words,
roughly twenty-five words a sentence. That is the same size whether the brief was three
words or three hundred: a dense frame with many regions and a lot of text runs longer, a
single quiet subject runs shorter, but a thin brief never buys a thin description.

**Observe, don't instruct.** Present tense, third person, declarative. No "you", no
"create", no "make sure", no "the AI should". No quality boosters — no "masterpiece",
"8K", "highly detailed", "award-winning".

**Hedge what you cannot be certain of.** An observer describing a picture says
"appears to be", "likely", "suggesting", and offers a pair — "a notebook
or a tablet", "wood or dark laminate" — when the thing is genuinely ambiguous. Do
this often; it is the natural register here. Be flatly definite only about what the
user fixed.

**Name colours with a modifier, almost never bare.** Deep navy, muted olive, pale
cream, warm terracotta, soft dusty rose, blue-grey, off-white, charcoal, brownish-
green. Hex codes only if the user gave them.

**Give the material, not just the noun.** Brushed metal, matte plastic, glossy
ceramic, coarse linen, weathered wood, frosted glass, grain, scuffs, condensation,
visible brush strokes, paper fibre.

**Enumerate; never summarise.** "Several items" and "various decorations" are not
descriptions. Say what each thing is. Write small counts as words — three, five,
twelve — and if something is partly hidden, say so and describe the visible part.

**People get their observable surface.** Build, posture, where they are looking,
expression, hair, skin tone, and each garment with its colour and material. Age is a
life stage or a decade — a child, a teenager, a young adult, middle-aged, elderly,
in her thirties — never a number of years. If a face is turned away or cropped, say
that instead of describing it.

**Objects by class, not by brand.** A silver laptop, a mirrorless camera, a compact
hatchback — unless the user named the brand. Photographic and design vocabulary is
welcome: shallow depth of field, bokeh, backlit, close-up, negative space,
grid, drop shadow.

**Everything holds together physically.** Shadows fall away from the light, reflections
match what is in front of the surface, scale is consistent between neighbouring
objects, and a surface reacts to what sits on it. If the user asked for something
impossible, describe it as the image shows it and let the rest of the scene stay
coherent around it.

## Language

The description is always in English, whatever language the request arrives in. The
only exception is text shown inside the image, which stays in its own script.

## Output format

Return one strictly valid JSON object on a single line, nothing before or after:

{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}
```
