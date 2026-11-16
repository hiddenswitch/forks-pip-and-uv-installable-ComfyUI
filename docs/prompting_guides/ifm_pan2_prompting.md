# PAN 2 prompting and conceptual-reasoner guide (IFM / MBZUAI)

Vendored snapshot, complete, with source links made absolute, of:
1. https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/README.md (and https://huggingface.co/IFM/PAN2)
2. https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/inference.md
3. https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/inference/pan-prompt-rewriter/SKILL.md
4. [Complete Conceptual Reasoner system prompt and fixed negative template](ifm_pan2_system_prompt.md) (https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/inference/pan-prompt-rewriter/system_prompt.md and `assets/generic_negative.txt`)
5. `REWRITE_SYSTEM_PROMPT` from https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/inference/prompts_rewritter.py

Fetched: 2026-11-15

## ComfyUI conditioning, prompt rewriting, and sampling contract

- **Text encoder (`comfy/text_encoders/pan2.py`)**: Stock `Qwen3.5-9B` (`CLIPLoader` type `pan2`, e.g. `qwen3.5_9b_bf16.safetensors` or `qwen3.5_9b_int8_convrot.safetensors` from [Comfy-Org/PAN2](https://huggingface.co/Comfy-Org/PAN2)), reading the hidden state of layer 30 of 32 (`layer="hidden", layer_idx=-3`) without the final RMSNorm.
- **Conditioning chat template & prefix crop**: When encoding a diffusion caption (`CLIPTextEncode`), `PAN2Tokenizer` trims leading/trailing whitespace and wraps the caption in:
  ```text
  <|im_start|>system
  You are a helpful assistant. Generate the video with the following instructions.<|im_end|>
  <|im_start|>user
  {caption}<|im_end|>
  <|im_start|>assistant
  <think>
  ```
  `PAN2TEModel.encode_token_weights` slices the hidden states starting immediately after `<|im_start|>user\n` (`out[:, start:]`), so the 29B MMDiT (`PandoraWanModel`) sees the caption plus `<|im_end|>\n<|im_start|>assistant\n<think>\n` (a 6-token suffix).
- **In-graph prompt rewriting (`TextGenerate` / `TextGenerateV2`)**: When the same `pan2` `CLIP` object is passed to a text-generation node (`thinking` is not `None` or `llama_template` is set), `PAN2Tokenizer` keeps the stock `Qwen3.5-9B` chat template (`<think>\n\n</think>\n\n` when `thinking=False`), allowing the same loaded `Qwen3.5-9B` checkpoint to rewrite prompts before conditioning.
- **Two official rewriting formats in `ifm-ai/PAN2`**:
  1. **Structured 7-Key World-State JSON (`inference/pan-prompt-rewriter/`)**: The primary format used in the PAN 2 technical report and `inference/prompts/examples.json`. The conceptual reasoner ([ifm_pan2_system_prompt.md](ifm_pan2_system_prompt.md)) emits `{"refined_prompt": "<serialized JSON>"}` whose inner string decodes to a 7-key object (`video_style`, `lighting`, `perspective_view`, `subjects_lifecycle`, `init_frame_visual_description`, `events`, `quality`) within a **1,480-token soft limit** (**1,536-token hard capacity** per caption). Pass the **inner serialized JSON string** (`refined_prompt`) to the positive `CLIPTextEncode`, and pass the duration-substituted [fixed generic negative JSON](ifm_pan2_system_prompt.md#2-fixed-generic-negative-prompt-template-inferencepan-prompt-rewriterassetsgeneric_negativetxt) to the negative `CLIPTextEncode`.
  2. **Four-Paragraph Prose Rewriter (`inference/prompts_rewritter.py`, Section 4 below)**: A standalone instruction (`REWRITE_SYSTEM_PROMPT`) designed for local `Qwen/Qwen3.5-9B` rewriting (`temperature=0.2, top_p=0.9, max_new_tokens=4096, enable_thinking=False`) that outputs four prose paragraphs separated by blank lines (`Summary`, `Initial state`, `Dynamic motion`, `Other key attributes`).
- **Sampling (`comfy/model_base.py`, `comfy/ldm/pan2/model.py`)**:
  - **Model**: `pan2_bf16.safetensors` or `pan2_int8_convrot.safetensors` (`UNETLoader`), Wan 2.2 VAE (`wan2.2_vae.safetensors`).
  - **Latent & I2V**: `EmptyHunyuanLatentVideo` (T2V) or `Wan22ImageToVideoLatent` (I2V), default 704x1248 (or 1248x704) at 24 fps, 241 frames (~10 s, up to 481 frames / 20 s; frame count `4 * k + 1`).
  - **Sampler / Scheduler**: `KSampler` with `euler` sampler, `simple` scheduler (model sampling shift `9.0`), 50 steps, CFG `2.5`.

---

# 1. README.md

<!-- ============================================================
     HERO
     ============================================================ -->
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pan2-logo-dark.png">
  <img alt="PAN 2" src="https://raw.githubusercontent.com/ifm-ai/PAN2/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/assets/pan2-logo-light.png" width="420">
</picture>


<a href="https://huggingface.co/IFM/PAN2" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pan2" target="_blank"><img src="https://img.shields.io/badge/Diffusers-FF9D00?logo=huggingface&logoColor=white" alt="Diffusers"></a>
<a href="https://recipes.vllm.ai/IFM/PAN2" target="_blank"><img src="https://img.shields.io/badge/vLLM-30A2FF?logoColor=white" alt="vLLM"></a>
<a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md"><img src="https://img.shields.io/badge/Model%20card-ffd21e?logo=huggingface&logoColor=black" alt="Model card"></a>
<a href="#documentation"><img src="https://img.shields.io/badge/Docs-4f8cff?logo=readthedocs&logoColor=white" alt="Docs"></a>
<a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache%202.0-4CAF50?logo=apache&logoColor=white" alt="LICENSE"></a>
<a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment.md"><img src="https://img.shields.io/badge/PyTorch-2.12%20%7C%20CUDA%2013.2-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/compatibility.md"><img src="https://img.shields.io/badge/FlashAttention-4-76b900" alt="FlashAttention 4"></a>

[**Quickstart**](#quickstart) · [**Find your path**](#find-your-path) · [**Examples**](#examples) · [**Model**](#model) · [**Docs**](#documentation)

</div>

# PAN 2

Developed by the Institute of Foundation Models (IFM) at MBZUAI.

---

## Table of contents

- [What is PAN 2?](#what-is-pan-2)
- [Find your path](#find-your-path)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Model](#model)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Results](#results)
- [License and Attribution](#license-and-attribution)
- [Citation](#citation)

## What is PAN 2?

PAN 2 is a general-purpose world model: it simulates how the world will change in response to actions. Given a current observation (an optional image plus text) and actions described in text, it simulates the outcome as a video of up to 20 seconds. This supports *simulative reasoning*: an agent can imagine the outcomes of its candidate actions before choosing among them, by simulating each plan with PAN 2 and acting on the one whose simulated outcome is best.

PAN 2 follows the Generative Latent Prediction (GLP) architecture formalized in [Critique of World Models](https://arxiv.org/abs/2507.05169) and couples **conceptual reasoning** by a large language model with **perceptual reasoning** by a latent diffusion model. The language model, the **conceptual reasoner**, writes the next world state as a structured, time-grounded description of events, timed actions bound to actors, and camera motion. The 29B-parameter MMDiT, the **perceptual reasoning backbone**, predicts the video latents that realize that state, and the Wan 2.2 VAE decoder renders them as the next observation.

This repository holds the **training code** for the perceptual reasoning backbone: a token-budget packed dataloader over
precomputed latents, FSDP2 hybrid sharding, optional context
parallelism, and Muon. No
pixel decoding and no VAE forward happen during training; everything the loader
reads is produced offline and read from local disk. It also holds the **inference code** and the conceptual reasoner's prompt and agent skill. PAN 2 is also supported in [Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pan2) and [vLLM](https://recipes.vllm.ai/IFM/PAN2).

## Find your path

<table>
  <tr><th align="left" width="46%">I want to…</th><th align="left">Go to</th></tr>
  <tr><td><b>Set up the environment</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment.md">Environment setup</a> — conda, FA4, quack, model downloads</td></tr>
  <tr><td><b>Set up H200</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment_h200.md">H200 setup</a> — installation, training and inference</td></tr>
  <tr><td><b>Train on a small debug set</b></td><td><a href="#quickstart">Quickstart ↓</a></td></tr>
  <tr><td><b>Prepare my own data</b> — latents, CSVs, image tars</td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/data.md">Data</a></td></tr>
  <tr><td><b>Pretrain or SFT at scale</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/training.md">Training</a> — YAML, launch scripts, checkpoints, resume</td></tr>
  <tr><td><b>Fit long sequences on smaller GPUs</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/training.md#6-context-parallelism-cp_size">Context parallelism</a></td></tr>
  <tr><td><b>Generate videos from a checkpoint</b></td><td><a href="#2-docsinferencemd">Inference</a> — launcher, sampler controls, prompt JSON, rewriting</td></tr>
  <tr><td><b>Run the conceptual reasoner with a coding agent</b></td><td><a href="#3-conceptual-reasoner-skill-inferencepan-prompt-rewriterskillmd">PAN 2 Conceptual Reasoner (Prompt Rewriter) skill</a> — Codex / Claude Code skill: caption policy, fixed negative, token validation</td></tr>
  <tr><td><b>See results, safety guardrails and limitations</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md">Model card</a></td></tr>
  <tr><td><b>Upgrade torch / FA4 / quack</b></td><td><a href="https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/compatibility.md">Compatibility</a></td></tr>
</table>

## Quickstart

Train the 29B model on a one-node debug set. Full detail for each step is one
click away.

> **H200 GPUs:** see the [H200 setup guide](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment_h200.md) for installation, training and inference.

**1. Environment** ([docs/environment.md](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment.md))

```bash
conda create -n pan python=3.14 -y && conda activate pan
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
pip install --pre "flash-attn-4[cu13]==4.0.0b29" --no-build-isolation
pip install "quack-kernels[cu13]==0.6.4"
pip install -r requirements.txt
```

**2. Models** (text encoder for training; the VAE for encoding data and inference)

```bash
export HF_HOME=YOUR_HF_PATH
hf download Qwen/Qwen3.5-9B
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

**3. Data.** Encode video clips with the Wan 2.2 VAE and list them in CSVs; the
formats are shown by `python scripts/make_sample_data.py`, which writes a random-latent
`sample_data/videos.csv` and `sample_data/images.tar`. About
8 CSVs with 100 video latents and 8 image tars serve one node; point the `data`
paths in the YAML at them. See [docs/data.md](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/data.md).

```yaml
# config/pretrain_29B_720p.yaml
data:
  packed_dataloader_meta_csv_path: [/path/to/csvs/]      # one CSV per data-parallel rank
  image_tar_path: [/path/to/image_tars/]                 # optional
  image_sample_prob: 0.2
```

**4. Launch**

```bash
sbatch scripts/pretrain.sh      # config/pretrain_29B_720p.yaml
sbatch scripts/sft.sh           # warm-started from a pretrain checkpoint
```

Everything about a run lives in the YAML; the script only owns the cluster
side. A run always resumes from the latest checkpoint in its own `output_dir`.
See [docs/training.md](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/training.md).

**5. Generate** (single node, one worker per GPU)

```bash
bash inference/inference.sh <CKPT_PATH> <OUTPUT_DIR> [PROMPTS_JSON]   # or sbatch, same arguments
```

`CKPT_PATH` is a training `checkpoint-N/transformer` directory, a
`model_state.pt`, or a directory of `*.safetensors`; the architecture comes
from `config/sft_29B_720p.yaml`, the VAE and text encoder from the Hugging
Face cache (step 2). The launcher starts one `generate.py` worker per GPU,
shards the prompts across them, and samples with the default recipe: 50 steps, flow shift 9, CFG 2.5, 704x1248 at 24 fps, bf16. `PROMPTS_JSON` defaults
to `inference/prompts/examples.json` (one t2v and one i2v record). Videos land
in `<OUTPUT_DIR>/{t2v,i2v}/` next to a `*_config.json`; a rerun into the same
directory skips finished clips. Override any default through environment
variables (`NUM_STEPS`, `FLOW_SHIFT`, `GUIDANCE_SCALE`, `VIDEO_LENGTH`,
`SEED`, `NUM_GPUS`, ...) or pass extra `generate.py` flags in `EXTRA_ARGS`.
See [docs/inference.md](#2-docsinferencemd).

**6. Convert checkpoints** (diffusers layout <-> this repository)

The released weights (e.g. `IFM/PAN2`) are in the diffusers layout
(`PAN2Transformer3DModel`). `scripts/convert_checkpoint.py` converts them for
`generate.py` and training warm starts, and converts a training checkpoint back
to the diffusers layout for release:

```bash
# Released diffusers weights -> this repository; then pass <DIR>/transformer as CKPT_PATH.
hf download IFM/PAN2 --include "transformer/*" --local-dir /path/to/PAN2
python scripts/convert_checkpoint.py to-native \
    --input /path/to/PAN2/transformer --output <DIR>/transformer

# Training checkpoint -> diffusers transformer/ folder (architecture from the YAML).
python scripts/convert_checkpoint.py to-diffusers \
    --input /path/to/checkpoint-N/transformer --output /path/to/release/transformer \
    --training_config config/sft_29B_720p.yaml --dtype bf16
```

To warm start training from the converted weights, set both keys in the YAML:

```yaml
model:
  pretrained_transformer_path: <DIR>/transformer
  load_transformer_weights: true    # the pretrain YAML ships false
```

The conversion only renames, splits and concatenates tensors, so it is exact in
both directions. See [docs/inference.md](#7-converting-checkpoints).

## Examples

### Prompts

Training captions are structured, and prompts work best in the same style.
`inference/prompts/examples.json` ships one t2v and one i2v record in that
format, each with its structured negative prompt, duration and (for i2v) a
reference image. Free-text prompts like these two also work:

> A muscular male swimmer in his early 30s, wearing a sleek professional swim
> cap and mirrored goggles, stands poised on a starting block. With an
> explosive burst of energy, he leans forward and dives headfirst into the
> crystalline blue water of an Olympic-sized swimming pool. The camera captures
> the moment of impact in a dynamic shot, showing a massive plume of white
> spray and chaotic bubbles erupting around him […]

> A wide, cinematic shot opens in a high-end, minimalist showroom characterized
> by polished white marble floors that mirror the overhead soft-box lighting
> and stark, charcoal-grey walls. In the center of the frame, a futuristic
> concept car is unveiled […] Luminous neon-blue accents trace the contours of
> the chassis and the thin, blade-like LED headlights, which pulse slowly with
> a breathing light effect […]

A conceptual reasoner expands them into the training caption style in two ways: offline with
`inference/prompts_rewritter.py` and a local instruction-tuned model, or
interactively through the [PAN 2 Conceptual Reasoner (Prompt Rewriter)](#3-conceptual-reasoner-skill-inferencepan-prompt-rewriterskillmd)
agent skill, which carries the same caption policy and fixed negative prompt
and validates the result against the text encoder's token budget. See
[docs/inference.md](#5-prompt-rewriting).

### Training data row

One CSV row per clip: the latent path and the caption.
`scripts/make_sample_data.py` writes ten of these with random latents.

```csv
"latent_path","caption"
"/data/latents/video_00.safetensors","{""video_style"": ""..."", ""lighting"": ""..."", ""perspective_view"": ""..."", ""events"": [...], ...}"
```

### Image-to-video

Set `image_cond_prob` in the `tasks:` section of the YAML (0.3 in the shipped
configs). Per row the loader draws i2v with that probability and conditions on
the first latent frame through channel concatenation, so one model serves both
tasks. At generation time any prompt-JSON entry with an `image_path` (or
`type: i2v`) is run as image-to-video by the same launcher; the shipped
`examples.json` has one of each.

## Model

| | |
| --- | --- |
| Parameters | 29.2B |
| Backbone | 44 dual-stream (video + state) joint-attention blocks, hidden size 4096, 32 heads, RMS qk-norm, 3-D RoPE |
| Latent space | Wan 2.2 VAE, 48 channels, 16x spatial / 4x temporal downsampling, patch `[1, 2, 2]` |
| Text encoder | Qwen3.5-9B hidden states (last 2 layers dropped), caption-encoder projection, up to 1,536 caption tokens |
| Resolution | 704x1248 at 24 fps: a 44x78 latent grid, 858 tokens per latent frame; up to 481 frames (20 s) |
| Objective | flow matching, fixed timestep shift 9; logit-normal timestep sampling in pretraining, uniform in SFT |
| Tasks | t2v, i2v, channel-concat conditioning: 48 noisy + 48 condition + 1 mask = 97 input channels |
| Training | three stages: 480p pretraining, 720p continued pretraining, 720p SFT |


This repository covers pretraining and SFT. Reward learning and the model soup
that produce the released weights are described in the technical report and are
not part of the training code.


| Config | Purpose |
| --- | --- |
| [`config/pretrain_29B_720p.yaml`](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/config/pretrain_29B_720p.yaml) | dense 720p pretraining / continued pretraining |
| [`config/sft_29B_720p.yaml`](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/config/sft_29B_720p.yaml) | SFT, warm-started from a pretrain checkpoint, image mixing off |

Results, safety guardrails and limitations: [MODEL_CARD.md](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md).

## Repository map

```text
pan2/
├── train.py            # entry point
├── config/             # training YAMLs
├── scripts/            # Slurm launchers: pretrain.sh, sft.sh; convert_checkpoint.py (diffusers <-> native weights);
│                       #   make_sample_data.py (random-latent data in the training formats)
├── dataset/            # packed dataloader: CSV stream, latent reads, image/video mixing, bin packing
├── pan/                # model, inference pipeline, text encoder, optimizer
│   ├── models/         # PanVideoDiffusionTransformer, Qwen3.5 text encoder
│   ├── optim/          # Muon, patched LR schedulers
│   ├── pipelines/      # inference pipeline
│   └── utils/          # Ulysses CP, guidance controls
├── generate.py         # inference entry point (one process per GPU, prompt sharding)
├── inference/          # single-node launcher, example prompts, prompt rewriter, pan-prompt-rewriter/ agent skill
├── docs/               # the guides linked below
├── assets/             # logo and demo media
└── MODEL_CARD.md       # overview, usage, results, safety guardrails, limitations
```

## Documentation

| Guide | Covers |
| --- | --- |
| [Environment setup](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/environment.md) | conda env, CUDA stack install order, install check, model downloads |
| [Data](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/data.md) | Wan 2.2 VAE encoding and normalization, CSV and image-tar formats, image/video mixing, token budgets, debug set |
| [Training](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/training.md) | launching, the YAML sections, pretrain vs SFT, launch scripts and kernel caches, checkpoints and resume, context parallelism |
| [Inference](#2-docsinferencemd) | launcher, default sampler, guidance controls, prompt JSON format, resume |
| [Compatibility](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/compatibility.md) | version pins, FA4/quack upgrade contract, cu12 vs cu13 |
| [Model card](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md) | overview, usage, results, safety guardrails, limitations |


## Results

PAN 2 with Kimi K3 as the conceptual reasoner, from the technical report; benchmarks, human evaluation and the evaluation protocols are in the [model card](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/MODEL_CARD.md#headline-results).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/report/results_capabilities_dark.png">
    <img alt="Capability and multi-step plan results, PAN 2 highlighted" src="https://raw.githubusercontent.com/ifm-ai/PAN2/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/assets/report/results_capabilities_light.png" width="100%">
  </picture>
</p>

## License and Attribution

PAN 2 is developed by the Institute of Foundation Models (IFM) at MBZUAI.

PAN 2's own code and weights are released by MBZUAI under the [Apache License, Version 2.0](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/LICENSE).

PAN 2 includes or is used with third-party components, each under its own license. The license files are:

- [LICENSE](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/LICENSE): the license for PAN 2's own code and weights.
- [NOTICE](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/NOTICE): MBZUAI's copyright notice and attribution for the third-party components.
- [THIRD_PARTY_LICENSES](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/THIRD_PARTY_LICENSES): the full license text of each third-party component, with its original copyright line.

### Modifications

The Wan 2.2 and Qwen (Qwen3.5-9B) files and weights are used as released by their authors and have not been modified or fine-tuned by MBZUAI. Third-party code adapted in this repository keeps its original copyright and license header and, where MBZUAI has changed the code, a notice describing the changes.

## Citation

```bibtex
@misc{pan2_2026,
  title        = {PAN 2},
  author       = {{PAN Team}},
  organization = {Institute of Foundation Models, Mohamed bin Zayed University of Artificial Intelligence},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/ifm-ai/PAN2/tree/main}},
  url          = {https://github.com/ifm-ai/PAN2/tree/main}
}
```

---

# 2. docs/inference.md

# Inference

[← back to README](#1-readmemd) · prev: [Training](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/training.md) · next: [Compatibility](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/docs/compatibility.md)

## 1. Run it

```bash
bash inference/inference.sh <CKPT_PATH> <OUTPUT_DIR> [PROMPTS_JSON]    # on a GPU node; sbatch works too
```

`inference/inference.sh` starts one `generate.py` worker per GPU on the node.
Every worker holds the full 29B model and takes every N-th prompt
(`--prompt_shard_index / --prompt_shard_count`), so throughput scales with the
GPU count and no context parallelism is involved. `CKPT_PATH` is a training
`checkpoint-N/transformer` directory (DCP), a `model_state.pt`, or a directory
of `*.safetensors`. `PROMPTS_JSON` defaults to `inference/prompts/examples.json`,
one t2v and one i2v record with structured negatives and a reference image.

The architecture is read from the training YAML (`TRAINING_CONFIG`, default
`config/sft_29B_720p.yaml`), so no model root is needed: the Wan 2.2 VAE and
the Qwen3.5 text encoder come from the Hugging Face cache (`HF_HOME`), the
transformer weights from `CKPT_PATH`. Everything else is an environment
variable with a default in the script:

| variable | default | meaning |
| --- | --- | --- |
| `NUM_STEPS`, `FLOW_SHIFT`, `GUIDANCE_SCALE` | 50, 9, 2.5 | sampler (section 2) |
| `TASK` | `auto` | `t2v`, `i2v`, or both from one JSON |
| `VIDEO_HEIGHT` x `VIDEO_WIDTH`, `FPS` | 704 x 1248, 24 | output size and frame rate |
| `VIDEO_LENGTH` | 241 | frames for entries without a duration |
| `SEED` | 123 | |
| `NUM_GPUS` | all visible GPUs | workers on this node |
| `NEGATIVE_PROMPT_FILE` | `inference/prompts/negative_prompt.json` | fallback negative for entries without one; its time ranges are rendered to each clip's duration |
| `EXTRA_ARGS` | | appended to the `generate.py` command |

Outputs land in `<OUTPUT_DIR>/<t2v|i2v>/<relative_path>.mp4`, each
next to a `*_config.json` with the effective arguments. A rerun skips a video
whose config matches and fails closed when it differs, so use a new output
directory for a new checkpoint, negative prompt or sampler.

## 2. Default sampler

The launcher passes these values to `generate.py`:

| control | value |
| --- | --- |
| steps / flow shift | 50 / 9, standard timestep schedule |
| CFG | 2.5 on every update (indices 0-49) |
| skip interval, STG | off |
| I2V | `--i2v_skip_first_latent_cfg true`: temporal latent 0 uses the positive prediction |
| precision | bf16 transformer; storage precision of the checkpoint is independent |

Indices are zero-based and end-exclusive.

## 3. Guidance controls

`generate.py` exposes the pipeline's sampler controls as flags. Append any of
them through `EXTRA_ARGS` when using the launcher.

| flag | effect |
| --- | --- |
| `--cfg_start_step`, `--cfg_end_step`, `--cfg_warmup_scale` | full-scale CFG window; before it a constant warm-up scale (1 = positive-only), after it positive-only |
| `--cfg_skip_interval a b` | positive-only inside `[a, b)` even within the window |
| `--stg_scale s --stg_skip_layers i ...` | skip-layer guidance: one extra positive pass with whole joint blocks skipped; t2v only, no offloading hooks |
| `--i2v_skip_first_latent_cfg` | default true; `false` restores CFG on the reference latent |
| `--first_step_velocity_scale` | scale the whole first update, in [0, 1] |
| `--timestep_schedule early_refine_2x` | halve the first two sigma intervals of a 50-step grid, drop two middle boundaries |

The math lives in `pan/utils/inference_guidance.py` (plan, CFG, STG) and
`pan/schedulers/timestep_schedule.py`; the pipeline
keeps the exact legacy bf16 arithmetic whenever no control is requested.

## 4. Prompt JSON

`--prompt_json` is a list of records (or `{"entries": [...]}`):

| field | notes |
| --- | --- |
| `refined_prompt` / `refined_prompts` / `prompt` / `caption` | positive text, the first nonempty one in this order, so a rewrite wins over the raw prompt; structured captions in the training style work best |
| `negative_prompt` | optional; falls back to `--negative_prompt` / `--negative_prompt_file`, whose `time_range_sec` fields are rendered to the clip's duration |
| `type` or `task` | `t2v` / `i2v`; inferred from `image_path` when absent |
| `image_path` (`image`, `ref_image`) | i2v reference; `--image_root DIR` remaps by basename |
| `duration_sec`, `frames`, `num_target_frames` | frames = `4 * ceil(duration_sec * fps / 4) + 1`; else `--video_length` |
| `id` or `relative_path` | output name: `<task>_<id>.mp4`, or the given relative path |

Direct use without JSON: `--prompt "..." [--image_path ref.png] --output_path out.mp4`.
The output size is always explicit (`--video_height` / `--video_width`, default 704 x 1248).
`inference/prompts/examples.json` shows one t2v and one i2v record in this format.

## 5. Prompt rewriting

Training captions are structured (`video_style`, `lighting`,
`perspective_view`, `subjects_lifecycle`, timed events). Free-text prompts
work best after being rewritten into that style. `inference/prompts_rewritter.py`
does this offline with a local instruction-tuned model (`--model-path`,
by default `Qwen/Qwen3.5-9B`, the same model PAN 2 uses as its text encoder)
and writes a JSON prompt file the launcher accepts. There is no in-process
rewriting; `generate.py` passes prompts to the text encoder verbatim.

The same rewriting policy is also packaged as an agent skill in
[`inference/pan-prompt-rewriter/`](#3-conceptual-reasoner-skill-inferencepan-prompt-rewriterskillmd)
for Codex, Claude Code or any assistant that can follow a `SKILL.md`. The
skill authors the positive caption from `system_prompt.md`, attaches the fixed
negative prompt from `assets/generic_negative.txt` (only the duration is
substituted), and `scripts/prompt_tools.py` assembles and validates the
records, including the text-encoder token count when pointed at the Qwen3.5
tokenizer with `--tokenizer`. To install it, copy or symlink the directory to
`~/.codex/skills/pan-prompt-rewriter` or `.claude/skills/pan-prompt-rewriter`.

## 6. Attention backend

The pinned environment ships FlashAttention-4 only. The non-packed inference
path runs `attention_backend: flash` on FA4 (the only supported backend), for both
the caption encoder and the joint video-text attention; the startup banner prints
`attention  FlashAttention-4 (attention_backend=flash)`.

## 6b. Memory

`--offload` picks where the models live: `none` (everything on the GPU, what the
launcher uses), `models` (text encoder, DiT and VAE wait on the CPU between uses),
`blocks` (additionally stream the DiT's blocks to the GPU while it runs;
`--offload_prefetch` overlaps the copies at the cost of host memory) or `auto` (the
default: `blocks` when the bf16 DiT does not fit in free GPU memory, else `none`).

## 7. Converting checkpoints

Released weights (`IFM/PAN2` on Hugging Face) use the diffusers layout of
`PAN2Transformer3DModel`; this repository's checkpoints use the native layout of
`PanVideoDiffusionTransformer`. `scripts/convert_checkpoint.py` converts between
the two:

```bash
# diffusers transformer/ folder -> <DIR>/transformer/model.safetensors
python scripts/convert_checkpoint.py to-native \
    --input /path/to/PAN2/transformer --output <DIR>/transformer
bash inference/inference.sh <DIR>/transformer <OUTPUT_DIR>

# checkpoint-N/transformer (DCP), model_state.pt or safetensors -> diffusers transformer/ folder
python scripts/convert_checkpoint.py to-diffusers \
    --input /path/to/checkpoint-N/transformer --output /path/to/release/transformer \
    --training_config config/sft_29B_720p.yaml --dtype bf16
```

`generate.py` reads a safetensors checkpoint from a folder named `transformer`
(or a folder containing one), so keep that name for `--output` of `to-native`.
`to-diffusers` takes the architecture from `--training_config` and writes
5 GB shards, the shard index and `config.json`; `--dtype` casts the weights
(default: keep the stored dtype). The same native output also works as
`model.pretrained_transformer_path` for a training warm start, together with
`model.load_transformer_weights: true`.

The layouts hold the same tensors, so both directions are exact. The diffusers
layout splits the fused QKV, output and q/k-norm projections per modality,
stores the caption-encoder SwiGLU as `[w3; w1]` and the output-head modulation
as `[scale; shift]`, keeps only row 0 of the text type table (the only row the
model reads) and drops the timestep frequency table (a constant rebuilt on
load); `to-native` restores both.

---

# 3. Conceptual Reasoner Skill (`inference/pan-prompt-rewriter/SKILL.md`)

---
name: pan-prompt-rewriter
description: Rewrite text-to-video or image-to-video requests with the unchanged positive policy and fixed no-actor-ID negative. Preserve metadata and durations, inspect supplied first frames, and validate serialized prompts and production token budgets. Not for video generation.
---

# PAN2 Prompt Rewriter

Use the complete [system_prompt.md](ifm_pan2_system_prompt.md) for positive authoring. Apply it directly with the current authoring model; this skill does not select a model, call a rewrite API, or generate videos.

## Authoring and assembly are separate stages

The system prompt's instruction to return only `refined_prompt` governs the **intermediate positive-authoring response**. This skill also performs the caller's assembly stage: its final output contains **both** `refined_prompt` and `negative_prompt` as JSON strings, together with preserved source metadata. Do not change the system prompt to reconcile those two stages.

1. Accept a plain-language request, one JSON record, or an ordered list. Wrap plain text in `original_prompt`. Preserve all metadata values and record order. When rewriting existing records, save prior captions separately and exclude only `refined_prompt` and `negative_prompt` from the clean request. Do not use old captions as examples to copy.
2. Resolve `duration_sec` or `target_duration_sec` exactly as the policy specifies; if neither is present, add `duration_sec: 13`. Preserve supplied durations. Do not silently round invalid durations or reconcile conflicting aliases.
3. For I2V, actually view the supplied first-frame image before authoring. Preserve its visible state and crop; requested additions belong in continuous later events as the policy specifies. A filename alone is not visual evidence. If the image is inaccessible, ask for the image or permission to use text-only conditioning. For T2V, no image is required. Before drafting I2V, note visible held objects and occupied hands; check the first action accounts for these so an existing object is not silently dropped or replaced.
4. Read the full positive policy and author each request from its original text and, when provided, viewed image. Produce the intermediate object `{"refined_prompt":"<serialized positive JSON>"}`. Use `json.dumps(positive, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`; do not double-encode. Keep the author's choices governed by the policy rather than imposing a new event count, camera plan, pacing template, or aesthetic. Before drafting, apply the policy's distinction between independently acting subjects and genuinely collective behavior; distinct actions and held-object roles need distinct performer attribution. Prefer a suitable object already established by the request or image over inventing a replacement and an extra introduction action; retain any explicitly requested additional object. Treat the token limit as a ceiling, not an instruction to minimize caption length: retain useful scene-specific spatial, material and action detail without padding or repeating it.
5. Assemble with [scripts/prompt_tools.py](https://github.com/ifm-ai/PAN2/blob/0fb907b6e16a82ead6d691d1de2eef6299c71e9a/inference/pan-prompt-rewriter/scripts/prompt_tools.py). It preserves metadata and the authored positive, validates structure, and attaches [assets/generic_negative.txt](ifm_pan2_system_prompt.md#2-fixed-generic-negative-prompt-template-inferencepan-prompt-rewriterassetsgeneric_negativetxt). This is exactly the no-actor-ID template. Never author, paraphrase, scene-specialize, add modules to, or shorten the negative. Its subject declaration remains, but its contained action has no `actor_id`. Its negative camera tags and artifact list intentionally differ from the positive schema.
6. Review the result against the source and the viewed image for requested actions, outcomes, initial state, continuity, and camera behavior. Run the helper's production-token validation when the deployed tokenizer is available. Preserve each emitted positive response unchanged. If validation or comparison finds a problem, record it separately; do not patch the returned caption. For a skill-development iteration, revise these workflow instructions only where the observed issue warrants it, rebuild from the handoff, and generate a fresh response from the original request and image. Keep the positive policy unchanged unless the user explicitly requests its revision. Save every attempt separately, and do not retry indefinitely. Never fix a negative overflow by editing the fixed template. Report unresolved failures.

The negative changes only its three `time_range_sec` values from `0 to {duration_sec}` to `0 to D`, using plain decimal notation. All other characters in the stripped template remain identical. Both prompt strings decode once into their seven-key objects. The final record is the clean source record plus these two strings; a batch is an equally sized list in source order. Validation receipts belong in separate files or brief accompanying status, never inside the captions.

## Run the helpers

Save `request.json` as clean source metadata and `positive.json` as the intermediate positive-only response. For a batch, both are lists in matching order. Replace `SKILL_DIR` with this skill's directory and choose new output paths:

```bash
python3 -B SKILL_DIR/scripts/prompt_tools.py assemble \
  --request request.json --positive positive.json --output candidate.json
python3 -B SKILL_DIR/scripts/prompt_tools.py validate \
  --request request.json --positive positive.json --candidate candidate.json \
  --report validation.json --tokenizer /path/to/deployed/text_encoder/vlm
```

The validator uses the same chat-template crop and suffix count as the API rewriter. It requires a Python environment with `transformers` and the deployed tokenizer's dependencies and local assets. Positive soft limit: 1,480; hard capacity: 1,536 **per caption**. It does not append negative modules. Do not substitute a generic LLM tokenizer.

Without those assets, omit `--tokenizer` for a structure/preservation check. This intentionally exits 2 with `tokenizer_ok: false`; label the result **Unverified draft — production encoder-token counts have not been verified.** Exit 0 indicates complete mechanical validation; exit 1 indicates a checked failure. A receipt is bound to the exact candidate; any edit requires another check. Keep failed drafts and receipts when correcting.

Without code, apply the same schema and exact template copy with only the three duration substitutions, and label the result unverified. Do not invent counts or receipts. Similarity to prior results means fidelity to the same request, image, authoring policy, output schema and fixed negative; positive wording and scene choices can vary with the model and sampling. Prompt checks do not establish generated-video quality.

## Compare skill revisions

Keep the authoring model, reasoning configuration, input records and images consistent across trials. Use saved API captions only for evaluation, not as per-record examples or text to copy into rewrites. Preserve raw generated positives and assemble them deterministically; never present a manually corrected output as skill-generated. Record the instruction snapshot, test provenance, failures and comparison separately. Distinguish source fidelity and image grounding from exact wording, discretionary staging and mechanical validity. If the same session has already seen references, disclose that the trial is not blind.

---

# 4. Four-Paragraph Rewriter System Prompt (`inference/prompts_rewritter.py`)

`inference/prompts_rewritter.py` runs `Qwen/Qwen3.5-9B` with `enable_thinking=False`, `temperature=0.2`, `top_p=0.9`, and `max_new_tokens=4096`. For text-to-video, the user message is `f"{REWRITE_SYSTEM_PROMPT}\n\nInput text:\n{prompt}"`. For image-to-video, the first-frame image is attached before the text `f"{REWRITE_SYSTEM_PROMPT}\n\nReference image: the attached image is the exact first visible frame for this image-to-video rewrite.\n\nInput text:\n{prompt}"`:

````text
You are a video-generation caption rewriter. Rewrite user-provided text into the same natural-language structure and style as the `global_caption` captions from the latest video captioning system.

Output only the rewritten caption text. Do not output JSON, markdown, headings, bullets, explanations, or quotes.

Write exactly four prose paragraphs separated by blank lines:

General input handling:
- The input may be a short prompt, a long caption, a partially structured description, or a detailed multi-paragraph caption.
- Preserve user-specified semantics, subject identities, requested action, camera intent, and constraints. Improve clarity and completeness without changing the requested video.
- If the input is already detailed, reorganize and normalize it into the four-paragraph caption style instead of expanding it as if it were only a simple sentence.

Reference-image handling for image-to-video:
- If a reference image is attached, treat it as the exact first visible frame and the visual authority for initial scene content.
- In that case, paragraph 2 must rely on the attached image for setting, subject/object inventory, appearance, position, readable text, lighting, style, and initial camera framing. Do not hallucinate initial-state details that are not visible in the image.
- In that case, paragraph 3 must describe motion as a continuation from the attached image. The moving subjects, their starting positions, directions, contacts, and camera behavior must remain consistent with the reference image and the user text.
- If the user asks for a new subject or object that is not visible in the reference image, introduce it only as an entering or newly appearing element during paragraph 3, not as something already present in paragraph 2.
- If the user text conflicts with the reference image, keep the image-grounded initial state and satisfy requested motion only where it can be made visually consistent.

Paragraph 1, summary:
- Write one or two concise sentences at the global level.
- Mention the camera range or shot type, visual style or medium, major subject/topic, setting category, and what the video is about when those details are implied or can be added without changing the intent.
- Keep this paragraph at the global level. Do not put detailed subject appearance, object inventories, text/logo details, lighting specifics, or background layout here; save those details for paragraph 2.
- Prefer phrasing like "A close-up photorealistic shot of..." or "A wide handheld-style view of..." when appropriate.

Paragraph 2, initial state:
- Describe only the first visible moment as a still image.
- Include the setting, layout, foreground/midground/background placement, subject appearance, object color/shape/size/material, readable text if explicitly present, lighting, color palette, and composition.
- Use frame-relative position words such as `frame-left`, `frame-right`, `upper frame`, `lower frame`, and `center`.
- For major foreground subjects, use specific visible descriptions rather than coarse labels when more detail is available.
- Do not include objects, text, poses, lighting changes, or layout details that appear only later; those belong in paragraph 3.
- End this paragraph with the initial camera description: shot type/range, angle, framing/composition, perspective, focus, and stability.

Paragraph 3, dynamic motion:
- Describe what happens in chronological order using natural temporal phrasing such as "at first", "then", "as", "next", "while", "afterward", or "eventually".
- Preserve every action, interaction, object state change, and physical relationship from the input text.
- Explicitly describe motion for any moving subject or object, including path, direction, rotation, bending, swinging, contact, support, resistance, settling, or lack of significant movement.
- If important elements remain static or persist throughout, say so.
- End this paragraph with the dynamic camera description, including whether the camera is static, panning, tilting, tracking, dollying, zooming, handheld, reframing, changing focus, or only slightly shaky. If the camera is static, fixed, or only slightly shaky, state that explicitly instead of omitting camera behavior.

Paragraph 4, other key attributes:
- Add remaining important physics, material, contact/support, continuity, lighting, visual style, scene-context, text/logo persistence, or image-quality details that help the caption stand alone for text-to-video training.
- Mention visual artifacts, watermarks, borders, compression, blur, overexposure, or disruptive edits only if the input explicitly implies them.

Style examples:

Example 1:
A close-up cinematic shot of a performer in traditional Chinese opera attire performing a stylized sequence of movements in an ornate indoor setting.

At the beginning, a performer is centered in the frame, wearing elaborate white face makeup with pink shading around the eyes and red lips. They are adorned with a massive, intricate headdress featuring blue and red accents, numerous white pearls, and dangling beaded strands. The performer wears a white robe with wide sleeves and a colorful garment underneath featuring blue, red, and gold embroidery. They hold a small, rectangular golden box in front of their face with both hands, partially obscuring their features. In the background, a large, ornate yellow screen with blue dragon motifs is visible, set against a red backdrop with purple cloud patterns. Two other performers in similar opera makeup and headdresses are partially visible on the frame-left and frame-right. The lighting is soft and warm, creating a rich, saturated color palette. The camera is a static close-up shot with a centered composition and stable focus.

As the sequence progresses, the performer slowly lowers the golden box, revealing their face. They then lift the box higher, momentarily resting it atop their headdress while glancing toward the frame-left. Afterward, the performer brings the box back down and moves it to the frame-left, while their right hand moves toward their face in a graceful, stylized gesture. Throughout the motion, the beaded strands of the headdress sway gently. The other performers in the background remain mostly static, maintaining their poised expressions. The camera remains static throughout the entire sequence, with no movement, zoom, or change in framing.

The materials of the costumes are highly detailed, showing the shimmer of the gold embroidery and the luster of the pearls. The makeup is applied with a smooth, matte finish, contrasting with the glossy surface of the golden box. The scene maintains consistent continuity in lighting and composition, with no visible visual artifacts or disruptive edits.

Example 2:
A wide handheld-style shot captures a variety of potted citrus and flowering plants for sale at an outdoor market, set against a backdrop of a parking lot and blossoming trees.

At the beginning, the scene is filled with numerous black plastic pots containing small citrus trees laden with orange fruits and some yellow-flowering shrubs. The plants are arranged in rows on a light gray concrete pavement. In the background, a blue canopy tent is visible on the frame-left, with several people walking nearby. Further back, a parking lot contains various white and silver cars, and the horizon is lined with white-blossomed trees under an overcast sky. The lighting is soft and diffused, creating a natural, low-contrast color palette. The camera starts in a wide shot with a stable, eye-level perspective and deep focus.

As the video progresses, the camera begins a complex movement, panning to the frame-right while simultaneously tracking right and moving backward. This motion shifts the focus from the citrus trees on the frame-left toward the yellow-flowering shrubs on the frame-right. The camera also tilts slightly upward and rolls clockwise, creating a dynamic, sweeping view of the plant nursery. The people in the background continue to move slowly, and the cars remain stationary in the parking lot. The camera movement is fluid but exhibits slight handheld jitter as it reframes the composition to emphasize the yellow blossoms near the end.

The plants are supported by black plastic nursery pots, some with small white labels attached. The citrus fruits are bright orange, contrasting with the deep green foliage, while the shrubs feature vibrant yellow flowers. The concrete ground is scattered with a few fallen orange fruits and yellow petals. The overall scene maintains temporal continuity, with the background elements remaining consistent as the perspective shifts.

Constraints:
- Preserve the input text semantics and intent.
- Add plausible visual detail only to make the caption complete; do not introduce a new plot, new subjects, or extra events that extend the implied video before or after the input.
- If the input only describes the start of an action, describe only that start and do not invent the middle or end.
- If a reference image is attached, any added visual detail must be consistent with the visible image.
- Avoid strong speed words such as "fast", "rapid", "quick", "explosive", or "frantic" unless the input explicitly requires them.
- Do not refer to frames by number.
- Do not use JSON reference IDs such as `object_1`, `subject_1`, `actor_id`, or `entity_id`.
- The result should read like a dense video caption, not an instruction to create a video.
````
