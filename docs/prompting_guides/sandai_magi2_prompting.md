# MAGI-2 Preview prompting guide (Sand.ai)

Vendored snapshot, complete, with source links made absolute, of:
1. https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/README.md
2. https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/inference/prompt_enhancement/prompts/i2v.md
3. [Complete T2V prompt-enhancement system prompt](sandai_magi2_t2v_system_prompt.md) (https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/inference/prompt_enhancement/prompts/t2v.md)
4. https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/assets/sample_enhanced_t2v.json and `assets/sample_000.txt`

Fetched: 2026-11-15

## ComfyUI conditioning, flattening, and sampling contract

- **Text encoder (`comfy/text_encoders/magi2.py`)**: `Qwen3.5-27B` (`CLIPLoader` type `magi2`), hidden states taken three layers from the top (`layer="hidden", layer_idx=-3`, without the final RMSNorm). No chat template is applied (`MAX_TOKENS = 7000`), and every CJK character is isolated before Qwen pre-tokenization.
- **T2V structured JSON flattening (`json_to_compact_markdown`)**: When a prompt passed to `CLIPTextEncode` or `Magi2ImageToVideo` is valid JSON containing `global_layer` (the schema emitted by [sandai_magi2_t2v_system_prompt.md](sandai_magi2_t2v_system_prompt.md)), `Magi2Tokenizer` automatically strips empty fields and flattens `global_layer`, `dynamic_layer`, and `reference_layer` into the compact training-time text form (`context:`, `aesthetics:`, `audio:`, `subjects:`, `objects:`, `timeline:`). You can pass either the raw JSON object emitted by the T2V prompt enhancer or its pre-flattened text.
- **I2V structured Markdown & `<Figure 1>` reference (`comfy_extras/nodes/nodes_magi2.py`)**: SandAI's I2V enhancer (`i2v.md`, Section 2 below) emits a six-key JSON object (`global_caption`, `reference_bank`, `shot_timeline`, `audio_event_timeline`, `visible_text`, `generation_requirements`), which `inference/prompt_enhancement/pe_markdown.py` renders into readable Markdown sections (`# global_caption`, `# reference_bank`, `# shot_timeline`, `# audio_event_timeline`). In ComfyUI, `Magi2ImageToVideo` points the prompt at `<Figure 1>` by adding `"reference_layer": ["The first frame refers to <Figure 1>"]` to JSON prompts or appending `reference_layer:The first frame refers to <Figure 1>` to Markdown/plain-text prompts, encodes the reference image with the fp32 Wan 2.2 VAE (`magi2_wan2.2_vae_fp32.safetensors`), and splices the reference tokens at `<Figure 1>`.
- **Sampling (`nodes_magi2.py`)**:
  - **Preview (`Magi2`)**: `EmptyMagi2Latent` (896x512 for 540p or 448x256 for 272p, 10.0 seconds = 125 video frames at 12.5 fps + 25 Hz audio latents), `LTXV Dual CFG Guider` (`video_cfg = 5.0`, `audio_cfg = 7.0`), `KSamplerSelect` `"uni_pc_flow"`, and `BasicScheduler` `"flow_unipc"` for 100 steps.
  - **Refiner (`Magi2Refiner`)**: `Magi2RefinerLatent` upscales the preview video latent to 1080p (1920x1088 at 25 fps) and pairs with `CFGGuider` (`cfg = 2.0`), `"uni_pc_flow"`, and `"flow_unipc"` for 5 steps.
  - **Decoding**: `VAEDecode` with `magi2_turbo_vae.safetensors` (or `magi2_wan2.2_vae_fp32.safetensors`) for video and `VAEDecodeAudio` with `stable-audio-open-1.0.safetensors` for 44.1 kHz stereo audio.

---

# 1. README.md

![magi-logo](https://raw.githubusercontent.com/SandAI-org/MAGI-1/main/figures/logo_black.png)

---

<p align="center">
  <a href="https://sand.ai"><img alt="Sand.ai Homepage" src="https://img.shields.io/badge/Sand.ai-Homepage-333333.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMjI3IDIyNS4wODVDMjI3IDIwMi4zMDMgMjI3IDE5MC45MTIgMjMxLjQzNyAxODIuMjExQzIzNS4zMzkgMTc0LjU1NyAyNDEuNTY2IDE2OC4zMzQgMjQ5LjIyNiAxNjQuNDM0QzI1Ny45MzMgMTYwIDI2OS4zMzIgMTYwIDI5Mi4xMjkgMTYwSDUwNy44NzFDNTA5LjI5NSAxNjAgNTEwLjY3NiAxNjAgNTEyLjAxNCAxNjAuMDAxQzUzMi4wODIgMTYwLjAxNyA1NDIuNjExIDE2MC4yNzcgNTUwLjc3NCAxNjQuNDM0QzU1OC40MzQgMTY4LjMzNCA1NjQuNjYxIDE3NC41NTcgNTY4LjU2MyAxODIuMjExQzU3MyAxOTAuOTEyIDU3MyAyMDIuMzAzIDU3MyAyMjUuMDg1VjI1Ni41NThDNTczIDI5MS4zMTkgNTczIDMwOC43IDU2NS4wMzUgMzIzLjI3OUM1NTguNzU2IDMzNC43NzIgNTQzLjU2NSAzNDYuMTEgNTIzLjA3OCAzNTkuNjA1QzUxNC42NzQgMzY1LjE0MSA1MTAuNDcyIDM2Ny45MDkgNTA1LjYzOSAzNjcuOTM2QzUwMC44MDYgMzY3Ljk2NCA0OTYuNTAzIDM2NS4yIDQ4Ny44OTYgMzU5LjY3MUw0ODcuODk2IDM1OS42N0w0NjYuNDY5IDM0NS45MDVDNDU2Ljg3NSAzMzkuNzQyIDQ1Mi4wNzggMzM2LjY2IDQ1Mi4wNzggMzMyLjIxOEM0NTIuMDc4IDMyNy43NzcgNDU2Ljg3NSAzMjQuNjk1IDQ2Ni40NjkgMzE4LjUzMUw1MjYuNzgyIDI3OS43ODVDNTM1LjI5MSAyNzQuMzE5IDU0MC40MzUgMjY0LjkwMyA1NDAuNDM1IDI1NC43OTRDNTQwLjQzNSAyMzguMzg2IDUyNy4xMjUgMjI1LjA4NSA1MTAuNzA1IDIyNS4wODVIMjg5LjI5NUMyNzIuODc1IDIyNS4wODUgMjU5LjU2NSAyMzguMzg2IDI1OS41NjUgMjU0Ljc5NEMyNTkuNTY1IDI2NC45MDMgMjY0LjcwOSAyNzQuMzE5IDI3My4yMTggMjc5Ljc4NUw1MTMuMTggNDMzLjk0MUM1NDIuNDQxIDQ1Mi43MzggNTU3LjA3MSA0NjIuMTM3IDU2NS4wMzUgNDc2LjcxNkM1NzMgNDkxLjI5NCA1NzMgNTA4LjY3NSA1NzMgNTQzLjQzNlY1NzQuOTE1QzU3MyA1OTcuNjk3IDU3MyA2MDkuMDg4IDU2OC41NjMgNjE3Ljc4OUM1NjQuNjYxIDYyNS40NDQgNTU4LjQzNCA2MzEuNjY2IDU1MC43NzQgNjM1LjU2NkM1NDIuMDY3IDY0MCA1MzAuNjY4IDY0MCA1MDcuODcxIDY0MEgyOTIuMTI5QzI2OS4zMzIgNjQwIDI1Ny45MzMgNjQwIDI0OS4yMjYgNjM1LjU2NkMyNDEuNTY2IDYzMS42NjYgMjM1LjMzOSA2MjUuNDQ0IDIzMS40MzcgNjE3Ljc4OUMyMjcgNjA5LjA4OCAyMjcgNTk3LjY5NyAyMjcgNTc0LjkxNVY1NDMuNDM2QzIyNyA1MDguNjc1IDIyNyA0OTEuMjk0IDIzNC45NjUgNDc2LjcxNkMyNDEuMjQ0IDQ2NS4yMjIgMjU2LjQzMyA0NTMuODg2IDI3Ni45MTggNDQwLjM5MkMyODUuMzIyIDQzNC44NTYgMjg5LjUyNSA0MzIuMDg4IDI5NC4zNTcgNDMyLjA2QzI5OS4xOSA0MzIuMDMyIDMwMy40OTQgNDM0Ljc5NyAzMTIuMSA0NDAuMzI2TDMzMy41MjcgNDU0LjA5MUMzNDMuMTIyIDQ2MC4yNTQgMzQ3LjkxOSA0NjMuMzM2IDM0Ny45MTkgNDY3Ljc3OEMzNDcuOTE5IDQ3Mi4yMiAzNDMuMTIyIDQ3NS4zMDEgMzMzLjUyOCA0ODEuNDY1TDMzMy41MjcgNDgxLjQ2NUwyNzMuMjIgNTIwLjIwOEMyNjQuNzA5IDUyNS42NzUgMjU5LjU2NSA1MzUuMDkxIDI1OS41NjUgNTQ1LjIwMkMyNTkuNTY1IDU2MS42MTIgMjcyLjg3NyA1NzQuOTE1IDI4OS4yOTkgNTc0LjkxNUg1MTAuNzAxQzUyNy4xMjMgNTc0LjkxNSA1NDAuNDM1IDU2MS42MTIgNTQwLjQzNSA1NDUuMjAyQzU0MC40MzUgNTM1LjA5MSA1MzUuMjkxIDUyNS42NzUgNTI2Ljc4IDUyMC4yMDhMMjg2LjgyIDM2Ni4wNTNDMjU3LjU2IDM0Ny4yNTYgMjQyLjkyOSAzMzcuODU3IDIzNC45NjUgMzIzLjI3OUMyMjcgMzA4LjcgMjI3IDI5MS4zMTkgMjI3IDI1Ni41NThWMjI1LjA4NVoiIGZpbGw9IiNGRkZGRkYiLz4KPC9zdmc+Cg=="></a>
  <a href="https://sand.ai/blog/magi-2-preview"><img alt="Tech Blog" src="https://img.shields.io/badge/Tech%20Blog-MAGI--2%20Preview-333333"></a>
  <a href="https://huggingface.co/sand-ai/MAGI-2-preview"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MAGI--2%20Preview-ffc107"></a>
  <a href="https://x.com/SandAI_HQ"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Sand.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/LICENSE"><img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache%202.0-green?logo=apache"></a>
</p>

# MAGI-2 Preview

Magi-2 Preview is a 114B-parameter unified audio-video generation model that
activates just 6B parameters per token. Built on MagiMoE and co-designed across
architecture, systems, and data, it explores an efficient path to scaling video
generation. The architecture, the training system built around it, and the data
pipeline are described in [MAGI-2 Preview: Scaling Video Generation Models
Efficiently](https://sand.ai/blog/magi-2-preview); the weights are on Hugging
Face at [sand-ai/MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview).

This repository is the inference code. It generates video from a text prompt
(T2V) or from a prompt plus a still image (I2V), with sound generated alongside
the video and muxed into the output file. Clips are 10 seconds long, which is
the only duration the model currently supports. Generation runs in two stages:
`magi2_preview` denoises the clip at low resolution, and `magi2_refiner` takes
that result up to 1080p.

## Requirements

- NVIDIA Hopper GPUs, 8 of them.
- Python 3.12 and a recent CUDA toolkit.
- `ffmpeg` on `PATH`, to mux the audio track. Without it the video is still
  written, just silently.

## Setup

### Docker

The published image, [sandai/magi-2-preview](https://hub.docker.com/r/sandai/magi-2-preview),
already has the dependencies built, including the ones that need a compiler:

```bash
docker pull sandai/magi-2-preview:latest
docker run --gpus all -it -v /path/to/ckpt:/workspace/ckpt sandai/magi-2-preview:latest
```

There is a tag per commit as well, `sandai/magi-2-preview:<commit>`. Name that
one when reporting a result, because `latest` moves; the image also records what
it was built from in `/etc/magi2-build-info`.

Building it yourself is only necessary to change a dependency version, or to
work somewhere the registry is not reachable:

```bash
docker build -t magi-2-preview:local .
```

### From source

```bash
pip install -r requirements.txt
```

MAGI-2 also needs [MagiAttention](https://github.com/SandAI-org/MagiAttention)
and [MagiCompiler](https://github.com/SandAI-org/MagiCompiler). The pinned
revisions are recorded as build args in the [Dockerfile](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/Dockerfile).

## Checkpoints

Neither transformer has been step-distilled, so the denoising step count is
where most of the wall-clock time goes. That is the difference between the
release that exists today and the one that follows:

| Release | Denoising steps | Weights |
| --- | --- | --- |
| MAGI-2 Preview, base | 100 preview + 5 refiner | [sand-ai/MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview) |
| MAGI-2 Preview, distilled | far fewer | Coming soon |

Everything the base model loads lives in that one repository, roughly 307 GB in
total. Weights are not bundled with the code; download them into `ckpt/` in the
repository root, which is gitignored:

```bash
pip install huggingface_hub
hf download sand-ai/MAGI-2-preview --local-dir ckpt
```

The directory names in that repository are the ones the configs already expect,
so nothing needs renaming afterwards:

```
ckpt/
├── preview/                            # preview stage: 56 safetensors shards + index
├── refiner/                            # refiner stage: 3 shards + index
├── text_encoder/                       # text encoder
├── vae/                                # video VAE
│   └── Wan2.2_VAE.pth
├── stable-audio-open-1.0/              # audio VAE
└── turbo_vae/                          # fast VAE decoder
    ├── TurboV3-Wan22-TinyShallow_7_7.json
    └── checkpoint.ckpt
```

Each directory name below links to the folder it comes from in the Hugging Face
repository:

| Directory | Size | Contents |
| --- | --- | --- |
| [`preview`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/preview) | 228 GB | Preview-stage transformer, released with MAGI-2 |
| [`text_encoder`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/text_encoder) | 56 GB | Text encoder, [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| [`refiner`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/refiner) | 14 GB | Refiner-stage transformer, released with MAGI-2 |
| [`stable-audio-open-1.0`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/stable-audio-open-1.0) | 5 GB | Audio VAE, decodes the generated audio latents |
| [`vae`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/vae) | 3 GB | Video VAE, from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| [`turbo_vae`](https://huggingface.co/sand-ai/MAGI-2-preview/tree/main/turbo_vae) | 2 GB | Distilled VAE decoder, used for decoding by default |

The configs under [`configs/`](https://github.com/SandAI-org/MAGI-2-preview/tree/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/configs) reference these as
`${MAGI2_CKPT_ROOT}/<name>`, and that variable defaults to `<repo>/ckpt`. To keep
weights somewhere else, point it at them rather than editing the configs:

```bash
export MAGI2_CKPT_ROOT=/data/magi2-weights
```

## Prompt enhancement

The captions the model was trained on are long and structured, so a short hand-
written prompt underuses it. The pipeline can optionally rewrite the input
through a prompt-enhancement (PE) step before encoding:

- text-to-video uses [`inference/prompt_enhancement/prompts/t2v.md`](sandai_magi2_t2v_system_prompt.md)
- image-to-video (when a first-frame image is provided) uses
  [`inference/prompt_enhancement/prompts/i2v.md`](#2-image-to-video-system-prompt-inferenceprompt_enhancementpromptsi2vmd)

Both templates ask an instruction-following LLM for a structured JSON caption of
the 10-second clip; the result is then rendered to readable Markdown and passed
downstream. The reference client in
[`inference/prompt_enhancement/`](https://github.com/SandAI-org/MAGI-2-preview/tree/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/inference/prompt_enhancement) talks to an
OpenAI-compatible endpoint — set `API_KEY` in
[`enhancer.py`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/inference/prompt_enhancement/enhancer.py) to enable it, or
subclass `LLMClient` for your own provider. Leave `API_KEY` empty to skip PE and
use the raw prompt as-is.

[`assets/`](https://github.com/SandAI-org/MAGI-2-preview/tree/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/assets) ships example prompts that are already PE-enhanced
(`sample_000.txt` and `sample_002.txt`). You can still generate without running
PE — leave `API_KEY` empty and pass any prompt you like.

## Running inference

[`scripts/run_demo.sh`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/scripts/run_demo.sh) launches
[`inference/pipeline/entry.py`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/inference/pipeline/entry.py) under `torchrun` on
every visible GPU. It generates at 1080p, with seed 42 and the batch in
[`assets/demo_samples.json`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/assets/demo_samples.json):

```bash
bash scripts/run_demo.sh
SAMPLES=my_samples.json bash scripts/run_demo.sh   # a different batch
OUTPUT_DIR=output/run7 bash scripts/run_demo.sh
```

The script takes `SAMPLES`, `OUTPUT_DIR`, `SEED` and `MASTER_PORT` from the
environment. Videos land in `$OUTPUT_DIR/sample_000.mp4` and up, numbered by
position in the batch.

A samples file is a JSON array with one entry per video. An entry carries its
prompt inline as `prompt` or as a path in `prompt_file`, and a first frame in
`image`; leaving `image` out makes it a T2V entry. The shipped batch runs the
three stills in `assets/` as I2V, the same three prompts again as T2V, and
`assets/sample_enhanced_t2v.json`.

For a single clip, call the entry point directly:

```bash
torchrun --nproc_per_node=8 inference/pipeline/entry.py \
    --prompt "a red fox in snow" --output output/
```

It also takes `--prompt-file`, `--image`, `--seed`, `--config`, `--output-width`
/ `--output-height`, `--num-inference-steps`, `--refiner-num-inference-steps`
and `--deterministic`. Of these only `--seed`, `--samples` and `--output` are
reachable through `run_demo.sh`.

1080p runs [`configs/magi2_refiner.json`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/configs/magi2_refiner.json): the
preview stage generates 512x896 and the refiner takes that to 1088x1920.
`magi2_refiner.json` extends
[`magi2_preview.json`](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/configs/magi2_preview.json) and carries only what the
refiner stage adds, so a shared setting is edited in one place.

1080p is a delivery tier, not the shape that gets generated. The VAE stride
constrains every generated dimension to a multiple of 16, so the tier generates
1088 wide rather than 1080, and the video is written at that generated shape.
Pass `--output-width` and `--output-height` to have the finished video rescaled
to an exact size, 1080x1920 the way the reference delivers the tier.

Four environment variables decide where each large component sits between
phases: `MAGI2_TEXT_ENC_OFFLOAD_MODE`, `MAGI2_PREVIEW_OFFLOAD_MODE`,
`MAGI2_REFINER_OFFLOAD_MODE` and `MAGI2_VAE_OFFLOAD_MODE`, each one of `cpu`,
`gpu` or `roundtrip`. The preview and the refiner default to `roundtrip`, staged
in and out around the stage that needs them, because at 1080p neither fits on an
80GB card next to the other's activations.

Decoding uses the distilled turbo decoder from `ckpt/turbo_vae`, a temporal
sliding window that runs on one rank per video. `MAGI2_DETERMINISTIC=1`, or
`--deterministic`, makes the MoE scatter and the attention kernels bit-exact at
some cost in speed. `MAGI2_SAVE_LATENT_PATH` writes the post-refiner latent of
each sample to that directory.

## License

Apache 2.0. See [LICENSE](https://github.com/SandAI-org/MAGI-2-preview/blob/f68a0f9bbccbea56e0177a9bd912abc3b18ffe61/LICENSE).

---

# 2. Image-to-Video System Prompt (`inference/prompt_enhancement/prompts/i2v.md`)

Complete template used by `inference/prompt_enhancement/enhancer.py` (`I2V_TASK`, response schema `I2VEnhanceResult`). Substitute `{user_input_text}` with the user prompt and `{variation_seed}` with `""`, attach the first-frame image as an `image_url` part in the user message, and render the returned JSON object to Markdown with `pe_markdown.render_markdown` (or follow the rendered Markdown structure in Section 3b below):

````text
You are an Audio Visual Director Script Caption Compiler for an Image-to-Video (I2V) generation model.
Your task is to analyze the provided input image and the user's free-form text prompt, enhance user prompt to be more specific and detailed, to output a professional, structured, generation-ready video caption. 
Since the starting visual state is already established by the image, your primary role is to **preserve the image's content** and **drastically enhance the motion, temporal dynamics, physics, and camera instructions** that the user left out.
The response must be a single valid JSON object and must follow the exact output schema below.

========================
Required Output Schema
========================
Return exactly this JSON structure:
{
"global_caption": {
    "core_caption": "",
    "format_and_structure": "",
    "narrative_and_emotional_design": "",
    "visual_design": "",
    "audio_design": "",
    "continuity_requirements": []
},
"reference_bank": [],
"shot_timeline": [],
"audio_event_timeline": [],
"visible_text": [],
"generation_requirements": []
}
Use valid JSON syntax.
Do not output markdown.
Do not output explanations outside the JSON.
Use empty arrays only when a section truly has no applicable content.

========================
Input Variables
========================
User input:
{user_input_text}

Optional variation seed:
{variation_seed}

========================
Language and Caption Voice
========================
Write all descriptive content in clear, simple English using direct, objective director-style language. For Chinese user input, translate descriptive content into English while keeping quoted speech and visible UI/signage text in the original language. Focus heavily on verbs and temporal shifts.
Good: "The camera pushes in slowly from the initial wide frame as the woman’s coat billows in the newly formed wind."
Bad: "The image will start to move and the woman should walk."

========================
I2V Core Principles
========================
1. The Image is the Absolute Baseline.
You must lock in the subjects, lighting, composition, style, and setting of the input image for the beginning of the timeline. The prompt cannot alter the past; it can only dictate the future of the frame.        
2. Hyper-Focus on Motion Enrichment.
Because the visual aesthetic is pre-defined, your "Director Fill" must focus on physical motion. When the user leaves motion unspecified, enrich it with:
- Precise joint articulation and kinematic realism for human/animal subjects to prevent limb distortion.
- Fluid dynamics for water, smoke, or weather.
- Camera trajectories (pan, tilt, track, crane, focal pull) that logically extend from the image's initial 2D composition into 3D space.
- Cause-and-effect physics (e.g., footsteps displacing dust, weight-bearing shifts).
3. Aesthetic Continuity.
Do not invent new lighting or styles unless bridging a user-requested transition. Analyze the image's contrast, color palette, and exposure, and explicitly instruct the model to maintain those exact parameters throughout the video.
4. Relational Grounding.
Important characters, objects, and scenes extracted from the image must be cataloged in the `reference_bank` so their identity remains strictly stable across the generated motion.

========================
Default Duration Rule
========================
The default duration is exactly 10.0 seconds. 
When the user specifies a duration, map it to 10 seconds. Otherwise, format the `shot_timeline` and `audio_event_timeline` to explicitly map a 10.0-second chronological evolution. I2V requires smooth temporal continuity, so default to single-shot continuous takes `[0.0, 10.0]` unless the user text implies a montage.

========================
Motion Director Fill Intensity
========================
1. Light enrichment:
Use when the user provides highly specific camera, physics, and action instructions. Add only stability requirements and audio-visual synchronization.
2. Medium enrichment:
Use when the user provides an action but no camera or physics details. Lock the camera movement to track the action, add environmental interactions (wind, gravity, fabric movement), and define a clear start, middle, and end state.
3. Strong enrichment (Static Image Awakening):
Use when the user input is abstract, generic, or just says "animate". You must invent a logically sound trajectory for the scene. Add subtle parallax camera movement, ambient environmental motion (e.g., breathing, leaves rustling, water rippling), and one distinct physical turning beat.

========================
Critical Decision Policy: 
Analyze, Filter, and Enhance
========================
Before writing the JSON, internally execute:
Step 1: Extract Ground Truth from the Image.
Identify the exact subject identity, spatial layout, lighting direction, and visual texture of the starting frame.
Step 2: Enhance User Text.
Compare user text to image ground truth. Enhance the user text to be more specific and detailed.
Step 3: Inject Category-Specific Motion (18 Categories).
Select the best primary content category below. Instead of pulling static aesthetic details, pull the category's specific **Motion, Physics, and Camera** defaults to animate the image. 
Step 4: Ensure Structural Rigidity.
The most common failure in I2V is structural collapse during high-speed movement. Explicitly describe how volume, anatomy, and physical boundaries are maintained during the motion you script.

========================
18 Content Categories: Motion and Physics Bank
========================
(Apply these physical rules to bring the static image to life based on its detected category)
1. Natural Landscapes and Weather
Camera: Slow drone push-in, lateral parallax shift to reveal 3D depth.
Motion: Clouds drifting, water surface tension rippling, wind swaying foreground foliage. Maintain stable horizon lines. 

2. Objects, Materials, and Micro-Motion
Camera: Probe-lens glide or slow orbit.
Motion: High-fidelity material response. Fluid viscosity, steam rising, fabric settling, or precise light refraction changing as the camera moves.

3. Animals, Wildlife, and Pets
Camera: Stabilized tracking, keeping the eyes on the focal plane.
Motion: Anatomically correct quadruped/bipedal gait, fur/feather movement reacting to wind and inertia, micro-expressions in the ears and eyes. Prevent structural limb morphing.

4. Sports, Fitness, and Outdoor Activity
Camera: High-speed gimbal follow, matching the velocity of the subject.
Motion: Weight transfer, muscle tension, ground-impact physics (dust, sweat, chalk). Fast kinematic motion must strictly maintain joint locking and volume.

5. Everyday Human Life and Social Moments
Camera: Soft handheld breathing, slow push-in for emotional intimacy.
Motion: Natural body language, chest rising with breath, subtle gaze shifting, secondary clothing movement.

6. Science Education and Physical Phenomena
Camera: Locked-off or smooth top-down track.
Motion: Strict adherence to real-world physics (diffusion, gravity, magnetism, thermodynamics). Left-to-right progression of chemical or physical states.

7. Art, Performance, and Cultural Events
Camera: Rhythmic lateral tracking or sweeping crane movement.
Motion: Graceful, choreographed human articulation. Cloth flowing with momentum, precise instrument interaction.

8. Food, Beverage, and Cooking
Camera: 45-degree tabletop macro push-in or slow pan.
Motion: Heat-based physics (bubbling, steam, melting), slow-motion pours with correct liquid viscosity, knife impact.

9. City Travel, Landmarks, and Architecture
Camera: Smooth hyperlapse or slow tilt to establish verticality.
Motion: Traffic flow, pedestrian crowds moving with independent trajectories, shifting reflections on glass as the camera translates.

10. Vehicles and Transportation
Camera: Rolling tracking shot or vehicle-mounted POV.
Motion: Wheel rotation matching ground speed, suspension reacting to terrain, aerodynamic drag on smoke or rain, accurate mechanical articulation.

11. Space Exploration and Astronomy
Camera: Extremely slow, frictionless orbital drift.
Motion: Zero-gravity physics, slow rotation of celestial bodies, lack of atmospheric distortion, harsh shadow shifts as objects rotate relative to the light source.

12. Fantasy, Mythology, and Supernatural
Camera: Dreamlike floating dolly or slow portal push-through.
Motion: Volumetric glowing particles, unnatural but internally consistent gravity (e.g., floating rocks), cloaks reacting to magical updrafts. 

13. Industrial Manufacturing, Construction
Camera: Mechanical linear tracking.
Motion: Rigid body dynamics, heavy impacts, robotic arm articulation with locked joints, sparks falling with gravity and bouncing on hard surfaces.

14. Technology Products and Digital Interfaces
Camera: Motion-control robotic orbit or macro edge slide.
Motion: Controlled screen UI animations, slow illumination of hardware LEDs, perfect rigid-body stability (no warping of straight metal edges).

15. Science Fiction and Cyberpunk
Camera: Rain-night street tracking, navigating through layered depth.
Motion: Flickering neon logic, steam venting upward, flying vehicles maintaining stable trajectories, cybernetic joint movement.

16. Abstract Graphics, Data, and Text
Camera: Isometric glide or zoom through layers.
Motion: Smooth easing (ease-in/ease-out), grid-locked transitions, particles following mathematical attractors. No organic morphing unless specified.

17. Medical, Health, and Life Sciences
Camera: Smooth 3D fly-through or stable clinical observation.
Motion: Anatomically strict biological processes (blood flow, cellular division, chest breathing). Zero fleshy distortion of tools or instruments.

18. Public Safety, Military, and Emergency
Camera: Documentary handheld follow with sudden responsive pans.
Motion: Urgent kinematic movement, chaotic but physically grounded debris fall, smoke reacting to wind and heat thermodynamics.

========================
Dimension-Level I2V Fill Rules
========================
1. Camera Movement: The camera must logically start from the exact framing of the input image. If the image is a macro shot, you cannot immediately transition to an extreme wide shot without a continuous pull-back.
2. Lighting & Color: Write explicit instructions to maintain the image's initial lighting setup. If a time-lapse or environmental change is triggered by the text, describe the transition of the light source.
3. Motion and Physical Logic (CRITICAL): You must define the start, process, and end states of the movement. Translate the user's verbs into precise physical instructions (e.g., instead of "he dances", use "he shifts his weight, articulating his knees, while his arms swing chronologically").
4. Visible Text: If the input image contains readable text/logos, add a continuity requirement to lock the text shape. Do not invent new UI or signs unless requested.

========================
Global Caption & Reference Bank Rules
========================
`global_caption.core_caption`: One sentence combining the image's starting state with the user's requested motion.
`global_caption.visual_design`: Describe the aesthetic of the *input image* so the model knows what to maintain.
`global_caption.continuity_requirements`: Must include rules about maintaining the image's exact character identities, background architecture, and preventing structural morphing/limb distortion during movement.
`reference_bank`: Define the image's subjects here. Under `appearance_or_design`, describe them exactly as they look in the starting state. 

========================
Final Output Checklist
========================
Before returning, verify:
- The JSON is valid and matches the exact schema requested.
- The prompt logic treats the image as unalterable past and the text as the trajectory for the future.
- The shot timeline includes heavy emphasis on physical physics, camera dynamics, and preventing distortion.
- The default duration is 10.0 seconds.
- The text outputs direct, professional script language.
````

---

# 3. Official Enhanced Prompt Examples

## 3a. T2V Enhanced JSON (`assets/sample_enhanced_t2v.json`)

````json
{
  "global_layer": {
    "context": "A single continuous take inside a small corner cafe on a rainy afternoon.",
    "description": "A barista lowers a cup onto a saucer and speaks one line to the customer while the camera pushes in.",
    "aesthetics": {
      "style": "naturalistic documentary cinematography",
      "contrast": "low to moderate with open shadow detail",
      "saturation": "restrained true-to-life saturation",
      "color_scheme": "warm amber interior against cool grey window light",
      "mood_atmosphere": "calm and unhurried"
    },
    "audio_baseline": {
      "ambience": "steady rain against glass, low espresso machine hum, occasional cup clatter",
      "dialogue": {
        "language": "English",
        "speaker_tags": ["<subject 1>"]
      }
    },
    "objects_static": [
      {
        "object_id": "<object 1>",
        "description": "a white ceramic cup resting on a matching saucer",
        "shape_and_color": "rounded white cup holding a pale crema surface",
        "texture": "smooth glazed ceramic",
        "relative_size": "small, about one third the width of the barista's hand",
        "position": "on the wooden counter in the lower right third of frame",
        "orientation": "handle turned toward the customer"
      }
    ],
    "video_metadata": {
      "duration": 10.0,
      "aspect_ratio": "16:9"
    },
    "lighting_baseline": {
      "conditions": "soft overcast daylight through a large window with warm pendant fill",
      "direction": "key from the window at camera left, warm fill from overhead pendants",
      "shadows": "soft edged with retained detail",
      "source_consistency": "window and pendant sources hold the same intensity and color for the full clip"
    },
    "environment_baseline": {
      "background_setting": "a small cafe interior with a wooden counter, shelves of stacked cups, and a rain streaked window behind"
    },
    "alive_subjects_static": [
      {
        "subject_id": "<subject 1>",
        "description": "a barista working behind the counter",
        "position": "behind the counter at frame center",
        "orientation": "facing camera, angled slightly to frame right",
        "visual_attributes": {
          "gender": "female",
          "ethnicity": "East Asian",
          "age_group": "late twenties",
          "facial_features": "round face, dark eyes, hair tied back in a low bun",
          "clothing": "dark green apron over a white shirt with rolled sleeves",
          "appearance_details": "a thin silver ring on the right hand"
        }
      }
    ],
    "camera_cinematography": {
      "overall_camera_style": "single continuous slow push-in on a stabilized head",
      "default_depth_of_field": "shallow, the barista sharp and the shelves softly separated",
      "lens_focal_length": "50mm"
    }
  },
  "dynamic_layer": {
    "timeline_segments": [
      {
        "segment_basic_info": {
          "timestamp_range": "00:00.0-00:05.0",
          "segment_description": "The barista lowers the cup onto the saucer with both hands and releases it, and the camera begins a slow push-in.",
          "active_background": "rain runs down the window behind her in continuous vertical streaks",
          "camera_delta": {
            "focus": "held on the barista's hands and face",
            "camera_position": "chest height, one meter back from the counter edge",
            "composition": "medium shot, barista at center, cup in the lower right third",
            "camera_movement": {
              "type": "push-in",
              "direction": "forward along the lens axis",
              "intensity": "slow and even"
            }
          }
        },
        "objects": [
          {
            "object_id": "<object 1>",
            "timestamp": "00:00.0-00:05.0",
            "number_of_objects": 1,
            "dynamic_state": {
              "state_change": "the cup travels from her hands down onto the saucer and comes to rest level and full",
              "motion_detail": "a short controlled descent that settles without a wobble, the crema surface swaying once and flattening"
            },
            "spatial_position": {
              "location": "on the counter in the lower right third",
              "relative_size": "small",
              "orientation": "handle turned toward the customer"
            }
          }
        ],
        "alive_subjects": [
          {
            "subject_id": "<subject 1>",
            "timestamp": "00:00.0-00:05.0",
            "number_of_subjects": 1,
            "spatial_position": {
              "location": "behind the counter at frame center",
              "relative_size": "upper body fills the central third",
              "orientation": "facing camera, shoulders square to the counter"
            },
            "action": {
              "primary_action": "setting the cup down and releasing it",
              "body_configuration": "both elbows tucked close, forearms lowering together, spine upright",
              "motion_detail": "fingers open away from the cup one at a time, hands withdraw to the counter edge and rest there",
              "interaction": "both hands hold the saucer rim, then leave it fully supported on the counter",
              "facial_expression": "eyes down on the cup, mouth relaxed"
            }
          }
        ],
        "text_render": [],
        "causal_events": [
          {
            "event_id": "event 1",
            "timestamp_range": "00:02.4-00:04.0",
            "trigger": "the barista releases the saucer",
            "effect_outcome": "the cup stays level on the counter and a thin ribbon of steam rises straight from the surface",
            "physics": "the saucer takes the full weight of the cup and the liquid surface returns to level"
          }
        ],
        "audio": {
          "dialogue_lines": [],
          "ambience_deltas": [
            {
              "timestamp_range": "00:00.0-00:05.0",
              "ambience": "rain and machine hum hold steady under a soft ceramic contact at the moment of placement"
            }
          ],
          "special_audio_events": [
            {
              "timestamp": "00:02.4",
              "sound_description": "a short soft ceramic tap as the cup meets the saucer",
              "visual_sync": "matches the frame where the cup base settles"
            }
          ]
        }
      },
      {
        "segment_basic_info": {
          "timestamp_range": "00:05.0-00:10.0",
          "segment_description": "The barista lifts her gaze to the customer, speaks one line, and holds a small closing smile as the push-in settles.",
          "active_background": "rain continues down the window at the same rate",
          "camera_delta": {
            "focus": "shifts from the cup to the barista's eyes and holds",
            "camera_position": "chest height, now half a meter from the counter edge",
            "composition": "medium close-up, barista at center, cup still visible in the lower right corner",
            "camera_movement": {
              "type": "push-in",
              "direction": "forward along the lens axis, decelerating to a stop",
              "intensity": "slowing to rest by the final second"
            }
          }
        },
        "objects": [
          {
            "object_id": "<object 1>",
            "timestamp": "00:05.0-00:10.0",
            "number_of_objects": 1,
            "dynamic_state": {
              "state_change": "the cup remains at rest on the saucer, still full",
              "motion_detail": "only the steam moves, drifting upward and thinning"
            },
            "spatial_position": {
              "location": "on the counter in the lower right corner",
              "relative_size": "small",
              "orientation": "handle turned toward the customer"
            }
          }
        ],
        "alive_subjects": [
          {
            "subject_id": "<subject 1>",
            "timestamp": "00:05.0-00:10.0",
            "number_of_subjects": 1,
            "spatial_position": {
              "location": "behind the counter at frame center",
              "relative_size": "head and shoulders fill the central half",
              "orientation": "facing camera directly"
            },
            "action": {
              "primary_action": "looking up and speaking one line, then holding still",
              "body_configuration": "chin lifts, shoulders stay level, hands rest on the counter edge",
              "motion_detail": "gaze rises to camera, lips articulate the line, then the mouth closes into a small smile that holds",
              "interaction": "addresses the customer standing at the counter off camera",
              "facial_expression": "attentive, warming into a brief smile"
            }
          }
        ],
        "text_render": [],
        "causal_events": [],
        "audio": {
          "dialogue_lines": [
            {
              "timestamp": "00:05.6-00:07.4",
              "speaker": "<subject 1>",
              "text": "Careful, it's still very hot.",
              "delivery": "warm and low, unhurried, close to the microphone"
            }
          ],
          "ambience_deltas": [
            {
              "timestamp_range": "00:07.4-00:10.0",
              "ambience": "the machine hum drops away and the rain remains alone under the final hold"
            }
          ],
          "special_audio_events": []
        }
      }
    ]
  }
}
````

## 3b. I2V Enhanced Markdown (`assets/sample_000.txt`)

````markdown
# global_caption

core_caption:A 10-second continuous animated sequence following a young girl from a dark, mist-filled forest to a sudden, bright clearing overlooking a vast ocean, shifting the emotional tone from mysterious tension to liberated relief.
format_and_structure:10-second Japanese anime-style narrative short. Features a linear chronological progression with a smooth environmental transition and one deliberate camera cut near the climax to emphasize emotional release.
narrative_and_emotional_design:The emotional arc is entirely visual and atmospheric, progressing from solitary, quiet movement in a confined, oppressive space to overwhelming awe, ending in peaceful relief in an expansive, bright environment.
visual_design:2D Japanese anime film aesthetic (8K high-definition finish). Begins with deep, desaturated greens, low-key lighting, wet textures, and volumetric mist. Progresses smoothly to bright, high-contrast cyan, brilliant sunlight, and a soft, hazy atmospheric perspective.
audio_design:Completely ambient and naturalistic. Driven by diegetic sounds: crunching footsteps on wet stones, rustling leaves, transitioning to swelling ocean waves, strong sea wind, and a subtle human breath. No dialogue or background music.

continuity_requirements:Strictly maintain the 2D Japanese anime aesthetic; do not shift to 3D CGI or photorealistic live-action styles. | Lock the character's identity and outfit: dark hair, light-colored coat, dark skirt, brown boots, and simple canvas backpack. | Ensure precise volume retention and anatomical consistency of the character's limbs during walking motions; prevent structural morphing. | Maintain spatial continuity of the background; do not let trees or ocean waves pop or warp into existence.

# reference_bank

## reference_bank1

reference_identifier:ref_girl
reference_type:character
label:Young Japanese Girl
role:Primary protagonist traversing the environment
description:A young female character in a 2D anime style, seen initially from behind, carrying a backpack.
appearance_or_design:Shoulder-length dark brown/black hair, wearing a loose light-colored coat, a dark mid-length skirt, brown boots, and a simple light-colored canvas backpack.

## reference_bank2

reference_identifier:ref_forest
reference_type:setting
label:Deep Forest Path
role:Starting environment providing a confined, mysterious atmosphere
description:A dense, dark woodland with a rocky, uneven path.
appearance_or_design:Towering pine and cedar trees, deep green hues, moss-covered rocks, wet stones reflecting faint light, scattered fallen leaves, and volumetric ground mist.

## reference_bank3

reference_identifier:ref_ocean
reference_type:setting
label:Vast Ocean
role:Climactic environment providing visual and emotional release
description:An expansive, bright coastal view suddenly revealed at the edge of the forest.
appearance_or_design:Bright blue water stretching to a distant horizon, sparkling surface reflections from intense sunlight, white crashing waves, and a clear blue sky.

# shot_timeline

## shot_timeline1

shot_identifier:shot_01_forest_to_ocean_reveal

time_range_seconds:0.0 | 8.0
scene_reference_identifier:ref_forest
shot_caption:Starting from the exact framing of the input image, the camera steadily follows the girl from behind as she walks down the damp, rocky path. As she progresses, the canopy thins and the dark green environment gradually brightens, revealing patches of blue sky. The wind picks up, blowing her hair and coat backwards. She slows her pace and raises her arm to push aside a dense layer of foreground branches. As the foliage parts, the camera pushes smoothly past her shoulder to reveal a vast, sparkling blue ocean. The sudden influx of intense sunlight heavily backlights her silhouette, and her body language freezes in awe.
camera_lighting_and_composition:Smooth, stabilized follow-camera from a mid-level behind the subject, seamlessly pushing forward. Lighting begins as low-key volumetric lighting (dark greens, cyan shadows) and dynamically transitions into high-key, blooming sunlight from the front-center as the ocean is revealed, creating strong spatial depth.
motion_and_physical_logic:Articulated bipedal walking gait with realistic weight transfer on the wet, uneven stones; her coat and skirt hem sway naturally with her momentum. From 3.0s, frontal wind introduces secondary motion to her hair and clothing. At 5.0s, strict arm kinematics are used to push aside rigid foreground foliage, preventing hand distortion. The ocean surface exhibits high-fidelity fluid dynamics with rolling waves.

reference_identifiers:ref_girl | ref_forest | ref_ocean

audio_event_identifiers:audio_footsteps | audio_wind_buildup | audio_branch_sweep | audio_ocean_waves

## shot_timeline2

shot_identifier:shot_02_profile_relief

time_range_seconds:8.0 | 10.0
scene_reference_identifier:ref_ocean
shot_caption:Cut to a 2/3 front-profile medium close-up of the girl. Bright sunlight illuminates her face. She closes her eyes, takes a deep breath, and her shoulders visibly drop as a subtle, soft smile forms on her lips. Her hair and collar flutter wildly yet beautifully in the strong sea breeze. The camera slowly pans sideways to the expansive ocean, leaving her face and filling the frame with the bright sky and gentle waves.
camera_lighting_and_composition:Medium close-up shifting to a slow lateral pan. Soft, diffused sunlight wraps around the subject's face, casting natural anime-style cel shading on her neck. High contrast between the foreground subject and the bright, airy ocean background.
motion_and_physical_logic:Facial micro-kinematics: eyelids closing smoothly, chest expanding steadily with an inhale, and shoulders dropping under gravity to signify relaxation. High-frequency fluid dynamics applied to hair strands and fabric reacting continuously to a sustained directional wind.

reference_identifiers:ref_girl | ref_ocean

audio_event_identifiers:audio_ocean_waves | audio_deep_breath | audio_seagulls

# audio_event_timeline

## audio_event_timeline1

audio_event_identifier:audio_footsteps
event_type:foley

time_range_seconds:0.0 | 5.0
source_reference_identifier:ref_girl
audio_caption:Rhythmic, slightly hurried crunching footsteps on wet stones and damp leaves, accompanied by subtle fabric rustling.
synchronization_note:Sync perfectly with the girl's walking cadence; fade out as her pace slows.

## audio_event_timeline2

audio_event_identifier:audio_wind_buildup
event_type:ambient

time_range_seconds:3.0 | 8.0
source_reference_identifier:ref_forest
audio_caption:A rising, whistling wind rustling through dense pine needles, growing progressively louder.
synchronization_note:Volume increases linearly alongside the visual brightening of the scene.

## audio_event_timeline3

audio_event_identifier:audio_branch_sweep
event_type:foley

time_range_seconds:5.0 | 6.5
source_reference_identifier:ref_girl
audio_caption:The distinct, sharp brushing and snapping sound of stiff leaves and branches being pushed aside.
synchronization_note:Sync precisely with the girl's hand contacting and moving the foreground foliage.

## audio_event_timeline4

audio_event_identifier:audio_ocean_waves
event_type:ambient

time_range_seconds:4.0 | 10.0
source_reference_identifier:ref_ocean
audio_caption:Distant muffled waves that suddenly swell into a loud, expansive, and rhythmic crashing of sea surf.
synchronization_note:Low-pass filter off and dramatic volume spike exactly at 6.0s when the ocean visually dominates the frame.

## audio_event_timeline5

audio_event_identifier:audio_deep_breath
event_type:foley

time_range_seconds:8.0 | 9.5
source_reference_identifier:ref_girl
audio_caption:A soft, deliberate human inhale and quiet, relieved exhale.
synchronization_note:Sync with the expansion of the character's chest and dropping shoulders in the close-up.

## audio_event_timeline6

audio_event_identifier:audio_seagulls
event_type:ambient

time_range_seconds:8.5 | 10.0
source_reference_identifier:ref_ocean
audio_caption:The faint, echoing cries of seagulls soaring over the water.
synchronization_note:Introduce seamlessly during the camera pan away from the girl.

generation_requirements:Lock the starting frame perfectly to the provided input image, preserving the exact lighting, framing, and character posture. | Do not use cheap transitions such as flash-to-white, sudden crossfades, or teleportation; the transition from forest to ocean must be achieved spatially through character movement and camera push-in. | Ensure zero generation of modern elements: no cars, buildings, smartphones, or text. | Strictly enforce 2D anime physics; avoid realistic fleshy textures or 3D video game rendering styles. | Maintain strict structural rigidity of the character's face and limbs during the cut at 8.0s; do not allow facial features to warp or slide.
````
