# PixelDiT prompting (NVIDIA)

Fetched 2026-10-19 from:
1. https://github.com/NVlabs/PixelDiT/blob/f7d7c35/t2i/README.md, its `t2i/inference.py`,
   `t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml` and `t2i/prompts.txt`
2. https://arxiv.org/abs/2511.20645 (PixelDiT: Pixel Diffusion Transformers for Image Generation, CVPR 2026)
3. https://arxiv.org/abs/2410.10629 (Sana, whose text-encoder and caption recipe PixelDiT's T2I configs follow)
4. https://huggingface.co/Comfy-Org/PixelDiT and the `image_pixeldit_t2i` workflow template

NVIDIA publishes no separate prompting guide for PixelDiT. What determines how it reads a prompt is the text encoder
setup, the system prompt it was trained behind, and the captions it was trained on, all below.

Reviewed: 2026-10-29. The moving `main/t2i/README.md` URL now returns 404;
the pinned source above remains available. This recipe describes PixelDiT-T2I
v1, not PixelDiT2 or the separate [PiD decoder](https://github.com/nv-tlabs/PiD).

## Model

- `nvidia/PixelDiT-1300M-1024px` (`pixeldit_t2i_v1.pth`): 1.3B parameters, pixel space (no VAE), 16 px patches,
  14 MM-DiT patch blocks and 2 pixel (PiT) blocks, trained at 512 px (stages 1 and 2, multi-aspect from stage 2) and
  1024 px (stage 3, multi-aspect). ComfyUI: `Comfy-Org/PixelDiT` `pixeldit_1300m_1024px_bf16.safetensors` (or mxfp8).
- License: NVIDIA Source Code License NSCLv1, non-commercial use only ("for research or evaluation purposes only").
- Text encoder: Gemma-2-2B-IT (`Efficient-Large-Model/gemma-2-2b-it`; ComfyUI `gemma_2_2b_it_elm_bf16.safetensors`,
  CLIPLoader type `pixeldit`), y_norm, 300 text rows.

## The system prompt (CHI) and the 300-row window

Every prompt is encoded behind Sana's Complex Human Instruction, the stage-3 config's `chi_prompt` joined with
newlines (ComfyUI prepends the same text in `comfy/text_encoders/pixeldit.py`):

```
Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt: 
```

The tokenizer reads `chi_prompt + prompt` padded and truncated to `len(chi) + 300 - 2` tokens (506), and the model
conditions on BOS plus the last 299 positions of that window (`select_index = [0] + range(-299, 0)`). So:

- A prompt keeps at most 299 tokens after the system prompt (its trailing space token merges into the prompt's first
  word); anything longer is silently cut at the end. Short prompts leave the tail of the system prompt inside the
  window, as in training.
- The negative prompt is encoded without the system prompt, padded to 300 tokens (`inference.py`). ComfyUI's
  tokenizer prepends the system prompt to any non-empty text, the negative included; an empty negative is encoded
  alone, as in the official code.

## The training captions

The stage-3 data config draws each image's caption from its original prompt and two VLM captions,
`external_caption_suffixes: ['', _InternVL2-26B, _VILA1-5-13B]`, weighted by their CLIP scores
(`clip_thr: 25.0`, `clip_thr_temperature: 0.1`: a caption below a CLIP score of 25 is dropped and the sampling is
sharp, nearly always the best-scoring one). Sana describes the same scheme: several VLMs caption each image
("Multiple VLMs can make the caption more accurate and more diverse") and a caption is sampled with
P(c_i) = exp(c_i / tau) / sum_j exp(c_j / tau). The trained register is therefore dense, literal English description
of what is visible, as long-form VLM captions write it, with the visual specifics the system prompt asks for:
colours, shapes, sizes, textures and spatial relationships. Text is dropped (empty caption) 10% of the time
(`class_dropout_prob: 0.1`).

`t2i/prompts.txt` (the README's example prompts) are long descriptive paragraphs, several closing on style or
camera tags, for example "Clean flat vector style, rounded lines, no shading."

## Sampling

- Official (`t2i/README.md`): FlowDPMSolver (multistep, order 2, `time_uniform_flow`), flow shift 4.0 at 1024 px,
  50 steps by default, the README command at CFG 2.75 with negative prompt
  `low quality, worst quality, over-saturated, blurry, deformed, watermark`; the paper reports 25 steps at
  guidance 4.5 for its text-to-image numbers.
- ComfyUI template `image_pixeldit_t2i`: `EmptyChromaRadianceLatentImage`, KSampler `er_sde`/`simple`, 30 steps,
  CFG 4, the same negative prompt, the model's own sampling (discrete flow, shift 4.0), `VAELoader` `pixel_space`.
- Sides are multiples of 16 (the patch size); trained around 1 MP at 1024 and at 512.
