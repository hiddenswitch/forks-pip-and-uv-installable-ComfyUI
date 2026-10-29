# Anima model card

Vendored snapshot from: https://huggingface.co/circlestone-labs/Anima

Fetched: 2026-06-03. Prompting and variant guidance updated: 2026-10-29.

---

---
license: other
license_name: circlestone-labs-non-commercial-license
license_link: LICENSE.md
tags:
  - diffusion-single-file
  - comfyui
---

<img src="https://huggingface.co/circlestone-labs/Anima/resolve/main/montage.jpg">

Anima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org. It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.

It is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.

# Versions
- Anima-Base
  - The pretrained, unrefined base model. Maximum flexibility, diversity, and style adherence. Use Base for training LoRAs.
- Anima-Aesthetic
  - Aesthetic fine-tune. Omit `score_*` tags from both prompts; human quality tags are optional.
- Anima-Turbo
  - Released distilled variant: use CFG 1 and 8–12 steps. It favors speed and a stronger default style over Base's diversity.

# Installing and running
Workflow:
<img src="https://huggingface.co/circlestone-labs/Anima/resolve/main/example.png" width="200">
The model is natively supported in ComfyUI. The above image contains a workflow; you can open it in ComfyUI or drag-and-drop to get the workflow. The model files go in their respective folders inside your model directory:
- anima-base-v1.0.safetensors goes in ComfyUI/models/diffusion_models
- qwen_3_06b_base.safetensors goes in ComfyUI/models/text_encoders
- qwen_image_vae.safetensors goes in ComfyUI/models/vae (this is the Qwen-Image VAE, you might already have it)

## Generation settings
- Works at resolutions between 512^2 and 1536^2 pixels.
- Base: 30-50 steps, CFG 4-5. Turbo uses the separate settings above.
- A variety of samplers work. Some of my favorites:
  - er_sde: neutral style, flat colors, sharp lines. I use this as a reasonable default.
  - euler: also recommended by the author for Aesthetic and Turbo.
  - euler_a: Softer, thinner lines. Can sometimes tend towards a 2.5D look. CFG can be pushed a bit higher than other samplers without burning the image.
  - dpmpp_2m_sde_gpu: similar in style to er_sde but can produce more variety and be more "creative". Depending on the prompt it can get too wild sometimes.
- If going for a more realistic / painterly look, the beta57 scheduler (ComfyUI RES4LYF custom node pack) can help make better textures, since it puts more emphasis on low-noise timesteps.

# Prompting
The model is trained on Danbooru-style tags, natural language captions, and combinations of tags and captions.
- Use lowercase for tags, and spaces instead of underscores. Score tags are the only tags that use underscores.
- Recommended positive prefix: "masterpiece, best quality, score_7, safe, "
- Recommended Base negative: "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
- When using a tag that is different between Danbooru and Gelbooru, prefer the Gelbooru version.
- Prompt weighting works, but needs a weight higher than typically used for SDXL. Example: "(chibi:2)"

## Tag order
[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] [series] [artist] [general tags]

Within each tag section, the tags can be in arbitrary order.

## Quality tags
Human score based: masterpiece, best quality, good quality, normal quality, low quality, worst quality

PonyV7 aesthetic model based: score_9, score_8, ..., score_1

For Base, you can use human quality tags, score tags, both, or neither. For Aesthetic, omit score tags from positive and negative prompts.

## Time period tags
Specific year: year 2025, year 2024, ...

Period: newest, recent, mid, early, old

## Meta tags
highres, absurdres, anime screenshot, jpeg artifacts, official art, etc

## Safety tags
safe, sensitive, nsfw, explicit

## Artist tags
Prefix artist with @. E.g. "@big chungus". **You must put @ in front of the artist.** The effect will be very weak if you don't.

## Full tag example
year 2025, newest, normal quality, score_5, highres, safe, 1girl, oomuro sakurako, yuru yuri, @nnn yryr, smile, brown hair, hat, solo, fur-trimmed gloves, open mouth, long hair, gift box, fang, skirt, red gloves, blunt bangs, gloves, one eye closed, shirt, brown eyes, santa costume, red hat, skin fang, twitter username, white background, holding bag, fur trim, simple background, brown skirt, bag, gift bag, looking at viewer, santa hat, ;d, red shirt, box, gift, fur-trimmed headwear, holding, red capelet, holding box, capelet

## Tag dropout
The model was trained with random tag dropout. You don't need to include every single relevant tag for the image.

## Dataset tags
To improve style and content diversity, the model was additionally trained on two non-anime datasets: LAION-POP (specifically the ye-pop version) and DeviantArt. Both were filtered to exclude photos. Because these datasets are qualitatively different from anime datasets, captions from them have been labeled with a "dataset tag". This occurs at the very beginning of a prompt followed by a newline. Optionally, the second line can contain either the image alt-text (ye-pop) or the title of the work (DeviantArt). Examples:


ye-pop<br>
For Sale: Others by Arun Prem<br>
Abstract, oil painting of three faceless, blue-skinned figures. Left: white, draped figure; center: yellow-shirted, dark-haired figure; right: red-veiled, dark-haired figure carrying another. Bold, textured colors, minimalist style.


deviantart<br>
Flame<br>
Digital painting of a fiery dragon with glowing yellow eyes, black horns, and a long, sinuous tail, perched on a glowing, molten rock formation. The background is a gradient of dark purple to orange.

## Natural language prompting tips
- Follow standard English capitalization rules for character and series names.
- If using pure natural langauge, more descriptive is better. Aim for at least 2 sentences. Extremely short prompts can give unexpected results.
- You can mix tags and natural language in arbitrary order.
- You can put quality / artist tags at the beginning of a natural language prompt.
  - "masterpiece, best quality, @big chungus. An anime girl with medium-length blonde hair is..."
- Name a character, then describe their basic appearance.
  - "Digital artwork of Fern from Sousou no Frieren, with long purple hair and purple eyes, wearing a black coat over a white dress with puffy sleeves..."
  - This is extra important when prompting for multiple characters. If you just list off character names with no description of appearance, the model can get confused.

# Model comparison
You may be interested in comparing Anima's outputs with other models. A ComfyUI workflow, anima_comparison.json, is provided. This workflow generates a grid of images where each model is a column and the rows are different seeds. It can be configured to compare any number of models you select by changing a few output nodes. Supported model architectures: Anima, SDXL, Lumina, Chroma, Newbie-Image. The default configuration compares Anima, NetaYume, and Newbie-Image.

# Limitations
- The model doesn't do realism well. This is intended. It is an anime / illustration / art focused model.
- The model may generate undesired content, especially if the prompt is short or lacking details.
  - Avoid this by using the appropriate safety tags in the positive and negative prompts, and by writing sufficiently detailed prompts.
- The model isn't great at text rendering. It can generally do single words and sometimes short phrases, but lengthy text rendering won't work well.
- The base version is a true base model. It hasn't been aesthetic tuned on a curated dataset. The default style is very plain and neutral, which is especially apparent if you don't use artist or quality tags.

# Finetuning Tips
- **Don't train the LLM adapter.** My own training script, diffusion-pipe, lets you set llm_adapter_lr=0 to completely disable training it, and the example config has this as a default.
  - Other trainers like sd-scripts have similar options that should be used.
  - The LLM adapter processes the text embeddings before they get to the diffusion model, and therefore has an outsized influence on the generated images. The adapter itself contains a surprising amount of knowledge and is easy to degrade by training it.
- Use a low learning rate. For a rank 32 LoRA, start with 2e-5 and adjust up or down from there.
  - As a base model, there is no aggressive aesthetic tuning or RLHF you need to overcome when finetuning.
  - The model has an extremely large and diverse amount of visual concepts baked in already. A light touch is all you need.
- [Example](https://civitai.com/models/2536147) of a style LoRA, with dataset and configs shared.

# License
Consult the current [CircleStone Labs license](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md) and the model card's distinction between model use and generated outputs. Anima also derives from Cosmos-Predict2-2B-Text2Image; the card links the applicable [NVIDIA license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

If you would like a commercial license, please email tdrussell@circlestone.ai

Built on NVIDIA Cosmos.
