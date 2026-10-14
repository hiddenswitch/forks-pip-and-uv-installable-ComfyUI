# Chroma1-HD model card

Vendored snapshot from: https://huggingface.co/lodestones/Chroma1-HD

Fetched: 2026-06-03

---

---
license: apache-2.0
pipeline_tag: text-to-image
---
# Chroma1-HD

Chroma1-HD is an **8.9B** parameter text-to-image foundational model based on **FLUX.1-schnell**. It is fully **Apache 2.0 licensed**, ensuring that anyone can use, modify, and build upon it.

As a **base model**, Chroma1 is intentionally designed to be an excellent starting point for **finetuning**. It provides a strong, neutral foundation for developers, researchers, and artists to create specialized models.

for the fast CFG "baked" version please go to [Chroma1-Flash](https://huggingface.co/lodestones/Chroma1-Flash).

### Key Features
*   **High-Performance Base:** 8.9B parameters, built on the powerful FLUX.1 architecture.
*   **Easily Finetunable:** Designed as an ideal checkpoint for creating custom, specialized models.
*   **Community-Driven & Open-Source:** Fully transparent with an Apache 2.0 license, and training history.
*   **Flexible by Design:** Provides a flexible foundation for a wide range of generative tasks.

## Special Thanks
A massive thank you to our supporters who make this project possible.
*   **Anonymous donor** whose incredible generosity funded the pretraining run and data collections. Your support has been transformative for open-source AI.
*   **Fictional.ai** for their fantastic support and for helping push the boundaries of open-source AI. You can try Chroma on their platform:

[![FictionalChromaBanner_1.png](https://huggingface.co/lodestones/Chroma1-HD/resolve/main/images/FictionalChromaBanner_1.png)](https://fictional.ai/?ref=chroma_hf)

## How to Use

### `diffusers` Library

install the requirements 

`pip install transformers diffusers sentencepiece accelerate`

```python
import torch
from diffusers import ChromaPipeline

pipe = ChromaPipeline.from_pretrained("lodestones/Chroma1-HD", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt =  ["low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator("cpu").manual_seed(433),
    num_inference_steps=40,
    guidance_scale=3.0,
    num_images_per_prompt=1,
).images[0]
image.save("chroma.png")
```

Quantized inference using gemlite

```py
import torch
from diffusers import ChromaPipeline

pipe = ChromaPipeline.from_pretrained("lodestones/Chroma1-HD", torch_dtype=torch.float16)
#pipe.enable_model_cpu_offload()

#######################################################
import gemlite
device = 'cuda:0'
processor = gemlite.helper.A8W8_int8_dynamic
#processor = gemlite.helper.A8W8_fp8_dynamic
#processor = gemlite.helper.A16W4_MXFP

for name, module in pipe.transformer.named_modules():
    module.name = name

def patch_linearlayers(model, fct):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.Linear):
            setattr(model, name, fct(layer, name))
        else:
            patch_linearlayers(layer, fct)

def patch_linear_to_gemlite(layer, name):
    layer = layer.to(device, non_blocking=True)
    try:
        return processor(device=device).from_linear(layer) 
    except Exception as exception:
        print('Skipping gemlite conversion for: ' + str(layer.name), exception)
        return layer

patch_linearlayers(pipe.transformer, patch_linear_to_gemlite)
torch.cuda.synchronize()
torch.cuda.empty_cache()

pipe.to(device)
pipe.transformer.forward = torch.compile(pipe.transformer.forward, fullgraph=True)
pipe.vae.forward = torch.compile(pipe.vae.forward, fullgraph=True)
#pipe.set_progress_bar_config(disable=True)
#######################################################

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt =  ["low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"]

import time
for _ in range(3):
    t_start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator("cpu").manual_seed(433),
        num_inference_steps=40,
        guidance_scale=3.0,
        num_images_per_prompt=1,
    ).images[0]
    t_end = time.time()
    print(f"Took: {t_end - t_start} secs.") #66.1242527961731 -> 27.72 sec

image.save("chroma.png")
```

ComfyUI
For advanced users and customized workflows, you can use Chroma with ComfyUI.

**Requirements:**
*   A working ComfyUI installation.
*   [Chroma checkpoint](https://huggingface.co/lodestones/Chroma) (latest version).
*   [T5 XXL Text Encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors).
*   [FLUX VAE](https://huggingface.co/lodestones/Chroma/resolve/main/ae.safetensors).
*   [Chroma Workflow JSON](https://huggingface.co/lodestones/Chroma1-HD/resolve/main/ComfyUI_Chroma1-HD_T2I-workflow.json).

![Chroma Workflow](https://huggingface.co/lodestones/Chroma1-HD/resolve/main/ComfyUI_Chroma1-HD_T2I-sample.png)
![Workflow Overview](https://huggingface.co/lodestones/Chroma1-HD/resolve/main/ComfyUI_Chroma1-HD_T2I-overview.png)

**Setup:**
1.  Place the `T5_xxl` model in your `ComfyUI/models/clip` folder.
2.  Place the `FLUX VAE` in your `ComfyUI/models/vae` folder.
3.  Place the `Chroma checkpoint` in your `ComfyUI/models/diffusion_models` folder.
4.  Load the Chroma workflow file into ComfyUI and run.

## Model Details
*   **Architecture:** Based on the 8.9B parameter FLUX.1-schnell model.
*   **Training Data:** Trained on a 5M sample dataset curated from a 20M pool, including artistic, photographic, and niche styles.
*   **Technical Report:** A comprehensive technical paper detailing the architectural modifications and training process is forthcoming.

## Intended Use
Chroma is intended to be used as a **base model** for researchers and developers to build upon. It is ideal for:
*   Finetuning on specific styles, concepts, or characters.
*   Research into generative model behavior, alignment, and safety.
*   As a foundational component in larger AI systems.

## Limitations and Bias Statement
Chroma is trained on a broad, filtered dataset from the internet. As such, it may reflect the biases and stereotypes present in its training data. The model is released in a state as is and has not been aligned with a specific safety filter.

Users are responsible for their own use of this model. It has the potential to generate content that may be considered harmful, explicit, or offensive. I encourage developers to implement appropriate safeguards and ethical considerations in their downstream applications.

## Summary of Architectural Modifications
*(For a full breakdown, tech report soon-ish.)*

*   **12B → 8.9B Parameters:**
    *   **TL;DR:** I replaced a 3.3B parameter timestep-encoding layer with a more efficient 250M parameter FFN, as the original was vastly oversized for its task.
*   **MMDiT Masking:**
    *   **TL;DR:** Masking T5 padding tokens enhanced fidelity and increased training stability by preventing the model from focusing on irrelevant `<pad>` tokens.
*   **Custom Timestep Distributions:**
    *   **TL;DR:** I implemented a custom timestep sampling distribution (`-x^2`) to prevent loss spikes and ensure the model trains effectively on both high-noise and low-noise regions.

## P.S
Chroma1-HD is not the old Chroma-v.50 it has been retrained from v.48

## Citation
```
@misc{rock2025chroma,
  author = {Lodestone Rock},
  title = {Chroma1-HD},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/lodestones/Chroma1-HD}},
}
```