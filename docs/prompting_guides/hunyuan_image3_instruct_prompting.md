# HunyuanImage 3.0 Instruct prompting references

Official sources, checked 2026-10-29:

- [Tencent's Instruct usage and examples](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0#hunyuanimage-30-instruct-instruction-reasoning-and-image-to-image-generation-including-editing-and-multi-image-fusion)
- [Official Instruct model card](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct)
- [Native system prompts](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/hunyuan_image_3/system_prompt.py)
- [Hunyuan Image 3.0 Prompt Handbook](https://docs.qq.com/doc/DUVVadmhCdG9qRXBU), linked by Tencent's [base-model guide](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/Hunyuan-Image3.md)
- [Local base-model guide and worked prompts](hunyuan_image3_prompting.md)

The handbook link is verified against Tencent's repository; its QQ document
could not be retrieved here. The local base-model snapshot is not a copy of
that handbook.

Instruct supports direct image generation, prompt rewriting, and reasoning
followed by rewriting and generation. Tencent names those tasks `image`,
`recaption`, and `think_recaption`; its recommended Instruct example uses
`en_unified` with `think_recaption`. `auto` is the text-output task.
These are native upstream option names; inspect the installed workflow's inputs
before mapping them to ComfyUI controls.

For multiple reference images, identify each image's role and order explicitly;
Tencent's example combines the logo from the first image with the material from
the second. Base-model DeepSeek prompt enhancement is a separate facility:
Instruct performs its own enhancement. Keep this distinction when selecting
between a supplied caption and automatic rewriting.

Tencent also publishes [Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil):
its recommended sampling count is eight steps, with the other recommended
Instruct parameters unchanged. The ordinary Instruct example uses 50 steps.
