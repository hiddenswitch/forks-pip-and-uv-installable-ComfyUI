# Native language-model workflows

The preferred LLM path is ComfyUI’s native text-generation nodes. They use
the loaded ComfyUI `CLIP` object, so tokenization, device placement,
DynamicVRAM, templates, and workflow serialization stay inside the normal
ComfyUI machinery.

## Native setup

Build a workflow with:

```text
CLIPLoader → TextGenerate → downstream conditioning/video/image node
```

`TextGenerate` accepts text and optional image, video, or audio inputs. Its
controls include maximum length, sampling mode, thinking mode, and an optional
model template. The node returns generated text and a separate thinking-text
output. `TextGenerateLTX2Prompt` provides
the native prompt-enhancement path for LTX-2 workflows.

Use the packaged workflow templates whenever possible; they already connect
the correct CLIP loader and generation node for the model family.

Read the checkpoint's [prompting and chat-template reference](prompting_guides/model_coverage.md#language-and-hosted-models).
Prompt-enhancer fine-tunes need their own task prompts; loading a stock model
with the same architecture does not reproduce the fine-tune's answer format.

## Models and memory

Place model files in the configured model directories and use the ordinary
model loaders. DynamicVRAM may eject a text encoder or other dependency when
the diffusion model needs its memory, then reload it when required. This is
the same policy used for image and video workflows; do not add a special
offloading node.

## Legacy Transformers nodes

`TransformersLoader` and `TransformersGenerate` remain available for existing
workflows that directly manage a Hugging Face Transformers model. Treat them
as an advanced compatibility path: they do not automatically receive all
native CLIP templates, memory-management decisions, or workflow conveniences.
