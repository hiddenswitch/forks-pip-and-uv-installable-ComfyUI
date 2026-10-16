"""Utilities for manipulating workflow prompt dicts."""
from __future__ import annotations

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TEXT_ENCODE_FIELDS: dict[str, list[str]] = {
    "CLIPTextEncode": ["text"],
    "CLIPTextEncodeSD3": ["clip_l", "clip_g", "t5xxl"],
    "TextEncodeQwenImageEdit": ["prompt"],
    "OneShotInstructTokenize": ["prompt"],
    "TransformersTranslationTokenize": ["prompt"],
    "TransformersTokenize": ["prompt"],
    "MiniMaxH3ImageToVideo": ["prompt"],
    "MiniMaxH3ReferenceToVideo": ["prompt"],
}


_STEPS_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "BasicScheduler",
    "Flux2Scheduler",
    "LTXVScheduler",
    "AlignYourStepsScheduler",
    "Ideogram4Scheduler",
})

# class_type -> seed field name
_SEED_FIELDS: dict[str, str] = {
    "KSampler": "seed",
    "KSamplerAdvanced": "seed",
    "RandomNoise": "noise_seed",
    "SamplerCustom": "noise_seed",
    "TransformersGenerate": "seed",
}

_CFG_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
})

_SAMPLER_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
})

_SCHEDULER_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "BasicScheduler",
})

_DENOISE_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
})

_LATENT_SIZE_CLASS_TYPES = frozenset({
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanImageLatent",
    "EmptyChromaRadianceLatentImage",
    "EmptyHiDreamO1LatentImage",
    "EmptyQwenImageLayeredLatentImage",
})

_CHECKPOINT_CLASS_TYPES = frozenset({
    "CheckpointLoaderSimple",
})

_DIFFUSION_MODEL_CLASS_TYPES = frozenset({
    "UNETLoader",
    "DiffusionModelLoader",
    "UnetLoaderGGUF",
})

_IMAGE_LOAD_CLASS_TYPES = frozenset({
    "LoadImage",
    "LoadImageFromURL",
    "ImageRequestParameter",
})

_VIDEO_LOAD_CLASS_TYPES = frozenset({
    "LoadVideo",
    "LoadVideoFromURL",
    "VHS_LoadVideo",
    "VHS_LoadVideoFFmpeg",
    "VideoRequestParameter",
})

_AUDIO_LOAD_CLASS_TYPES = frozenset({
    "LoadAudio",
    "LoadAudioFromURL",
    "AudioRequestParameter",
})


from comfy_execution.graph_utils import is_link as _is_node_ref


def _replace_field_in_nodes(prompt: dict, class_types: frozenset, field: str, value) -> dict:
    node_ids = [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in class_types
        and field in node.get("inputs", {})
    ]
    if not node_ids:
        return prompt
    prompt = copy.deepcopy(prompt)
    for nid in node_ids:
        prompt[nid]["inputs"][field] = value
    return prompt


def _find_text_encoder_in_predecessors(prompt: dict, start_node_id: str) -> Optional[str]:
    visited: set[str] = set()
    stack = [start_node_id]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if node is None:
            continue
        if node.get("class_type", "") in _TEXT_ENCODE_FIELDS:
            return nid
        for val in node.get("inputs", {}).values():
            if _is_node_ref(val):
                ref_id = str(val[0])
                if ref_id not in visited:
                    stack.append(ref_id)
    return None



def _find_positive_text_encoder_via_positive_input(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        positive_ref = node.get("inputs", {}).get("positive")
        if not _is_node_ref(positive_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(positive_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_guider(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") != "BasicGuider":
            continue
        cond_ref = node.get("inputs", {}).get("conditioning")
        if not _is_node_ref(cond_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(cond_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_title(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") not in _TEXT_ENCODE_FIELDS:
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title or "(prompt)" in title:
            return node_id
    return None


def _find_sole_text_encoder(prompt: dict) -> Optional[str]:
    text_nodes = [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in _TEXT_ENCODE_FIELDS
    ]
    if len(text_nodes) == 1:
        return text_nodes[0]
    return None


def find_positive_text_encoder(prompt: dict) -> Optional[str]:
    return (
        _find_positive_text_encoder_via_positive_input(prompt)
        or _find_positive_text_encoder_via_guider(prompt)
        or _find_positive_text_encoder_via_title(prompt)
        or _find_sole_text_encoder(prompt)
    )


def replace_prompt_text(prompt: dict, text: str) -> dict:
    """Return a copy of *prompt* with the positive text encoding node's text replaced.

    Raises ``ValueError`` if no suitable text encoding node is found.
    """
    node_id = find_positive_text_encoder(prompt)
    if node_id is None:
        raise ValueError("Could not find a positive text encoding node to replace")

    prompt = copy.deepcopy(prompt)
    node = prompt[node_id]
    class_type = node["class_type"]
    fields = _TEXT_ENCODE_FIELDS[class_type]
    for field in fields:
        if field in node["inputs"]:
            node["inputs"][field] = text
    return prompt



def _find_negative_text_encoder_via_negative_input(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        negative_ref = node.get("inputs", {}).get("negative")
        if not _is_node_ref(negative_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(negative_ref[0]))
        if result is not None:
            return result
    return None


def _find_negative_text_encoder_via_title(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") not in _TEXT_ENCODE_FIELDS:
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if "negative" in title:
            return node_id
    return None


def find_negative_text_encoder(prompt: dict) -> Optional[str]:
    return (
        _find_negative_text_encoder_via_negative_input(prompt)
        or _find_negative_text_encoder_via_title(prompt)
    )


def replace_negative_prompt_text(prompt: dict, text: str) -> dict:
    """Return a copy of *prompt* with the negative text encoding node's text replaced.

    Raises ``ValueError`` if no suitable negative text encoding node is found.
    """
    node_id = find_negative_text_encoder(prompt)
    if node_id is None:
        raise ValueError("Could not find a negative text encoding node to replace")

    prompt = copy.deepcopy(prompt)
    node = prompt[node_id]
    class_type = node["class_type"]
    fields = _TEXT_ENCODE_FIELDS[class_type]
    for field in fields:
        if field in node["inputs"]:
            node["inputs"][field] = text
    return prompt



def find_steps_nodes(prompt: dict) -> list[str]:
    """Return node IDs of all nodes that have a ``steps`` input."""
    return [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in _STEPS_CLASS_TYPES
        and "steps" in node.get("inputs", {})
    ]


def replace_steps(prompt: dict, steps: int) -> dict:
    """Return a copy of *prompt* with all sampler/scheduler step counts replaced."""
    return _replace_field_in_nodes(prompt, _STEPS_CLASS_TYPES, "steps", steps)



def find_seed_nodes(prompt: dict) -> list[tuple[str, str]]:
    """Return ``(node_id, field_name)`` pairs for all nodes with a seed input."""
    results = []
    for nid, node in prompt.items():
        class_type = node.get("class_type", "")
        field = _SEED_FIELDS.get(class_type)
        if field is not None and field in node.get("inputs", {}):
            results.append((nid, field))
    return results


def replace_seed(prompt: dict, seed: int) -> dict:
    """Return a copy of *prompt* with all seed values replaced."""
    pairs = find_seed_nodes(prompt)
    if not pairs:
        return prompt
    prompt = copy.deepcopy(prompt)
    for nid, field in pairs:
        prompt[nid]["inputs"][field] = seed
    return prompt


# filesystem-loader -> URL-loader class_type
_MEDIA_LOADER_TO_URL: dict[str, str] = {
    "LoadImage": "LoadImageFromURL",
    "LoadVideo": "LoadVideoFromURL",
    "LoadAudio": "LoadAudioFromURL",
}

_MEDIA_INPUT_FIELDS: dict[str, str] = {
    "VHS_LoadVideo": "video",
    "VHS_LoadVideoFFmpeg": "video",
}


def _find_media_nodes(prompt: dict, class_types: frozenset) -> list[str]:
    """Return node IDs of media-loading nodes matching *class_types*."""
    return [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in class_types
    ]


def _replace_media(
    prompt: dict,
    values: list[str],
    class_types: frozenset,
) -> dict:
    """Generic replacement for image / video / audio loading nodes.

    Filesystem loaders (``LoadImage``, ``LoadVideo``, ``LoadAudio``) are
    converted to their ``*FromURL`` counterparts.  Nodes that already accept
    a ``value`` input simply have it updated.
    """
    node_ids = _find_media_nodes(prompt, class_types)
    if not node_ids or not values:
        return prompt
    prompt = copy.deepcopy(prompt)
    for i, nid in enumerate(node_ids):
        if i >= len(values):
            break
        node = prompt[nid]
        class_type = node["class_type"]
        url_class = _MEDIA_LOADER_TO_URL.get(class_type)
        if url_class is not None:
            node["class_type"] = url_class
            node["inputs"] = {"value": values[i]}
            node.pop("_meta", None)
        else:
            field = _MEDIA_INPUT_FIELDS.get(class_type, "value")
            node["inputs"][field] = values[i]
    return prompt


# --image

def find_image_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of image loading nodes."""
    return _find_media_nodes(prompt, _IMAGE_LOAD_CLASS_TYPES)


def replace_images(prompt: dict, images: list[str]) -> dict:
    """Return a copy of *prompt* with image loading nodes replaced."""
    return _replace_media(prompt, images, _IMAGE_LOAD_CLASS_TYPES)


# --video

def find_video_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of video loading nodes."""
    return _find_media_nodes(prompt, _VIDEO_LOAD_CLASS_TYPES)


def replace_videos(prompt: dict, videos: list[str]) -> dict:
    """Return a copy of *prompt* with video loading nodes replaced."""
    return _replace_media(prompt, videos, _VIDEO_LOAD_CLASS_TYPES)


# --audio

def find_audio_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of audio loading nodes."""
    return _find_media_nodes(prompt, _AUDIO_LOAD_CLASS_TYPES)


def replace_audios(prompt: dict, audios: list[str]) -> dict:
    """Return a copy of *prompt* with audio loading nodes replaced."""
    return _replace_media(prompt, audios, _AUDIO_LOAD_CLASS_TYPES)


def replace_cfg(prompt: dict, cfg: float) -> dict:
    return _replace_field_in_nodes(prompt, _CFG_CLASS_TYPES, "cfg", cfg)


def replace_sampler(prompt: dict, sampler_name: str) -> dict:
    return _replace_field_in_nodes(prompt, _SAMPLER_CLASS_TYPES, "sampler_name", sampler_name)


def replace_scheduler(prompt: dict, scheduler: str) -> dict:
    return _replace_field_in_nodes(prompt, _SCHEDULER_CLASS_TYPES, "scheduler", scheduler)


def replace_denoise(prompt: dict, denoise: float) -> dict:
    return _replace_field_in_nodes(prompt, _DENOISE_CLASS_TYPES, "denoise", denoise)


def replace_width(prompt: dict, width: int) -> dict:
    return _replace_field_in_nodes(prompt, _LATENT_SIZE_CLASS_TYPES, "width", width)


def replace_height(prompt: dict, height: int) -> dict:
    return _replace_field_in_nodes(prompt, _LATENT_SIZE_CLASS_TYPES, "height", height)


def replace_batch_size(prompt: dict, batch_size: int) -> dict:
    return _replace_field_in_nodes(prompt, _LATENT_SIZE_CLASS_TYPES, "batch_size", batch_size)


def replace_checkpoint(prompt: dict, ckpt_name: str) -> dict:
    return _replace_field_in_nodes(prompt, _CHECKPOINT_CLASS_TYPES, "ckpt_name", ckpt_name)


def replace_diffusion_model(prompt: dict, unet_name: str) -> dict:
    return _replace_field_in_nodes(prompt, _DIFFUSION_MODEL_CLASS_TYPES, "unet_name", unet_name)


# --add-lora / --compile
#
# Node class_types that emit a MODEL link that a LoRA or TorchCompileModel
# node can splice after. The set mirrors the loaders we enumerate in the
# ComfyUI loaders registry; keep in sync with base_nodes.py when adding new
# root-level model producers.
_MODEL_PRODUCER_CLASS_TYPES = frozenset({
    "CheckpointLoader",
    "CheckpointLoaderSimple",
    "unCLIPCheckpointLoader",
    "UNETLoader",
    "DiffusionModelLoader",
    "ImageOnlyCheckpointLoader",
    "UnetLoaderGGUF",
    "NunchakuFluxDiTLoader",
})

# class_types that already accept a MODEL input *and* a MODEL output, i.e.
# they are themselves LoRA / model-mod stack stages. When present after a
# root loader, the new LoRA is spliced at the END of this stack (closest to
# the loader's output, farthest from the sampler) so it behaves like a
# prepended entry in a LoRA chain.
_MODEL_PASSTHROUGH_CLASS_TYPES = frozenset({
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ModelSamplingDiscrete",
    "ModelSamplingContinuousEDM",
    "ModelSamplingSD3",
    "ModelSamplingFlux",
    "ModelSamplingAuraFlow",
    "ModelSamplingStableCascade",
    "ModelSamplingLTXV",
    "CFGOverride",
})

_CLIP_PRODUCER_CLASS_TYPES = frozenset({
    "CheckpointLoader",
    "CheckpointLoaderSimple",
    "unCLIPCheckpointLoader",
    "CLIPLoader",
    "DualCLIPLoader",
    "TripleCLIPLoader",
    "QuadrupleCLIPLoader",
})


def _allocate_node_id(prompt: dict) -> str:
    """Return a fresh node id that doesn't collide with existing keys.

    Workflow ids in converted-subgraph form can look like ``"75:72"``, and UI
    format uses arbitrary strings. Rather than trying to parse, we pick an
    integer above the largest integer-looking id we see. This keeps new ids
    visually distinct from subgraph ids.
    """
    max_int = 0
    for nid in prompt.keys():
        # Strip the subgraph prefix like "75:" if present so compound ids
        # don't prevent us from finding the real maximum.
        tail = nid.rsplit(":", 1)[-1]
        try:
            max_int = max(max_int, int(tail))
        except ValueError:
            continue
    candidate = max_int + 1
    while str(candidate) in prompt:
        candidate += 1
    return str(candidate)


def _find_model_splice_point(prompt: dict) -> Optional[str]:
    """Return the MODEL loader node id the new LoRA should be spliced AFTER.

    The LoRA must load "as early as possible" — walking backward from the
    sampler, the node directly after the root loader. Returning the root
    loader itself means the splice lands between the loader and whatever
    passthrough chain (existing LoRAs, ``ModelSamplingFlux`` etc.) it
    already feeds, so the user's LoRA applies on top of the base weights
    before any other patch.
    """
    roots = [nid for nid, node in prompt.items()
             if node.get("class_type") in _MODEL_PRODUCER_CLASS_TYPES]
    if not roots:
        return None
    # Disambiguate when multiple loaders exist: pick the one whose MODEL
    # output reaches a sampler.
    sampler_feeding = [r for r in roots if _produces_for(prompt, r, _SAMPLER_UP_CLASS_TYPES)]
    return (sampler_feeding or roots)[0]


def _find_clip_splice_point(prompt: dict) -> Optional[str]:
    """Return the CLIP loader node id the new LoRA should be spliced AFTER.

    Like :func:`_find_model_splice_point`, we splice right at the root so
    the LoRA runs before any existing LoRA / CLIP patch chain.
    """
    roots = [nid for nid, node in prompt.items()
             if node.get("class_type") in _CLIP_PRODUCER_CLASS_TYPES]
    if not roots:
        return None
    return roots[0]


def _find_model_chain_tail_from_root(prompt: dict, root: str) -> str:
    """Return the MODEL chain tail that starts at *root*.

    The tail is the last MODEL-passthrough node before the chain reaches a
    sampler, guider, or other non-passthrough consumer. Multi-model workflows
    such as Ideogram4 have separate conditional and unconditional roots; each
    root needs its own compile wrapper.
    """
    tail = root
    while True:
        consumers = _consumers_of_model_output(prompt, tail)
        if len(consumers) != 1:
            return tail
        cid = consumers[0]
        if prompt[cid].get("class_type") not in _MODEL_PASSTHROUGH_CLASS_TYPES:
            return tail
        tail = cid


def _find_model_chain_tail(prompt: dict) -> Optional[str]:
    """Return the node id at the END of the MODEL chain (just before the sampler).

    Opposite end from :func:`_find_model_splice_point`. ``torch.compile``
    should wrap the final patched model, so the TorchCompileModel splice
    point sits AFTER all LoRAs / ``ModelSampling*`` stages and before guider
    nodes. Walks forward from the earliest root loader through passthroughs
    until the next consumer is a sampler / guider (non-passthrough).
    """
    root = _find_model_splice_point(prompt)
    if root is None:
        return None
    return _find_model_chain_tail_from_root(prompt, root)


# Class types whose presence downstream of a loader means the loader is the
# "main" MODEL source (used to disambiguate when multiple loaders exist).
_SAMPLER_UP_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    "CFGGuider",
    "BasicGuider",
})


def _produces_for(prompt: dict, source_id: str, target_class_types: frozenset, max_depth: int = 8) -> bool:
    """Return True if *source_id*'s MODEL output flows into a node whose class is in *target_class_types*."""
    frontier = {source_id}
    visited: set[str] = set()
    for _ in range(max_depth):
        if not frontier:
            return False
        next_frontier: set[str] = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            if prompt.get(nid, {}).get("class_type") in target_class_types:
                return True
            for consumer in _consumers_of_model_output(prompt, nid):
                next_frontier.add(consumer)
        frontier = next_frontier
    return False


def _consumers_of_output(prompt: dict, source_id: str, slot: int) -> list[str]:
    """Return node ids that reference ``[source_id, slot]`` in any input."""
    out = []
    for nid, node in prompt.items():
        for value in node.get("inputs", {}).values():
            if _is_node_ref(value) and str(value[0]) == source_id and int(value[1]) == slot:
                out.append(nid)
                break
    return out


def _consumers_of_model_output(prompt: dict, source_id: str) -> list[str]:
    """MODEL is output 0 for every loader and LoraLoader in core nodes."""
    return _consumers_of_output(prompt, source_id, 0)


def _consumers_of_clip_output(prompt: dict, source_id: str) -> list[str]:
    """CLIP output slot depends on class_type; enumerate explicitly."""
    source_cls = prompt.get(source_id, {}).get("class_type", "")
    slot = {
        "CheckpointLoader": 1,
        "CheckpointLoaderSimple": 1,
        "unCLIPCheckpointLoader": 1,
        "CLIPLoader": 0,
        "DualCLIPLoader": 0,
        "TripleCLIPLoader": 0,
        "QuadrupleCLIPLoader": 0,
        "LoraLoader": 1,
    }.get(source_cls)
    if slot is None:
        return []
    return _consumers_of_output(prompt, source_id, slot)


def _parse_add_lora_spec(spec: str) -> tuple[str, float, float]:
    """Parse ``name[:strength_model[:strength_clip]]`` → ``(name, sm, sc)``.

    A single numeric suffix applies to both model and clip. Works for URI
    names that contain colons (``hf://…``, ``https://…``) and for Windows
    paths (``C:\\loras\\foo``): a colon is only consumed as a weight
    separator when the suffix actually parses as a float.
    """
    # 3-part form: name + strength_model + strength_clip (two trailing floats).
    parts3 = spec.rsplit(":", 2)
    if len(parts3) == 3:
        name, a, b = parts3
        try:
            return name, float(a), float(b)
        except ValueError:
            pass
    # 2-part form: name + single strength (one trailing float).
    parts2 = spec.rsplit(":", 1)
    if len(parts2) == 2:
        name, b = parts2
        try:
            w = float(b)
            return name, w, w
        except ValueError:
            pass
    return spec, 1.0, 1.0


def _materialize_lora_name(name_or_path: str) -> str:
    """Pass *name_or_path* straight through to :class:`LoraLoader`.

    LoraLoader → ``get_full_path_or_raise("loras", ...)`` →
    ``comfy.model_downloader.get_or_download`` already handles:
      * bare filenames (looked up in ``models/loras/``)
      * local paths (resolved through folder_paths)
      * ``hf://owner/repo/path`` URIs
      * ``https://civitai.com/...`` and generic ``http(s)://`` URLs
      * any fsspec scheme (``s3://``, ``gcs://``, ``file://``, ...)
    The workflow-side combo validator is relaxed in
    :func:`comfy.cmd.execution.validate_inputs` to allow URIs through.
    """
    return name_or_path


def add_loras(prompt: dict, specs: list[str]) -> dict:
    """Return a copy of *prompt* with a ``LoraLoader`` inserted per *spec*.

    Each entry in *specs* has the form ``name[:strength_model[:strength_clip]]``
    where ``name`` is a bare filename, a local path, or a URL (fsspec). A
    single trailing float applies to both model and clip.

    The LoRA is spliced immediately after the tail of any existing LoRA /
    model-sampling chain — the chain closest to the loader rather than to
    the sampler — so it composes identically to a user-edited LoRA stack
    and doesn't skip over downstream patches like ``ModelSamplingFlux``.

    When the workflow has no CLIP-producing loader (rare, e.g. video-only
    pipelines where text conditioning comes from a separate encoder), the
    inserted node is ``LoraLoaderModelOnly``.
    """
    if not specs:
        return prompt

    prompt = copy.deepcopy(prompt)

    for spec in specs:
        raw_name, strength_model, strength_clip = _parse_add_lora_spec(spec)
        lora_name = _materialize_lora_name(raw_name)

        model_tail = _find_model_splice_point(prompt)
        if model_tail is None:
            logger.warning("--add-lora %r: no MODEL producer found, skipping", spec)
            continue
        clip_tail = _find_clip_splice_point(prompt)

        model_slot = 0  # all producers/passthroughs in our set emit MODEL at slot 0
        model_consumers = _consumers_of_model_output(prompt, model_tail)

        new_id = _allocate_node_id(prompt)

        if clip_tail is not None:
            clip_slot = {
                "CheckpointLoader": 1,
                "CheckpointLoaderSimple": 1,
                "unCLIPCheckpointLoader": 1,
                "CLIPLoader": 0,
                "DualCLIPLoader": 0,
                "TripleCLIPLoader": 0,
                "QuadrupleCLIPLoader": 0,
                "LoraLoader": 1,
            }[prompt[clip_tail]["class_type"]]
            clip_consumers = _consumers_of_clip_output(prompt, clip_tail)

            prompt[new_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": [model_tail, model_slot],
                    "clip": [clip_tail, clip_slot],
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                },
                "_meta": {"title": f"LoRA: {lora_name}"},
            }
            for cid in model_consumers:
                _rewire_inputs(prompt[cid], (model_tail, model_slot), (new_id, 0))
            for cid in clip_consumers:
                _rewire_inputs(prompt[cid], (clip_tail, clip_slot), (new_id, 1))
        else:
            prompt[new_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [model_tail, model_slot],
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                },
                "_meta": {"title": f"LoRA: {lora_name}"},
            }
            for cid in model_consumers:
                _rewire_inputs(prompt[cid], (model_tail, model_slot), (new_id, 0))

    return prompt


def _rewire_inputs(node: dict, old: tuple[str, int], new: tuple[str, int]) -> None:
    """Rewrite every ``[old_id, old_slot]`` reference in *node*'s inputs to *new*."""
    for key, value in list(node.get("inputs", {}).items()):
        if _is_node_ref(value) and str(value[0]) == old[0] and int(value[1]) == old[1]:
            node["inputs"][key] = [new[0], new[1]]


def enable_compile(prompt: dict) -> dict:
    """Return a copy of *prompt* with a ``TorchCompileModel`` spliced after
    every independent MODEL chain tail.

    ``torch.compile`` applied to the diffusion transformer typically gives a
    2–4× step-time speedup after the first warmup step; the tradeoff is
    ~30s–2min of compile time on first run and additional VRAM during
    graph capture. We splice at the end of each model chain so the compile
    wraps the model + any LoRAs or ModelSampling* patches the user added.
    """
    prompt = copy.deepcopy(prompt)
    if any(n.get("class_type") == "TorchCompileModel" for n in prompt.values()):
        # Workflow already opts in; don't double-wrap.
        return prompt
    roots = [
        nid for nid, node in prompt.items()
        if node.get("class_type") in _MODEL_PRODUCER_CLASS_TYPES
        and _consumers_of_model_output(prompt, nid)
    ]
    if not roots:
        logger.warning("--compile: no MODEL producer found, skipping")
        return prompt

    compiled_tails: set[str] = set()
    for root in roots:
        chain_tail = _find_model_chain_tail_from_root(prompt, root)
        if chain_tail in compiled_tails:
            continue
        model_consumers = _consumers_of_model_output(prompt, chain_tail)
        if not model_consumers:
            continue
        new_id = _allocate_node_id(prompt)
        prompt[new_id] = {
            "class_type": "TorchCompileModel",
            "inputs": {
                "model": [chain_tail, 0],
                "mode": "default",
            },
            "_meta": {"title": "torch.compile (auto)"},
        }
        for cid in model_consumers:
            _rewire_inputs(prompt[cid], (chain_tail, 0), (new_id, 0))
        compiled_tails.add(chain_tail)
    return prompt
