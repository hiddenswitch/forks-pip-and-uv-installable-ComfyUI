"""MAGI-2 text conditioning: Qwen3.5-27B hidden states of the raw prompt, two layers below the top.

Structured (JSON) prompts are flattened to the compact text form MAGI-2 was trained on, the
tokenizer isolates every CJK character before Qwen's own pre-tokenization, and prompts are cut
at 7000 tokens. No chat template is applied. The prompt flattening is SandAI's (Apache-2.0).
"""

import json

from tokenizers import Regex, pre_tokenizers

from . import qwen35
from .. import sd1_clip

MAX_TOKENS = 7000
MODEL_TYPE = "qwen35_27b"
_CJK = r"([ᄀ-ᇿ⺀-꓏ꡀ-힯豈-﫿︰-﹏･-ￜ\U00020000-\U0002FFFF])"


def _strip_empty(obj):
    if isinstance(obj, dict):
        return {key: value for key, value in ((key, _strip_empty(value)) for key, value in obj.items())
                if value is not None and value != [] and value != {}}
    if isinstance(obj, list):
        return [_strip_empty(value) for value in obj if value is not None]
    return obj


def _one_line(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def json_to_compact_markdown(raw_json):
    """The training-time text form of MAGI-2's structured prompt JSON."""
    obj = _strip_empty(json.loads(raw_json.strip()))
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return raw_json
    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = []
    for label, key in (("style", "style"), ("mood", "mood_atmosphere"), ("color", "color_scheme")):
        value = _one_line(aesthetics.get(key))
        if value:
            aesthetic_parts.append(f"{label}={value}")
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            visual = _as_dict(subject.get("visual_attributes"))
            attrs = []
            for key in ("gender", "age_group", "ethnicity", "clothing", "appearance_details"):
                if visual.get(key) in (None, [], {}):
                    continue
                value = _one_line(visual[key])
                if value:
                    attrs.append(f"{key}={value}")
            row = " - " + " | ".join(part for part in (_one_line(subject.get("subject_id")), _one_line(subject.get("description"))) if part)
            extra = "; ".join(part for part in (_one_line(subject.get("position")), _one_line(subject.get("orientation"))) if part)
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for item in objects:
            if not isinstance(item, dict):
                continue
            row = " - " + " | ".join(part for part in (_one_line(item.get("object_id")), _one_line(item.get("description"))) if part)
            details = "; ".join(part for part in (_one_line(item.get("shape_and_color")), _one_line(item.get("position"))) if part)
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            timestamp = _one_line(basic.get("timestamp_range"))
            description = _one_line(basic.get("segment_description"))
            head = f" - {timestamp}" if timestamp else " -"
            if description:
                head += f" {description}"
            lines.append(head.rstrip())
            for line in _as_dict(segment.get("audio")).get("dialogue_lines", []) or []:
                if not isinstance(line, dict):
                    continue
                speaker, text, stamp = _one_line(line.get("speaker")), _one_line(line.get("text")), _one_line(line.get("timestamp"))
                if text:
                    entry = (f"   - dialogue {speaker}: " if speaker else "   - dialogue: ") + text
                    if stamp:
                        entry += f" ({stamp})"
                    lines.append(entry)
            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                action = _as_dict(alive.get("action"))
                parts = [part for part in (_one_line(action.get(key)) for key in ("primary_action", "interaction", "facial_expression")) if part]
                if parts:
                    lines.append(f"   - action {_one_line(alive.get('subject_id'))}: " + "; ".join(parts))
            for item in segment.get("objects", []) or []:
                if not isinstance(item, dict):
                    continue
                state = _as_dict(item.get("dynamic_state"))
                parts = [part for part in (_one_line(state.get(key)) for key in ("state_change", "motion_detail")) if part]
                if parts:
                    lines.append(f"   - objects {_one_line(item.get('object_id'))}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())
    return "\n".join(line for line in lines if line.strip())


def normalize_prompt(prompt):
    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return prompt
    if isinstance(parsed, (dict, list)):
        return json_to_compact_markdown(prompt)
    return prompt


def _isolate_cjk(tokenizer):
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(pattern=Regex(_CJK), behavior="isolated"), tokenizer.backend_tokenizer.pre_tokenizer])


class Magi2SDTokenizer(qwen35.Qwen35Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, embedding_size=5120, embedding_key=MODEL_TYPE)
        _isolate_cjk(self.tokenizer)

    def clone(self):
        tokenizer = super().clone()
        _isolate_cjk(tokenizer.tokenizer)
        return tokenizer

    def _encode(self, text, **kwargs):
        return self.tokenizer([normalize_prompt(text)], max_length=MAX_TOKENS, truncation=True, **kwargs)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids = self._encode(text)["input_ids"][0] or [self.pad_token]
        return [[(token, 1.0, index + 1) if return_word_ids else (token, 1.0) for index, token in enumerate(ids)]]

    def token_indices(self, text, target):
        """Indices of the prompt tokens overlapping the first occurrence of target in the normalized prompt."""
        normalized = normalize_prompt(text)
        start = normalized.find(target)
        if start < 0:
            return []
        end = start + len(target)
        offsets = self._encode(text, return_offsets_mapping=True)["offset_mapping"][0]
        return [index for index, (first, last) in enumerate(offsets)
                if not (first == 0 and last == 0 and index != 0) and max(first, start) < min(last, end)]


class Magi2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=MODEL_TYPE, tokenizer=Magi2SDTokenizer)

    def token_indices(self, text, target):
        return self.sd_tokenizer.token_indices(text, target)


class Magi2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        clip_model = lambda **kw: qwen35.Qwen35ClipModel(**kw, layer="hidden", layer_idx=-3, model_type=MODEL_TYPE)
        super().__init__(device=device, dtype=dtype, name=MODEL_TYPE, clip_model=clip_model, model_options=model_options)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Magi2TEModel_(Magi2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Magi2TEModel_
