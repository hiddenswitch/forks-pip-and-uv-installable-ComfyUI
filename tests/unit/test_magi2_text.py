import importlib
import json
import os
from unittest.mock import patch

import pytest
import torch

from comfy.text_encoders.magi2 import Magi2Tokenizer, normalize_prompt

from . import magi2_reference

TOKENIZER_DIR = os.environ.get("MAGI2_TOKENIZER_DIR")
pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or TOKENIZER_DIR is None,
                                reason="needs MAGI2_REFERENCE_SRC and MAGI2_TOKENIZER_DIR (the release's text_encoder tokenizer files)")

STRUCTURED = json.dumps({
    "global_layer": {
        "context": "A rainy  night street", "description": "Neon\nsigns reflect in puddles.",
        "aesthetics": {"style": "cinematic", "mood_atmosphere": "moody", "color_scheme": None},
        "audio_baseline": {"dialogue": {"language": "English", "speaker_tags": ["A", "B"]}, "ambience": "rain"},
        "alive_subjects_static": [{"subject_id": "A", "description": "a woman (30s)", "position": "left", "orientation": "facing right",
                                   "visual_attributes": {"gender": "female", "clothing": "red coat", "ethnicity": []}}],
        "objects_static": [{"object_id": "umbrella", "description": "black umbrella", "shape_and_color": "round, black"}],
    },
    "dynamic_layer": {"timeline_segments": [{
        "segment_basic_info": {"timestamp_range": "0-5s", "segment_description": "She turns."},
        "audio": {"dialogue_lines": [{"speaker": "A", "text": "Wait for me!", "timestamp": "2s"}, {"text": ""}]},
        "alive_subjects": [{"subject_id": "A", "action": {"primary_action": "turns", "facial_expression": "smiles"}}],
        "objects": [{"object_id": "umbrella", "dynamic_state": {"motion_detail": "drips"}}]}]},
    "reference_layer": ["The first frame refers to <Figure 1>"],
}, ensure_ascii=False)
PROMPTS = [
    "A cat (white) walks across a sunny kitchen.reference_layer:The first frame refers to <Figure 1>",
    "一只猫在厨房里走动，阳光明媚。日本語の説明も。 <Figure 1> emoji 😀  tabs\t\tand\nnewlines",
    "embedding:not_an_embedding \\(escaped\\) [brackets] {braces}",
    STRUCTURED,
]


def _reference_encoder():
    magi2_reference.load()
    qwen35 = importlib.import_module("inference.model.qwen35")
    with patch("transformers.Qwen3_5TextModel.from_pretrained", lambda *args, **kwargs: torch.nn.Linear(1, 1)):
        return qwen35, qwen35.Qwen35TextEncoder(TOKENIZER_DIR, device="cpu", skip_layer=2)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_tokens_match_sandai_text_encoder(prompt):
    qwen35, reference = _reference_encoder()
    expected = reference.tokenizer([reference._normalize_prompt(prompt)], padding="longest", max_length=reference.max_length, truncation=True)["input_ids"][0]
    tokenizer = Magi2Tokenizer()
    for current in (tokenizer, tokenizer.clone()):
        tokens = current.tokenize_with_weights(prompt)["qwen35_27b"][0]
        assert [token for token, _ in tokens] == expected
        assert current.token_indices(prompt, "<Figure 1>") == (reference.get_target_token_indices(prompt, "<Figure 1>") or [])
    assert normalize_prompt(prompt) == reference._normalize_prompt(prompt)
