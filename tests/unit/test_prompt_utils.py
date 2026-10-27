import importlib.resources
import importlib.util
import json

import pytest

from comfy.component_model.prompt_utils import (
    find_positive_text_encoder,
    find_negative_text_encoder,
    replace_prompt_text,
    replace_negative_prompt_text,
    find_steps_nodes,
    replace_steps,
    find_seed_nodes,
    replace_seed,
    replace_sampler,
    find_image_load_nodes,
    replace_images,
    find_video_load_nodes,
    replace_videos,
    find_audio_load_nodes,
    replace_audios,
    _is_node_ref,
    _TEXT_ENCODE_FIELDS,
    _STEPS_CLASS_TYPES,
    _IMAGE_LOAD_CLASS_TYPES,
)
from tests.inference import workflows


def _all_workflow_files():
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


def _load_workflow(workflow_file) -> dict:
    return json.loads(workflow_file.read_text(encoding="utf8"))


class TestFindPositiveTextEncoder:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_finds_positive_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        node_id = find_positive_text_encoder(prompt)
        assert node_id is not None, f"Could not find positive text encoder in {workflow_name}"
        assert node_id in prompt
        class_type = prompt[node_id]["class_type"]
        assert class_type in _TEXT_ENCODE_FIELDS


class TestReplacePromptText:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        replacement = "a test prompt for unit testing"
        result = replace_prompt_text(prompt, replacement)

        node_id = find_positive_text_encoder(prompt)
        class_type = result[node_id]["class_type"]
        fields = _TEXT_ENCODE_FIELDS[class_type]

        for field in fields:
            if field in result[node_id]["inputs"]:
                assert result[node_id]["inputs"][field] == replacement

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_prompt_text(prompt, "mutated?")
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplacePromptTextSpecific:
    def test_simple_two_node_positive_negative(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 123,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"text": "original positive", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
            },
            "3": {
                "inputs": {"text": "original negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
            },
        }
        result = replace_prompt_text(prompt, "new positive")
        assert result["2"]["inputs"]["text"] == "new positive"
        assert result["3"]["inputs"]["text"] == "original negative"

    def test_single_clip_text_encode(self):
        prompt = {
            "1": {
                "inputs": {"text": "sole prompt", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "replaced")
        assert result["1"]["inputs"]["text"] == "replaced"

    def test_sd3_multi_prompt(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {
                    "clip_l": "original l",
                    "clip_g": "original g",
                    "t5xxl": "original t5",
                    "empty_padding": "none",
                    "clip": ["4", 0],
                },
                "class_type": "CLIPTextEncodeSD3",
            },
            "3": {
                "inputs": {"text": "negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "new prompt")
        assert result["2"]["inputs"]["clip_l"] == "new prompt"
        assert result["2"]["inputs"]["clip_g"] == "new prompt"
        assert result["2"]["inputs"]["t5xxl"] == "new prompt"
        assert result["3"]["inputs"]["text"] == "negative"

    def test_guider_chain(self):
        prompt = {
            "1": {
                "inputs": {"guider": ["2", 0]},
                "class_type": "SamplerCustomAdvanced",
            },
            "2": {
                "inputs": {"conditioning": ["3", 0], "model": ["5", 0]},
                "class_type": "BasicGuider",
            },
            "3": {
                "inputs": {"guidance": 3, "conditioning": ["4", 0]},
                "class_type": "FluxGuidance",
            },
            "4": {
                "inputs": {"text": "flux prompt", "clip": ["6", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
        }
        result = replace_prompt_text(prompt, "replaced flux")
        assert result["4"]["inputs"]["text"] == "replaced flux"

    def test_qwen_image_edit(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {
                    "prompt": "edit instruction",
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "image": ["6", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
            "3": {
                "inputs": {
                    "prompt": "",
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "image": ["6", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
        }
        result = replace_prompt_text(prompt, "new edit")
        assert result["2"]["inputs"]["prompt"] == "new edit"
        assert result["3"]["inputs"]["prompt"] == ""

    def test_oneshot_instruct_tokenize(self):
        prompt = {
            "1": {
                "inputs": {"prompt": "instruct me", "model": ["2", 0]},
                "class_type": "OneShotInstructTokenize",
            },
        }
        result = replace_prompt_text(prompt, "new instruction")
        assert result["1"]["inputs"]["prompt"] == "new instruction"

    def test_transformers_translation_tokenize(self):
        prompt = {
            "1": {
                "inputs": {"prompt": "translate this", "model": ["2", 0]},
                "class_type": "TransformersTranslationTokenize",
            },
        }
        result = replace_prompt_text(prompt, "translate that")
        assert result["1"]["inputs"]["prompt"] == "translate that"

    def test_transformers_tokenize(self):
        prompt = {
            "1": {
                "inputs": {"prompt": "tokenize this", "model": ["2", 0]},
                "class_type": "TransformersTokenize",
            },
        }
        result = replace_prompt_text(prompt, "tokenize that")
        assert result["1"]["inputs"]["prompt"] == "tokenize that"

    def test_cfg_guider_chain(self):
        prompt = {
            "1": {
                "inputs": {
                    "cfg": 5,
                    "model": ["5", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "CFGGuider",
            },
            "2": {
                "inputs": {"text": "original positive", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": "original negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "new positive")
        assert result["2"]["inputs"]["text"] == "new positive"
        assert result["3"]["inputs"]["text"] == "original negative"

    def test_cfg_guider_with_passthrough(self):
        prompt = {
            "1": {
                "inputs": {"guider": ["2", 0]},
                "class_type": "SamplerCustomAdvanced",
            },
            "2": {
                "inputs": {
                    "cfg": 5,
                    "model": ["6", 0],
                    "positive": ["3", 0],
                    "negative": ["5", 0],
                },
                "class_type": "CFGGuider",
            },
            "3": {
                "inputs": {"guidance": 3, "conditioning": ["4", 0]},
                "class_type": "FluxGuidance",
            },
            "4": {
                "inputs": {"text": "flux prompt", "clip": ["7", 0]},
                "class_type": "CLIPTextEncode",
            },
            "5": {
                "inputs": {"text": "", "clip": ["7", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "replaced flux")
        assert result["4"]["inputs"]["text"] == "replaced flux"
        assert result["5"]["inputs"]["text"] == ""

    def test_raises_on_no_text_encoder(self):
        prompt = {
            "1": {
                "inputs": {"width": 512, "height": 512},
                "class_type": "EmptyLatentImage",
            },
        }
        with pytest.raises(ValueError, match="Could not find"):
            replace_prompt_text(prompt, "test")


class TestReplaceSteps:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_steps_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_steps = any(
            node.get("class_type", "") in _STEPS_CLASS_TYPES
            and "steps" in node.get("inputs", {})
            for node in prompt.values()
        )
        if not has_steps:
            pytest.skip(f"{workflow_name} has no step-bearing nodes")

        result = replace_steps(prompt, 42)
        for nid in find_steps_nodes(prompt):
            assert result[nid]["inputs"]["steps"] == 42

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_steps_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_steps = any(
            node.get("class_type", "") in _STEPS_CLASS_TYPES
            and "steps" in node.get("inputs", {})
            for node in prompt.values()
        )
        if not has_steps:
            pytest.skip(f"{workflow_name} has no step-bearing nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_steps(prompt, 99)
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceStepsSpecific:
    def test_ksampler_steps(self):
        prompt = {
            "1": {
                "inputs": {"seed": 1, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
        }
        result = replace_steps(prompt, 50)
        assert result["1"]["inputs"]["steps"] == 50

    def test_multiple_step_nodes(self):
        prompt = {
            "1": {
                "inputs": {"seed": 1, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"steps": 30, "scheduler": "normal", "model": ["3", 0]},
                "class_type": "BasicScheduler",
            },
        }
        result = replace_steps(prompt, 10)
        assert result["1"]["inputs"]["steps"] == 10
        assert result["2"]["inputs"]["steps"] == 10

    def test_no_step_nodes_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_steps(prompt, 50)
        assert result is prompt  # same object, no copy needed

    def test_flux2_scheduler(self):
        prompt = {
            "1": {
                "inputs": {"steps": 25, "shift": 1.0, "model": ["2", 0]},
                "class_type": "Flux2Scheduler",
            },
        }
        result = replace_steps(prompt, 15)
        assert result["1"]["inputs"]["steps"] == 15

    def test_ksampler_advanced_steps(self):
        prompt = {
            "1": {
                "inputs": {"steps": 30, "cfg": 8.0, "start_at_step": 0, "end_at_step": 30},
                "class_type": "KSamplerAdvanced",
            },
        }
        result = replace_steps(prompt, 40)
        assert result["1"]["inputs"]["steps"] == 40


class TestReplaceSeed:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_seed_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        pairs = find_seed_nodes(prompt)
        if not pairs:
            pytest.skip(f"{workflow_name} has no seed-bearing nodes")

        result = replace_seed(prompt, 42)
        for nid, field in pairs:
            assert result[nid]["inputs"][field] == 42

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_seed_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if not find_seed_nodes(prompt):
            pytest.skip(f"{workflow_name} has no seed-bearing nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_seed(prompt, 99)
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceSeedSpecific:
    def test_ksampler_seed(self):
        prompt = {
            "1": {
                "inputs": {"seed": 12345, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
        }
        result = replace_seed(prompt, 99999)
        assert result["1"]["inputs"]["seed"] == 99999

    def test_random_noise_seed(self):
        prompt = {
            "1": {
                "inputs": {"noise_seed": 1038979},
                "class_type": "RandomNoise",
            },
        }
        result = replace_seed(prompt, 42)
        assert result["1"]["inputs"]["noise_seed"] == 42

    def test_sampler_custom_seed(self):
        prompt = {
            "1": {
                "inputs": {"noise_seed": 555, "cfg": 8.0},
                "class_type": "SamplerCustom",
            },
        }
        result = replace_seed(prompt, 111)
        assert result["1"]["inputs"]["noise_seed"] == 111

    def test_transformers_generate_seed(self):
        prompt = {
            "1": {
                "inputs": {"seed": 2013744903, "max_tokens": 512},
                "class_type": "TransformersGenerate",
            },
        }
        result = replace_seed(prompt, 0)
        assert result["1"]["inputs"]["seed"] == 0

    def test_multiple_seed_nodes(self):
        prompt = {
            "1": {
                "inputs": {"seed": 100, "steps": 20},
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"noise_seed": 200},
                "class_type": "RandomNoise",
            },
        }
        result = replace_seed(prompt, 777)
        assert result["1"]["inputs"]["seed"] == 777
        assert result["2"]["inputs"]["noise_seed"] == 777

    def test_no_seed_nodes_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_seed(prompt, 42)
        assert result is prompt


def test_replace_sampler_updates_sampler_select_nodes():
    prompt = {
        "1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    }

    result = replace_sampler(prompt, "euler")

    assert result["1"]["inputs"]["sampler_name"] == "euler"


class TestReplaceImages:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_images_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_images = any(
            node.get("class_type", "") in _IMAGE_LOAD_CLASS_TYPES
            for node in prompt.values()
        )
        if not has_images:
            pytest.skip(f"{workflow_name} has no image load nodes")

        test_uri = "https://example.com/test.png"
        node_ids = find_image_load_nodes(prompt)
        result = replace_images(prompt, [test_uri] * len(node_ids))

        for nid in node_ids:
            node = result[nid]
            assert node["class_type"] in ("LoadImageFromURL", "ImageRequestParameter")
            assert node["inputs"]["value"] == test_uri

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_images_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_images = any(
            node.get("class_type", "") in _IMAGE_LOAD_CLASS_TYPES
            for node in prompt.values()
        )
        if not has_images:
            pytest.skip(f"{workflow_name} has no image load nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_images(prompt, ["https://example.com/test.png"])
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceImagesSpecific:
    def test_load_image_converted_to_url(self):
        prompt = {
            "1": {
                "inputs": {"image": "example.png", "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"},
            },
        }
        result = replace_images(prompt, ["https://example.com/photo.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"] == {"value": "https://example.com/photo.png"}
        assert "_meta" not in result["1"]

    def test_load_image_from_url_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "https://old.com/img.png"},
                "class_type": "LoadImageFromURL",
            },
        }
        result = replace_images(prompt, ["https://new.com/img.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"]["value"] == "https://new.com/img.png"

    def test_image_request_parameter_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old_path.png"},
                "class_type": "ImageRequestParameter",
            },
        }
        result = replace_images(prompt, ["s3://bucket/new.png"])
        assert result["1"]["inputs"]["value"] == "s3://bucket/new.png"

    def test_multiple_images_assigned_in_order(self):
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
            "2": {
                "inputs": {"image": "b.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, ["https://x.com/1.png", "https://x.com/2.png"])
        assert result["1"]["inputs"]["value"] == "https://x.com/1.png"
        assert result["2"]["inputs"]["value"] == "https://x.com/2.png"

    def test_fewer_images_than_nodes(self):
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
            "2": {
                "inputs": {"image": "b.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, ["https://x.com/only.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"]["value"] == "https://x.com/only.png"
        # second node stays as LoadImage
        assert result["2"]["class_type"] == "LoadImage"
        assert result["2"]["inputs"]["image"] == "b.png"

    def test_no_image_nodes_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_images(prompt, ["https://x.com/img.png"])
        assert result is prompt

    def test_empty_images_list_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, [])
        assert result is prompt


class TestFindNegativeTextEncoder:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_finds_negative_in_workflows_with_negative_input(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_negative_input = any(
            _is_node_ref(node.get("inputs", {}).get("negative"))
            for node in prompt.values()
        )
        if not has_negative_input:
            pytest.skip(f"{workflow_name} has no node with negative conditioning input")
        has_text_encoder = any(
            node.get("class_type") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encoder:
            pytest.skip(f"{workflow_name} has no text encoder")

        node_id = find_negative_text_encoder(prompt)
        assert node_id is not None, f"Could not find negative text encoder in {workflow_name}"
        assert node_id in prompt
        class_type = prompt[node_id]["class_type"]
        assert class_type in _TEXT_ENCODE_FIELDS


class TestReplaceNegativePromptText:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_negative_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if find_negative_text_encoder(prompt) is None:
            pytest.skip(f"{workflow_name} has no negative text encoder")

        replacement = "a test negative prompt"
        result = replace_negative_prompt_text(prompt, replacement)

        node_id = find_negative_text_encoder(prompt)
        class_type = result[node_id]["class_type"]
        fields = _TEXT_ENCODE_FIELDS[class_type]

        for field in fields:
            if field in result[node_id]["inputs"]:
                assert result[node_id]["inputs"][field] == replacement

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_negative_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if find_negative_text_encoder(prompt) is None:
            pytest.skip(f"{workflow_name} has no negative text encoder")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_negative_prompt_text(prompt, "mutated?")
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceNegativePromptTextSpecific:
    def test_ksampler_positive_negative(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 123,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"text": "original positive", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
            },
            "3": {
                "inputs": {"text": "original negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
            },
        }
        result = replace_negative_prompt_text(prompt, "new negative")
        assert result["3"]["inputs"]["text"] == "new negative"
        assert result["2"]["inputs"]["text"] == "original positive"

    def test_negative_via_title(self):
        prompt = {
            "1": {
                "inputs": {"text": "positive prompt", "clip": ["3", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive"},
            },
            "2": {
                "inputs": {"text": "negative prompt", "clip": ["3", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative"},
            },
        }
        result = replace_negative_prompt_text(prompt, "replaced negative")
        assert result["2"]["inputs"]["text"] == "replaced negative"
        assert result["1"]["inputs"]["text"] == "positive prompt"

    def test_qwen_image_edit_negative(self):
        prompt = {
            "3": {
                "inputs": {
                    "seed": 1,
                    "positive": ["76", 0],
                    "negative": ["77", 0],
                },
                "class_type": "KSampler",
            },
            "76": {
                "inputs": {
                    "prompt": "edit instruction",
                    "clip": ["38", 0],
                    "vae": ["39", 0],
                    "image": ["93", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
            "77": {
                "inputs": {
                    "prompt": "",
                    "clip": ["38", 0],
                    "vae": ["39", 0],
                    "image": ["93", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
        }
        result = replace_negative_prompt_text(prompt, "ugly, bad quality")
        assert result["77"]["inputs"]["prompt"] == "ugly, bad quality"
        assert result["76"]["inputs"]["prompt"] == "edit instruction"

    def test_raises_on_no_negative_encoder(self):
        prompt = {
            "1": {
                "inputs": {"text": "sole prompt", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        with pytest.raises(ValueError, match="Could not find"):
            replace_negative_prompt_text(prompt, "test")

    def test_cfg_guider_negative(self):
        prompt = {
            "1": {
                "inputs": {
                    "cfg": 5,
                    "model": ["5", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "CFGGuider",
            },
            "2": {
                "inputs": {"text": "original positive", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": "original negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_negative_prompt_text(prompt, "new negative")
        assert result["3"]["inputs"]["text"] == "new negative"
        assert result["2"]["inputs"]["text"] == "original positive"

    def test_negative_through_passthrough(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"text": "positive", "clip": ["5", 0]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": "negative", "clip": ["5", 0]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"conditioning": ["3", 0]},
                "class_type": "ConditioningZeroOut",
            },
        }
        result = replace_negative_prompt_text(prompt, "replaced negative")
        assert result["3"]["inputs"]["text"] == "replaced negative"
        assert result["2"]["inputs"]["text"] == "positive"


class TestReplaceVideosSpecific:
    def test_load_video_converted_to_url(self):
        prompt = {
            "1": {
                "inputs": {"file": "clip.mp4"},
                "class_type": "LoadVideo",
                "_meta": {"title": "Load Video"},
            }
        }
        result = replace_videos(prompt, ["https://example.com/clip.mp4"])
        assert result["1"]["class_type"] == "LoadVideoObjectFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/clip.mp4"
        assert "_meta" not in result["1"]

    def test_load_video_from_url_value_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.mp4"},
                "class_type": "LoadVideoFromURL",
            }
        }
        result = replace_videos(prompt, ["https://example.com/new.mp4"])
        assert result["1"]["class_type"] == "LoadVideoFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/new.mp4"

    def test_video_request_parameter_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.mp4", "frame_load_cap": 100},
                "class_type": "VideoRequestParameter",
            }
        }
        result = replace_videos(prompt, ["https://example.com/new.mp4"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.mp4"

    def test_vhs_load_video_video_field_updated(self):
        prompt = {
            "1": {
                "inputs": {"video": "old.mp4", "frame_load_cap": 100},
                "class_type": "VHS_LoadVideo",
            }
        }
        result = replace_videos(prompt, ["new.mp4"])
        assert result["1"]["class_type"] == "VHS_LoadVideo"
        assert result["1"]["inputs"]["video"] == "new.mp4"

    def test_multiple_video_nodes(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
            "2": {"inputs": {"value": "b.mp4"}, "class_type": "LoadVideoFromURL"},
        }
        result = replace_videos(prompt, ["url1.mp4", "url2.mp4"])
        assert result["1"]["inputs"]["value"] == "url1.mp4"
        assert result["2"]["inputs"]["value"] == "url2.mp4"

    def test_more_videos_than_nodes(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, ["url1.mp4", "url2.mp4", "url3.mp4"])
        assert result["1"]["inputs"]["value"] == "url1.mp4"

    def test_no_video_nodes_unchanged(self):
        prompt = {
            "1": {"inputs": {"text": "hello"}, "class_type": "CLIPTextEncode"},
        }
        result = replace_videos(prompt, ["url1.mp4"])
        assert result is prompt  # no copy when nothing to replace

    def test_empty_videos_unchanged(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, [])
        assert result is prompt

    def test_does_not_mutate_original(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, ["url1.mp4"])
        assert prompt["1"]["class_type"] == "LoadVideo"
        assert result["1"]["class_type"] == "LoadVideoObjectFromURL"


class TestReplaceAudiosSpecific:
    def test_load_audio_converted_to_url(self):
        prompt = {
            "1": {
                "inputs": {"audio": "track.mp3"},
                "class_type": "LoadAudio",
                "_meta": {"title": "Load Audio"},
            }
        }
        result = replace_audios(prompt, ["https://example.com/track.mp3"])
        assert result["1"]["class_type"] == "LoadAudioFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/track.mp3"
        assert "_meta" not in result["1"]

    def test_load_audio_from_url_value_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.wav"},
                "class_type": "LoadAudioFromURL",
            }
        }
        result = replace_audios(prompt, ["https://example.com/new.wav"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.wav"

    def test_audio_request_parameter_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.wav"},
                "class_type": "AudioRequestParameter",
            }
        }
        result = replace_audios(prompt, ["https://example.com/new.wav"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.wav"

    def test_no_audio_nodes_unchanged(self):
        prompt = {
            "1": {"inputs": {"text": "hello"}, "class_type": "CLIPTextEncode"},
        }
        result = replace_audios(prompt, ["url.wav"])
        assert result is prompt

    def test_does_not_mutate_original(self):
        prompt = {
            "1": {"inputs": {"audio": "a.mp3"}, "class_type": "LoadAudio"},
        }
        result = replace_audios(prompt, ["url.mp3"])
        assert prompt["1"]["class_type"] == "LoadAudio"
        assert result["1"]["class_type"] == "LoadAudioFromURL"


def _real_nodes_available() -> bool:
    return importlib.util.find_spec("comfy.nodes.package") is not None


def _load_template_workflow(template_id: str) -> dict | None:
    try:
        from comfyui_workflow_templates import get_asset_path, iter_templates
    except ImportError:
        return None
    for t in iter_templates():
        if t.template_id == template_id:
            json_assets = [a for a in t.assets if a.filename.endswith(".json")]
            if json_assets:
                path = get_asset_path(t.template_id, json_assets[0].filename)
                with open(path) as f:
                    return json.load(f)
    return None


@pytest.mark.skipif(not _real_nodes_available(), reason="node system not available")
class TestConvertedTemplatePromptReplacement:

    @pytest.fixture(scope="class")
    def real_nodes(self):
        from comfy.nodes.package import import_all_nodes_in_workspace
        from comfy.execution_context import context_add_custom_nodes
        nodes = import_all_nodes_in_workspace()
        with context_add_custom_nodes(nodes):
            yield nodes

    def _convert(self, workflow: dict, real_nodes) -> dict:
        from comfy.component_model.workflow_convert import convert_ui_to_api
        return convert_ui_to_api(workflow)

    def test_klein_text_to_image_positive_prompt(self, real_nodes):
        workflow = _load_template_workflow("image_flux2_klein_text_to_image")
        if workflow is None:
            pytest.skip("image_flux2_klein_text_to_image template not found")
        api = self._convert(workflow, real_nodes)
        node_id = find_positive_text_encoder(api)
        assert node_id is not None, (
            "Could not find positive text encoder in converted Klein workflow"
        )
        assert api[node_id]["class_type"] in _TEXT_ENCODE_FIELDS
        result = replace_prompt_text(api, "a girl with red hair")
        assert result[node_id]["class_type"] in _TEXT_ENCODE_FIELDS


class TestFindMediaNodes:
    def test_find_video_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadVideo"},
            "2": {"inputs": {}, "class_type": "LoadVideoFromURL"},
            "3": {"inputs": {}, "class_type": "CLIPTextEncode"},
        }
        assert set(find_video_load_nodes(prompt)) == {"1", "2"}

    def test_find_audio_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadAudio"},
            "2": {"inputs": {}, "class_type": "AudioRequestParameter"},
            "3": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert set(find_audio_load_nodes(prompt)) == {"1", "2"}

    def test_find_no_video_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert find_video_load_nodes(prompt) == []

    def test_find_no_audio_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert find_audio_load_nodes(prompt) == []


# ---------------------------------------------------------------------------
# --add-lora / --compile
# ---------------------------------------------------------------------------

from comfy.component_model.prompt_utils import (  # noqa: E402
    add_loras,
    enable_compile,
    _parse_add_lora_spec,
    _find_model_splice_point,
    _find_model_chain_tail,
    _find_clip_splice_point,
)


def _sdxl_like_workflow() -> dict:
    """Minimal Checkpoint → KSampler graph with one pre-existing LoraLoader."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["1", 1],
            "lora_name": "existing.safetensors",
            "strength_model": 0.7, "strength_clip": 0.7,
        }},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat", "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
            "seed": 0, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
    }


def _flux_like_workflow() -> dict:
    """UNETLoader + DualCLIPLoader + SamplerCustomAdvanced (no Checkpoint)."""
    return {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors", "weight_dtype": "default"}},
        "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5.sft", "clip_name2": "l.sft", "type": "flux"}},
        "12": {"class_type": "ModelSamplingFlux", "inputs": {
            "model": ["10", 0], "max_shift": 1.15, "base_shift": 0.5, "width": 1024, "height": 1024,
        }},
        "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat", "clip": ["11", 0]}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["13", 0]}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "17": {"class_type": "BasicScheduler", "inputs": {"model": ["12", 0], "scheduler": "normal", "steps": 20, "denoise": 1.0}},
        "18": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["15", 0], "guider": ["14", 0], "sampler": ["16", 0], "sigmas": ["17", 0], "latent_image": ["18", 0],
        }},
    }


class TestParseAddLoraSpec:
    def test_bare_name(self):
        assert _parse_add_lora_spec("foo.safetensors") == ("foo.safetensors", 1.0, 1.0)

    def test_name_weight(self):
        assert _parse_add_lora_spec("foo.safetensors:0.8") == ("foo.safetensors", 0.8, 0.8)

    def test_name_model_clip_weights(self):
        assert _parse_add_lora_spec("foo.safetensors:0.8:0.4") == ("foo.safetensors", 0.8, 0.4)

    def test_windows_path_with_colon_not_misparsed(self):
        # C:\loras\foo.safetensors has a colon but no float suffix — treat as name.
        assert _parse_add_lora_spec(r"C:\loras\foo.safetensors") == (r"C:\loras\foo.safetensors", 1.0, 1.0)

    def test_hf_uri(self):
        name, sm, sc = _parse_add_lora_spec("hf://owner/repo/lora.safetensors:0.5")
        assert name == "hf://owner/repo/lora.safetensors"
        assert sm == 0.5 and sc == 0.5


class TestAddLorasSdxl:
    def test_splice_point_is_checkpoint_loader(self):
        wf = _sdxl_like_workflow()
        assert _find_model_splice_point(wf) == "1"
        assert _find_clip_splice_point(wf) == "1"

    def test_splice_point_is_earliest_predecessor_not_tail(self):
        """The new LoRA must splice AFTER the Checkpoint, BEFORE the
        existing LoraLoader — not at the chain tail next to the sampler."""
        wf = _sdxl_like_workflow()
        patched = add_loras(wf, ["new.safetensors:0.9"])

        new_ids = [nid for nid, node in patched.items()
                   if node["class_type"] == "LoraLoader" and node["inputs"].get("lora_name") == "new.safetensors"]
        assert len(new_ids) == 1
        new_id = new_ids[0]

        # New LoRA's inputs come from the Checkpoint.
        assert patched[new_id]["inputs"]["model"] == ["1", 0]
        assert patched[new_id]["inputs"]["clip"] == ["1", 1]
        # Existing LoraLoader's model/clip got rewired to the new LoRA's outputs.
        assert patched["2"]["inputs"]["model"] == [new_id, 0]
        assert patched["2"]["inputs"]["clip"] == [new_id, 1]
        # VAE output of Checkpoint is untouched (slot 2 not rewired).
        assert patched["7"]["inputs"]["vae"] == ["1", 2]
        # KSampler still points at the existing LoraLoader, unchanged.
        assert patched["6"]["inputs"]["model"] == ["2", 0]

    def test_strength_parsing(self):
        wf = _sdxl_like_workflow()
        patched = add_loras(wf, ["a.safetensors:0.8"])
        node = next(n for n in patched.values()
                    if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == "a.safetensors")
        assert node["inputs"]["strength_model"] == 0.8
        assert node["inputs"]["strength_clip"] == 0.8

    def test_separate_model_and_clip_strengths(self):
        wf = _sdxl_like_workflow()
        patched = add_loras(wf, ["a.safetensors:0.8:0.2"])
        node = next(n for n in patched.values()
                    if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == "a.safetensors")
        assert node["inputs"]["strength_model"] == 0.8
        assert node["inputs"]["strength_clip"] == 0.2

    def test_multiple_loras_stack(self):
        wf = _sdxl_like_workflow()
        patched = add_loras(wf, ["a.safetensors:0.5", "b.safetensors:0.3"])
        new_loras = [n for n in patched.values()
                     if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") in ("a.safetensors", "b.safetensors")]
        assert len(new_loras) == 2

    def test_deepcopy_doesnt_mutate_input(self):
        wf = _sdxl_like_workflow()
        before = json.dumps(wf, sort_keys=True)
        add_loras(wf, ["a.safetensors"])
        assert json.dumps(wf, sort_keys=True) == before


class TestAddLorasFlux:
    def test_model_only_when_clip_chain_is_separate(self):
        """Flux workflows have a separate DualCLIPLoader; the inserted LoRA
        uses LoraLoader (both model+clip) since both producers exist."""
        wf = _flux_like_workflow()
        patched = add_loras(wf, ["flux_style.safetensors:0.6"])
        new = next(n for n in patched.values()
                   if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == "flux_style.safetensors")
        assert new["inputs"]["model"] == ["10", 0]
        assert new["inputs"]["clip"] == ["11", 0]

    def test_splices_before_model_sampling_flux(self):
        """ModelSamplingFlux is a passthrough; the LoRA must land between
        the UNETLoader and ModelSamplingFlux so the patch is applied to the
        base weights before time-step reshaping."""
        wf = _flux_like_workflow()
        patched = add_loras(wf, ["flux_style.safetensors"])
        new_id = next(nid for nid, n in patched.items()
                      if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == "flux_style.safetensors")
        assert patched["12"]["inputs"]["model"] == [new_id, 0]


class TestAddLorasUrls:
    def test_hf_uri_passes_through_unchanged(self):
        wf = _sdxl_like_workflow()
        patched = add_loras(wf, ["hf://owner/repo/lora.safetensors:0.5"])
        node = next(n for n in patched.values()
                    if n["class_type"] == "LoraLoader"
                    and n["inputs"].get("lora_name") == "hf://owner/repo/lora.safetensors")
        assert node["inputs"]["strength_model"] == 0.5

    def test_civitai_url_passes_through_unchanged(self):
        wf = _sdxl_like_workflow()
        url = "https://civitai.com/api/download/models/12345"
        patched = add_loras(wf, [url])
        assert any(n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == url
                   for n in patched.values())


class TestEnableCompile:
    def test_splices_at_chain_tail_not_root(self):
        """TorchCompileModel must wrap the FINAL patched model (after LoRAs
        and ModelSamplingFlux) so the compiled graph includes all patches."""
        wf = _flux_like_workflow()
        patched = enable_compile(wf)
        compile_id = next(nid for nid, n in patched.items() if n["class_type"] == "TorchCompileModel")
        # Compile input is the ModelSamplingFlux (chain tail), not the UNETLoader.
        assert patched[compile_id]["inputs"]["model"] == ["12", 0]
        # The guider/scheduler now point at the compile node.
        assert patched["14"]["inputs"]["model"] == [compile_id, 0]
        assert patched["17"]["inputs"]["model"] == [compile_id, 0]

    def test_compile_after_add_lora_wraps_lora(self):
        wf = _flux_like_workflow()
        patched = add_loras(wf, ["foo.safetensors"])
        patched = enable_compile(patched)
        compile_id = next(nid for nid, n in patched.items() if n["class_type"] == "TorchCompileModel")
        # Compile's upstream should be ModelSamplingFlux which feeds from the new LoRA.
        assert patched[compile_id]["inputs"]["model"] == ["12", 0]
        new_lora_id = next(nid for nid, n in patched.items()
                           if n["class_type"] == "LoraLoader" and n["inputs"].get("lora_name") == "foo.safetensors")
        # ModelSamplingFlux's model input is the newly inserted LoRA.
        assert patched["12"]["inputs"]["model"] == [new_lora_id, 0]

    def test_compile_wraps_each_dual_model_branch(self):
        wf = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "ideogram4-unconditional.safetensors"},
            },
            "2": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "ideogram4.safetensors"},
            },
            "3": {
                "class_type": "DualModelGuider",
                "inputs": {"model": ["6", 0], "model_negative": ["1", 0], "conditioning": ["4", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "prompt"},
            },
            "5": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {"guider": ["3", 0]},
            },
            "6": {
                "class_type": "CFGOverride",
                "inputs": {"model": ["2", 0], "cfg": 3, "start_percent": 0.7, "end_percent": 1},
            },
        }

        patched = enable_compile(wf)
        compile_nodes = {
            tuple(node["inputs"]["model"]): nid
            for nid, node in patched.items()
            if node["class_type"] == "TorchCompileModel"
        }

        assert set(compile_nodes) == {("1", 0), ("6", 0)}
        assert patched["3"]["inputs"]["model_negative"] == [compile_nodes[("1", 0)], 0]
        assert patched["3"]["inputs"]["model"] == [compile_nodes[("6", 0)], 0]
        assert patched["6"]["inputs"]["model"] == ["2", 0]
        assert patched["5"]["inputs"]["guider"] == ["3", 0]

    def test_idempotent_when_workflow_has_torch_compile(self):
        wf = _sdxl_like_workflow()
        wf["99"] = {"class_type": "TorchCompileModel", "inputs": {"model": ["2", 0]}}
        patched = enable_compile(wf)
        compiled = [nid for nid, n in patched.items() if n["class_type"] == "TorchCompileModel"]
        assert compiled == ["99"]

    def test_no_op_when_no_model_producer(self):
        wf = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}}}
        patched = enable_compile(wf)
        assert not any(n["class_type"] == "TorchCompileModel" for n in patched.values())


class TestFindChainTail:
    def test_tail_is_tail_of_lora_chain_and_model_sampling(self):
        wf = _flux_like_workflow()
        # UNET (10) → ModelSamplingFlux (12) → BasicGuider/BasicScheduler
        assert _find_model_chain_tail(wf) == "12"

    def test_tail_stops_before_guider(self):
        """BasicGuider takes MODEL but is not treated as a passthrough for
        this purpose — compile must wrap before the guider."""
        wf = _flux_like_workflow()
        tail = _find_model_chain_tail(wf)
        # The guider (14) is a consumer of tail but not IS tail.
        assert tail != "14"


@pytest.mark.parametrize("class_type", ["EmptyFlux2LatentImage", "EmptyHunyuanImageLatent", "EmptyChromaRadianceLatentImage", "EmptyHiDreamO1LatentImage", "EmptyQwenImageLayeredLatentImage"])
def test_dimension_overrides_reach_model_specific_image_latents(class_type):
    from comfy.component_model.prompt_utils import replace_batch_size, replace_height, replace_width
    prompt = {"6": {"class_type": class_type, "inputs": {"width": 256, "height": 256, "batch_size": 1}}}
    prompt = replace_batch_size(replace_height(replace_width(prompt, 1024), 768), 2)
    assert prompt["6"]["inputs"] == {"width": 1024, "height": 768, "batch_size": 2}
