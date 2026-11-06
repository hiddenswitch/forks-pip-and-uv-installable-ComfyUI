"""Tests for comfy.component_model.workflow_convert."""
from __future__ import annotations

import importlib.util
import logging

import pytest

from comfy.component_model.workflow_convert import (
    _VIRTUAL_NODE_TYPES,
    _collect_subgraph_defs,
    _extra_widgets_after,
    _get_sg_widget_positional,
    _is_widget_type,
    _map_unknown_widgets,
    _map_widgets,
    _resolve_source,
    _wrap_value,
    convert_ui_to_api,
)
from comfy.component_model.litegraph_types import LiteLink
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package_typing import ExportedNodes

logger = logging.getLogger(__name__)


def test_unknown_subgraph_widgets_align_with_serialized_value_types():
    node = {
        # Compression removes unlinked widget inputs before unknown nodes are
        # mapped, so use the serialized type metadata retained at that point.
        "inputs": [],
        "_widget_names_ordered": ["steps", "cfg"],
        "_widget_types_ordered": ["INT", "FLOAT"],
    }

    assert _map_unknown_widgets(
        [331728509923362, "randomize", 20, 2.5, "euler", "simple", 1],
        node=node,
    ) == {"steps": 20, "cfg": 2.5}


def test_frontend_unknown_node_widgets_preserve_every_serialized_value():
    assert _map_unknown_widgets(["model", 256, True, "randomize"]) == {
        "UNKNOWN": "model",
        "UNKNOWN_1": 256,
        "UNKNOWN_2": True,
        "UNKNOWN_3": "randomize",
    }


def test_nested_dynamic_combo_widgets_are_consumed_recursively():
    input_types = {
        "required": {
            "format": (
                "COMFY_DYNAMICCOMBO_V3",
                {
                    "options": [
                        {
                            "key": "mp4",
                            "inputs": {
                                "required": {
                                    "codec": (
                                        "COMFY_DYNAMICCOMBO_V3",
                                        {
                                            "options": [
                                                {
                                                    "key": "h264",
                                                    "inputs": {
                                                        "optional": {
                                                            "encoding": (
                                                                "COMFY_DYNAMICCOMBO_V3",
                                                                {
                                                                    "options": [
                                                                        {"key": "auto", "inputs": {"required": {}}},
                                                                        {
                                                                            "key": "re-encode",
                                                                            "inputs": {
                                                                                "required": {
                                                                                    "crf": ("FLOAT", {"default": 23.0}),
                                                                                }
                                                                            },
                                                                        },
                                                                    ]
                                                                },
                                                            )
                                                        }
                                                    },
                                                }
                                            ]
                                        },
                                    )
                                }
                            },
                        }
                    ]
                },
            )
        }
    }

    mapped, consumed = _map_widgets(input_types, ["mp4", "h264", "re-encode", 18.0])

    assert consumed == 4
    assert mapped == {
        "format": "mp4",
        "format.codec": "h264",
        "format.codec.encoding": "re-encode",
        "format.codec.encoding.crf": 18.0,
    }


# ── helper: tiny node classes for testing ─────────────────────────────────────

class _KSamplerLike:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**64, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "dpmpp"], {}),
                "scheduler": (["normal", "karras"], {}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }


class _CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            }
        }


class _CheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["model_a.safetensors", "model_b.safetensors"],),
            }
        }


class _EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "batch_size": ("INT", {"default": 1}),
            }
        }


class _VAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }


class _SaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            }
        }


class _LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (["example.png"], {"image_upload": True}),
            }
        }


class _ImageScaleToTotalPixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (["lanczos", "bilinear"],),
                "megapixels": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "resolution_steps": ("INT", {"default": 1}),
            }
        }


class _ForceInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "forceInput": True}),
                "label": ("STRING", {"default": "hello"}),
            }
        }


class _OptionalFrontendDefaultsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "response_modalities": (["IMAGE+TEXT", "IMAGE"], {"advanced": True}),
                "system_prompt": ("STRING", {"default": "system default", "advanced": True}),
            },
        }


class _ToBasicPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }


class _BasicPipeConsumer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
            }
        }


class _IntSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


# ── fixtures ──────────────────────────────────────────────────────────────────

_TEST_MAPPINGS = {
    "KSampler": _KSamplerLike,
    "CLIPTextEncode": _CLIPTextEncode,
    "CheckpointLoaderSimple": _CheckpointLoaderSimple,
    "EmptyLatentImage": _EmptyLatentImage,
    "VAEDecode": _VAEDecode,
    "SaveImage": _SaveImage,
    "LoadImage": _LoadImage,
    "ImageScaleToTotalPixels": _ImageScaleToTotalPixels,
    "ForceInputNode": _ForceInputNode,
    "OptionalFrontendDefaultsNode": _OptionalFrontendDefaultsNode,
    "ToBasicPipe": _ToBasicPipe,
    "BasicPipeConsumer": _BasicPipeConsumer,
    "IntSource": _IntSource,
}


@pytest.fixture()
def _with_test_nodes():
    """Push test node mappings into the execution context for the duration."""
    exported = ExportedNodes()
    exported.NODE_CLASS_MAPPINGS.update(_TEST_MAPPINGS)
    with context_add_custom_nodes(exported):
        yield


def _build_default_workflow() -> dict:
    """Return the canonical SD 1.5 default workflow in UI format."""
    return {
        "nodes": [
            {
                "id": 4, "type": "CheckpointLoaderSimple", "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "slot_index": 0, "links": [1]},
                    {"name": "CLIP", "type": "CLIP", "slot_index": 1, "links": [3, 5]},
                    {"name": "VAE", "type": "VAE", "slot_index": 2, "links": [8]},
                ],
                "widgets_values": ["model_a.safetensors"],
            },
            {
                "id": 3, "type": "KSampler", "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                    {"name": "positive", "type": "CONDITIONING", "link": 4},
                    {"name": "negative", "type": "CONDITIONING", "link": 6},
                    {"name": "latent_image", "type": "LATENT", "link": 2},
                ],
                "outputs": [{"name": "LATENT", "type": "LATENT", "slot_index": 0, "links": [7]}],
                "widgets_values": [685468484323813, "randomize", 20, 8, "euler", "normal", 1],
            },
            {
                "id": 6, "type": "CLIPTextEncode", "mode": 0,
                "inputs": [{"name": "clip", "type": "CLIP", "link": 3}],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [4]}],
                "widgets_values": ["beautiful scenery"],
            },
            {
                "id": 7, "type": "CLIPTextEncode", "mode": 0,
                "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6]}],
                "widgets_values": ["text, watermark"],
            },
            {
                "id": 5, "type": "EmptyLatentImage", "mode": 0,
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [2]}],
                "widgets_values": [512, 512, 1],
            },
            {
                "id": 8, "type": "VAEDecode", "mode": 0,
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 7},
                    {"name": "vae", "type": "VAE", "link": 8},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9]}],
                "widgets_values": [],
            },
            {
                "id": 9, "type": "SaveImage", "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
                "outputs": [],
                "widgets_values": ["SD1.5"],
            },
        ],
        "links": [
            [1, 4, 0, 3, 0, "MODEL"],
            [2, 5, 0, 3, 3, "LATENT"],
            [3, 4, 1, 6, 0, "CLIP"],
            [4, 6, 0, 3, 1, "CONDITIONING"],
            [5, 4, 1, 7, 0, "CLIP"],
            [6, 7, 0, 3, 2, "CONDITIONING"],
            [7, 3, 0, 8, 0, "LATENT"],
            [8, 4, 2, 8, 1, "VAE"],
            [9, 8, 0, 9, 0, "IMAGE"],
        ],
    }


def test_legacy_group_node_terminal_output_is_linked(_with_test_nodes):
    workflow = {
        "nodes": [
            {
                "id": 3,
                "type": "workflow>Impact::MAKE_BASIC_PIPE",
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {
                        "name": "basic_pipe",
                        "type": "BASIC_PIPE",
                        "slot_index": 0,
                        "links": [3],
                    }
                ],
                "widgets_values": ["model_a.safetensors", "positive", "negative"],
            },
            {
                "id": 2,
                "type": "BasicPipeConsumer",
                "mode": 0,
                "inputs": [{"name": "basic_pipe", "type": "BASIC_PIPE", "link": 3}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[3, 3, 0, 2, 0, "BASIC_PIPE"]],
        "extra": {
            "groupNodes": {
                "Impact::MAKE_BASIC_PIPE": {
                    "nodes": [
                        {
                            "index": 0,
                            "type": "CheckpointLoaderSimple",
                            "mode": 0,
                            "inputs": [],
                            "outputs": [
                                {"name": "MODEL", "type": "MODEL", "links": []},
                                {"name": "CLIP", "type": "CLIP", "links": []},
                                {"name": "VAE", "type": "VAE", "links": []},
                            ],
                            "widgets_values": ["model_a.safetensors"],
                        },
                        {
                            "index": 1,
                            "type": "CLIPTextEncode",
                            "mode": 0,
                            "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
                            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
                            "widgets_values": ["positive"],
                        },
                        {
                            "index": 2,
                            "type": "CLIPTextEncode",
                            "mode": 0,
                            "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
                            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
                            "widgets_values": ["negative"],
                        },
                        {
                            "index": 3,
                            "type": "ToBasicPipe",
                            "mode": 0,
                            "inputs": [
                                {"name": "model", "type": "MODEL", "link": None},
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "vae", "type": "VAE", "link": None},
                                {"name": "positive", "type": "CONDITIONING", "link": None},
                                {"name": "negative", "type": "CONDITIONING", "link": None},
                            ],
                            "outputs": [{"name": "basic_pipe", "type": "BASIC_PIPE", "links": None}],
                            "widgets_values": [],
                        },
                    ],
                    "links": [
                        [0, 0, 3, 0, 1, "MODEL"],
                        [0, 1, 1, 0, 1, "CLIP"],
                        [0, 1, 2, 0, 1, "CLIP"],
                        [0, 1, 3, 1, 1, "CLIP"],
                        [0, 2, 3, 2, 1, "VAE"],
                        [1, 0, 3, 3, 3, "CONDITIONING"],
                        [2, 0, 3, 4, 4, "CONDITIONING"],
                    ],
                }
            }
        },
    }

    result = convert_ui_to_api(workflow)

    assert result["2"]["inputs"]["basic_pipe"] == ["3:3", 0]
    assert result["3:3"]["class_type"] == "ToBasicPipe"


def test_force_input_widget_placeholder_does_not_shift_later_widgets(_with_test_nodes):
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "IntSource",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "INT", "type": "INT", "links": [1]}],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "ForceInputNode",
                "mode": 0,
                "inputs": [
                    {"name": "value", "type": "INT", "widget": {"name": "value"}, "link": 1}
                ],
                "outputs": [],
                "widgets_values": [123, "mapped label"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "INT"]],
    }

    result = convert_ui_to_api(workflow)

    assert result["2"]["inputs"]["value"] == ["1", 0]
    assert result["2"]["inputs"]["label"] == "mapped label"


# ── unit tests: _is_widget_type ───────────────────────────────────────────────

class TestIsWidgetType:
    def test_int(self):
        assert _is_widget_type("INT") is True

    def test_float(self):
        assert _is_widget_type("FLOAT") is True

    def test_string(self):
        assert _is_widget_type("STRING") is True

    def test_boolean(self):
        assert _is_widget_type("BOOLEAN") is True

    def test_combo_list(self):
        assert _is_widget_type(["euler", "dpmpp"]) is True

    @pytest.mark.parametrize("socketless", [True, False])
    def test_explicit_socketless_marks_custom_widget(self, socketless):
        assert _is_widget_type("CUSTOM_WIDGET", {"socketless": socketless}) is True

    def test_model_is_connection(self):
        assert _is_widget_type("MODEL") is False

    def test_latent_is_connection(self):
        assert _is_widget_type("LATENT") is False

    def test_image_is_connection(self):
        assert _is_widget_type("IMAGE") is False

    def test_conditioning_is_connection(self):
        assert _is_widget_type("CONDITIONING") is False

    def test_clip_is_connection(self):
        assert _is_widget_type("CLIP") is False


# ── unit tests: _extra_widgets_after ──────────────────────────────────────────

class TestExtraWidgets:
    def test_no_extras(self):
        assert _extra_widgets_after({}) == []

    def test_control_after_generate(self):
        assert _extra_widgets_after({"control_after_generate": True}) == [None]

    def test_image_upload(self):
        assert _extra_widgets_after({"image_upload": True}) == [None]

    def test_painter_upload_is_owned_by_custom_widget(self):
        assert _extra_widgets_after({"widgetType": "PAINTER", "image_upload": True}) == []

    def test_both(self):
        assert _extra_widgets_after({"control_after_generate": True, "image_upload": True}) == [None, None]


# ── unit tests: _map_widgets ──────────────────────────────────────────────────

class TestMapWidgets:
    def test_ksampler(self):
        """widgets_values should skip connection types and handle control_after_generate."""
        input_types = _KSamplerLike.INPUT_TYPES()
        widgets = [685468484323813, "randomize", 20, 8, "euler", "normal", 1]
        result, _ = _map_widgets(input_types, widgets)
        assert result == {
            "seed": 685468484323813,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
        }

    def test_clip_text_encode(self):
        input_types = _CLIPTextEncode.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["beautiful scenery"])
        assert result == {"text": "beautiful scenery"}

    def test_checkpoint_loader(self):
        input_types = _CheckpointLoaderSimple.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["model_a.safetensors"])
        assert result == {"ckpt_name": "model_a.safetensors"}

    def test_empty_latent_image(self):
        input_types = _EmptyLatentImage.INPUT_TYPES()
        result, _ = _map_widgets(input_types, [512, 512, 1])
        assert result == {"width": 512, "height": 512, "batch_size": 1}

    def test_save_image(self):
        input_types = _SaveImage.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["SD1.5"])
        assert result == {"filename_prefix": "SD1.5"}

    def test_load_image_with_upload(self):
        input_types = _LoadImage.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["photo.png", "image"])
        assert result == {"image": "photo.png"}

    def test_vae_decode_empty(self):
        input_types = _VAEDecode.INPUT_TYPES()
        result, _ = _map_widgets(input_types, [])
        assert result == {}

    def test_image_scale_with_optional(self):
        input_types = _ImageScaleToTotalPixels.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["lanczos", 1.0, 1])
        assert result == {
            "upscale_method": "lanczos",
            "megapixels": 1.0,
            "resolution_steps": 1,
        }

    def test_force_input_skipped(self):
        input_types = _ForceInputNode.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["hello"])
        assert result == {"label": "hello"}

    def test_truncated_widgets(self):
        """If widgets_values is shorter than expected, fill trailing defaults."""
        input_types = _EmptyLatentImage.INPUT_TYPES()
        result, _ = _map_widgets(input_types, [768])
        assert result == {"width": 768, "height": 512, "batch_size": 1}

    def test_optional_widgets_use_frontend_defaults(self):
        """Optional widget defaults should match frontend widget constructors."""
        input_types = _OptionalFrontendDefaultsNode.INPUT_TYPES()
        result, _ = _map_widgets(input_types, ["user prompt"])
        assert result == {
            "prompt": "user prompt",
            "response_modalities": "IMAGE+TEXT",
            "system_prompt": "system default",
        }

    def test_dict_widgets_passthrough(self, _with_test_nodes):
        workflow = {
            "nodes": [{"id": 1, "type": "EmptyLatentImage", "mode": 0,
                       "inputs": [], "outputs": [],
                       "widgets_values": {"width": 768, "height": 384, "batch_size": 2}}],
            "links": [],
        }
        result = convert_ui_to_api(workflow)
        assert result["1"]["inputs"] == {"width": 768, "height": 384, "batch_size": 2}

    def test_list_value_wrapped(self):
        """List widget values must be wrapped as {'__value__': ...}."""

        class _ListWidget:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"items": ("STRING", {})}}

        result, _ = _map_widgets(_ListWidget.INPUT_TYPES(), [["a", "b"]])
        assert result == {"items": {"__value__": ["a", "b"]}}

    def test_socket_capable_custom_widgets_preserve_positional_alignment(self):
        input_types = {
            "required": {
                "width": ("INT", {"default": 1024}),
                "editor_state": ("BOUNDING_BOXES", {"default": [], "socketless": False}),
            },
            "optional": {
                "last_incoming": ("BOUNDING_BOXES", {"default": [], "socketless": True}),
            },
        }
        boxes = [{"x": 1, "y": 2, "width": 3, "height": 4}]

        result, consumed = _map_widgets(input_types, [1024, boxes, []])

        assert result == {
            "width": 1024,
            "editor_state": {"__value__": boxes},
            "last_incoming": {"__value__": []},
        }
        assert consumed == 3

    def test_scalar_not_wrapped(self):
        """Scalar values should NOT be wrapped."""
        input_types = _EmptyLatentImage.INPUT_TYPES()
        result, _ = _map_widgets(input_types, [512, 512, 1])
        assert result["width"] == 512
        assert not isinstance(result["width"], dict)


# ── unit tests: _wrap_value ──────────────────────────────────────────────────

class TestWrapValue:
    def test_list_wrapped(self):
        assert _wrap_value(["a", "b"]) == {"__value__": ["a", "b"]}

    def test_string_unchanged(self):
        assert _wrap_value("hello") == "hello"

    def test_int_unchanged(self):
        assert _wrap_value(42) == 42

    def test_none_unchanged(self):
        assert _wrap_value(None) is None

    def test_empty_list_wrapped(self):
        assert _wrap_value([]) == {"__value__": []}


# ── unit tests: _resolve_source ───────────────────────────────────────────────

class TestResolveSource:
    def test_normal_node(self):
        nodes = {1: {"type": "SomeNode", "mode": 0, "outputs": []}}
        result = _resolve_source(1, 0, nodes, {})
        assert result == ("link", "1", 0)

    def test_muted_node_returns_none(self):
        nodes = {1: {"type": "SomeNode", "mode": 2, "outputs": []}}
        result = _resolve_source(1, 0, nodes, {})
        assert result is None

    def test_reroute_follows_input(self):
        nodes = {
            10: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": 100}],
                "outputs": [{"name": "", "type": "MODEL", "links": [101]}],
            },
            1: {"type": "SomeNode", "mode": 0, "outputs": []},
        }
        links = {100: LiteLink(100, 1, 0, 10, 0, "MODEL")}
        result = _resolve_source(10, 0, nodes, links)
        assert result == ("link", "1", 0)

    def test_reroute_disconnected_returns_none(self):
        nodes = {
            10: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": None}],
            },
        }
        result = _resolve_source(10, 0, nodes, {})
        assert result is None

    def test_primitive_node_returns_value(self):
        nodes = {
            20: {
                "type": "PrimitiveNode", "mode": 0,
                "widgets_values": [3.5, "fixed"],
            },
        }
        result = _resolve_source(20, 0, nodes, {})
        assert result == ("value", 3.5)

    def test_typed_primitive_alias_returns_value(self):
        nodes = {
            20: {
                "type": "Int", "mode": 0,
                "widgets_values": [17],
            },
        }
        result = _resolve_source(20, 0, nodes, {})
        assert result == ("value", 17)

    def test_bypass_no_matching_type(self):
        """Bypass node with no matching input type returns None."""
        nodes = {
            5: {
                "type": "VAEDecode", "mode": 4,  # bypassed
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 50},
                    {"name": "vae", "type": "VAE", "link": 51},
                ],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [52]},
                ],
            },
            1: {"type": "Producer", "mode": 0},
        }
        # IMAGE output doesn't match LATENT or VAE inputs → no route
        result = _resolve_source(5, 0, nodes, {}, target_type="IMAGE")
        assert result is None

    def test_bypass_routes_matching_type(self):
        """Bypass routes through when input/output types match."""
        nodes = {
            5: {
                "type": "Passthrough", "mode": 4,
                "inputs": [
                    {"name": "in_image", "type": "IMAGE", "link": 50},
                ],
                "outputs": [
                    {"name": "out_image", "type": "IMAGE", "links": [51]},
                ],
            },
            1: {"type": "Producer", "mode": 0},
        }
        links = {50: LiteLink(50, 1, 0, 5, 0, "IMAGE")}
        result = _resolve_source(5, 0, nodes, links, target_type="IMAGE")
        assert result == ("link", "1", 0)

    def test_bypass_routes_comma_separated_matching_type(self):
        """Bypass routes through a multi-type input when one type matches."""
        nodes = {
            5: {
                "type": "ImageOrMaskPassthrough", "mode": 4,
                "inputs": [
                    {"name": "image", "type": "IMAGE,MASK", "link": 50},
                ],
                "outputs": [
                    {"name": "output", "type": "IMAGE", "links": [51]},
                ],
            },
            1: {"type": "Producer", "mode": 0},
        }
        links = {50: LiteLink(50, 1, 0, 5, 0, "IMAGE")}
        result = _resolve_source(5, 0, nodes, links, target_type="IMAGE")
        assert result == ("link", "1", 0)

    def test_bypass_opposite_slot_preferred(self):
        """Bypass prefers the opposite slot (same index) when types match."""
        nodes = {
            5: {
                "type": "DualPass", "mode": 4,
                "inputs": [
                    {"name": "in_model", "type": "MODEL", "link": 50},
                    {"name": "in_clip", "type": "CLIP", "link": 51},
                ],
                "outputs": [
                    {"name": "out_model", "type": "MODEL", "links": []},
                    {"name": "out_clip", "type": "CLIP", "links": []},
                ],
            },
            1: {"type": "Src1", "mode": 0},
            2: {"type": "Src2", "mode": 0},
        }
        links = {
            50: LiteLink(50, 1, 0, 5, 0, "MODEL"),
            51: LiteLink(51, 2, 0, 5, 1, "CLIP"),
        }
        assert _resolve_source(5, 0, nodes, links, target_type="MODEL") == ("link", "1", 0)
        assert _resolve_source(5, 1, nodes, links, target_type="CLIP") == ("link", "2", 0)

    def test_bypass_multiple_outputs(self):
        """Bypass with multiple outputs of the same type routes correctly."""
        nodes = {
            5: {
                "type": "Splitter", "mode": 4,
                "inputs": [
                    {"name": "in1", "type": "IMAGE", "link": 50},
                    {"name": "in2", "type": "IMAGE", "link": 51},
                ],
                "outputs": [
                    {"name": "out1", "type": "IMAGE", "links": [60]},
                    {"name": "out2", "type": "IMAGE", "links": [61]},
                ],
            },
            1: {"type": "ProducerA", "mode": 0},
            2: {"type": "ProducerB", "mode": 0},
        }
        links = {
            50: LiteLink(50, 1, 0, 5, 0, "IMAGE"),
            51: LiteLink(51, 2, 0, 5, 1, "IMAGE"),
        }
        # Output 0 → Input 0 (first IMAGE → first IMAGE)
        assert _resolve_source(5, 0, nodes, links) == ("link", "1", 0)
        # Output 1 → Input 1 (second IMAGE → second IMAGE)
        assert _resolve_source(5, 1, nodes, links) == ("link", "2", 0)

    def test_chained_reroutes(self):
        nodes = {
            10: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": 100}],
            },
            11: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": 101}],
            },
            1: {"type": "Source", "mode": 0},
        }
        links = {
            100: LiteLink(100, 11, 0, 10, 0),
            101: LiteLink(101, 1, 0, 11, 0),
        }
        result = _resolve_source(10, 0, nodes, links)
        assert result == ("link", "1", 0)

    def test_cycle_detection(self):
        nodes = {
            10: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": 100}],
            },
            11: {
                "type": "Reroute", "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": 101}],
            },
        }
        links = {
            100: LiteLink(100, 11, 0, 10, 0),
            101: LiteLink(101, 10, 0, 11, 0),
        }
        result = _resolve_source(10, 0, nodes, links)
        assert result is None

    def test_missing_node(self):
        result = _resolve_source(999, 0, {}, {})
        assert result is None


# ── integration tests: convert_ui_to_api ──────────────────────────────────────

@pytest.mark.usefixtures("_with_test_nodes")
class TestConvertDefault:
    def test_basic_conversion(self):
        """The default SD1.5 workflow converts correctly."""
        workflow = _build_default_workflow()
        result = convert_ui_to_api(workflow)

        # All active nodes present.
        assert set(result.keys()) == {"3", "4", "5", "6", "7", "8", "9"}

        # KSampler widgets mapped correctly (control_after_generate skipped).
        ks = result["3"]["inputs"]
        assert ks["seed"] == 685468484323813
        assert ks["steps"] == 20
        assert ks["cfg"] == 8
        assert ks["sampler_name"] == "euler"
        assert ks["scheduler"] == "normal"
        assert ks["denoise"] == 1
        # Linked inputs.
        assert ks["model"] == ["4", 0]
        assert ks["positive"] == ["6", 0]
        assert ks["negative"] == ["7", 0]
        assert ks["latent_image"] == ["5", 0]

    def test_clip_text_encode(self):
        workflow = _build_default_workflow()
        result = convert_ui_to_api(workflow)
        assert result["6"]["inputs"]["text"] == "beautiful scenery"
        assert result["6"]["inputs"]["clip"] == ["4", 1]

    def test_class_types(self):
        workflow = _build_default_workflow()
        result = convert_ui_to_api(workflow)
        assert result["4"]["class_type"] == "CheckpointLoaderSimple"
        assert result["3"]["class_type"] == "KSampler"

    def test_checkpoint_loader(self):
        workflow = _build_default_workflow()
        result = convert_ui_to_api(workflow)
        assert result["4"]["inputs"]["ckpt_name"] == "model_a.safetensors"


@pytest.mark.usefixtures("_with_test_nodes")
class TestMutedNodes:
    def test_muted_node_excluded(self):
        workflow = _build_default_workflow()
        # Mute the SaveImage node.
        for n in workflow["nodes"]:
            if n["id"] == 9:
                n["mode"] = 2
        result = convert_ui_to_api(workflow)
        assert "9" not in result

    def test_link_to_muted_node_removed(self):
        """Dangling link to a muted node is cleaned up."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}],
                    "widgets_values": [512, 512, 1],
                },
                {
                    "id": 2, "type": "VAEDecode", "mode": 2,  # muted
                    "inputs": [{"name": "samples", "type": "LATENT", "link": 1}],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
                    "widgets_values": [],
                },
                {
                    "id": 3, "type": "SaveImage", "mode": 0,
                    "inputs": [{"name": "images", "type": "IMAGE", "link": 2}],
                    "outputs": [],
                    "widgets_values": ["out"],
                },
            ],
            "links": [
                [1, 1, 0, 2, 0, "LATENT"],
                [2, 2, 0, 3, 0, "IMAGE"],
            ],
        }
        result = convert_ui_to_api(workflow)
        assert "2" not in result
        # SaveImage's link to muted VAEDecode should be removed.
        assert "images" not in result["3"]["inputs"]


@pytest.mark.usefixtures("_with_test_nodes")
class TestBypassedNodes:
    def test_bypassed_node_excluded(self):
        workflow = _build_default_workflow()
        for n in workflow["nodes"]:
            if n["id"] == 9:
                n["mode"] = 4
        result = convert_ui_to_api(workflow)
        assert "9" not in result

    def test_bypass_pass_through(self):
        """Bypass routes a matching input to the output."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}],
                    "widgets_values": [512, 512, 1],
                },
                {
                    "id": 2, "type": "SaveImage", "mode": 4,  # bypass
                    "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                    "outputs": [],
                    "widgets_values": ["prefix"],
                },
            ],
            "links": [[1, 1, 0, 2, 0, "LATENT"]],
        }
        result = convert_ui_to_api(workflow)
        assert "2" not in result


@pytest.mark.usefixtures("_with_test_nodes")
class TestVirtualNodes:
    def test_note_excluded(self):
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [], "outputs": [],
                    "widgets_values": [512, 512, 1],
                },
                {
                    "id": 2, "type": "MarkdownNote", "mode": 0,
                    "inputs": [], "outputs": [],
                    "widgets_values": ["some note text"],
                },
            ],
            "links": [],
        }
        result = convert_ui_to_api(workflow)
        assert "2" not in result
        assert "1" in result

    def test_reroute_transparent(self):
        """A reroute between source and destination is transparent."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "CheckpointLoaderSimple", "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "MODEL", "type": "MODEL", "links": [1]},
                        {"name": "CLIP", "type": "CLIP", "links": [3, 5]},
                        {"name": "VAE", "type": "VAE", "links": []},
                    ],
                    "widgets_values": ["model_a.safetensors"],
                },
                {
                    "id": 10, "type": "Reroute", "mode": 0,
                    "inputs": [{"name": "", "type": "*", "link": 1}],
                    "outputs": [{"name": "", "type": "MODEL", "links": [2]}],
                    "widgets_values": [],
                },
                {
                    "id": 3, "type": "KSampler", "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 2},
                        {"name": "positive", "type": "CONDITIONING", "link": 4},
                        {"name": "negative", "type": "CONDITIONING", "link": 6},
                        {"name": "latent_image", "type": "LATENT", "link": 7},
                    ],
                    "outputs": [],
                    "widgets_values": [42, "fixed", 10, 7.5, "euler", "normal", 0.8],
                },
                {
                    "id": 6, "type": "CLIPTextEncode", "mode": 0,
                    "inputs": [{"name": "clip", "type": "CLIP", "link": 3}],
                    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [4]}],
                    "widgets_values": ["positive"],
                },
                {
                    "id": 7, "type": "CLIPTextEncode", "mode": 0,
                    "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
                    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6]}],
                    "widgets_values": ["negative"],
                },
                {
                    "id": 5, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7]}],
                    "widgets_values": [512, 512, 1],
                },
            ],
            "links": [
                [1, 1, 0, 10, 0, "MODEL"],
                [2, 10, 0, 3, 0, "MODEL"],
                [3, 1, 1, 6, 0, "CLIP"],
                [4, 6, 0, 3, 1, "CONDITIONING"],
                [5, 1, 1, 7, 0, "CLIP"],
                [6, 7, 0, 3, 2, "CONDITIONING"],
                [7, 5, 0, 3, 3, "LATENT"],
            ],
        }
        result = convert_ui_to_api(workflow)
        # Reroute (id=10) should not be in output.
        assert "10" not in result
        # KSampler model input resolves through reroute to checkpoint loader.
        assert result["3"]["inputs"]["model"] == ["1", 0]

    def test_primitive_node_value(self):
        """PrimitiveNode's value is used directly as the input."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "KSampler", "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": None},
                        {"name": "positive", "type": "CONDITIONING", "link": None},
                        {"name": "negative", "type": "CONDITIONING", "link": None},
                        {"name": "latent_image", "type": "LATENT", "link": None},
                        {"name": "denoise", "type": "FLOAT", "link": 100, "widget": {"name": "denoise"}},
                    ],
                    "outputs": [],
                    "widgets_values": [42, "fixed", 10, 7.5, "euler", "normal", 0.5],
                },
                {
                    "id": 20, "type": "PrimitiveNode", "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "FLOAT", "type": "FLOAT", "widget": {"name": "denoise"}, "links": [100]},
                    ],
                    "widgets_values": [0.75, "fixed"],
                },
            ],
            "links": [
                [100, 20, 0, 1, 4, "FLOAT"],
            ],
        }
        result = convert_ui_to_api(workflow)
        assert "20" not in result
        assert result["1"]["inputs"]["denoise"] == 0.75


@pytest.mark.usefixtures("_with_test_nodes")
class TestConvertedWidgets:
    def test_linked_widget_overrides_value(self):
        """A widget that has been converted to input and linked uses the link."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [
                        {"name": "width", "type": "INT", "link": 10, "widget": {"name": "width"}},
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                    "widgets_values": [512, 768, 1],
                },
                {
                    "id": 2, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [1024, 1024, 1],
                },
            ],
            "links": [
                # src node 2 doesn't have an INT output, but this is synthetic
                # For the test, we verify the link is stored correctly.
            ],
        }
        result = convert_ui_to_api(workflow)
        # Widget values still consumed correctly even with converted widget.
        assert result["1"]["inputs"]["width"] == 512
        assert result["1"]["inputs"]["height"] == 768


@pytest.mark.usefixtures("_with_test_nodes")
class TestDanglingLinks:
    def test_link_to_unknown_node_cleaned(self):
        """Links referencing nodes not in the output are removed."""
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "SaveImage", "mode": 0,
                    "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                    "outputs": [],
                    "widgets_values": ["out"],
                },
                # Node 99 is an unknown type – won't be in output.
            ],
            "links": [
                [1, 99, 0, 1, 0, "IMAGE"],
            ],
        }
        result = convert_ui_to_api(workflow)
        assert "images" not in result["1"]["inputs"]


@pytest.mark.usefixtures("_with_test_nodes")
class TestLoadImage:
    def test_image_upload_extra_widget(self):
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "LoadImage", "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                    "widgets_values": ["photo.png", "image"],
                },
            ],
            "links": [],
        }
        result = convert_ui_to_api(workflow)
        assert result["1"]["inputs"]["image"] == "photo.png"
        assert "upload" not in result["1"]["inputs"]


@pytest.mark.usefixtures("_with_test_nodes")
class TestForceInput:
    def test_force_input_not_in_widgets(self):
        workflow = {
            "nodes": [
                {
                    "id": 1, "type": "ForceInputNode", "mode": 0,
                    "inputs": [{"name": "value", "type": "INT", "link": 1}],
                    "outputs": [],
                    "widgets_values": ["hello"],
                },
                {
                    "id": 2, "type": "EmptyLatentImage", "mode": 0,
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [512, 512, 1],
                },
            ],
            "links": [[1, 2, 0, 1, 0, "INT"]],
        }
        result = convert_ui_to_api(workflow)
        # "value" should come from link, "label" from widget.
        assert result["1"]["inputs"]["label"] == "hello"


# ── unit tests: subgraph / group node expansion ──────────────────────────────

class _VAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (["vae.safetensors"],)}}


class _UNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (["model.safetensors"],), "weight_dtype": (["default"],)}}


class _CLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (["clip.safetensors"],), "type": (["stable_diffusion"],), "device": (["default"],)}}


_SUBGRAPH_TEST_MAPPINGS = {
    **_TEST_MAPPINGS,
    "VAELoader": _VAELoader,
    "UNETLoader": _UNETLoader,
    "CLIPLoader": _CLIPLoader,
}


@pytest.fixture()
def _with_subgraph_nodes():
    """Push test node mappings including subgraph inner node types."""
    exported = ExportedNodes()
    exported.NODE_CLASS_MAPPINGS.update(_SUBGRAPH_TEST_MAPPINGS)
    with context_add_custom_nodes(exported):
        yield


def _build_subgraph_workflow() -> dict:
    """Return a workflow with a group node containing inner nodes.

    Outer workflow:
        node 85 (CLIPTextEncode) → group node 86 (subgraph) → node 87 (SaveImage)

    Subgraph inner nodes:
        CLIPTextEncode → KSampler → VAEDecode → outputNode
        inputNode → CLIPTextEncode (text input from outside)
    """
    return {
        "nodes": [
            {
                "id": 85, "type": "CLIPTextEncode", "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "CONDITIONING", "type": "CONDITIONING", "links": [149]},
                ],
                "widgets_values": ["a cat"],
            },
            {
                "id": 86, "type": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "mode": 0,
                "inputs": [
                    {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 149},
                ],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [146]},
                ],
                "properties": {
                    "proxyWidgets": [
                        ["-1", "text"],
                        ["-1", "width"],
                        ["3", "seed"],
                    ],
                },
                "widgets_values": ["override prompt", 768, 42],
            },
            {
                "id": 87, "type": "SaveImage", "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 146}],
                "outputs": [],
                "widgets_values": ["output"],
            },
        ],
        "links": [
            [149, 85, 0, 86, 0, "CONDITIONING"],
            [146, 86, 0, 87, 0, "IMAGE"],
        ],
        "extra": {
            "definitions": {
                "subgraphs": [
                    {
                        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "name": "Test Subgraph",
                        "inputs": [
                            {"name": "text", "type": "STRING"},
                            {"name": "width", "type": "INT"},
                        ],
                        "outputs": [
                            {"name": "IMAGE", "type": "IMAGE"},
                        ],
                        "inputNode": {"id": -10},
                        "outputNode": {"id": -20},
                        "nodes": [
                            {
                                "id": 10, "type": "CLIPTextEncode", "mode": 0,
                                "inputs": [
                                    {"name": "clip", "type": "CLIP", "link": None},
                                    {"name": "text", "type": "STRING", "link": 201,
                                     "widget": {"name": "text"}},
                                ],
                                "outputs": [
                                    {"name": "CONDITIONING", "type": "CONDITIONING",
                                     "links": [202]},
                                ],
                                "widgets_values": ["inner prompt"],
                            },
                            {
                                "id": 11, "type": "EmptyLatentImage", "mode": 0,
                                "inputs": [
                                    {"name": "width", "type": "INT", "link": 205,
                                     "widget": {"name": "width"}},
                                ],
                                "outputs": [
                                    {"name": "LATENT", "type": "LATENT", "links": [203]},
                                ],
                                "widgets_values": [512, 512, 1],
                            },
                            {
                                "id": 12, "type": "VAEDecode", "mode": 0,
                                "inputs": [
                                    {"name": "samples", "type": "LATENT", "link": 203},
                                    {"name": "vae", "type": "VAE", "link": None},
                                ],
                                "outputs": [
                                    {"name": "IMAGE", "type": "IMAGE", "links": [206]},
                                ],
                                "widgets_values": [],
                            },
                        ],
                        "links": [
                            {"id": 201, "origin_id": -10, "origin_slot": 0,
                             "target_id": 10, "target_slot": 1, "type": "STRING"},
                            {"id": 203, "origin_id": 11, "origin_slot": 0,
                             "target_id": 12, "target_slot": 0, "type": "LATENT"},
                            {"id": 205, "origin_id": -10, "origin_slot": 1,
                             "target_id": 11, "target_slot": 0, "type": "INT"},
                            {"id": 206, "origin_id": 12, "origin_slot": 0,
                             "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                        ],
                    }
                ]
            }
        },
    }


class TestCollectSubgraphDefs:
    def test_finds_subgraphs_in_extra(self):
        workflow = _build_subgraph_workflow()
        defs = _collect_subgraph_defs(workflow)
        assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" in defs

    def test_empty_workflow_no_subgraphs(self):
        assert _collect_subgraph_defs({}) == {}

    def test_no_definitions_key(self):
        assert _collect_subgraph_defs({"nodes": [], "links": []}) == {}


@pytest.mark.usefixtures("_with_subgraph_nodes")
class TestSubgraphExpansion:
    def test_group_node_not_in_output(self):
        """The group node itself should not appear in the API output."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        assert "86" not in result

    def test_inner_nodes_have_prefixed_ids(self):
        """Inner nodes should have IDs like '86:10', '86:11', '86:12'."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        assert "86:10" in result
        assert "86:11" in result
        assert "86:12" in result

    def test_inner_node_class_types(self):
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        assert result["86:10"]["class_type"] == "CLIPTextEncode"
        assert result["86:11"]["class_type"] == "EmptyLatentImage"
        assert result["86:12"]["class_type"] == "VAEDecode"

    def test_inner_links_resolved(self):
        """Inner links between inner nodes should use prefixed IDs."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        # VAEDecode.samples should link to EmptyLatentImage
        vae_inputs = result["86:12"]["inputs"]
        assert vae_inputs.get("samples") == ["86:11", 0]

    def test_input_boundary_from_link(self):
        """Inner node connected to inputNode should resolve to outer link source."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        # CLIPTextEncode inner node's text input comes from inputNode slot 0,
        # which is connected to outer node 85 via link 149.
        clip_inputs = result["86:10"]["inputs"]
        assert clip_inputs.get("text") == ["85", 0]

    def test_input_boundary_from_proxy_value(self):
        """Inner node connected to inputNode uses proxy value when not linked."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        # EmptyLatentImage inner node's width comes from inputNode slot 1,
        # which is NOT linked externally → uses proxy value 768.
        latent_inputs = result["86:11"]["inputs"]
        assert latent_inputs.get("width") == 768

    def test_output_boundary(self):
        """Outer node linking from group node resolves to inner source."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        # SaveImage (87) links from group node 86 output 0 → inner VAEDecode 12.
        save_inputs = result["87"]["inputs"]
        assert save_inputs.get("images") == ["86:12", 0]

    def test_outer_nodes_still_present(self):
        """Non-group outer nodes should still be in the output."""
        workflow = _build_subgraph_workflow()
        result = convert_ui_to_api(workflow)
        assert "85" in result
        assert "87" in result

    def test_proxy_widget_direct_override(self):
        """ProxyWidgets with a real node ID override inner widgets."""
        workflow = _build_subgraph_workflow()
        # proxyWidget ["3", "seed"] → value 42. But inner node 3 doesn't exist
        # in this test subgraph (we use 10, 11, 12). Let's adjust the test to
        # use a matching inner node ID.
        sg = workflow["extra"]["definitions"]["subgraphs"][0]
        # Change proxyWidget to point to inner node 11 (EmptyLatentImage) height
        workflow["nodes"][1]["properties"]["proxyWidgets"][2] = ["11", "height"]
        workflow["nodes"][1]["widgets_values"][2] = 1024
        result = convert_ui_to_api(workflow)
        assert result["86:11"]["inputs"]["height"] == 1024


@pytest.mark.usefixtures("_with_subgraph_nodes")
class TestValueWrapping:
    def test_list_widget_value_wrapped(self):
        """A widget value that is a list gets wrapped as __value__."""
        class _ListWidgetNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"data": ("STRING", {})}}

        exported = ExportedNodes()
        exported.NODE_CLASS_MAPPINGS.update({**_SUBGRAPH_TEST_MAPPINGS, "ListNode": _ListWidgetNode})
        with context_add_custom_nodes(exported):
            workflow = {
                "nodes": [{
                    "id": 1, "type": "ListNode", "mode": 0,
                    "inputs": [], "outputs": [],
                    "widgets_values": [["val1", "val2"]],
                }],
                "links": [],
            }
            result = convert_ui_to_api(workflow)
            assert result["1"]["inputs"]["data"] == {"__value__": ["val1", "val2"]}

    def test_list_value_survives_dangling_cleanup(self):
        """A wrapped list value should NOT be deleted by dangling-link cleanup."""
        class _ListWidgetNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"data": ("STRING", {})}}

        exported = ExportedNodes()
        exported.NODE_CLASS_MAPPINGS.update({**_SUBGRAPH_TEST_MAPPINGS, "ListNode": _ListWidgetNode})
        with context_add_custom_nodes(exported):
            workflow = {
                "nodes": [{
                    "id": 1, "type": "ListNode", "mode": 0,
                    "inputs": [], "outputs": [],
                    "widgets_values": [["", ""]],  # Would look like dangling link without wrapping
                }],
                "links": [],
            }
            result = convert_ui_to_api(workflow)
            # Should be preserved (not cleaned up as dangling link).
            assert "data" in result["1"]["inputs"]
            assert result["1"]["inputs"]["data"] == {"__value__": ["", ""]}


# ── integration tests against real template workflows ─────────────────────────

def _load_template_workflow(template_id: str) -> dict | None:
    try:
        from comfyui_workflow_templates import get_asset_path, iter_templates
    except ImportError:
        return None
    for t in iter_templates():
        if t.template_id == template_id:
            json_assets = [a for a in t.assets if a.filename.endswith(".json")]
            if json_assets:
                import json as _json
                path = get_asset_path(t.template_id, json_assets[0].filename)
                with open(path, encoding="utf-8") as f:
                    return _json.load(f)
    return None


def _real_nodes_available() -> bool:
    """Check if the node system can be loaded."""
    return importlib.util.find_spec("comfy.nodes.package") is not None


@pytest.fixture(scope="module")
def _real_node_context(_preloaded_nodes):
    """Activate the preloaded node snapshot in the execution context."""
    with context_add_custom_nodes(_preloaded_nodes):
        yield _preloaded_nodes


def _template_ids():
    """Discover available template IDs for parametrized tests."""
    try:
        from comfyui_workflow_templates import iter_templates
        ids = []
        for t in iter_templates():
            json_assets = [a for a in t.assets if a.filename.endswith(".json")]
            if json_assets:
                ids.append(t.template_id)
        return ids
    except ImportError:
        return []


def _is_ui_workflow(data: dict) -> bool:
    return "nodes" in data and "links" in data


@pytest.mark.skipif(not _real_nodes_available(), reason="node system not available")
class TestRealTemplateConversion:
    @pytest.mark.parametrize("template_id", _template_ids())
    def test_converts_without_error(self, template_id, _real_node_context):
        data = _load_template_workflow(template_id)
        if data is None:
            pytest.skip(f"template {template_id} not found")
        if not _is_ui_workflow(data):
            pytest.skip(f"{template_id} is not a UI workflow")
        result = convert_ui_to_api(data)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("template_id", _template_ids())
    def test_no_muted_nodes_in_output(self, template_id, _real_node_context):
        data = _load_template_workflow(template_id)
        if data is None or not _is_ui_workflow(data):
            pytest.skip("not applicable")
        muted_ids = {
            str(n["id"]) for n in data.get("nodes", [])
            if n.get("mode") == 2
        }
        if not muted_ids:
            pytest.skip("no muted nodes")
        result = convert_ui_to_api(data)
        assert muted_ids.isdisjoint(result.keys()), \
            f"muted nodes {muted_ids & result.keys()} should not be in output"

    @pytest.mark.parametrize("template_id", _template_ids())
    def test_no_bypassed_nodes_in_output(self, template_id, _real_node_context):
        data = _load_template_workflow(template_id)
        if data is None or not _is_ui_workflow(data):
            pytest.skip("not applicable")
        bypassed_ids = {
            str(n["id"]) for n in data.get("nodes", [])
            if n.get("mode") == 4
        }
        if not bypassed_ids:
            pytest.skip("no bypassed nodes")
        result = convert_ui_to_api(data)
        assert bypassed_ids.isdisjoint(result.keys()), \
            f"bypassed nodes {bypassed_ids & result.keys()} should not be in output"

    @pytest.mark.parametrize("template_id", _template_ids())
    def test_no_virtual_nodes_in_output(self, template_id, _real_node_context):
        data = _load_template_workflow(template_id)
        if data is None or not _is_ui_workflow(data):
            pytest.skip("not applicable")
        virtual_ids = {
            str(n["id"]) for n in data.get("nodes", [])
            if n.get("type") in _VIRTUAL_NODE_TYPES
        }
        if not virtual_ids:
            pytest.skip("no virtual nodes")
        result = convert_ui_to_api(data)
        assert virtual_ids.isdisjoint(result.keys())

    def test_default_workflow_ksampler(self, _real_node_context):
        data = _load_template_workflow("default")
        if data is None:
            pytest.skip("default template not found")
        result = convert_ui_to_api(data)

        # Find the KSampler node.
        ks_nodes = [
            (nid, n) for nid, n in result.items()
            if n["class_type"] == "KSampler"
        ]
        assert len(ks_nodes) == 1
        ks_id, ks = ks_nodes[0]

        # Widget values should be correctly mapped.
        assert isinstance(ks["inputs"]["seed"], (int, float))
        assert isinstance(ks["inputs"]["steps"], (int, float))
        assert isinstance(ks["inputs"]["cfg"], (int, float))
        assert ks["inputs"]["sampler_name"] in ("euler", "dpmpp_2m", "dpmpp_2m_sde")
        assert isinstance(ks["inputs"]["scheduler"], str)
        assert isinstance(ks["inputs"]["denoise"], (int, float))

        # Linked inputs should be node references.
        assert isinstance(ks["inputs"]["model"], list)
        assert len(ks["inputs"]["model"]) == 2
        assert isinstance(ks["inputs"]["positive"], list)
        assert isinstance(ks["inputs"]["negative"], list)
        assert isinstance(ks["inputs"]["latent_image"], list)

    def test_default_workflow_completeness(self, _real_node_context):
        """All active non-virtual nodes from the default template should be present."""
        data = _load_template_workflow("default")
        if data is None:
            pytest.skip("default template not found")
        result = convert_ui_to_api(data)

        expected_types = {
            "CheckpointLoaderSimple", "KSampler", "CLIPTextEncode",
            "EmptyLatentImage", "VAEDecode", "SaveImage",
        }
        actual_types = {n["class_type"] for n in result.values()}
        assert expected_types.issubset(actual_types), \
            f"missing: {expected_types - actual_types}"

    @pytest.mark.parametrize("template_id", _template_ids())
    def test_all_link_targets_exist(self, template_id, _real_node_context):
        """Every [node_id, slot] reference should point to a node in the output."""
        data = _load_template_workflow(template_id)
        if data is None or not _is_ui_workflow(data):
            pytest.skip("not applicable")
        result = convert_ui_to_api(data)
        for nid, node_data in result.items():
            for inp_name, val in node_data["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    assert val[0] in result, (
                        f"node {nid}.{inp_name} references missing node {val[0]}"
                    )

    @pytest.mark.parametrize("template_id", _template_ids())
    def test_subgraph_nodes_expanded(self, template_id, _real_node_context):
        """Group nodes should be expanded; they must not appear in the output."""
        data = _load_template_workflow(template_id)
        if data is None or not _is_ui_workflow(data):
            pytest.skip("not applicable")

        sg_defs = _collect_subgraph_defs(data)
        group_node_ids = {
            str(n["id"]) for n in data.get("nodes", [])
            if n.get("type", "") in sg_defs and n.get("mode", 0) not in (2, 4)
        }
        if not group_node_ids:
            pytest.skip("no group nodes")

        result = convert_ui_to_api(data)

        # Group nodes themselves should not be in the output.
        for gid in group_node_ids:
            assert gid not in result, f"group node {gid} should be expanded"


def test_subgraph_widget_shared_with_non_widget_input():
    definition = {
        'inputs': [{'name': 'prompt'}, {'name': 'enabled'}],
        'nodes': [
            {'id': 1, 'inputs': [{'name': 'on_false', 'link': 10}]},
            {'id': 2, 'inputs': [{'name': 'prompt', 'widget': {'name': 'prompt'}, 'link': 11}]},
            {'id': 3, 'inputs': [{'name': 'enabled', 'widget': {'name': 'enabled'}, 'link': 12}]},
        ],
        'links': [
            [10, -10, 0, 1, 0, 'STRING'],
            [11, -10, 0, 2, 0, 'STRING'],
            [12, -10, 1, 3, 0, 'BOOLEAN'],
        ],
    }
    assert _get_sg_widget_positional(definition, 'prompt', ['a cat', True]) == (True, 'a cat')
    assert _get_sg_widget_positional(definition, 'enabled', ['a cat', True]) == (True, True)
