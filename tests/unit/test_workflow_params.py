"""Tests for workflow parameter discovery and application.

Two layers:

* **Synthetic unit tests** exercise each predicate against minimal, controlled
  workflows.

* **Parameterized real-workflow tests** at the bottom of this file iterate over
  every directory under ``tests/data/workflows/`` that contains both a
  ``workflow.json`` (UI or API format) and an ``expected.json`` sidecar. Drop
  new workflows in there to extend coverage without touching test code.
"""
from __future__ import annotations

import json
from importlib.metadata import packages_distributions
from importlib.resources import files
from pathlib import Path

import pytest
from packaging.version import Version

from comfy.cli_args_types import Configuration
from comfy.entrypoints.workflow import _apply_overrides
from comfy.entrypoints.workflow_params import (
    Param,
    TIER_ADVANCED,
    TIER_COMMON,
    TIER_HEADLINE,
    apply,
    apply_role,
    assign_ui_groups,
    class_type_roles,
    count_input_slots,
    discover,
    easy_pack_nodes,
    frontend_widget_pool,
    linear_app_inputs,
    params_by_address,
    params_by_role,
    primitive_nodes,
    promoted_widgets_metadata,
    prompt_polarity,
    set_node_pairs,
    titled_nodes,
    turbo_switches,
    workflow_extra_metadata,
)


# ── synthetic unit tests: frontend_widget_pool ────────────────────────────────


def _api(node_id: str, class_type: str, **inputs) -> tuple[str, dict]:
    return node_id, {"class_type": class_type, "inputs": inputs}


def test_frontend_widget_pool_emits_one_param_per_non_link_widget():
    api = dict(
        [
            _api("3", "KSampler", seed=42, steps=20, cfg=8.0, model=["1", 0]),
            _api("1", "CheckpointLoaderSimple", ckpt_name="model_a.safetensors"),
        ]
    )
    out = list(frontend_widget_pool(api, ui=None))
    by_addr = {p.address: p for p in out}
    assert ("3", "seed") in by_addr
    assert ("3", "steps") in by_addr
    assert ("3", "cfg") in by_addr
    assert ("1", "ckpt_name") in by_addr
    # `model` is a link, must not be promoted to a Param
    assert ("3", "model") not in by_addr


def test_frontend_widget_pool_infers_widget_types():
    api = dict(
        [
            _api(
                "1", "Anything",
                an_int=5, a_float=1.25, a_string="hello", a_bool=True,
                a_combo_choice="euler",
            ),
        ]
    )
    by_name = {p.widget_name: p for p in frontend_widget_pool(api, None)}
    assert by_name["an_int"].type == "INT"
    assert by_name["a_float"].type == "FLOAT"
    assert by_name["a_string"].type == "STRING"
    assert by_name["a_bool"].type == "BOOLEAN"
    # Plain string from a combo selection still types as STRING; only list
    # values type as COMBO. Frontend rule doesn't introspect choices.
    assert by_name["a_combo_choice"].type == "STRING"


def test_frontend_widget_pool_propagates_node_meta_title_to_label():
    api = {
        "5": {
            "class_type": "WanVideoSampler",
            "inputs": {"seed": 42},
            "_meta": {"title": "My Sampler"},
        }
    }
    out = list(frontend_widget_pool(api, None))
    assert out[0].label == "My Sampler"


def test_frontend_widget_pool_treats_string_node_id_links_as_links():
    """Link form is [src_id, slot]; src_id is a string in API output."""
    api = {
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["1", 2]},
        }
    }
    out = list(frontend_widget_pool(api, None))
    assert out == []


def test_frontend_widget_pool_skips_unreferenced_typed_primitives():
    api = {
        "1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "preset"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "active prompt"}},
    }
    out = list(frontend_widget_pool(api, None))
    assert {p.address for p in out} == {("2", "text")}


# ── synthetic unit tests: discover() orchestration ────────────────────────────


def test_discover_runs_predicates_and_dedupes_by_address():
    api = dict([_api("1", "Foo", x=1, y=2)])
    params = discover(api)
    addrs = {p.address for p in params}
    assert addrs == {("1", "x"), ("1", "y")}
    # Stage 1 has only the frontend predicate; every Param should record it
    for p in params:
        assert "frontend_widget_pool" in p.source_predicates


def test_discover_ranks_advanced_tier_by_default():
    api = dict([_api("1", "Foo", x=1)])
    params = discover(api)
    assert params[0].tier == TIER_ADVANCED


def test_discover_sorted_stably_by_node_then_widget():
    api = dict(
        [
            _api("10", "Foo", b=1, a=2),
            _api("2", "Foo", b=3, a=4),
        ]
    )
    addrs = [p.address for p in discover(api)]
    assert addrs == [("10", "a"), ("10", "b"), ("2", "a"), ("2", "b")]


def _turbo_switch_workflow(enabled: bool = False) -> dict:
    return {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["model", 0], "lora_name": "turbo.safetensors", "strength_model": 1.0,
        }},
        "toggle": {"class_type": "PrimitiveBoolean", "inputs": {"value": enabled}},
        "model_switch": {"class_type": "ComfySwitchNode", "inputs": {
            "switch": ["toggle", 0], "on_false": ["model", 0], "on_true": ["lora", 0],
        }},
        "full_steps": {"class_type": "PrimitiveInt", "inputs": {"value": 20}},
        "turbo_steps": {"class_type": "PrimitiveInt", "inputs": {"value": 4}},
        "step_switch": {"class_type": "ComfySwitchNode", "inputs": {
            "switch": ["toggle", 0], "on_false": ["full_steps", 0], "on_true": ["turbo_steps", 0],
        }},
        "scheduler": {"class_type": "BasicScheduler", "inputs": {
            "model": ["model_switch", 0], "steps": ["step_switch", 0], "scheduler": "simple", "denoise": 1.0,
        }},
    }


def test_turbo_switch_requires_paired_lora_and_step_switches():
    workflow = _turbo_switch_workflow()
    turbo = turbo_switches(workflow, None)

    assert [(p.address, p.flag_name, p.value, p.roles) for p in turbo] == [
        (("toggle", "value"), "turbo", False, {"turbo"}),
    ]

    del workflow["scheduler"]["inputs"]["steps"]
    assert turbo_switches(workflow, None) == []


def test_turbo_switch_allows_acceleration_lora_layered_on_baseline_lora():
    workflow = _turbo_switch_workflow()
    workflow["baseline_lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["model", 0],
            "lora_name": "style.safetensors",
            "strength_model": 1.0,
        },
    }
    workflow["lora"]["inputs"]["model"] = ["baseline_lora", 0]
    workflow["model_switch"]["inputs"]["on_false"] = ["baseline_lora", 0]

    turbo = turbo_switches(workflow, None)

    assert len(turbo) == 1
    assert turbo[0].address == ("toggle", "value")


def test_turbo_switch_supports_ui_converted_primitive_widget_names():
    workflow = _turbo_switch_workflow()
    workflow["toggle"]["inputs"] = {"widget_0": False}

    turbo = turbo_switches(workflow, None)

    assert len(turbo) == 1
    assert turbo[0].widget_name == "widget_0"


def test_discover_exposes_one_turbo_boolean_for_paired_switches():
    params = discover(_turbo_switch_workflow())
    turbo = params_by_role(params, "turbo")

    assert len(turbo) == 1
    assert turbo[0].flag_name == "turbo"
    assert turbo[0].type == "BOOLEAN"


@pytest.mark.parametrize(
    "enabled,steps,expected_full,expected_turbo",
    [
        (False, None, 20, 4),
        (False, 12, 12, 4),
        (True, None, 20, 4),
        (True, 2, 20, 2),
    ],
)
def test_turbo_toggle_selects_matching_default_steps_before_explicit_override(
    enabled, steps, expected_full, expected_turbo
):
    workflow = _turbo_switch_workflow(enabled=not enabled)
    output = _apply_overrides(
        workflow,
        Configuration(set=[f"toggle.inputs.value={str(enabled).lower()}"], steps=steps),
    )

    assert output["toggle"]["inputs"]["value"] is enabled
    assert output["full_steps"]["inputs"]["value"] == expected_full
    assert output["turbo_steps"]["inputs"]["value"] == expected_turbo


def test_h3_acceleration_profile_uses_diffusers_base_schedule_and_allows_overrides():
    workflow = _turbo_switch_workflow(enabled=False)
    workflow["lora"]["inputs"]["lora_name"] = (
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    workflow["sampler"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
    }

    base = _apply_overrides(workflow, Configuration())
    explicit = _apply_overrides(
        workflow,
        Configuration(steps=17, sampler="heun"),
    )

    assert base["full_steps"]["inputs"]["value"] == 49
    assert base["sampler"]["inputs"]["sampler_name"] == "euler"
    assert explicit["full_steps"]["inputs"]["value"] == 17
    assert explicit["sampler"]["inputs"]["sampler_name"] == "heun"


def test_bundled_workflow_catalog_turbo_detection_is_complete():
    cache_root = Path(__file__).parent / "playwright_cache"
    template_dir = max(
        (path for path in cache_root.iterdir() if path.is_dir()),
        key=lambda path: Version(path.name.split("+", 1)[0]),
    )
    detected = {}
    for path in template_dir.glob("*.json"):
        workflow = json.loads(path.read_text())
        if not isinstance(workflow, dict) or not workflow:
            continue
        raw_params = turbo_switches(workflow, None)
        if not raw_params:
            continue
        step_pairs = []
        for param in raw_params:
            step_switch = next(
                node
                for switch_id, node in workflow.items()
                if isinstance(node, dict)
                and node.get("class_type") in {"ComfySwitchNode", "SwitchNode"}
                and node.get("inputs", {}).get("switch") == [param.node_id, 0]
                and any(
                    isinstance(consumer, dict)
                    and consumer.get("inputs", {}).get("steps") == [str(switch_id), 0]
                    for consumer in workflow.values()
                )
            )
            inputs = step_switch["inputs"]
            step_pairs.append((
                workflow[str(inputs["on_false"][0])]["inputs"]["value"],
                workflow[str(inputs["on_true"][0])]["inputs"]["value"],
            ))
        friendly = params_by_role(discover(workflow), "turbo")
        assert len(friendly) == 1, path.name
        assert len(friendly[0].addresses) == len(raw_params), path.name
        assert len({param.value for param in raw_params}) == 1, path.name
        assert len(set(step_pairs)) == 1, path.name
        disable_sets = [
            f"{node_id}.inputs.{widget_name}=false"
            for node_id, widget_name in friendly[0].addresses
        ]
        disabled = _apply_overrides(workflow, Configuration(set=disable_sets))
        overridden = _apply_overrides(workflow, Configuration(set=disable_sets, steps=13))
        for param in raw_params:
            assert disabled[param.node_id]["inputs"][param.widget_name] is False, path.name
            assert overridden[param.node_id]["inputs"][param.widget_name] is False, path.name
            step_switch = next(
                node
                for switch_id, node in workflow.items()
                if isinstance(node, dict)
                and node.get("class_type") in {"ComfySwitchNode", "SwitchNode"}
                and node.get("inputs", {}).get("switch") == [param.node_id, 0]
                and any(
                    isinstance(consumer, dict)
                    and consumer.get("inputs", {}).get("steps") == [str(switch_id), 0]
                    for consumer in workflow.values()
                )
            )
            full_source = str(step_switch["inputs"]["on_false"][0])
            turbo_source = str(step_switch["inputs"]["on_true"][0])
            assert overridden[full_source]["inputs"]["value"] == 13, path.name
            assert overridden[turbo_source]["inputs"]["value"] == step_pairs[0][1], path.name
        detected[path.stem] = (raw_params[0].value, *step_pairs[0], len(raw_params))

    assert detected == {
        "image_anima_base_v1": (False, 30, 8, 1),
        "image_anima_lllite_any_control_to_image": (False, 30, 8, 1),
        "image_anima_lllite_depth_control_to_image": (False, 30, 8, 1),
        "image_anima_lllite_image_inpainting": (True, 30, 8, 1),
        "image_chrono_edit_14B": (False, 20, 4, 1),
        "image_firered_image_edit1_1": (False, 40, 8, 1),
        "image_flux2": (False, 20, 8, 1),
        "image_flux2_fp8": (False, 20, 8, 1),
        "image_flux2_text_to_image": (False, 20, 8, 1),
        "image_qwen_Image_2512": (False, 50, 4, 1),
        "image_qwen_Image_2512_controlnet": (False, 50, 4, 1),
        "image_qwen_image": (False, 20, 8, 1),
        "image_qwen_image_edit": (False, 20, 4, 1),
        "image_qwen_image_edit_2509": (True, 20, 4, 1),
        "image_qwen_image_edit_2511": (False, 40, 4, 1),
        "image_qwen_image_edit_2511_int8": (False, 40, 4, 1),
        "image_qwen_image_union_control_lora": (False, 20, 4, 1),
        "template_qwen_Image_2512_360_lora": (False, 50, 4, 1),
        "template_qwen_image_edit_2511_systms_action": (True, 40, 4, 1),
        "template_qwen_multiangle.app": (True, 40, 4, 1),
        "templates-1_click_multiple_character_angles-v1.0": (True, 40, 4, 8),
        "templates-portrait_light_migration": (False, 20, 4, 1),
        "templates_rob_portrait_light_migration.app": (False, 20, 4, 1),
        "video_bernini_r_image_editing": (True, 40, 6, 1),
        "video_bernini_r_video_editing": (True, 40, 6, 1),
        "video_minimax_h3_i2v": (False, 20, 8, 1),
        "video_minimax_h3_i2v_continuation": (False, 20, 8, 1),
        "video_minimax_h3_multiframe_reference": (False, 20, 4, 1),
        "video_minimax_h3_r2v": (False, 20, 4, 1),
        "video_minimax_h3_t2v": (False, 20, 8, 1),
        "video_wan2_2_14B_i2v": (False, 20, 4, 1),
        "video_wan2_2_14B_t2v": (False, 20, 4, 1),
        "video_wan21_scail2_character_replacement": (True, 40, 6, 2),
        "video_wan21_scail2_character_replacement_int8": (True, 40, 6, 2),
        "video_wan_dancer": (True, 25, 6, 1),
        "video_wan_vace_flf2v": (False, 20, 6, 1),
        "video_wan_vace_inpainting": (True, 20, 6, 1),
        "video_wanmove_480p": (False, 20, 6, 1),
    }


def test_discover_collapses_equivalent_flags_and_suffixes_distinct_collisions():
    api = dict(
        [
            _api("1", "UNETLoader", unet_name="same.safetensors"),
            _api("2", "UNETLoader", unet_name="same.safetensors"),
            _api("3", "UNETLoader", unet_name="other.safetensors"),
        ]
    )

    params = [
        p
        for p in discover(api)
        if p.class_type == "UNETLoader" and p.widget_name == "unet_name"
    ]

    assert [(p.node_id, p.value, p.flag_name) for p in params] == [
        ("1", "same.safetensors", "unet-1"),
        ("3", "other.safetensors", "unet-2"),
    ]


def test_ideogram_default_workflow_has_unique_dynamic_flags():
    workflow_path = files("tests.inference.workflows").joinpath("ideogram4-0.json")

    workflow = json.loads(workflow_path.read_text())
    params = discover(workflow)
    flags = [p.flag_name for p in params if p.flag_name is not None]

    assert len(flags) == len(set(flags))
    assert "unet" not in flags
    assert {"unet-1", "unet-2"}.issubset(flags)
    assert "primitivestringmultiline" not in flags
    assert not any(flag.startswith("primitivestringmultiline-") for flag in flags)


# ── synthetic unit tests: helpers ─────────────────────────────────────────────


def test_params_by_address_finds_existing_param():
    api = dict([_api("3", "KSampler", seed=42, steps=20)])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    assert seed is not None
    assert seed.value == 42


def test_params_by_address_returns_none_for_missing():
    api = dict([_api("3", "KSampler", seed=42)])
    assert params_by_address(discover(api), "99", "seed") is None


def test_params_by_role_filters_on_role_set():
    p1 = Param(node_id="1", class_type="K", widget_name="seed", value=0, roles={"seed"})
    p2 = Param(node_id="2", class_type="L", widget_name="x", value=0, roles={"prompt"})
    assert params_by_role([p1, p2], "seed") == [p1]


def test_params_by_role_empty_when_no_matches():
    api = dict([_api("1", "Foo", x=1)])
    # Stage 1 hasn't introduced any roles yet
    assert params_by_role(discover(api), "prompt") == []


# ── synthetic unit tests: apply() ─────────────────────────────────────────────


def test_apply_sets_widget_value_and_returns_a_copy():
    api = dict([_api("3", "KSampler", seed=42, steps=20)])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    out = apply(api, seed, 1234)
    assert out["3"]["inputs"]["seed"] == 1234
    # Original untouched
    assert api["3"]["inputs"]["seed"] == 42


def test_apply_rejects_ui_format():
    ui = {"nodes": [], "links": []}
    p = Param(node_id="1", class_type="X", widget_name="x", value=0)
    with pytest.raises(ValueError, match="API-format"):
        apply(ui, p, 1)


def test_apply_raises_when_node_missing():
    api = dict([_api("3", "KSampler", seed=42)])
    p = Param(node_id="99", class_type="X", widget_name="x", value=0)
    with pytest.raises(KeyError):
        apply(api, p, 1)


# ── synthetic unit tests: class_type_roles ────────────────────────────────────


def test_class_type_roles_tags_ksampler_widgets():
    api = dict(
        [
            _api(
                "3", "KSampler",
                seed=42, steps=20, cfg=8.0,
                sampler_name="euler", scheduler="normal", denoise=1,
                model=["1", 0],
            ),
        ]
    )
    out = list(class_type_roles(api, None))
    by_widget = {p.widget_name: p for p in out}
    assert by_widget["seed"].roles == {"seed"}
    assert by_widget["steps"].roles == {"steps"}
    assert by_widget["cfg"].roles == {"cfg"}
    assert by_widget["sampler_name"].roles == {"sampler"}
    assert by_widget["scheduler"].roles == {"scheduler"}
    assert by_widget["denoise"].roles == {"denoise"}
    assert by_widget["denoise"].type == "FLOAT"
    # All class_type_roles output sits at TIER_COMMON
    for p in out:
        assert p.tier == TIER_COMMON


def test_class_type_roles_tags_random_noise_seed_field():
    api = dict([_api("5", "RandomNoise", noise_seed=12345)])
    out = list(class_type_roles(api, None))
    assert len(out) == 1
    assert out[0].widget_name == "noise_seed"
    assert out[0].roles == {"seed"}


def test_class_type_roles_tags_loaders_and_latent():
    api = dict(
        [
            _api("1", "CheckpointLoaderSimple", ckpt_name="model_a.safetensors"),
            _api("2", "UNETLoader", unet_name="unet_b.safetensors"),
            _api("5", "EmptyLatentImage", width=512, height=768, batch_size=2),
            _api("9", "LoadImage", image="cat.png"),
            _api("10", "LoadVideo", value="walk.mp4"),
            _api("11", "LoadAudio", value="bgm.wav"),
        ]
    )
    out = list(class_type_roles(api, None))
    by_addr = {p.address: p for p in out}
    assert by_addr[("1", "ckpt_name")].roles == {"checkpoint"}
    assert by_addr[("2", "unet_name")].roles == {"unet"}
    assert by_addr[("5", "width")].roles == {"width"}
    assert by_addr[("5", "height")].roles == {"height"}
    assert by_addr[("5", "batch_size")].roles == {"batch_size"}
    assert by_addr[("9", "image")].roles == {"image_input"}
    assert by_addr[("10", "value")].roles == {"video_input"}
    assert by_addr[("11", "value")].roles == {"audio_input"}


def test_class_type_roles_skips_links():
    api = dict([_api("3", "KSampler", seed=42, model=["1", 0])])
    out = list(class_type_roles(api, None))
    addrs = {p.address for p in out}
    assert ("3", "seed") in addrs


def test_class_type_roles_text_encode_widgets_get_generic_tag():
    api = dict(
        [
            _api("6", "CLIPTextEncode", text="positive prompt", clip=["4", 1]),
        ]
    )
    out = list(class_type_roles(api, None))
    assert len(out) == 1
    assert out[0].roles == {"text_encode"}


# ── synthetic unit tests: prompt_polarity ─────────────────────────────────────


def test_prompt_polarity_disambiguates_positive_via_sampler_input():
    api = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["6", 0], "negative": ["7", 0], "seed": 1, "steps": 20},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "good things", "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "bad things", "clip": ["4", 1]},
        },
    }
    out = list(prompt_polarity(api, None))
    by_role = {next(iter(p.roles)): p for p in out}
    assert by_role["prompt"].node_id == "6"
    assert by_role["negative_prompt"].node_id == "7"


def test_prompt_polarity_falls_back_to_sole_encoder_for_positive():
    api = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "prompt only"}},
    }
    out = list(prompt_polarity(api, None))
    roles = {r for p in out for r in p.roles}
    assert "prompt" in roles
    assert "negative_prompt" not in roles


@pytest.mark.parametrize(
    "class_type",
    ["MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"],
)
def test_prompt_polarity_recognizes_minimax_h3_conditioning_prompt(class_type):
    api = {
        "conditioning": {
            "class_type": class_type,
            "inputs": {"prompt": "original prompt"},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"positive": ["conditioning", 0]},
        },
    }

    params = discover(api)
    prompt_params = params_by_role(params, "prompt")
    assert [(p.node_id, p.widget_name) for p in prompt_params] == [
        ("conditioning", "prompt")
    ]

    out = apply_role(api, "prompt", "replacement", params=params)
    assert out["conditioning"]["inputs"]["prompt"] == "replacement"
    assert api["conditioning"]["inputs"]["prompt"] == "original prompt"


def test_prompt_polarity_emits_nothing_when_no_text_encoder():
    api = dict([_api("1", "Foo", x=1)])
    assert list(prompt_polarity(api, None)) == []


# ── synthetic unit tests: set_node_pairs ──────────────────────────────────────


def _ui(nodes: list[dict], links: list[list]) -> dict:
    return {"nodes": nodes, "links": links}


def test_set_node_pairs_tags_upstream_widgets_at_headline_tier():
    api = {
        "11": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {"video": "AstronautWalking2.mp4", "force_rate": 16},
        },
        "64": {"class_type": "SetNode", "inputs": {}},
    }
    ui = _ui(
        nodes=[
            {"id": 11, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [100]}]},
            {"id": 64, "type": "SetNode", "title": "Set_Input_Video",
             "inputs": [{"name": "*", "link": 100}], "widgets_values": ["Input_Video"]},
        ],
        links=[[100, 11, 0, 64, 0, "IMAGE"]],
    )
    out = list(set_node_pairs(api, ui))
    assert all(p.tier == TIER_HEADLINE for p in out)
    assert {p.widget_name for p in out} == {"video", "force_rate"}
    assert all(p.roles == {"set:input_video"} for p in out)
    assert all(p.node_id == "11" for p in out)


def test_set_node_pairs_skips_when_ui_missing():
    api = {"1": {"class_type": "Foo", "inputs": {"x": 1}}}
    assert list(set_node_pairs(api, ui=None)) == []


def test_set_node_pairs_ignores_non_set_titles():
    ui = _ui(
        nodes=[
            {"id": 1, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [9]}]},
            {"id": 2, "type": "SetNode", "title": "MyNode",
             "inputs": [{"name": "*", "link": 9}], "widgets_values": ["X"]},
        ],
        links=[[9, 1, 0, 2, 0, "IMAGE"]],
    )
    api = {"1": {"class_type": "VHS_LoadVideoFFmpeg", "inputs": {"video": "x.mp4"}}}
    assert list(set_node_pairs(api, ui)) == []


def test_set_node_pairs_skips_when_input_link_missing():
    ui = _ui(
        nodes=[
            {"id": 2, "type": "SetNode", "title": "Set_Foo",
             "inputs": [{"name": "*", "link": None}]},
        ],
        links=[],
    )
    assert list(set_node_pairs({}, ui)) == []


def test_set_node_pairs_skips_when_source_widget_is_a_link():
    """The source's own connected inputs (links) should not be promoted."""
    api = {
        "1": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {"video": "x.mp4", "image_input": ["99", 0]},
        },
        "2": {"class_type": "SetNode", "inputs": {}},
    }
    ui = _ui(
        nodes=[
            {"id": 1, "type": "VHS_LoadVideoFFmpeg",
             "outputs": [{"name": "IMAGE", "links": [10]}]},
            {"id": 2, "type": "SetNode", "title": "Set_Foo",
             "inputs": [{"name": "*", "link": 10}], "widgets_values": ["Foo"]},
        ],
        links=[[10, 1, 0, 2, 0, "IMAGE"]],
    )
    out = list(set_node_pairs(api, ui))
    assert {p.widget_name for p in out} == {"video"}


def test_set_node_pairs_slugifies_complex_names():
    ui = _ui(
        nodes=[
            {"id": 1, "type": "Foo", "outputs": [{"name": "x", "links": [1]}]},
            {"id": 2, "type": "SetNode", "title": "Set_My-Cool Name!",
             "inputs": [{"name": "*", "link": 1}], "widgets_values": ["x"]},
        ],
        links=[[1, 1, 0, 2, 0, "X"]],
    )
    api = {"1": {"class_type": "Foo", "inputs": {"v": 1}}}
    out = list(set_node_pairs(api, ui))
    assert out and out[0].roles == {"set:my_cool_name"}


# ── synthetic unit tests: apply_role ──────────────────────────────────────────


def test_apply_role_writes_to_every_param_with_role():
    api = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}},
        "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2}},
    }
    out = apply_role(api, "seed", 999)
    assert out["3"]["inputs"]["seed"] == 999
    assert out["5"]["inputs"]["noise_seed"] == 999
    # other widgets untouched
    assert out["3"]["inputs"]["steps"] == 20


def test_apply_role_returns_input_unchanged_when_no_match():
    api = dict([_api("1", "Foo", x=1)])
    out = apply_role(api, "seed", 999)
    assert out is api  # short-circuit when nothing to do


def test_apply_role_does_not_mutate_input():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    out = apply_role(api, "seed", 999)
    assert api["3"]["inputs"]["seed"] == 1
    assert out["3"]["inputs"]["seed"] == 999


def test_apply_role_rejects_ui_format():
    ui = {"nodes": [], "links": []}
    with pytest.raises(ValueError, match="API-format"):
        apply_role(ui, "seed", 1)


def test_apply_role_reuses_caller_supplied_params_list():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}}}
    params = discover(api)
    # Apply two roles back-to-back without re-running discover
    out = apply_role(api, "seed", 42, params=params)
    out = apply_role(out, "steps", 100, params=params)
    assert out["3"]["inputs"]["seed"] == 42
    assert out["3"]["inputs"]["steps"] == 100


def test_apply_role_matches_replace_seed_behavior():
    """Regression: apply_role('seed', N) is equivalent to prompt_utils.replace_seed(N)."""
    from comfy.component_model.prompt_utils import replace_seed
    api = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}},
        "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2}},
    }
    assert apply_role(api, "seed", 999) == replace_seed(api, 999)


def test_apply_role_matches_replace_steps_behavior():
    from comfy.component_model.prompt_utils import replace_steps
    api = {
        "3": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": 8}},
        "9": {"class_type": "BasicScheduler", "inputs": {"steps": 30, "denoise": 1.0}},
    }
    assert apply_role(api, "steps", 50) == replace_steps(api, 50)


def test_apply_role_steps_updates_active_linked_primitive_through_switch():
    api = {
        "sched": {
            "class_type": "Flux2Scheduler",
            "inputs": {"steps": ["switch", 0], "width": 512, "height": 512},
        },
        "switch": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["use_fast", 0],
                "on_false": ["full_steps", 0],
                "on_true": ["fast_steps", 0],
            },
        },
        "use_fast": {"class_type": "PrimitiveBoolean", "inputs": {"value": False}},
        "full_steps": {"class_type": "PrimitiveInt", "inputs": {"value": 20}},
        "fast_steps": {"class_type": "PrimitiveInt", "inputs": {"value": 8}},
    }

    out = apply_role(api, "steps", 2)

    assert out["sched"]["inputs"]["steps"] == ["switch", 0]
    assert out["full_steps"]["inputs"]["value"] == 2
    assert out["fast_steps"]["inputs"]["value"] == 8
    assert api["full_steps"]["inputs"]["value"] == 20


def test_apply_role_prompt_matches_replace_prompt_text_when_text_encoder_present():
    from comfy.component_model.prompt_utils import replace_prompt_text
    api = {
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
    }
    assert apply_role(api, "prompt", "new") == replace_prompt_text(api, "new")


# ── synthetic unit tests: primitive_nodes ─────────────────────────────────────


def test_primitive_nodes_typed_primitive_int_tagged_at_headline():
    api = {
        "1": {
            "class_type": "PrimitiveInt",
            "inputs": {"value": 42},
            "_meta": {"title": "My Steps"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {"steps": ["1", 0]},
        },
    }
    out = list(primitive_nodes(api, None))
    assert len(out) == 1
    p = out[0]
    assert p.roles == {"primitive:my_steps"}
    assert p.tier == TIER_HEADLINE
    assert p.value == 42
    assert p.label == "My Steps"


def test_primitive_nodes_typed_primitive_uses_node_id_when_untitled():
    api = {
        "5": {"class_type": "PrimitiveString", "inputs": {"value": "hi"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
    }
    out = list(primitive_nodes(api, None))
    assert out[0].roles == {"primitive:node_5"}


def test_primitive_nodes_skips_unreferenced_typed_primitive():
    api = {
        "1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "preset"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "active prompt"}},
    }
    assert list(primitive_nodes(api, None)) == []


def test_primitive_nodes_legacy_primitivenode_tags_consumer_widget():
    """Legacy PrimitiveNode is virtual; its value lands on the consumer's widget."""
    api = {
        "20": {
            "class_type": "KSampler",
            "inputs": {"seed": 1234, "steps": 20},
        },
    }
    ui = _ui(
        nodes=[
            {"id": 10, "type": "PrimitiveNode", "title": "User Seed",
             "outputs": [{"name": "INT", "links": [50]}],
             "widgets_values": [1234]},
            {"id": 20, "type": "KSampler",
             "inputs": [{"name": "seed", "link": 50, "widget": {"name": "seed"}}]},
        ],
        links=[[50, 10, 0, 20, 0, "INT"]],
    )
    out = list(primitive_nodes(api, ui))
    assert len(out) == 1
    p = out[0]
    assert p.node_id == "20"
    assert p.widget_name == "seed"
    assert p.roles == {"primitive:user_seed"}
    assert p.tier == TIER_HEADLINE


def test_primitive_nodes_skips_typed_primitives_when_value_is_a_link():
    api = {
        "1": {"class_type": "PrimitiveInt", "inputs": {"value": ["99", 0]}},
    }
    assert list(primitive_nodes(api, None)) == []


# ── synthetic unit tests: easy_pack_nodes ─────────────────────────────────────


def test_easy_pack_nodes_tags_easy_seed_as_seed_role():
    api = {"3": {"class_type": "easy seed", "inputs": {"seed": 42}}}
    out = list(easy_pack_nodes(api, None))
    assert len(out) == 1
    assert out[0].roles == {"seed"}
    assert out[0].tier == TIER_HEADLINE


def test_easy_pack_nodes_tags_easy_positive_negative_as_prompt_roles():
    api = {
        "4": {"class_type": "easy positive", "inputs": {"positive": "good"}},
        "5": {"class_type": "easy negative", "inputs": {"negative": "bad"}},
    }
    out = list(easy_pack_nodes(api, None))
    by_role = {next(iter(p.roles)): p for p in out}
    assert by_role["prompt"].value == "good"
    assert by_role["negative_prompt"].value == "bad"
    assert by_role["prompt"].tier == TIER_HEADLINE
    assert by_role["negative_prompt"].tier == TIER_HEADLINE


def test_easy_pack_nodes_ignores_unrelated_easy_classes():
    api = {"7": {"class_type": "easy showAnything", "inputs": {"any": "x"}}}
    assert list(easy_pack_nodes(api, None)) == []


# ── synthetic unit tests: titled_nodes ────────────────────────────────────────


def test_titled_nodes_promotes_titled_node_widgets_to_common_tier():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}}}
    ui = _ui(
        nodes=[{"id": 3, "type": "KSampler", "title": "Main Sampler"}],
        links=[],
    )
    out = list(titled_nodes(api, ui))
    assert {p.widget_name for p in out} == {"seed", "steps"}
    assert all(p.tier == TIER_COMMON for p in out)
    assert all(p.label == "Main Sampler" for p in out)
    assert all(p.roles == {"title:main_sampler"} for p in out)


def test_titled_nodes_skips_set_get_primitive_titles():
    api = {
        "1": {"class_type": "Foo", "inputs": {"x": 1}},
        "2": {"class_type": "Foo", "inputs": {"x": 2}},
        "3": {"class_type": "PrimitiveInt", "inputs": {"value": 3}},
    }
    ui = _ui(
        nodes=[
            {"id": 1, "type": "SetNode", "title": "Set_X"},
            {"id": 2, "type": "GetNode", "title": "Get_X"},
            {"id": 3, "type": "PrimitiveInt", "title": "x"},
        ],
        links=[],
    )
    assert list(titled_nodes(api, ui)) == []


def test_titled_nodes_returns_empty_without_ui():
    assert list(titled_nodes({"1": {"class_type": "Foo", "inputs": {"x": 1}}}, None)) == []


# ── synthetic unit tests: workflow_extra_metadata ─────────────────────────────


def test_workflow_extra_metadata_honours_explicit_parameters():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {
            "parameters": [
                {"node_id": "3", "widget_name": "seed", "role": "seed", "label": "Seed"},
                {"node_id": "3", "widget_name": "steps"},
            ]
        },
    }
    out = list(workflow_extra_metadata(api, ui))
    by_widget = {p.widget_name: p for p in out}
    assert by_widget["seed"].roles == {"seed"}
    assert by_widget["seed"].label == "Seed"
    assert by_widget["seed"].tier == TIER_HEADLINE
    assert by_widget["steps"].roles == set()


def test_workflow_extra_metadata_drops_invalid_addresses():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {"parameters": [
            {"node_id": "99", "widget_name": "x"},
            {"node_id": "3", "widget_name": "missing_widget"},
        ]},
    }
    assert list(workflow_extra_metadata(api, ui)) == []


def test_workflow_extra_metadata_returns_empty_when_extra_missing():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    assert list(workflow_extra_metadata(api, None)) == []
    assert list(workflow_extra_metadata(api, {"nodes": [], "links": []})) == []


# ── synthetic unit tests: promoted_widgets_metadata ───────────────────────────


def test_promoted_widgets_metadata_honours_promotion_entries():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {"promotionEntries": [
            {"interiorNodeId": "3", "widgetName": "seed"},
        ]},
    }
    out = list(promoted_widgets_metadata(api, ui))
    assert len(out) == 1
    assert out[0].roles == {"frontend_promoted"}
    assert out[0].tier == TIER_HEADLINE


def test_promoted_widgets_metadata_accepts_promotions_alias():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {"promotions": [
            {"interiorNodeId": "3", "widgetName": "seed"},
        ]},
    }
    assert len(list(promoted_widgets_metadata(api, ui))) == 1


def test_promoted_widgets_metadata_returns_empty_when_no_promotion_block():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    assert list(promoted_widgets_metadata(api, {"nodes": [], "links": []})) == []


# ── synthetic unit tests: _unbypass_extra_loaders ─────────────────────────────


def test_unbypass_extra_loaders_flips_enough_mode4_nodes_to_meet_target():
    from comfy.entrypoints.workflow import _unbypass_extra_loaders
    ui = _ui(
        nodes=[
            {"id": 1, "type": "LoadImage", "mode": 0},
            {"id": 2, "type": "LoadImage", "mode": 4},
            {"id": 3, "type": "LoadImage", "mode": 4},
            {"id": 4, "type": "LoadImage", "mode": 4},
        ],
        links=[],
    )
    out = _unbypass_extra_loaders(ui, "images", target_count=3)
    modes = [n["mode"] for n in out["nodes"]]
    # First active stays mode 0, two more flip to 0, fourth stays bypassed.
    assert modes.count(0) == 3
    assert modes.count(4) == 1


def test_unbypass_extra_loaders_no_op_when_active_already_sufficient():
    from comfy.entrypoints.workflow import _unbypass_extra_loaders
    ui = _ui(
        nodes=[
            {"id": 1, "type": "LoadImage", "mode": 0},
            {"id": 2, "type": "LoadImage", "mode": 0},
            {"id": 3, "type": "LoadImage", "mode": 4},
        ],
        links=[],
    )
    out = _unbypass_extra_loaders(ui, "images", target_count=2)
    assert out is ui  # short-circuit


def test_unbypass_extra_loaders_targets_only_requested_kind():
    from comfy.entrypoints.workflow import _unbypass_extra_loaders
    ui = _ui(
        nodes=[
            {"id": 1, "type": "LoadImage", "mode": 0},
            {"id": 2, "type": "VHS_LoadVideoFFmpeg", "mode": 4},
        ],
        links=[],
    )
    out = _unbypass_extra_loaders(ui, "images", target_count=2)
    # Video loader's mode untouched
    video = next(n for n in out["nodes"] if n["type"] == "VHS_LoadVideoFFmpeg")
    assert video["mode"] == 4


# ── synthetic unit tests: linear_app_inputs (frontend "app mode") ─────────────


def test_linear_app_inputs_promotes_extra_linear_data_inputs():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {"linearData": {"inputs": [[3, "seed"]], "outputs": []}},
    }
    out = list(linear_app_inputs(api, ui))
    assert len(out) == 1
    assert out[0].roles == {"app_input"}
    assert out[0].tier == TIER_HEADLINE
    assert out[0].flag_name == "set-3-seed"


def test_linear_app_inputs_drops_invalid_pairs():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler"}],
        "links": [],
        "extra": {"linearData": {"inputs": [
            [99, "seed"],            # node missing
            [3, "missing_widget"],   # widget missing
            ["3"],                   # short tuple
            "not_a_pair",
        ]}},
    }
    assert list(linear_app_inputs(api, ui)) == []


def test_linear_app_inputs_returns_empty_without_extra_block():
    assert list(linear_app_inputs({}, None)) == []
    assert list(linear_app_inputs({}, {"nodes": [], "links": []})) == []


# ── synthetic unit tests: assign_ui_groups ────────────────────────────────────


def test_assign_ui_groups_buckets_params_by_containing_group():
    workflow = {
        "nodes": [
            {"id": 1, "type": "Foo", "pos": [100, 100]},
            {"id": 2, "type": "Foo", "pos": [600, 600]},
        ],
        "links": [],
        "groups": [
            {"title": "Inputs",  "bounding": [0, 0, 200, 200]},
            {"title": "Outputs", "bounding": [500, 500, 200, 200]},
        ],
    }
    p1 = Param(node_id="1", class_type="Foo", widget_name="x", value=1)
    p2 = Param(node_id="2", class_type="Foo", widget_name="y", value=2)
    out = assign_ui_groups(workflow, [p1, p2])
    assert out == {"Inputs": [p1], "Outputs": [p2]}


def test_assign_ui_groups_empty_bucket_for_orphans():
    workflow = {
        "nodes": [{"id": 99, "type": "Foo", "pos": [9999, 9999]}],
        "links": [],
        "groups": [{"title": "G", "bounding": [0, 0, 100, 100]}],
    }
    p = Param(node_id="99", class_type="Foo", widget_name="x", value=1)
    out = assign_ui_groups(workflow, [p])
    assert out == {"": [p]}


def test_assign_ui_groups_no_groups_returns_single_bucket():
    workflow = {"nodes": [{"id": 1, "type": "Foo", "pos": [0, 0]}], "links": []}
    p = Param(node_id="1", class_type="Foo", widget_name="x", value=1)
    out = assign_ui_groups(workflow, [p])
    assert out == {"": [p]}


def test_assign_ui_groups_smallest_group_wins_when_nested():
    workflow = {
        "nodes": [{"id": 1, "type": "Foo", "pos": [110, 110]}],
        "links": [],
        "groups": [
            {"title": "Outer", "bounding": [0, 0, 1000, 1000]},
            {"title": "Inner", "bounding": [100, 100, 200, 200]},
        ],
    }
    p = Param(node_id="1", class_type="Foo", widget_name="x", value=1)
    out = assign_ui_groups(workflow, [p])
    assert out == {"Inner": [p]}


# ── synthetic unit tests: count_input_slots ───────────────────────────────────


def test_count_input_slots_counts_active_and_bypassed_loaders():
    ui = _ui(
        nodes=[
            {"id": 1, "type": "LoadImage", "mode": 0},
            {"id": 2, "type": "LoadImage", "mode": 4},
            {"id": 3, "type": "LoadImage", "mode": 4},
            {"id": 4, "type": "VHS_LoadVideoFFmpeg", "mode": 0},
            {"id": 5, "type": "LoadAudio", "mode": 4},
        ],
        links=[],
    )
    counts = count_input_slots(ui)
    assert counts["images"] == (1, 3)
    assert counts["videos"] == (1, 1)
    assert counts["audios"] == (0, 1)


def test_count_input_slots_handles_api_format():
    api = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {"class_type": "VHS_LoadVideo", "inputs": {"value": "b.mp4"}},
    }
    counts = count_input_slots(api)
    assert counts["images"] == (1, 1)
    assert counts["videos"] == (1, 1)


def test_count_input_slots_returns_zero_when_no_loaders():
    ui = _ui(nodes=[{"id": 1, "type": "KSampler", "mode": 0}], links=[])
    assert count_input_slots(ui) == {"images": (0, 0), "videos": (0, 0), "audios": (0, 0)}


def test_kontext_template_input_count_includes_bypassed_image():
    matches = [
        path for path in _builtin_template_paths()
        if path.name == "flux_kontext_dev_basic.json"
    ]
    if not matches:
        pytest.skip("flux_kontext_dev_basic template not installed")
    workflow = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    counts = count_input_slots(workflow)
    active, total = counts["images"]
    assert total > active, (
        f"kontext should have at least one bypassed image (active={active}, total={total})"
    )


# ── synthetic unit tests: discover() merges roles across predicates ───────────


def test_discover_merges_roles_from_class_type_and_polarity():
    api = {
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "seed": 42}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}},
    }
    params = discover(api)
    text = params_by_address(params, "6", "text")
    assert text is not None
    # Picked up by both class_type_roles (text_encode) and prompt_polarity (prompt)
    assert text.roles == {"text_encode", "prompt"}
    assert "class_type_roles" in text.source_predicates
    assert "prompt_polarity" in text.source_predicates
    assert "frontend_widget_pool" in text.source_predicates


def test_discover_skips_unreferenced_typed_primitives():
    api = {
        "1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "preset"}},
        "2": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "used"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0]}},
    }
    params = discover(api)
    addrs = {p.address for p in params}
    assert ("1", "value") not in addrs
    assert ("2", "value") in addrs


def test_discover_lifts_tier_to_common_for_role_tagged_params():
    api = dict([_api("3", "KSampler", seed=42, steps=20, model=["1", 0])])
    params = discover(api)
    seed = params_by_address(params, "3", "seed")
    assert seed.tier == TIER_COMMON


def test_discover_keeps_advanced_tier_for_unknown_class_widgets():
    api = dict([_api("5", "WanVideoBlockSwap", blocks_to_swap=25, offload_img=True)])
    params = discover(api)
    for p in params:
        assert p.tier == TIER_ADVANCED


def test_params_by_role_finds_role_tagged_params():
    api = dict([_api("3", "KSampler", seed=42, steps=20, cfg=8.0, model=["1", 0])])
    params = discover(api)
    assert len(params_by_role(params, "seed")) == 1
    assert len(params_by_role(params, "steps")) == 1
    assert len(params_by_role(params, "cfg")) == 1


# ── parameterized real-workflow tests ─────────────────────────────────────────
#
# Add entries to ``WORKFLOW_FIXTURES`` to extend coverage. Each name maps to
# ``tests/data/workflows/<name>.json``. An optional sidecar
# ``tests/data/workflows/<name>.expected.json`` lets a fixture additionally
# assert ``min_total_params`` (int) and ``roles_present`` (list[str]); when
# absent, only the universal invariants run. Anything that doesn't fit those
# two keys belongs in a normal pytest function below — write the assertion
# directly rather than extending the sidecar schema.

_WORKFLOWS_DIR = files("tests").joinpath("data", "workflows")

WORKFLOW_FIXTURES: list[str] = [
    "yt_bgswap_v01",
]


def _load_workflow(name: str) -> dict:
    """Load ``<name>.json``, booting the node system if UI-format.

    `convert_ui_to_api` needs `import_all_nodes_in_workspace` to populate
    the node registry. Unknown classes (e.g. WanVideoWrapper) survive via
    `preserve_unknown_nodes=True`.
    """
    workflow = json.loads((_WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if "nodes" in workflow and "links" in workflow:
        from comfy.nodes.package import import_all_nodes_in_workspace
        from comfy.nodes_context import get_nodes
        if len(get_nodes()) == 0:
            import_all_nodes_in_workspace()
    return workflow


def _maybe_load_expected(name: str) -> dict | None:
    path = _WORKFLOWS_DIR / f"{name}.expected.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.mark.parametrize("name", WORKFLOW_FIXTURES, ids=WORKFLOW_FIXTURES)
def test_real_workflow(name: str):
    workflow = _load_workflow(name)
    params = discover(workflow)

    # Universal invariants: every param resolves and none holds a link value.
    from comfy.entrypoints.workflow_params import _is_link, _to_api
    api, _ = _to_api(workflow)
    for p in params:
        assert p.node_id in api, (
            f"param node_id {p.node_id!r} missing from API workflow"
        )
        assert p.widget_name in (api[p.node_id].get("inputs") or {}), (
            f"param widget {p.widget_name!r} missing on node {p.node_id} ({p.class_type})"
        )
        assert not _is_link(p.value), (
            f"param {p.address} has a link value {p.value!r}"
        )

    expected = _maybe_load_expected(name)
    if expected is None:
        return
    if "min_total_params" in expected:
        assert len(params) >= expected["min_total_params"], (
            f"discover() returned {len(params)} params; expected at least "
            f"{expected['min_total_params']}"
        )
    for role in expected.get("roles_present", []):
        assert params_by_role(params, role), f"role {role!r} not present"


# Case-specific assertions that don't fit the sidecar schema go here as plain
# tests. Keep them thin and named after what they verify.

def _builtin_template_paths() -> list:
    paths = []
    for package in sorted(packages_distributions()):
        if not package.startswith("comfyui_workflow_templates_"):
            continue
        templates = files(package).joinpath("templates")
        if not templates.is_dir():
            continue
        paths.extend(
            path for path in templates.iterdir()
            if path.is_file() and path.name.endswith(".json")
        )
    return sorted(paths, key=lambda path: (str(path.parent), path.name))


def _collect_builtin_templates() -> list:
    paths = []
    for p in _builtin_template_paths():
        try:
            wf = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_ui = isinstance(wf, dict) and "nodes" in wf and "links" in wf
        is_api = (
            isinstance(wf, dict)
            and wf
            and all(isinstance(v, dict) and "class_type" in v for v in wf.values())
        )
        if is_ui or is_api:
            paths.append(p)
    return [pytest.param(p, id=p.stem) for p in paths]


@pytest.mark.parametrize("path", _collect_builtin_templates())
def test_builtin_templates_smoke(path: Path):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    from comfy.nodes.package_typing import ExportedNodes
    params = discover(workflow, node_mappings=ExportedNodes())

    from comfy.entrypoints.workflow_params import _is_link, _to_api
    api, _ = _to_api(workflow, node_mappings=ExportedNodes())
    for p in params:
        assert p.node_id in api, f"{path.name}: param node {p.node_id!r} missing from API workflow"
        assert p.widget_name in (api[p.node_id].get("inputs") or {}), (
            f"{path.name}: widget {p.widget_name!r} missing on node {p.node_id} ({p.class_type})"
        )
        assert not _is_link(p.value), f"{path.name}: param {p.address} has link value"


def test_qwen_image_layered_sampler_parameter_values():
    from comfy.cmd.workflow_templates import resolve_template
    from comfy.nodes.package_typing import ExportedNodes

    workflow = json.loads(Path(resolve_template("image_qwen_image_layered")).read_text(encoding="utf-8"))
    params = discover(workflow, node_mappings=ExportedNodes())

    steps = params_by_role(params, "steps")
    cfg = params_by_role(params, "cfg")
    assert [(param.value, param.type) for param in steps] == [(20, "INT")]
    assert [(param.value, param.type) for param in cfg] == [(2.5, "FLOAT")]


def test_qwen_image_layered_blueprint_sampler_parameter_values():
    from comfy.nodes.package_typing import ExportedNodes

    blueprint = (
        files("comfy.blueprints")
        .joinpath("Image to Layers(Qwen-Image-Layered).json")
    )
    workflow = json.loads(blueprint.read_text(encoding="utf-8"))
    params = discover(workflow, node_mappings=ExportedNodes())

    assert [(param.value, param.type) for param in params_by_role(params, "steps")] == [(20, "INT")]
    assert [(param.value, param.type) for param in params_by_role(params, "cfg")] == [(2.5, "FLOAT")]
    assert [(param.value, param.type) for param in params_by_role(params, "sampler")] == [("euler", "STRING")]
    assert [(param.value, param.type) for param in params_by_role(params, "scheduler")] == [("simple", "STRING")]
    assert [(param.value, param.type) for param in params_by_role(params, "denoise")] == [(1, "FLOAT")]


def test_yt_bgswap_v01_image_loader_is_loadimage_node_42():
    workflow = _load_workflow("yt_bgswap_v01")
    params = discover(workflow)
    image_inputs = params_by_role(params, "image_input")
    assert len(image_inputs) == 1
    assert image_inputs[0].node_id == "42"
    assert image_inputs[0].class_type == "LoadImage"


def test_yt_bgswap_v01_set_node_pairs_promote_named_variables_to_headline():
    """Set_Model, Set_Prompt, Set_VAE, Set_CLIP, Set_VideoIn each bless their
    upstream loader/encoder node, which lands them at headline tier with a
    set:<name> role. The exact source nodes are baked into the workflow:
    Set_Model←6 (WanVideoModelLoader), Set_VAE←8 (WanVideoVAELoader),
    Set_CLIP←7 (LoadWanVideoT5TextEncoder), Set_VideoIn←11 (VHS_LoadVideoFFmpeg),
    Set_Prompt←73 (WanVideoTextEncode).
    """
    workflow = _load_workflow("yt_bgswap_v01")
    params = discover(workflow)

    expected = {
        "set:model":    ("6",  "WanVideoModelLoader"),
        "set:vae":      ("8",  "WanVideoVAELoader"),
        "set:clip":     ("7",  "LoadWanVideoT5TextEncoder"),
        "set:videoin":  ("11", "VHS_LoadVideoFFmpeg"),
        "set:prompt":   ("73", "WanVideoTextEncode"),
    }
    for role, (expected_node_id, expected_class) in expected.items():
        matches = params_by_role(params, role)
        assert matches, f"role {role!r} has no params"
        assert all(p.tier == TIER_HEADLINE for p in matches), (
            f"role {role!r} should be headline tier"
        )
        assert all(p.node_id == expected_node_id for p in matches), (
            f"role {role!r} should target node {expected_node_id} ({expected_class})"
        )
        assert all(p.class_type == expected_class for p in matches)
