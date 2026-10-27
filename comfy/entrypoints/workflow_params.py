"""Workflow parameter discovery and application."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..component_model.workflow_convert import convert_ui_to_api, is_ui_workflow
from ..component_model import prompt_utils as _pu

logger = logging.getLogger(__name__)


_DIRECT_ROLES: dict[tuple[str, str], str] = {}
_ROLE_TYPES = {
    "steps": "INT",
    "cfg": "FLOAT",
    "sampler": "STRING",
    "scheduler": "STRING",
    "denoise": "FLOAT",
    "seed": "INT",
    "width": "INT",
    "height": "INT",
    "batch_size": "INT",
}


def _seed_direct_roles() -> None:
    if _DIRECT_ROLES:
        return
    for ct in _pu._STEPS_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "steps")] = "steps"
    for ct in _pu._CFG_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "cfg")] = "cfg"
    for ct in _pu._SAMPLER_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "sampler_name")] = "sampler"
    for ct in _pu._SCHEDULER_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "scheduler")] = "scheduler"
    for ct in _pu._DENOISE_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "denoise")] = "denoise"
    for ct in _pu._LATENT_SIZE_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "width")] = "width"
        _DIRECT_ROLES[(ct, "height")] = "height"
        _DIRECT_ROLES[(ct, "batch_size")] = "batch_size"
    for ct in _pu._CHECKPOINT_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "ckpt_name")] = "checkpoint"
    for ct in _pu._DIFFUSION_MODEL_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "unet_name")] = "unet"
    for ct, field_name in _pu._SEED_FIELDS.items():
        _DIRECT_ROLES[(ct, field_name)] = "seed"
    # Filesystem and URL loaders both expose a media-bearing widget; the role
    # is the same regardless of which form the workflow uses. Tagging only —
    # the URL-rewrite that --image performs lives at apply-time, not here.
    for ct in _pu._IMAGE_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "image")] = "image_input"
        _DIRECT_ROLES[(ct, "value")] = "image_input"
    for ct in _pu._VIDEO_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "video")] = "video_input"
        _DIRECT_ROLES[(ct, "value")] = "video_input"
    for ct in _pu._AUDIO_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "audio")] = "audio_input"
        _DIRECT_ROLES[(ct, "value")] = "audio_input"
    for ct, fields in _pu._TEXT_ENCODE_FIELDS.items():
        for field_name in fields:
            _DIRECT_ROLES[(ct, field_name)] = "text_encode"


_seed_direct_roles()


TIER_HEADLINE = 0
TIER_COMMON = 1
TIER_ADVANCED = 2


@dataclass
class Param:
    node_id: str
    class_type: str
    widget_name: str
    value: Any
    type: str = "ANY"
    options: list[Any] = field(default_factory=list)
    roles: set[str] = field(default_factory=set)
    tier: int = TIER_ADVANCED
    flag_name: str | None = None
    label: str | None = None
    source_predicates: list[str] = field(default_factory=list)
    targets: list[tuple[str, str]] = field(default_factory=list)

    @property
    def address(self) -> tuple[str, str]:
        return (self.node_id, self.widget_name)

    @property
    def addresses(self) -> list[tuple[str, str]]:
        return [self.address, *self.targets]


Predicate = Callable[[dict, dict | None], Iterable[Param]]


def _is_link(value: Any) -> bool:
    """Mirror the API-format link shape: [src_node_id, src_slot_idx]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _referenced_node_ids(api: dict) -> set[str]:
    referenced: set[str] = set()
    for node in api.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if _is_link(value):
                referenced.add(str(value[0]))
    return referenced


def _is_unreferenced_typed_primitive(referenced: set[str], node_id: str, class_type: str) -> bool:
    return class_type in _TYPED_PRIMITIVE_CLASSES and str(node_id) not in referenced


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, list):
        return "COMBO"
    return "ANY"


def _literal_value_source(api: dict, value: Any, seen: set[tuple[str, str]] | None = None):
    if not _is_link(value):
        return value
    source = _resolve_linked_input_source(api, str(value[0]), "value", seen)
    if source is None:
        return None
    source_node_id, source_widget = source
    source_node = api.get(source_node_id)
    if not isinstance(source_node, dict):
        return None
    source_inputs = source_node.get("inputs") or {}
    result = source_inputs.get(source_widget)
    return None if _is_link(result) else result


def _primitive_value_widget(node: dict) -> str | None:
    inputs = node.get("inputs") or {}
    if "value" in inputs:
        return "value"
    if "widget_0" in inputs:
        return "widget_0"
    return next((name for name, value in inputs.items() if not _is_link(value)), None)


def _resolve_linked_input_source(
    api: dict,
    node_id: str,
    widget_name: str,
    seen: set[tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Return the concrete API widget that currently feeds a linked input."""
    if seen is None:
        seen = set()
    node = api.get(str(node_id))
    if not isinstance(node, dict):
        return None
    if widget_name == "value" and node.get("class_type") in _TYPED_PRIMITIVE_CLASSES:
        widget_name = _primitive_value_widget(node) or widget_name
    key = (str(node_id), widget_name)
    if key in seen:
        return None
    seen.add(key)

    inputs = node.get("inputs") or {}
    value = inputs.get(widget_name)
    if value is None:
        return None

    if not _is_link(value):
        return key

    source_node_id = str(value[0])
    source_node = api.get(source_node_id)
    if not isinstance(source_node, dict):
        return None

    source_type = source_node.get("class_type") or ""
    source_inputs = source_node.get("inputs") or {}
    if source_type in _TYPED_PRIMITIVE_CLASSES:
        value_widget = _primitive_value_widget(source_node)
        if value_widget is None:
            return None
        primitive_value = source_inputs.get(value_widget)
        if _is_link(primitive_value):
            return _resolve_linked_input_source(api, source_node_id, value_widget, seen)
        return (source_node_id, value_widget)

    if source_type in {"ComfySwitchNode", "SwitchNode"}:
        switch_value = _literal_value_source(api, source_inputs.get("switch"), seen)
        branch = "on_true" if bool(switch_value) else "on_false"
        if branch in source_inputs:
            return _resolve_linked_input_source(api, source_node_id, branch, seen)

    return None


def frontend_widget_pool(api: dict, ui: dict | None) -> list[Param]:
    out: list[Param] = []
    referenced = _referenced_node_ids(api)
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        if _is_unreferenced_typed_primitive(referenced, str(node_id), class_type):
            continue
        title = ((node.get("_meta") or {}).get("title")) if isinstance(node.get("_meta"), dict) else None
        for widget_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    label=title,
                    source_predicates=["frontend_widget_pool"],
                )
            )
    return out


def class_type_roles(api: dict, ui: dict | None) -> list[Param]:
    out: list[Param] = []
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        for widget_name, value in (node.get("inputs") or {}).items():
            role = _DIRECT_ROLES.get((class_type, widget_name))
            if role is None:
                continue
            if _is_link(value):
                source = _resolve_linked_input_source(api, str(node_id), widget_name)
                if source is None:
                    continue
                source_node_id, source_widget = source
                source_node = api.get(source_node_id)
                if not isinstance(source_node, dict):
                    continue
                source_value = (source_node.get("inputs") or {}).get(source_widget)
                if source_value is None or _is_link(source_value):
                    continue
                out.append(
                    Param(
                        node_id=source_node_id,
                        class_type=source_node.get("class_type") or "",
                        widget_name=source_widget,
                        value=source_value,
                        type=_ROLE_TYPES.get(role, _infer_type(source_value)),
                        roles={role},
                        tier=TIER_COMMON,
                        flag_name=role.replace("_", "-"),
                        source_predicates=["class_type_roles:linked_source"],
                    )
                )
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_ROLE_TYPES.get(role, _infer_type(value)),
                    roles={role},
                    tier=TIER_COMMON,
                    flag_name=role.replace("_", "-"),
                    source_predicates=["class_type_roles"],
                )
            )
    return out


def turbo_switches(api: dict, ui: dict | None) -> list[Param]:
    model_toggles: set[str] = set()
    step_toggles: set[str] = set()

    for node_id, node in api.items():
        if not isinstance(node, dict) or node.get("class_type") not in {"ComfySwitchNode", "SwitchNode"}:
            continue
        inputs = node.get("inputs") or {}
        switch = inputs.get("switch")
        if not _is_link(switch):
            continue
        toggle_id = str(switch[0])
        toggle = api.get(toggle_id)
        if not isinstance(toggle, dict) or toggle.get("class_type") != "PrimitiveBoolean":
            continue

        on_true = inputs.get("on_true")
        on_false = inputs.get("on_false")
        true_node = api.get(str(on_true[0])) if _is_link(on_true) else None
        true_is_acceleration_layer = (
            isinstance(true_node, dict)
            and "loraloader" in str(true_node.get("class_type", "")).lower()
            and true_node.get("inputs", {}).get("model") == on_false
        )
        if true_is_acceleration_layer:
            model_toggles.add(toggle_id)

        full_steps = _literal_value_source(api, on_false)
        turbo_steps = _literal_value_source(api, on_true)
        feeds_steps = any(
            candidate.get("class_type") in _pu._STEPS_CLASS_TYPES
            and candidate.get("inputs", {}).get("steps") == [str(node_id), 0]
            for candidate in api.values()
            if isinstance(candidate, dict)
        )
        if feeds_steps and isinstance(full_steps, int) and isinstance(turbo_steps, int) and full_steps > turbo_steps:
            step_toggles.add(toggle_id)

    out: list[Param] = []
    for toggle_id in sorted(model_toggles & step_toggles):
        toggle = api[toggle_id]
        widget_name = _primitive_value_widget(toggle)
        value = toggle.get("inputs", {}).get(widget_name) if widget_name else None
        if not isinstance(value, bool):
            continue
        out.append(Param(
            node_id=toggle_id,
            class_type="PrimitiveBoolean",
            widget_name=widget_name,
            value=value,
            type="BOOLEAN",
            roles={"turbo"},
            tier=TIER_HEADLINE,
            flag_name="turbo",
            label="Turbo",
            source_predicates=["turbo_switches"],
        ))
    return out


def _slug(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _kebab(s: str) -> str:
    return _slug(s).replace("_", "-")


def set_node_pairs(api: dict, ui: dict | None) -> list[Param]:
    if ui is None:
        return []

    nodes_by_id = {n.get("id"): n for n in (ui.get("nodes") or [])}
    links_by_id = {link[0]: link for link in (ui.get("links") or []) if link}

    out: list[Param] = []
    for node in ui.get("nodes") or []:
        if node.get("type") != "SetNode":
            continue
        title = node.get("title") or ""
        if not title.startswith("Set_"):
            continue
        slug = _slug(title[len("Set_"):])
        if not slug:
            continue

        inputs = node.get("inputs") or []
        if not inputs:
            continue
        link_id = inputs[0].get("link")
        link = links_by_id.get(link_id) if link_id is not None else None
        if link is None:
            continue
        src_node_id = link[1]
        src_node = nodes_by_id.get(src_node_id)
        if src_node is None:
            continue

        api_node = api.get(str(src_node_id))
        if not isinstance(api_node, dict):
            continue
        class_type = api_node.get("class_type") or src_node.get("type") or ""
        title_label = title[len("Set_"):].replace("_", " ")

        for widget_name, value in (api_node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(src_node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={f"set:{slug}"},
                    tier=TIER_HEADLINE,
                    flag_name=f"set-{_kebab(slug)}-{_kebab(widget_name)}",
                    label=title_label,
                    source_predicates=["set_node_pairs"],
                )
            )
    return out


_TYPED_PRIMITIVE_CLASSES = frozenset({
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveBoolean",
    "String",
    "StringMultiline",
    "Int",
    "Float",
    "Boolean",
})


def primitive_nodes(api: dict, ui: dict | None) -> list[Param]:
    out: list[Param] = []
    referenced = _referenced_node_ids(api)

    # Typed primitives via API (no UI required).
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        if class_type not in _TYPED_PRIMITIVE_CLASSES:
            continue
        if _is_unreferenced_typed_primitive(referenced, str(node_id), class_type):
            continue
        title = ((node.get("_meta") or {}).get("title")) if isinstance(node.get("_meta"), dict) else None
        slug = _slug(title) if title else f"node_{node_id}"
        for widget_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={f"primitive:{slug}"},
                    tier=TIER_HEADLINE,
                    flag_name=_kebab(slug),
                    label=title,
                    source_predicates=["primitive_nodes"],
                )
            )

    # Legacy PrimitiveNode requires UI to find consumers.
    if ui is None:
        return out

    nodes_by_id = {n.get("id"): n for n in (ui.get("nodes") or [])}
    links_by_id = {link[0]: link for link in (ui.get("links") or []) if link}

    for node in ui.get("nodes") or []:
        if node.get("type") != "PrimitiveNode":
            continue
        title = node.get("title") or ""
        slug = _slug(title) if title else f"node_{node.get('id')}"

        outputs = node.get("outputs") or []
        for slot_idx, output in enumerate(outputs):
            for link_id in output.get("links") or []:
                link = links_by_id.get(link_id)
                if link is None:
                    continue
                dst_node_id = link[3]
                dst_slot = link[4]
                dst_node = nodes_by_id.get(dst_node_id)
                if dst_node is None:
                    continue
                dst_inputs = dst_node.get("inputs") or []
                if dst_slot >= len(dst_inputs):
                    continue
                widget = (dst_inputs[dst_slot].get("widget") or {})
                widget_name = widget.get("name")
                if not widget_name:
                    continue
                api_node = api.get(str(dst_node_id))
                if not isinstance(api_node, dict):
                    continue
                value = (api_node.get("inputs") or {}).get(widget_name)
                if value is None or _is_link(value):
                    continue
                out.append(
                    Param(
                        node_id=str(dst_node_id),
                        class_type=api_node.get("class_type") or dst_node.get("type") or "",
                        widget_name=widget_name,
                        value=value,
                        type=_infer_type(value),
                        roles={f"primitive:{slug}"},
                        tier=TIER_HEADLINE,
                        flag_name=_kebab(slug),
                        label=title or None,
                        source_predicates=["primitive_nodes"],
                    )
                )
    return out


_EASY_PACK_ROLES: dict[tuple[str, str], str] = {
    ("easy seed", "seed"): "seed",
    ("easy positive", "positive"): "prompt",
    ("easy negative", "negative"): "negative_prompt",
}


def easy_pack_nodes(api: dict, ui: dict | None) -> list[Param]:
    out: list[Param] = []
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        for widget_name, value in (node.get("inputs") or {}).items():
            role = _EASY_PACK_ROLES.get((class_type, widget_name))
            if role is None:
                continue
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={role},
                    tier=TIER_HEADLINE,
                    flag_name=role.replace("_", "-"),
                    source_predicates=["easy_pack_nodes"],
                )
            )
    return out


def titled_nodes(api: dict, ui: dict | None) -> list[Param]:
    if ui is None:
        return []

    out: list[Param] = []
    for node in ui.get("nodes") or []:
        title = node.get("title")
        if not title or not isinstance(title, str):
            continue
        if title.startswith(("Set_", "Get_")):
            continue
        node_type = node.get("type") or ""
        if node_type in {"PrimitiveNode", "Note", "MarkdownNote", "Reroute"}:
            continue
        if node_type in _TYPED_PRIMITIVE_CLASSES:
            continue

        nid = str(node.get("id"))
        api_node = api.get(nid)
        if not isinstance(api_node, dict):
            continue
        class_type = api_node.get("class_type") or node_type

        for widget_name, value in (api_node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=nid,
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={f"title:{_slug(title)}"},
                    tier=TIER_COMMON,
                    flag_name=f"{_kebab(title)}-{_kebab(widget_name)}",
                    label=title,
                    source_predicates=["titled_nodes"],
                )
            )
    return out


def workflow_extra_metadata(api: dict, ui: dict | None) -> list[Param]:
    source = ui if ui is not None else api
    extra = source.get("extra") if isinstance(source, dict) else None
    if not isinstance(extra, dict):
        return []
    entries = extra.get("parameters")
    if not isinstance(entries, list):
        return []

    out: list[Param] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("node_id") or "")
        widget_name = entry.get("widget_name") or ""
        if not node_id or not widget_name:
            continue
        api_node = api.get(node_id)
        if not isinstance(api_node, dict):
            continue
        value = (api_node.get("inputs") or {}).get(widget_name)
        if value is None or _is_link(value):
            continue
        role = entry.get("role")
        roles = {role} if isinstance(role, str) and role else set()
        out.append(
            Param(
                node_id=node_id,
                class_type=api_node.get("class_type") or "",
                widget_name=widget_name,
                value=value,
                type=_infer_type(value),
                roles=roles,
                tier=TIER_HEADLINE,
                flag_name=entry.get("flag") if isinstance(entry.get("flag"), str) else None,
                label=entry.get("label") if isinstance(entry.get("label"), str) else None,
                source_predicates=["workflow_extra_metadata"],
            )
        )
    return out


def linear_app_inputs(api: dict, ui: dict | None) -> list[Param]:
    """Honour the frontend's "App Mode" inputs at ``extra.linearData.inputs``.

    Format (from ``src/stores/appModeStore.ts``): list of ``[node_id, widget_name]``
    pairs that the workflow author explicitly promoted to user-facing inputs
    in the linear/app builder. Authoritative — these are the canonical
    parameters when present.
    """
    source = ui if ui is not None else api
    extra = source.get("extra") if isinstance(source, dict) else None
    if not isinstance(extra, dict):
        return []
    linear_data = extra.get("linearData")
    if not isinstance(linear_data, dict):
        return []
    pairs = linear_data.get("inputs")
    if not isinstance(pairs, list):
        return []

    out: list[Param] = []
    for entry in pairs:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        node_id, widget_name = entry[0], entry[1]
        node_id = str(node_id) if node_id is not None else ""
        if not node_id or not isinstance(widget_name, str):
            continue
        api_node = api.get(node_id)
        if not isinstance(api_node, dict):
            continue
        value = (api_node.get("inputs") or {}).get(widget_name)
        if value is None or _is_link(value):
            continue
        out.append(
            Param(
                node_id=node_id,
                class_type=api_node.get("class_type") or "",
                widget_name=widget_name,
                value=value,
                type=_infer_type(value),
                roles={"app_input"},
                tier=TIER_HEADLINE,
                flag_name=f"set-{_kebab(node_id)}-{_kebab(widget_name)}",
                source_predicates=["linear_app_inputs"],
            )
        )
    return out


def promoted_widgets_metadata(api: dict, ui: dict | None) -> list[Param]:
    source = ui if ui is not None else api
    extra = source.get("extra") if isinstance(source, dict) else None
    if not isinstance(extra, dict):
        return []
    entries = extra.get("promotionEntries") or extra.get("promotions")
    if not isinstance(entries, list):
        return []

    out: list[Param] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("interiorNodeId") or entry.get("node_id") or "")
        widget_name = entry.get("widgetName") or entry.get("widget_name") or ""
        if not node_id or not widget_name:
            continue
        api_node = api.get(node_id)
        if not isinstance(api_node, dict):
            continue
        value = (api_node.get("inputs") or {}).get(widget_name)
        if value is None or _is_link(value):
            continue
        out.append(
            Param(
                node_id=node_id,
                class_type=api_node.get("class_type") or "",
                widget_name=widget_name,
                value=value,
                type=_infer_type(value),
                roles={"frontend_promoted"},
                tier=TIER_HEADLINE,
                flag_name=f"set-{_kebab(node_id)}-{_kebab(widget_name)}",
                source_predicates=["promoted_widgets_metadata"],
            )
        )
    return out


_IMAGE_INPUT_CLASSES = frozenset(_pu._IMAGE_LOAD_CLASS_TYPES) | frozenset({
    "BatchImagesNode", "GeminiInputFiles", "OpenAIInputFiles",
    "LumaReferenceNode", "LumaConceptsNode",
    "MeshyTextureNode", "MeshyAnimateModelNode", "MeshyRefineNode",
})
_VIDEO_INPUT_CLASSES = frozenset(_pu._VIDEO_LOAD_CLASS_TYPES) | frozenset({
    "VHS_LoadVideo", "VHS_LoadVideoFFmpeg",
})
_AUDIO_INPUT_CLASSES = frozenset(_pu._AUDIO_LOAD_CLASS_TYPES) | frozenset({
    "VHS_LoadAudio",
})


def assign_ui_groups(workflow: dict, params: list[Param]) -> dict[str, list[Param]]:
    """Bucket *params* by the UI workflow's groups.

    Returns ``{group_title: [Param, ...]}``. Params whose source node falls
    outside every group bounding box (or when no groups exist) bucket under
    the empty-string key.

    LiteGraph groups carry ``bounding=[x, y, w, h]`` and nodes carry
    ``pos=[x, y]``; we pick the most specific (smallest-area) group that
    contains the node so nested groups behave intuitively.
    """
    if not isinstance(workflow, dict) or "nodes" not in workflow:
        return {"": list(params)}
    groups = workflow.get("groups") or []
    if not isinstance(groups, list) or not groups:
        return {"": list(params)}

    boxes: list[tuple[float, str]] = []
    rect_by_title: dict[str, tuple[float, float, float, float]] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        bb = g.get("bounding")
        title = g.get("title")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4 or not isinstance(title, str):
            continue
        x, y, w, h = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        rect_by_title[title] = (x, y, w, h)
        boxes.append((w * h, title))
    boxes.sort()  # smallest-area group wins

    pos_by_id: dict[str, tuple[float, float]] = {}
    for node in workflow.get("nodes") or []:
        nid = node.get("id")
        pos = node.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2 and nid is not None:
            pos_by_id[str(nid)] = (float(pos[0]), float(pos[1]))

    out: dict[str, list[Param]] = {}
    for p in params:
        pos = pos_by_id.get(p.node_id)
        bucket = ""
        if pos is not None:
            for _area, title in boxes:
                rect = rect_by_title[title]
                if rect[0] <= pos[0] <= rect[0] + rect[2] and rect[1] <= pos[1] <= rect[1] + rect[3]:
                    bucket = title
                    break
        out.setdefault(bucket, []).append(p)
    return out


def count_input_slots(workflow: dict) -> dict[str, tuple[int, int]]:
    """Return ``{kind: (active, total)}`` for image/video/audio loader nodes.

    Reads the UI workflow when available so bypassed (mode==4) loaders count
    toward ``total``. For API-format workflows, ``active == total``.
    """
    active = {"images": 0, "videos": 0, "audios": 0}
    total = {"images": 0, "videos": 0, "audios": 0}

    if isinstance(workflow, dict) and "nodes" in workflow and "links" in workflow:
        for node in workflow.get("nodes") or []:
            ntype = node.get("type") or ""
            mode = node.get("mode", 0)
            kind = (
                "images" if ntype in _IMAGE_INPUT_CLASSES
                else "videos" if ntype in _VIDEO_INPUT_CLASSES
                else "audios" if ntype in _AUDIO_INPUT_CLASSES
                else None
            )
            if kind is None:
                continue
            total[kind] += 1
            if mode not in (2, 4):
                active[kind] += 1
    else:
        for node in (workflow or {}).values():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type") or ""
            kind = (
                "images" if ctype in _IMAGE_INPUT_CLASSES
                else "videos" if ctype in _VIDEO_INPUT_CLASSES
                else "audios" if ctype in _AUDIO_INPUT_CLASSES
                else None
            )
            if kind is None:
                continue
            total[kind] += 1
            active[kind] += 1

    return {k: (active[k], total[k]) for k in active}


def prompt_polarity(api: dict, ui: dict | None) -> list[Param]:
    out: list[Param] = []
    positive_node = _pu.find_positive_text_encoder(api)
    negative_node = _pu.find_negative_text_encoder(api)

    def _emit(node_id: str | None, role: str) -> None:
        if node_id is None:
            return
        node = api.get(node_id) or {}
        class_type = node.get("class_type") or ""
        fields = _pu._TEXT_ENCODE_FIELDS.get(class_type, [])
        for field_name in fields:
            if field_name not in (node.get("inputs") or {}):
                continue
            value = node["inputs"][field_name]
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=field_name,
                    value=value,
                    type=_infer_type(value),
                    roles={role},
                    tier=TIER_COMMON,
                    flag_name=role.replace("_", "-"),
                    source_predicates=["prompt_polarity"],
                )
            )

    _emit(positive_node, "prompt")
    _emit(negative_node, "negative_prompt")
    return out


_PREDICATES: list[tuple[str, Predicate]] = [
    ("frontend_widget_pool", frontend_widget_pool),
    ("class_type_roles", class_type_roles),
    ("turbo_switches", turbo_switches),
    ("prompt_polarity", prompt_polarity),
    ("set_node_pairs", set_node_pairs),
    ("primitive_nodes", primitive_nodes),
    ("easy_pack_nodes", easy_pack_nodes),
    ("titled_nodes", titled_nodes),
    ("workflow_extra_metadata", workflow_extra_metadata),
    ("promoted_widgets_metadata", promoted_widgets_metadata),
    ("linear_app_inputs", linear_app_inputs),
]


def _merge(into: dict[tuple[str, str], Param], candidate: Param) -> None:
    existing = into.get(candidate.address)
    if existing is None:
        into[candidate.address] = candidate
        return
    existing.roles |= candidate.roles
    if candidate.tier < existing.tier:
        existing.tier = candidate.tier
    for source in candidate.source_predicates:
        if source not in existing.source_predicates:
            existing.source_predicates.append(source)
    if existing.flag_name is None and candidate.flag_name is not None:
        existing.flag_name = candidate.flag_name
    if existing.label is None and candidate.label is not None:
        existing.label = candidate.label
    if candidate.type != "ANY" and (
        existing.type == "ANY"
        or any(source.startswith("class_type_roles") for source in candidate.source_predicates)
    ):
        existing.type = candidate.type
    if not existing.options and candidate.options:
        existing.options = candidate.options


def _to_api(workflow: dict, *, node_mappings=None) -> tuple[dict, dict | None]:
    if is_ui_workflow(workflow):
        return convert_ui_to_api(workflow, node_mappings=node_mappings), workflow
    return workflow, None


def _rank(params: list[Param]) -> list[Param]:
    return sorted(params, key=lambda p: (p.tier, p.node_id, p.widget_name))


def _param_value_key(value: Any) -> str:
    try:
        return json_dumps_stable(value)
    except TypeError:
        return repr(value)


def json_dumps_stable(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _merge_equivalent_flag_params(params: list[Param]) -> list[Param]:
    out: list[Param] = []
    seen: dict[tuple[str, int, str, str], Param] = {}
    for param in params:
        if not param.flag_name:
            out.append(param)
            continue
        key = (param.flag_name, param.tier, param.type, _param_value_key(param.value))
        existing = seen.get(key)
        if existing is None:
            seen[key] = param
            out.append(param)
            continue
        existing.roles |= param.roles
        for source in param.source_predicates:
            if source not in existing.source_predicates:
                existing.source_predicates.append(source)
        if existing.label is None and param.label is not None:
            existing.label = param.label
        if not existing.options and param.options:
            existing.options = param.options
        for address in param.addresses:
            if address != existing.address and address not in existing.targets:
                existing.targets.append(address)
    return out


def _disambiguate_duplicate_flag_names(params: list[Param]) -> list[Param]:
    by_flag: dict[str, list[Param]] = {}
    for param in params:
        if param.flag_name:
            by_flag.setdefault(param.flag_name, []).append(param)

    for flag_name, flagged in by_flag.items():
        if len(flagged) <= 1:
            continue
        for idx, param in enumerate(flagged, start=1):
            param.flag_name = f"{flag_name}-{idx}"
    return params


def _normalize_flag_names(params: list[Param]) -> list[Param]:
    """Collapse equivalent generated flags and suffix non-equivalent collisions."""
    return _disambiguate_duplicate_flag_names(_merge_equivalent_flag_params(params))


def discover(workflow: dict, *, node_mappings=None) -> list[Param]:
    api, ui = _to_api(workflow, node_mappings=node_mappings)
    candidates: dict[tuple[str, str], Param] = {}
    for _name, predicate in _PREDICATES:
        for cand in predicate(api, ui):
            _merge(candidates, cand)
    return _normalize_flag_names(_rank(list(candidates.values())))


def params_by_role(params: Iterable[Param], role: str) -> list[Param]:
    return [p for p in params if role in p.roles]


def params_by_address(
    params: Iterable[Param], node_id: str, widget_name: str
) -> Param | None:
    for p in params:
        if p.node_id == str(node_id) and p.widget_name == widget_name:
            return p
    return None


def apply(workflow: dict, param: Param, value: Any) -> dict:
    """Return a copy of *workflow* (API format) with *param*'s widget set to *value*."""
    if is_ui_workflow(workflow):
        raise ValueError(
            "apply() expects an API-format workflow; call convert_ui_to_api first"
        )
    out = copy.deepcopy(workflow)
    for node_id, widget_name in param.addresses:
        node = out.get(node_id)
        if not isinstance(node, dict):
            raise KeyError(f"node {node_id!r} not in workflow")
        inputs = node.setdefault("inputs", {})
        inputs[widget_name] = value
    return out


def format_params_text(params: list[Param], *, show_all: bool = False) -> str:
    sections = [
        (TIER_HEADLINE, "Headline parameters (Set_<Name>, primitives, easy-pack)"),
        (TIER_COMMON, "Common parameters (sampler/prompt/loader/dimensions/...)"),
    ]
    if show_all:
        sections.append((TIER_ADVANCED, "Advanced parameters (every other non-disabled widget)"))

    lines: list[str] = []
    for tier, heading in sections:
        rows = [p for p in params if p.tier == tier]
        if not rows:
            continue
        if lines:
            lines.append("")
        lines.append(f"# {heading}")
        for p in rows:
            roles = ",".join(sorted(p.roles)) if p.roles else "-"
            label = f" ({p.label})" if p.label else ""
            lines.append(
                f"  [{roles}]  {p.class_type}.{p.widget_name}  "
                f"node={p.node_id}{label}  value={p.value!r}"
            )

    if not show_all:
        n_advanced = sum(1 for p in params if p.tier == TIER_ADVANCED)
        if n_advanced:
            if lines:
                lines.append("")
            lines.append(f"({n_advanced} advanced params hidden — pass --all to show.)")

    return "\n".join(lines)


def apply_role(
    workflow: dict,
    role: str,
    value: Any,
    *,
    params: list[Param] | None = None,
) -> dict:
    if is_ui_workflow(workflow):
        raise ValueError(
            "apply_role() expects an API-format workflow; call convert_ui_to_api first"
        )
    if params is None:
        params = discover(workflow)
    matches = params_by_role(params, role)
    if not matches:
        return workflow
    out = copy.deepcopy(workflow)
    for p in matches:
        for node_id, widget_name in p.addresses:
            node = out.get(node_id)
            if not isinstance(node, dict):
                continue
            inputs = node.setdefault("inputs", {})
            if _is_link(inputs.get(widget_name)):
                continue
            inputs[widget_name] = value
    return out
