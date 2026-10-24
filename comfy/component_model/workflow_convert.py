"""
Convert UI (LiteGraph) workflow format to API format.

The UI format stores workflows with ``nodes``, ``links``, and ``widgets_values``.
The API format stores workflows as ``{node_id: {"class_type": ..., "inputs": ...}}``.

Conversion requires node INPUT_TYPES, so the node system must be booted first
(``import_all_nodes_in_workspace``).

The logic mirrors the frontend ``graphToPrompt`` implementation from
``ComfyUI_frontend/src/utils/executionUtil.ts`` and its helper classes.
"""
from __future__ import annotations

import contextvars
from copy import deepcopy
import logging
import re
from types import MappingProxyType
from typing import Final, Optional

from .litegraph_types import LiteLink

logger = logging.getLogger(__name__)

# Node mappings active during a ``convert_ui_to_api`` call. Used by deep
# helpers (e.g. ``_get_inner_widget_value``) to look up INPUT_TYPES on inner
# subgraph nodes without threading the parameter through every recursion.
_active_node_mappings: contextvars.ContextVar = contextvars.ContextVar(
    "_active_node_mappings", default=None,
)

_WIDGET_TYPES: Final[frozenset[str]] = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

_VIRTUAL_NODE_TYPES: Final[frozenset[str]] = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Int",
    "Float",
    "String",
    "StringMultiline",
    "Boolean",
    "Note",
    "MarkdownNote",
    "Label (rgthree)",
    "SetNode",
    "GetNode",
})

_MODE_ALWAYS: Final[int] = 0
_MODE_ON_EVENT: Final[int] = 1
_MODE_NEVER: Final[int] = 2
_MODE_ON_TRIGGER: Final[int] = 3
_MODE_BYPASS: Final[int] = 4

_SUBGRAPH_INPUT_NODE_ID: Final[int] = -10
_SUBGRAPH_OUTPUT_NODE_ID: Final[int] = -20

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I,
)

_FRONTEND_INJECTED_WIDGETS: Final[MappingProxyType[str, tuple[tuple[str, object], ...]]] = MappingProxyType({
    "PreviewAny": (("previewMode", False),),
    "LoadAudio": (("audioUI", ""),),
    "SaveAudio": (("audioUI", ""),),
    "PreviewAudio": (("audioUI", ""),),
    "SaveAudioMP3": (("audioUI", ""),),
    "SaveAudioOpus": (("audioUI", ""),),
    "Preview3D": (("image", ""),),
    "SaveGLB": (("image", ""),),
    "RecordAudio": (("audio", ""),),
    # CustomCombo declares a single ``choice`` input in its V3 schema and
    # relies on ``accept_all_inputs=True`` + a frontend-defined widget
    # extension to render an ``index`` field plus user-authored
    # ``optionN`` entries. graphToPrompt serializes these by name even
    # though they aren't in INPUT_TYPES, so we have to mirror the naming
    # convention here.
    "CustomCombo": (
        ("index", 0),
        ("option1", ""),
        ("option2", ""),
        ("option3", ""),
        ("option4", ""),
        ("option5", ""),
        ("option6", ""),
        ("option7", ""),
        ("option8", ""),
        ("option9", ""),
        ("option10", ""),
        ("option11", ""),
        ("option12", ""),
        ("option13", ""),
    ),
})

_FRONTEND_OPTIONAL_INJECTED_WIDGETS: Final[frozenset[str]] = frozenset({"CustomCombo"})

_FRONTEND_WIDGET_SERIALIZATION_OVERRIDES: Final[
    MappingProxyType[str, tuple[tuple[str, object], ...]]
] = MappingProxyType({
    # ComfyUI_frontend/src/extensions/core/load3d.ts adds these widgets after
    # model_file and before width/height. graphToPrompt serializes actual
    # widget order, not just INPUT_TYPES order.
    "Load3D": (
        ("model_file", ""),
        ("upload 3d model", "upload3dmodel"),
        ("upload extra resources", "uploadExtraResources"),
        ("clear", "clear"),
        ("image", ""),
        ("width", 1024),
        ("height", 1024),
    ),
})

_PRIMITIVE_VALUE_NODE_TYPES: Final[frozenset[str]] = frozenset({
    "PrimitiveNode",
    "Int",
    "Float",
    "String",
    "StringMultiline",
    "Boolean",
})


def _is_widget_type(type_spec, opts=None) -> bool:
    # The camera preview extension declares a non-serializing UI widget.
    if opts and opts.get("widgetType") == "CAMERA_ANGLE_VIEW":
        return False
    if isinstance(type_spec, list):
        return True
    if not isinstance(type_spec, str):
        return False
    if type_spec in _WIDGET_TYPES:
        return True
    if type_spec == "COMFY_DYNAMICCOMBO_V3":
        return True
    if opts:
        # V3 custom widget inputs use ``socketless`` to control whether their
        # widget also exposes an input socket.  Both values still describe a
        # widget: ``True`` is widget-only, while ``False`` is widget + socket.
        # Checking truthiness here dropped socket-capable widgets such as
        # COLORS and BOUNDING_BOXES and shifted every later serialized value.
        if "socketless" in opts:
            return True
        if opts.get("widgetType"):
            return True
    return False


def _fix_unhashable_str_subclass(val):
    """Custom nodes define ``AnyType(str)`` with ``__eq__`` but no ``__hash__``,
    which Python 3 makes unhashable.  Restore ``str.__hash__`` on the type."""
    if isinstance(val, str) and type(val) is not str and getattr(type(val), '__hash__', None) is None:
        type(val).__hash__ = str.__hash__


def _input_type_and_opts(entry) -> tuple:
    if isinstance(entry, (list, tuple)):
        if len(entry) == 0:
            return None, {}
        type_spec = entry[0]
        _fix_unhashable_str_subclass(type_spec)
        opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        return type_spec, opts
    _fix_unhashable_str_subclass(entry)
    return entry, {}


_SEED_CONTROL_NAMES = frozenset({"seed", "noise_seed"})
_SEED_CONTROL_MODES = frozenset({"fixed", "randomize", "increment", "decrement"})
_MAX_SEED = 0xffffffffffffffff


def _extra_widgets_after(opts: dict, name: str = "", type_spec=None) -> list[str | None]:
    extras: list[str | None] = []
    has_seed_control = opts.get("control_after_generate")
    # The ComfyUI frontend also adds seed control widgets for INT fields
    # named "seed" or "noise_seed" even without control_after_generate,
    # so match that behavior for correct widget value alignment.
    if (
        "control_after_generate" not in opts
        and not has_seed_control
        and name in _SEED_CONTROL_NAMES
        and type_spec == "INT"
    ):
        has_seed_control = True
    if has_seed_control:
        extras.append(None)
    # PAINTER owns its upload interaction inside the custom widget and does not
    # serialize a separate upload-button value after the mask path.
    if (
        opts.get("widgetType") != "PAINTER"
        and (opts.get("image_upload") or opts.get("video_upload") or opts.get("audio_upload"))
    ):
        extras.append(None)
    return extras


def _iter_graph_nodes(workflow: dict):
    nodes = workflow.get("nodes")
    if isinstance(nodes, list):
        yield from nodes

    subgraphs = workflow.get("subgraphs")
    if isinstance(subgraphs, list):
        for subgraph in subgraphs:
            if isinstance(subgraph, dict):
                yield from _iter_graph_nodes(subgraph)

    definitions = workflow.get("definitions")
    if isinstance(definitions, dict):
        yield from _iter_graph_nodes(definitions)

    extra = workflow.get("extra")
    if isinstance(extra, dict):
        group_nodes = extra.get("groupNodes")
        if isinstance(group_nodes, dict):
            for group_node in group_nodes.values():
                if isinstance(group_node, dict):
                    yield from _iter_graph_nodes(group_node)


def _next_seed_value(base_seed: int, control: str, index: int, random_seed) -> int:
    if control == "randomize":
        return int(random_seed())
    if control == "increment":
        return (base_seed + index) % (_MAX_SEED + 1)
    if control == "decrement":
        return (base_seed - index) % (_MAX_SEED + 1)
    return base_seed


def apply_ui_seed_quantity(
    workflow: dict,
    index: int,
    *,
    seed: int | None = None,
    random_seed=None,
    node_mappings=None,
) -> dict:
    """Return a UI workflow copy with seed widgets advanced for a quantity run.

    This mirrors frontend queueing semantics by reading the hidden
    ``control_after_generate`` widget saved after ``seed`` / ``noise_seed``.
    """
    if random_seed is None:
        import random as _random
        random_seed = lambda: _random.SystemRandom().randint(0, _MAX_SEED)
    if node_mappings is None:
        from ..nodes_context import get_nodes
        node_mappings = get_nodes()

    workflow = deepcopy(workflow)
    for node in _iter_graph_nodes(workflow):
        if not isinstance(node, dict):
            continue
        widgets_values = node.get("widgets_values")
        if not isinstance(widgets_values, list):
            continue
        class_def = _get_node_class(node_mappings, node.get("type", ""))
        input_types = _get_input_types(class_def) if class_def is not None else None
        if not input_types:
            continue

        idx = 0
        for name, entry in list(input_types.get("required", {}).items()) + list(input_types.get("optional", {}).items()):
            type_spec, opts = _input_type_and_opts(entry)
            if not _is_widget_type(type_spec, opts):
                continue
            if opts.get("forceInput"):
                continue

            value_idx = idx
            idx += 1
            extras = _extra_widgets_after(opts, name=name, type_spec=type_spec)
            control_idx = idx if extras and idx < len(widgets_values) else None
            idx += len(extras)

            if name not in _SEED_CONTROL_NAMES or type_spec != "INT" or value_idx >= len(widgets_values):
                continue

            control = "fixed"
            if control_idx is not None and widgets_values[control_idx] in _SEED_CONTROL_MODES:
                control = widgets_values[control_idx]
            base_seed = seed if seed is not None else widgets_values[value_idx]
            try:
                base_seed = int(base_seed)
            except (TypeError, ValueError):
                base_seed = 0
            widgets_values[value_idx] = _next_seed_value(base_seed, control, index, random_seed)

    return workflow


def _wrap_value(val, type_spec=None):
    if type_spec == "CURVE" and val is not None:
        return {"__type__": "CURVE", "__value__": val}
    return {"__value__": val} if isinstance(val, list) else val


def _frontend_widget_default(type_spec, opts: dict):
    """Mirror frontend widget constructor defaults.

    This follows the current ComfyUI frontend widget constructors used by
    ``graphToPrompt`` after the workflow has been configured:
    - STRING/TEXTAREA -> default or ""
    - INT/FLOAT -> default or 0
    - BOOLEAN -> default or False
    - COMBO -> explicit default, otherwise first option, otherwise "Loading..."
      for remote combos
    """
    widget_type = opts.get("widgetType", type_spec)

    if isinstance(type_spec, list) or widget_type == "COMBO":
        if "default" in opts:
            return opts["default"]
        if isinstance(type_spec, list) and type_spec:
            return type_spec[0]
        if isinstance(opts.get("options"), list) and opts["options"]:
            return opts["options"][0]
        if opts.get("remote"):
            return "Loading..."
        return None

    if widget_type == "COMFY_DYNAMICCOMBO_V3":
        if "default" in opts:
            return opts["default"]
        options = opts.get("options", [])
        if options and isinstance(options[0], dict):
            key = options[0].get("key")
            if key is not None:
                return getattr(key, 'value', key)
        return None

    if widget_type in ("STRING", "TEXTAREA"):
        return opts.get("default", "")
    if widget_type == "INT":
        return opts.get("default", 0)
    if widget_type == "FLOAT":
        return opts.get("default", 0)
    if widget_type == "BOOLEAN":
        return opts.get("default", False)
    if widget_type in ("COLOR", "CURVE", "BOUNDING_BOX", "MARKDOWN", "CHART", "GALLERIA"):
        return opts.get("default")

    return opts.get("default")


def _serialized_widget_names_ordered(node: dict | None) -> list[str]:
    if node is None:
        return []

    stashed = node.get('_widget_names_ordered')
    if isinstance(stashed, list):
        return [name for name in stashed if isinstance(name, str)]

    ordered_names: list[str] = []
    for inp in node.get('inputs', []) or []:
        widget = inp.get('widget')
        if isinstance(widget, dict):
            name = widget.get('name') or inp.get('name')
            if isinstance(name, str):
                ordered_names.append(name)
    return ordered_names


def _map_widgets(input_types: dict, widgets_values: list, node: dict | None = None) -> tuple[dict[str, object], int]:
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})

    result: dict[str, object] = {}
    idx = 0
    in_optional = False
    serialized_widget_names = set(_serialized_widget_names_ordered(node))

    for name, entry in list(required.items()) + list(optional.items()):
        if not in_optional and name in optional:
            in_optional = True

        type_spec, opts = _input_type_and_opts(entry)

        if not _is_widget_type(type_spec, opts):
            continue
        if opts.get("forceInput"):
            if name in serialized_widget_names and idx < len(widgets_values):
                idx += 1
            continue

        if idx < len(widgets_values):
            val = widgets_values[idx]
            # Coerce empty-string placeholders: some saved workflows write ""
            # for INT/FLOAT fields that were left blank in the UI.  Fall back
            # to the default so the prompt passes validation.
            if val == "" and type_spec in ("INT", "FLOAT") and "default" in opts:
                val = opts["default"]
            result[name] = _wrap_value(val, type_spec)
            idx += 1
        else:
            default_value = _frontend_widget_default(type_spec, opts)
            if default_value is None:
                continue
            result[name] = _wrap_value(default_value, type_spec)

        for extra_name in _extra_widgets_after(opts, name=name, type_spec=type_spec):
            if idx < len(widgets_values):
                if extra_name is not None:
                    result[extra_name] = _wrap_value(widgets_values[idx])
                idx += 1

        if type_spec == "COMFY_DYNAMICCOMBO_V3":
            idx = _consume_dynamic_combo_subwidgets(
                name, result.get(name), opts, widgets_values, idx, result,
            )

    return result, idx


def _consume_dynamic_combo_subwidgets(
    parent_name: str,
    selected_key,
    opts: dict,
    widgets_values: list,
    idx: int,
    result: dict,
) -> int:
    options = opts.get("options", [])
    if not options or selected_key is None:
        return idx

    matched_option = None
    for option in options:
        if not isinstance(option, dict):
            continue
        key = option.get("key")
        if key == selected_key or getattr(key, "value", None) == selected_key:
            matched_option = option
            break
    if matched_option is None:
        return idx

    sub_inputs = matched_option.get("inputs", {})
    for section in ("required", "optional"):
        for sub_name, sub_entry in sub_inputs.get(section, {}).items():
            sub_type, sub_opts = _input_type_and_opts(sub_entry)
            if not _is_widget_type(sub_type, sub_opts):
                continue
            if sub_opts.get("forceInput"):
                continue
            dotted = f"{parent_name}.{sub_name}"
            if idx < len(widgets_values):
                sub_value = widgets_values[idx]
                result[dotted] = _wrap_value(sub_value, sub_type)
                idx += 1
            else:
                sub_value = _frontend_widget_default(sub_type, sub_opts)
                if sub_value is None:
                    continue
                result[dotted] = _wrap_value(sub_value, sub_type)

            # The frontend does not add the auto seed control widget for
            # dynamic combo sub-inputs (only top-level seed/noise_seed widgets
            # get it). Only honor explicit control_after_generate here.
            if sub_opts.get("control_after_generate"):
                if idx < len(widgets_values):
                    idx += 1

            # ``dynamicComboWidget`` calls the regular node-input factory for
            # each selected sub-input. A nested DynamicCombo therefore adds
            # its own selected widgets recursively (for example SaveVideo's
            # format.codec.encoding hierarchy), immediately after its parent.
            if sub_type == "COMFY_DYNAMICCOMBO_V3":
                idx = _consume_dynamic_combo_subwidgets(
                    dotted,
                    sub_value,
                    sub_opts,
                    widgets_values,
                    idx,
                    result,
                )

    return idx


def _map_widgets_dict(input_types: dict, widgets_values: dict) -> dict[str, object]:
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    all_inputs = {**required, **optional}

    result: dict[str, object] = {}
    for name, entry in all_inputs.items():
        type_spec, opts = _input_type_and_opts(entry)
        if _is_widget_type(type_spec, opts) and not opts.get("forceInput"):
            if name in widgets_values:
                result[name] = _wrap_value(widgets_values[name], type_spec)
            else:
                default_value = _frontend_widget_default(type_spec, opts)
                if default_value is not None:
                    result[name] = _wrap_value(default_value, type_spec)
    return result


def __unknown_widget_value(val):
    if isinstance(val, (str, int, float, bool)):
        return val
    import json as _json
    return _json.dumps(val)


def _unknown_widget_value_matches_type(value, type_spec) -> bool:
    if type_spec == "BOOLEAN":
        return isinstance(value, bool)
    if type_spec == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_spec == "FLOAT":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_spec == "STRING":
        return isinstance(value, str)
    return True


def _align_unknown_widget_values(node: dict, widgets_values: list, ordered_names: list[str]) -> list[object] | None:
    types = node.get('_widget_types_ordered')
    if not isinstance(types, list) or len(types) != len(ordered_names):
        widget_inputs = {
            (inp.get('widget') or {}).get('name') or inp.get('name'): inp
            for inp in node.get('inputs', []) or []
            if isinstance(inp.get('widget'), dict)
        }
        types = [widget_inputs.get(name, {}).get('type') for name in ordered_names]
    if not all(types):
        return None

    matches = {
        idx: (idx,)
        for idx, value in enumerate(widgets_values)
        if _unknown_widget_value_matches_type(value, types[0])
    }
    for type_spec in types[1:]:
        next_matches: dict[int, tuple[int, ...]] = {}
        for idx, value in enumerate(widgets_values):
            if not _unknown_widget_value_matches_type(value, type_spec):
                continue
            previous = [match for end, match in matches.items() if end < idx]
            if previous:
                next_matches[idx] = (*max(previous, key=lambda match: match[0]), idx)
        matches = next_matches
        if not matches:
            return None

    indices = min(matches.values(), key=lambda match: (match[-1] - match[0] + 1 - len(match), match[0]))
    return [widgets_values[idx] for idx in indices]


def _map_unknown_widgets(widgets_values, node: dict | None = None) -> dict[str, object]:
    if isinstance(widgets_values, dict):
        return {k: _wrap_value(v) for k, v in widgets_values.items()}
    if isinstance(widgets_values, list) and widgets_values:
        ordered_names: list[str] = []
        if node is not None:
            stashed = node.get('_widget_names_ordered')
            if isinstance(stashed, list):
                ordered_names = [n for n in stashed if isinstance(n, str)]
            if not ordered_names:
                for inp in node.get('inputs', []) or []:
                    widget = inp.get('widget')
                    if isinstance(widget, dict):
                        name = widget.get('name') or inp.get('name')
                        if isinstance(name, str):
                            ordered_names.append(name)
        if ordered_names:
            aligned_values = _align_unknown_widget_values(node, widgets_values, ordered_names)
            if aligned_values is not None:
                return {
                    name: _wrap_value(value)
                    for name, value in zip(ordered_names, aligned_values)
                }
            out: dict[str, object] = {}
            for i, name in enumerate(ordered_names):
                if i >= len(widgets_values):
                    break
                out[name] = _wrap_value(widgets_values[i])
            if out:
                return out
        if node is not None:
            return {f"widget_{i}": _wrap_value(v) for i, v in enumerate(widgets_values)}
        return {
            "UNKNOWN" if i == 0 else f"UNKNOWN_{i}": _wrap_value(__unknown_widget_value(value))
            for i, value in enumerate(widgets_values)
        }
    return {}


def _map_frontend_widget_override(
    class_type: str,
    widgets_values,
) -> tuple[dict[str, object], int] | None:
    layout = _FRONTEND_WIDGET_SERIALIZATION_OVERRIDES.get(class_type)
    if layout is None or not isinstance(widgets_values, list):
        return None

    result: dict[str, object] = {}
    for idx, (name, default_value) in enumerate(layout):
        if idx < len(widgets_values):
            result[name] = _wrap_value(widgets_values[idx])
        else:
            result[name] = _wrap_value(default_value)
    return result, len(layout)


def _serialized_widget_input_names(node: dict) -> set[str]:
    names: set[str] = set()
    for inp in node.get('inputs', []) or []:
        name = inp.get('name')
        if isinstance(name, str) and inp.get('widget') is not None:
            names.add(name)
    return names


def _get_subgraph_boundary_name(sg_node: dict, slot: int, sg_def: dict | None = None) -> str | None:
    if sg_def:
        sg_inputs = sg_def.get('inputs', []) or []
        if slot < len(sg_inputs):
            name = sg_inputs[slot].get('name')
            if isinstance(name, str):
                return name

    inputs = sg_node.get('inputs', []) or []
    if slot < len(inputs):
        name = inputs[slot].get('name')
        if isinstance(name, str):
            return name

    return None


def _get_subgraph_outer_input_slot(
    sg_node: dict,
    boundary_name: str | None,
) -> tuple[int | None, dict | None]:
    if boundary_name is None:
        return None, None

    for idx, inp in enumerate(sg_node.get('inputs', []) or []):
        if inp.get('name') == boundary_name:
            return idx, inp

    return None, None


def _types_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if a == "*" or b == "*":
        return True
    a_types = {part.strip() for part in a.split(",") if part.strip()}
    b_types = {part.strip() for part in b.split(",") if part.strip()}
    if "*" in a_types or "*" in b_types:
        return True
    if a_types and b_types and not a_types.isdisjoint(b_types):
        return True
    return False


def _get_bypass_slot_index(
    inputs: list[dict], outputs: list[dict], slot: int, target_type: str | None,
) -> int:
    if not target_type or target_type == "*":
        return slot if slot < len(inputs) else 0

    out_type = outputs[slot].get("type") if slot < len(outputs) else None

    if slot < len(inputs):
        opp_type = inputs[slot].get("type")
        if _types_match(opp_type, out_type) and _types_match(opp_type, target_type):
            return slot

    for i, inp in enumerate(inputs):
        if inp.get("type") == target_type:
            return i

    for i, inp in enumerate(inputs):
        inp_type = inp.get("type")
        if _types_match(inp_type, out_type) and _types_match(inp_type, target_type):
            return i

    return -1


def _resolve_source(
    src_node_id: int,
    src_slot: int,
    nodes_by_id: dict[int, dict],
    links: dict[int, LiteLink],
    visited: set | None = None,
    target_type: str | None = None,
) -> tuple | None:
    if visited is None:
        visited = set()
    key = (src_node_id, src_slot)
    if key in visited:
        return None
    visited.add(key)

    node = nodes_by_id.get(src_node_id)
    if node is None:
        return None

    node_type = node.get("type", "")
    mode = node.get("mode", 0)

    if node_type == "Reroute":
        inputs = node.get("inputs", [])
        if inputs and inputs[0].get("link") is not None:
            link = links.get(inputs[0]["link"])
            if link:
                return _resolve_source(
                    link.src_node, link.src_slot, nodes_by_id, links,
                    visited, target_type,
                )
        return None

    if node_type in _PRIMITIVE_VALUE_NODE_TYPES:
        wv = node.get("widgets_values", [])
        if wv:
            return ("value", wv[0])
        return None

    if mode == _MODE_NEVER:
        return None

    if mode == _MODE_BYPASS:
        outputs = node.get("outputs", [])
        node_inputs = node.get("inputs", [])
        if src_slot >= len(outputs) or not node_inputs:
            return None
        out_type = outputs[src_slot].get("type")
        bypass_type = target_type or out_type
        match_idx = _get_bypass_slot_index(
            node_inputs, outputs, src_slot, bypass_type,
        )
        if match_idx == -1:
            return None
        inp = node_inputs[match_idx]
        link_id = inp.get("link")
        if link_id is not None:
            link = links.get(link_id)
            if link:
                return _resolve_source(
                    link.src_node, link.src_slot,
                    nodes_by_id, links, visited, bypass_type,
                )
        return None

    return ("link", str(src_node_id), src_slot)


def _convert_legacy_group_node(name: str, defn: dict) -> dict:
    """Convert a legacy ``extra.groupNodes`` definition to modern subgraph format.

    Legacy group nodes (``workflow>NAME``) store inner nodes with ``index``
    instead of ``id``, and links as
    ``[src_index, src_slot, dst_index, dst_slot, ???, type]``.
    """
    # Build nodes with ``id`` = ``index``
    nodes = []
    for inner in defn.get("nodes", []):
        node = dict(inner)
        if "index" in node and "id" not in node:
            node["id"] = node["index"]
        # Ensure inner nodes have proper inputs with link refs
        nodes.append(node)

    # Convert links: legacy format uses indices, not global IDs.
    # Create proper LiteGraph link arrays: [link_id, src_id, src_slot, dst_id, dst_slot, type]
    links = []
    for link_id, raw_link in enumerate(defn.get("links", []), start=1):
        if len(raw_link) >= 5:
            src_index = raw_link[0]
            src_slot = raw_link[1]
            dst_index = raw_link[2]
            dst_slot = raw_link[3]
            link_type = raw_link[5] if len(raw_link) > 5 else None
            links.append({
                "id": link_id,
                "origin_id": src_index,
                "origin_slot": src_slot,
                "target_id": dst_index,
                "target_slot": dst_slot,
                "type": link_type,
            })

    # Wire up link references on inner node inputs/outputs
    for link_info in links:
        src_id = link_info["origin_id"]
        src_slot = link_info["origin_slot"]
        dst_id = link_info["target_id"]
        dst_slot = link_info["target_slot"]

        # Add link ref to destination node's input
        for node in nodes:
            if node.get("id") == dst_id:
                inputs = node.get("inputs", [])
                # Find input at the correct slot, accounting for the legacy
                # ``dst_slot`` which is relative to the total input count
                # (including linked inputs from inner connections).
                if isinstance(inputs, list):
                    while len(inputs) <= dst_slot:
                        inputs.append({"name": f"input_{len(inputs)}", "type": link_info.get("type", "*")})
                    if inputs[dst_slot].get("link") is None:
                        inputs[dst_slot]["link"] = link_info["id"]
                node["inputs"] = inputs
                break

        # Add link ref to source node's output
        for node in nodes:
            if node.get("id") == src_id:
                outputs = node.get("outputs", [])
                if isinstance(outputs, list):
                    while len(outputs) <= src_slot:
                        outputs.append({"name": f"output_{len(outputs)}", "type": link_info.get("type", "*"), "links": []})
                    if isinstance(outputs[src_slot].get("links"), list):
                        outputs[src_slot]["links"].append(link_info["id"])
                break

    sg_id = f"workflow>{name}"
    return {
        "id": sg_id,
        "name": name,
        "nodes": nodes,
        "links": links,
    }


def _collect_subgraph_defs(workflow: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}

    def _search(obj, depth=0):
        if depth > 4 or not isinstance(obj, dict):
            return
        sgs = obj.get("subgraphs")
        if isinstance(sgs, list):
            for sg in sgs:
                if isinstance(sg, dict) and "id" in sg:
                    result[sg["id"]] = sg
        for v in obj.values():
            if isinstance(v, dict):
                _search(v, depth + 1)

    _search(workflow)

    # Also collect legacy group nodes from extra.groupNodes
    extra = workflow.get("extra", {})
    group_nodes = extra.get("groupNodes", {})
    if isinstance(group_nodes, dict):
        for name, defn in group_nodes.items():
            if isinstance(defn, dict):
                sg_id = f"workflow>{name}"
                if sg_id not in result:
                    result[sg_id] = _convert_legacy_group_node(name, defn)

    return result


def _parse_link(raw) -> LiteLink:
    if isinstance(raw, dict):
        return LiteLink.from_dict(raw)
    return LiteLink.from_list(raw)


def _matches_legacy_api_input(inp: dict) -> bool:
    return not (inp.get('widget') is not None and inp.get('link') is None and not inp.get('label'))


def _compress_widget_input_slots(workflow: dict) -> dict:
    workflow = deepcopy(workflow)

    def _compress_graph_nodes(nodes, links, dict_links: bool):
        if not isinstance(nodes, list):
            return
        for node in nodes:
            inputs = node.get('inputs')
            if not isinstance(inputs, list):
                continue
            widget_names_ordered: list[str] = []
            widget_types_ordered: list[str] = []
            for inp in inputs:
                widget = inp.get('widget')
                if isinstance(widget, dict):
                    name = widget.get('name') or inp.get('name')
                    if isinstance(name, str):
                        widget_names_ordered.append(name)
                        widget_types_ordered.append(inp.get('type'))
            if widget_names_ordered:
                node['_widget_names_ordered'] = widget_names_ordered
                node['_widget_types_ordered'] = widget_types_ordered
            compressed_inputs = [inp for inp in inputs if _matches_legacy_api_input(inp)]
            node['inputs'] = compressed_inputs

            for input_index, inp in enumerate(compressed_inputs):
                link_id = inp.get('link')
                if link_id is None or not links:
                    continue
                for raw_link in links:
                    if dict_links:
                        if raw_link.get('id') == link_id:
                            raw_link['target_slot'] = input_index
                            break
                    else:
                        if len(raw_link) > 4 and raw_link[0] == link_id:
                            raw_link[4] = input_index
                            break

    def _compress_subgraphs(subgraphs):
        if not isinstance(subgraphs, list):
            return
        for subgraph in subgraphs:
            _compress_graph_nodes(
                subgraph.get('nodes', []),
                subgraph.get('links', []),
                dict_links=True,
            )
            definitions = subgraph.get('definitions', {})
            if isinstance(definitions, dict):
                _compress_subgraphs(definitions.get('subgraphs'))

    _compress_graph_nodes(workflow.get('nodes', []), workflow.get('links', []), dict_links=False)
    definitions = workflow.get('definitions', {})
    if isinstance(definitions, dict):
        _compress_subgraphs(definitions.get('subgraphs'))
    return workflow


def _build_inner_links(sg_def: dict) -> dict[int, LiteLink]:
    inner_links: dict[int, LiteLink] = {}
    for raw in sg_def.get("links", []):
        link = _parse_link(raw)
        inner_links[link.link_id] = link
    return inner_links


class _NodeDTO:
    __slots__ = (
        'exec_id', 'node', 'graph_links', 'graph_nodes_by_id',
        'subgraph_node_path', 'sg_node_exec_id',
        'sg_def', 'inner_links', 'proxy_overrides', 'id_remap',
        'clobbered_wv',
    )

    def __init__(self, node, subgraph_node_path, graph_links, graph_nodes_by_id,
                 sg_node_exec_id=None, sg_def=None, remapped_nid=None):
        self.node = node
        self.subgraph_node_path = list(subgraph_node_path)
        nid = remapped_nid if remapped_nid is not None else node['id']
        self.exec_id = ':'.join(str(x) for x in [*subgraph_node_path, nid])
        self.graph_links = graph_links
        self.graph_nodes_by_id = graph_nodes_by_id
        self.sg_node_exec_id = sg_node_exec_id
        self.sg_def = sg_def
        self.inner_links = _build_inner_links(sg_def) if sg_def else {}
        self.proxy_overrides: dict[tuple[int, str], object] = {}
        self.id_remap: dict[int, int] = {}
        self.clobbered_wv: list | None = None


def _connected_widget_source_for_slot(sg_def: dict, slot: int) -> tuple[int, str] | None:
    for raw in sg_def.get('links', []):
        link = _parse_link(raw)
        if link.src_node != _SUBGRAPH_INPUT_NODE_ID or link.src_slot != slot:
            continue

        target_node = next(
            (inner for inner in sg_def.get('nodes', []) if inner.get('id') == link.dst_node),
            None,
        )
        if target_node is None:
            continue

        target_inputs = target_node.get('inputs', [])
        if link.dst_slot >= len(target_inputs):
            continue

        target_input = target_inputs[link.dst_slot]
        widget = target_input.get('widget')
        if widget is None:
            continue

        widget_name = widget.get('name') if isinstance(widget, dict) else None
        if widget_name is None:
            widget_name = target_input.get('name')
        if not widget_name:
            continue

        try:
            return int(target_node['id']), str(widget_name)
        except (KeyError, TypeError, ValueError):
            continue

    return None


def _compute_proxy_overrides(sg_node, parent_overrides=None):
    proxy_widgets = sg_node.get('properties', {}).get('proxyWidgets', [])
    wv = sg_node.get('widgets_values', [])
    if isinstance(wv, dict):
        wv = []

    overrides: dict[tuple[int, str], object] = {}
    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        pw_target, pw_name = pw[0], pw[1]
        if str(pw_target) == '-1':
            continue
        if pw_idx >= len(wv):
            continue
        try:
            inner_nid = int(pw_target)
        except (ValueError, TypeError):
            continue
        val = wv[pw_idx]
        if parent_overrides:
            nid = sg_node['id']
            parent_val = parent_overrides.get((nid, pw_name))
            if parent_val is not None:
                val = parent_val
        overrides[(inner_nid, pw_name)] = val

    return overrides


def _ensure_global_id_uniqueness(
    workflow: dict, sg_defs: dict[str, dict],
) -> dict[str, dict[int, int]]:
    """Mirror the frontend serialized subgraph deduplication pass.

    The frontend pre-deduplicates serialized subgraph node IDs in definition
    order using the root graph's ``lastNodeId`` as the running counter. It
    does *not* pre-scan all subgraph IDs to raise the counter first; it only
    advances ``lastNodeId`` when encountering non-conflicting IDs or when
    allocating a replacement for a collision.
    """
    outer_ids = {n['id'] for n in workflow.get('nodes', [])
                 if isinstance(n.get('id'), int)}
    last_node_id = int(workflow.get('last_node_id', max(outer_ids) if outer_ids else 0))
    subgraph_defs = workflow.get('definitions', {}).get('subgraphs', [])

    used_ids: set[int] = set(outer_ids)
    remaps: dict[str, dict[int, int]] = {}

    for sg in subgraph_defs:
        if not isinstance(sg, dict) or 'id' not in sg:
            continue
        sg_id = sg['id']
        id_remap: dict[int, int] = {}
        for n in sg.get('nodes', []):
            nid = n.get('id')
            if not isinstance(nid, int) or nid < 0:
                continue
            if nid in used_ids:
                while last_node_id + 1 in used_ids:
                    last_node_id += 1
                last_node_id += 1
                id_remap[nid] = last_node_id
                used_ids.add(last_node_id)
            else:
                used_ids.add(nid)
                if nid > last_node_id:
                    last_node_id = nid
        if id_remap:
            remaps[sg_id] = id_remap

    return remaps


def _expand_subgraph(sg_node, sg_def, subgraph_node_path,
                     sg_defs, dto_map, sg_exec_id, id_remaps=None):
    sg_dto = dto_map[sg_exec_id]
    instance_path = [int(x) for x in sg_dto.exec_id.split(':')]

    inner_links = sg_dto.inner_links
    inner_nodes_by_id: dict[int, dict] = {}
    for n in sg_def.get('nodes', []):
        nid = n.get('id')
        if nid is not None:
            inner_nodes_by_id[nid] = n

    sg_uuid = sg_def.get('id', '')
    id_remap: dict[int, int] = id_remaps.get(sg_uuid, {}) if id_remaps else {}
    sg_dto.id_remap = id_remap

    for inner_node in sg_def.get('nodes', []):
        inner_nid = inner_node.get('id')
        if inner_nid is None or inner_nid < 0:
            continue

        remapped_nid = id_remap.get(inner_nid)
        class_type = inner_node.get('type', '')
        mode = inner_node.get('mode', 0)

        if _is_subgraph_type(class_type, sg_defs):
            child_sg_def = sg_defs[class_type]
            child_dto = _NodeDTO(
                inner_node, instance_path, inner_links, inner_nodes_by_id,
                sg_node_exec_id=sg_exec_id, sg_def=child_sg_def,
                remapped_nid=remapped_nid,
            )
            child_dto.proxy_overrides = _compute_proxy_overrides(
                inner_node,
                parent_overrides=sg_dto.proxy_overrides,
            )
            dto_map[child_dto.exec_id] = child_dto

            _expand_subgraph(
                inner_node, child_sg_def, instance_path,
                sg_defs, dto_map, child_dto.exec_id,
                id_remaps=id_remaps,
            )
        else:
            dto = _NodeDTO(
                inner_node, instance_path, inner_links, inner_nodes_by_id,
                sg_node_exec_id=sg_exec_id, remapped_nid=remapped_nid,
            )
            dto_map[dto.exec_id] = dto


def _build_dto_map(workflow, sg_defs):
    links: dict[int, LiteLink] = {}
    for raw in workflow.get('links', []):
        link = LiteLink.from_list(raw)
        links[link.link_id] = link

    nodes_by_id: dict[int, dict] = {}
    for node in workflow.get('nodes', []):
        nid = node.get('id')
        if nid is not None:
            nodes_by_id[nid] = node

    id_remaps = _ensure_global_id_uniqueness(workflow, sg_defs)
    dto_map: dict[str, _NodeDTO] = {}

    for node in workflow.get('nodes', []):
        nid = node.get('id')
        if nid is None:
            continue

        mode = node.get('mode', 0)
        class_type = node.get('type', '')

        if _is_subgraph_type(class_type, sg_defs):
            sg_def = sg_defs[class_type]
            sg_dto = _NodeDTO(node, [], links, nodes_by_id, sg_def=sg_def)
            sg_dto.proxy_overrides = _compute_proxy_overrides(node)
            dto_map[sg_dto.exec_id] = sg_dto

            if mode not in (_MODE_NEVER, _MODE_BYPASS):
                _expand_subgraph(
                    node, sg_def, [], sg_defs, dto_map, sg_dto.exec_id,
                    id_remaps=id_remaps,
                )
        else:
            dto = _NodeDTO(node, [], links, nodes_by_id)
            dto_map[dto.exec_id] = dto

    return dto_map


def _get_inner_widget_value(inner_node: dict, target_name: str, node_mappings) -> tuple[bool, object]:
    """Find a widget's value on an inner subgraph node by replicating the frontend
    widget construction order.

    Mirrors ``resolvePromotedWidgetAtHost`` (frontend ``core/graph/subgraph/
    resolveConcretePromotedWidget.ts``): finds ``node.widgets[name]`` and returns
    its ``.value``. The widget index in ``widgets_values`` is determined by
    walking ``INPUT_TYPES`` in declaration order, with the same extra-widget
    logic as ``_map_widgets`` (seed control, image/video/audio upload buttons,
    DynamicCombo sub-widgets).
    """
    if node_mappings is None:
        return False, None

    class_type = inner_node.get('type')
    class_def = _get_node_class(node_mappings, class_type) if class_type else None
    if class_def is None:
        return False, None

    input_types = _get_input_types(class_def)
    if not input_types:
        return False, None

    inner_wv = inner_node.get('widgets_values', [])
    if not isinstance(inner_wv, list):
        return False, None

    # Walk widgets in the same order _map_widgets does. Track the widget index
    # used to address widgets_values, advancing past extras (seed control,
    # upload buttons, DynamicCombo sub-widgets) so the lookup matches the
    # frontend's ``node.widgets`` array indexing.
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    idx = 0
    for name, entry in list(required.items()) + list(optional.items()):
        type_spec, opts = _input_type_and_opts(entry)
        if not _is_widget_type(type_spec, opts):
            continue
        if opts.get("forceInput"):
            continue
        if name == target_name:
            if idx < len(inner_wv):
                return True, inner_wv[idx]
            default_value = _frontend_widget_default(type_spec, opts)
            return (True, default_value) if default_value is not None else (False, None)
        idx += 1
        for _ in _extra_widgets_after(opts, name=name, type_spec=type_spec):
            idx += 1
        if type_spec == "COMFY_DYNAMICCOMBO_V3":
            # Dynamic combo: skip past sub-widgets for the SELECTED option.
            selected_key = inner_wv[idx - 1] if (idx - 1) < len(inner_wv) else None
            options = opts.get("options", []) if isinstance(opts, dict) else []
            matched_option = None
            for option in options:
                if not isinstance(option, dict):
                    continue
                key = option.get("key")
                if key == selected_key or getattr(key, "value", None) == selected_key:
                    matched_option = option
                    break
            if matched_option is not None:
                sub_inputs = matched_option.get("inputs", {})
                for section in ("required", "optional"):
                    for _sub_name, sub_entry in sub_inputs.get(section, {}).items():
                        sub_type, sub_opts = _input_type_and_opts(sub_entry)
                        if not _is_widget_type(sub_type, sub_opts):
                            continue
                        if sub_opts.get("forceInput"):
                            continue
                        idx += 1
                        if sub_opts.get("control_after_generate"):
                            idx += 1

    return False, None


def _get_sg_widget_by_slot(sg_node, slot, sg_def=None):
    inp_name = _get_subgraph_boundary_name(sg_node, slot, sg_def)
    if inp_name is None:
        return False, None

    _, slot_input = _get_subgraph_outer_input_slot(sg_node, inp_name)
    if slot_input is not None and slot_input.get('widget') is None:
        return False, None

    proxy_widgets = sg_node.get('properties', {}).get('proxyWidgets', [])
    wv = sg_node.get('widgets_values', [])
    if isinstance(wv, dict):
        wv = []

    if slot < len(proxy_widgets) and slot < len(wv):
        pw = proxy_widgets[slot]
        if isinstance(pw, (list, tuple)) and len(pw) >= 2 and pw[1] == inp_name:
            return True, wv[slot]

    # Legacy group-node format: proxyWidgets entries with sentinel '-1' nodeId
    # are matched by widget name only.
    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        if str(pw[0]) == '-1' and pw[1] == inp_name:
            if pw_idx < len(wv):
                return True, wv[pw_idx]
            return False, None

    if sg_def and wv and not proxy_widgets:
        return _get_sg_widget_positional(sg_def, inp_name, wv)

    # Outer widgets_values is empty (subgraph not customised) and a proxy
    # widget is registered: read the inner promoted widget's actual value
    # by following the inner link from the subgraph_input boundary slot to
    # the target inner node's input. Mirrors ``PromotedWidgetView.value`` ->
    # ``resolveAtHost`` (frontend ``core/graph/subgraph/promotedWidgetView.ts``
    # line 131-135), which falls through to the interior widget's value when
    # ``useWidgetValueStore`` has no override for that proxy.
    if sg_def and not wv and proxy_widgets:
        source = _connected_widget_source_for_slot(sg_def, slot)
        if source is not None:
            source_node_id, source_widget_name = source
            target_node = next(
                (inner for inner in sg_def.get('nodes', []) if inner.get('id') == source_node_id),
                None,
            )
            if target_node is not None:
                found, value = _get_inner_widget_value(
                    target_node, source_widget_name, _active_node_mappings.get(),
                )
                if found:
                    return True, value

        # Walk the subgraph's inner links to find the inner node that consumes
        # this boundary slot.
        for raw in sg_def.get('links', []):
            link = _parse_link(raw)
            if link.src_node != _SUBGRAPH_INPUT_NODE_ID or link.src_slot != slot:
                continue
            target_node = None
            for inner in sg_def.get('nodes', []):
                if inner.get('id') == link.dst_node:
                    target_node = inner
                    break
            if target_node is None:
                continue
            # Determine the inner widget name from the target input slot.
            target_inputs = target_node.get('inputs', [])
            if link.dst_slot >= len(target_inputs):
                continue
            target_input = target_inputs[link.dst_slot]
            widget = target_input.get('widget')
            inner_widget_name = None
            if isinstance(widget, dict):
                inner_widget_name = widget.get('name')
            if inner_widget_name is None:
                inner_widget_name = target_input.get('name')
            if not inner_widget_name:
                continue
            return _get_inner_widget_value(
                target_node, inner_widget_name, _active_node_mappings.get(),
            )

    return False, None


def _get_sg_widget_positional(sg_def, boundary_name, wv):
    inner_links = _build_inner_links(sg_def)
    inner_nodes_by_id: dict[int, dict] = {}
    for n in sg_def.get('nodes', []):
        nid = n.get('id')
        if nid is not None:
            inner_nodes_by_id[nid] = n

    sg_def_inputs = sg_def.get('inputs', [])

    slot_is_widget: dict[int, bool] = {}
    for link in inner_links.values():
        if link.src_node != _SUBGRAPH_INPUT_NODE_ID:
            continue
        slot = link.src_slot
        if slot_is_widget.get(slot, False):
            continue
        target_node = inner_nodes_by_id.get(link.dst_node)
        if target_node is None:
            slot_is_widget[slot] = False
            continue
        for tinp in target_node.get('inputs', []):
            if tinp.get('link') == link.link_id:
                slot_is_widget[slot] = 'widget' in tinp
                break
        else:
            slot_is_widget[slot] = False

    wv_idx = 0
    for slot_idx, sg_inp in enumerate(sg_def_inputs):
        if not slot_is_widget.get(slot_idx, False):
            continue
        if sg_inp.get('name') == boundary_name:
            if wv_idx < len(wv):
                return True, wv[wv_idx]
            return False, None
        wv_idx += 1

    return False, None


def _get_set_get_channel_name(node: dict) -> str:
    """Extract the channel name from a SetNode or GetNode."""
    wv = node.get('widgets_values', [])
    if isinstance(wv, list) and wv:
        return str(wv[0]) if wv[0] else ''
    if isinstance(wv, dict):
        return str(wv.get('Constant', ''))
    return ''


def _build_set_node_map(dto_map: dict[str, '_NodeDTO']) -> dict[str, '_NodeDTO']:
    """Build a mapping from SetNode channel name to its DTO."""
    result: dict[str, '_NodeDTO'] = {}
    for dto in dto_map.values():
        if dto.node.get('type') == 'SetNode':
            name = _get_set_get_channel_name(dto.node)
            if name:
                result[name] = dto
    return result


def _resolve_dto_input(dto, slot, dto_map, visited=None, skip_boundary_widgets=False,
                       set_node_map=None):
    if visited is None:
        visited = set()
    uid = f"{dto.exec_id}[I]{slot}"
    if uid in visited:
        return None
    visited.add(uid)

    inputs = dto.node.get('inputs', [])
    if slot >= len(inputs):
        return None

    inp = inputs[slot]
    link_id = inp.get('link')
    if link_id is None:
        return None

    link = dto.graph_links.get(link_id)
    if link is None:
        return None

    if dto.sg_node_exec_id is not None and link.src_node == _SUBGRAPH_INPUT_NODE_ID:
        sg_dto = dto_map.get(dto.sg_node_exec_id)
        if sg_dto is None:
            return None

        sg_inp_idx = link.src_slot
        boundary_name = _get_subgraph_boundary_name(sg_dto.node, sg_inp_idx, sg_dto.sg_def)
        outer_slot_idx, outer_input = _get_subgraph_outer_input_slot(sg_dto.node, boundary_name)
        outer_link_id = outer_input.get('link') if outer_input is not None else None

        if outer_link_id is None:
            if skip_boundary_widgets:
                return None
            found, val = _get_sg_widget_by_slot(sg_dto.node, sg_inp_idx, sg_dto.sg_def)
            return ("value", val) if found else None

        if outer_slot_idx is None:
            return None

        return _resolve_dto_input(sg_dto, outer_slot_idx, dto_map, visited,
                                  set_node_map=set_node_map)

    src_nid = link.src_node
    if dto.sg_node_exec_id:
        _sg_parent = dto_map.get(dto.sg_node_exec_id)
        if _sg_parent and _sg_parent.id_remap:
            src_nid = _sg_parent.id_remap.get(src_nid, src_nid)
    src_exec_id = ':'.join(str(x) for x in [*dto.subgraph_node_path, src_nid])
    src_dto = dto_map.get(src_exec_id)
    if src_dto is None:
        return None

    return _resolve_dto_output(
        src_dto, link.src_slot, inp.get('type'), dto_map, visited,
        set_node_map=set_node_map,
    )


def _resolve_dto_output(dto, slot, target_type, dto_map, visited, set_node_map=None):
    uid = f"{dto.exec_id}[O]{slot}"
    if uid in visited:
        return None
    visited.add(uid)

    mode = dto.node.get('mode', 0)

    if mode == _MODE_BYPASS:
        idx = _get_bypass_slot_index(
            dto.node.get('inputs', []), dto.node.get('outputs', []),
            slot, target_type,
        )
        if idx == -1:
            return None
        return _resolve_dto_input(dto, idx, dto_map, visited,
                                  set_node_map=set_node_map)

    if mode == _MODE_NEVER:
        return None

    class_type = dto.node.get('type', '')

    if dto.sg_def is not None:
        return _resolve_sg_output(dto, slot, target_type, dto_map, visited,
                                  set_node_map=set_node_map)

    if class_type == 'Reroute':
        return _resolve_dto_input(dto, slot, dto_map, visited,
                                  set_node_map=set_node_map)

    if class_type == 'SetNode':
        # SetNode passes through its input slot 0, like Reroute
        return _resolve_dto_input(dto, 0, dto_map, visited,
                                  set_node_map=set_node_map)

    if class_type == 'GetNode':
        # GetNode retrieves the value from the matching SetNode by channel name
        channel_name = _get_set_get_channel_name(dto.node)
        if set_node_map and channel_name:
            set_dto = set_node_map.get(channel_name)
            if set_dto:
                return _resolve_dto_input(set_dto, 0, dto_map, visited,
                                          set_node_map=set_node_map)
        return None

    if class_type in _PRIMITIVE_VALUE_NODE_TYPES:
        # PrimitiveNode is a virtual node. See
        # ComfyUI_frontend src/lib/litegraph/src/subgraph/ExecutableNodeDTO.ts
        # ``resolveOutput``: for a virtual node it calls ``getInputLink`` and
        # recurses via resolveInput. PrimitiveNode has its widget promoted
        # to an OUTPUT (no graph inputs), so getInputLink returns undefined
        # and resolveOutput returns undefined. That "undefined" then lets
        # graphToPrompt keep whatever widget value was pre-populated on the
        # downstream consumer (the widget value PrimitiveNode.applyToGraph
        # pushed at graph-load time). Returning ``('value', ...)`` here
        # made the resolved PrimitiveNode widget value overwrite the inner
        # node's stored widget value, contradicting the frontend — breaks
        # template_contact_sheet-step_3.app where KlingFirstLastFrameNode
        # inside a subgraph should end up with duration=3 (inner widget)
        # but was getting duration=5 (outer PrimitiveNode).
        return None

    return ("link", dto.exec_id, slot)


def _resolve_sg_output(sg_dto, slot, target_type, dto_map, visited, set_node_map=None):
    # Frontend 1.53 keeps the first serialized link to each input slot when
    # loading topology, including subgraph outputs with stale linkIds lists.
    links_to_try = [
        link for link in sg_dto.inner_links.values()
        if link.dst_node == _SUBGRAPH_OUTPUT_NODE_ID and link.dst_slot == slot
    ][:1]

    for link in links_to_try:
        src_nid = link.src_node
        if sg_dto.id_remap:
            src_nid = sg_dto.id_remap.get(src_nid, src_nid)
        inner_exec_id = f"{sg_dto.exec_id}:{src_nid}"
        inner_dto = dto_map.get(inner_exec_id)
        if inner_dto is None:
            continue
        result = _resolve_dto_output(
            inner_dto, link.src_slot, target_type, dto_map, visited,
            set_node_map=set_node_map,
        )
        if result is not None:
            return result
    result = _resolve_legacy_sg_output(
        sg_dto, slot, target_type, dto_map, visited, set_node_map=set_node_map,
    )
    if result is not None:
        return result
    return None


def _resolve_legacy_sg_output(sg_dto, slot, target_type, dto_map, visited, set_node_map=None):
    """Resolve outputs for legacy ``extra.groupNodes`` definitions.

    Legacy group nodes do not serialize modern subgraph output boundary nodes.
    The host node still has output slots, so mirror the frontend's legacy
    behaviour by matching that host output to a terminal inner node output with
    the same type/name.
    """
    if "outputNode" in (sg_dto.sg_def or {}):
        return None
    outer_outputs = sg_dto.node.get('outputs', [])
    if slot >= len(outer_outputs):
        return None

    outer_output = outer_outputs[slot]
    outer_type = outer_output.get('type') or target_type
    outer_name = outer_output.get('name')
    prefix = f"{sg_dto.exec_id}:"

    candidates = []
    for inner_dto in dto_map.values():
        if inner_dto.sg_node_exec_id != sg_dto.exec_id:
            continue
        if not inner_dto.exec_id.startswith(prefix):
            continue
        if ':' in inner_dto.exec_id[len(prefix):]:
            continue
        for output_slot, output in enumerate(inner_dto.node.get('outputs', [])):
            if outer_type is not None and output.get('type') != outer_type:
                continue
            if outer_name and output.get('name') not in (None, outer_name):
                continue
            output_links = output.get('links')
            if output_links not in (None, []):
                continue
            candidates.append((inner_dto, output_slot))

    if not candidates:
        return None

    exact_name_matches = [
        candidate for candidate in candidates
        if candidate[0].node.get('outputs', [])[candidate[1]].get('name') == outer_name
    ]
    if exact_name_matches:
        candidates = exact_name_matches

    if len(candidates) != 1:
        return None

    inner_dto, output_slot = candidates[0]
    return _resolve_dto_output(
        inner_dto, output_slot, target_type, dto_map, visited,
        set_node_map=set_node_map,
    )


def _is_subgraph_type(class_type: str, sg_defs: dict[str, dict]) -> bool:
    return class_type in sg_defs


def is_ui_workflow(workflow: dict) -> bool:
    """Return True if *workflow* is a UI/LiteGraph workflow (not API format)."""
    return "nodes" in workflow and "links" in workflow


def convert_ui_to_api(
    workflow: dict,
    *,
    preserve_unknown_nodes: bool = True,
    node_mappings=None,
) -> dict:
    """Convert a UI (LiteGraph) workflow dict to API format.

    Uses a DTO-based approach mirroring the frontend's ``graphToPrompt``
    and ``ExecutableNodeDTO`` for correct subgraph resolution at any
    nesting depth.

    Args:
        workflow: A UI (LiteGraph) workflow dict.
        preserve_unknown_nodes: When True (default), unknown node types
            retain their ``class_type`` string and all serialized widget
            values.  When False, unknown nodes get ``class_type: None``
            and only minimal widget data — matching the frontend's
            ``graphToPrompt`` behaviour for nodes missing from
            ``/object_info``.

    Raises:
        RuntimeError: If node system is not loaded.
    """
    if node_mappings is None:
        from ..nodes_context import get_nodes
        node_mappings = get_nodes()

    _node_mappings_token = _active_node_mappings.set(node_mappings)
    try:
        return _convert_ui_to_api_impl(workflow, preserve_unknown_nodes, node_mappings)
    finally:
        _active_node_mappings.reset(_node_mappings_token)


def _convert_ui_to_api_impl(workflow, preserve_unknown_nodes, node_mappings):
    workflow = _compress_widget_input_slots(workflow)
    sg_defs = _collect_subgraph_defs(workflow)

    dto_map = _build_dto_map(workflow, sg_defs)
    set_node_map = _build_set_node_map(dto_map)

    api_workflow: dict[str, dict] = {}

    for dto in dto_map.values():
        node = dto.node
        class_type = node.get('type', '')
        mode = node.get('mode', 0)

        if class_type in _VIRTUAL_NODE_TYPES:
            continue
        if _is_subgraph_type(class_type, sg_defs):
            continue
        if mode in (_MODE_NEVER, _MODE_BYPASS):
            continue

        class_def = _get_node_class(node_mappings, class_type)
        input_types = _get_input_types(class_def) if class_def is not None else None
        is_unknown = class_def is None or input_types is None

        widgets_values = node.get('widgets_values')
        _wv_consumed = 0
        if is_unknown:
            if class_def is None and not preserve_unknown_nodes:
                # Frontend-parity mode: errorNodeWidgets.ts creates UNKNOWN
                # widgets only from list-format widgets_values (has .length).
                # Dict-format widgets_values have no .length → no widgets.
                if isinstance(widgets_values, list):
                    api_inputs = _map_unknown_widgets(widgets_values)
                else:
                    api_inputs = {}
                use_class_type = None
            else:
                # Preserve mode (default): keep class_type and all values.
                api_inputs = _map_unknown_widgets(widgets_values, node=node)
                use_class_type = class_type
            logger.debug("Unknown node type %r (id=%s); preserving in API output",
                         class_type, node.get('id'))
        elif isinstance(widgets_values, list):
            overridden_mapping = _map_frontend_widget_override(
                class_type,
                widgets_values,
            )
            if overridden_mapping is not None:
                api_inputs, _wv_consumed = overridden_mapping
            else:
                api_inputs, _wv_consumed = _map_widgets(input_types, widgets_values, node=node)
            use_class_type = class_type
        elif isinstance(widgets_values, dict):
            api_inputs = _map_widgets_dict(input_types, widgets_values)
            use_class_type = class_type
        else:
            api_inputs, _wv_consumed = _map_widgets(input_types, [], node=node)
            use_class_type = class_type

        if dto.sg_node_exec_id and not is_unknown:
            sg_dto = dto_map.get(dto.sg_node_exec_id)
            if sg_dto:
                all_inputs = {**input_types.get('required', {}),
                              **input_types.get('optional', {})}
                nid = node['id']
                for (ovr_nid, wname), val in sg_dto.proxy_overrides.items():
                    if ovr_nid != nid or val is None:
                        continue
                    if wname not in all_inputs:
                        continue
                    ts, opts = _input_type_and_opts(all_inputs[wname])
                    if _is_widget_type(ts, opts) and not opts.get('forceInput'):
                        api_inputs[wname] = _wrap_value(val)

        if not is_unknown and input_types:
            _all_input_names = (
                set(input_types.get('required', {}).keys())
                | set(input_types.get('optional', {}).keys())
            )
        else:
            _all_input_names = None
        _all_input_names_from_serialized = _serialized_widget_input_names(node)
        if _all_input_names is None:
            _all_input_names = set(_all_input_names_from_serialized)
        else:
            _all_input_names |= _all_input_names_from_serialized

        _widget_input_names: set[str] = set()
        if input_types:
            for _wn, _we in {**input_types.get('required', {}),
                             **input_types.get('optional', {})}.items():
                _wts, _wopts = _input_type_and_opts(_we)
                if _is_widget_type(_wts, _wopts):
                    _widget_input_names.add(_wn)
        _widget_input_names |= _all_input_names_from_serialized

        for i, inp in enumerate(node.get('inputs', [])):
            inp_name = inp.get('name')
            if inp_name is None:
                continue
            link_id = inp.get('link')
            if link_id is None or link_id not in dto.graph_links:
                continue

            _frontend_unknown = class_def is None and not preserve_unknown_nodes
            resolved = _resolve_dto_input(dto, i, dto_map,
                                          skip_boundary_widgets=_frontend_unknown,
                                          set_node_map=set_node_map)
            if resolved is None:
                if inp.get('widget') is not None:
                    # Mirror PrimitiveNode.applyToGraph: a directly-connected
                    # PrimitiveNode pushes its widget value onto the consuming
                    # widget BEFORE serialization (see ComfyUI_frontend
                    # src/extensions/core/widgetInputs.ts applyToGraph). This
                    # only fires for direct edges; subgraph-boundary traversal
                    # puts link.src_node at _SUBGRAPH_INPUT_NODE_ID so the
                    # graph_nodes_by_id lookup misses and the inner widget
                    # value stays — matching the frontend, which only mutates
                    # the subgraph's own boundary widget in that case.
                    link = dto.graph_links.get(link_id)
                    if link is not None:
                        src_node = dto.graph_nodes_by_id.get(link.src_node)
                        if src_node and src_node.get('type') == 'PrimitiveNode':
                            wv = src_node.get('widgets_values', [])
                            if wv and (_all_input_names is None or inp_name in _all_input_names):
                                api_inputs[inp_name] = _wrap_value(wv[0])
                                continue
                    if inp_name in api_inputs:
                        continue
                api_inputs.pop(inp_name, None)
            elif resolved[0] == 'value':
                # In frontend-parity mode, unknown nodes have all widgets
                # renamed to UNKNOWN by errorNodeWidgets.ts, so
                # PrimitiveNode.applyToGraph() can't find the target widget
                # and the generic virtual node resolution returns undefined.
                if _frontend_unknown:
                    pass
                elif link_id is not None or _all_input_names is None or inp_name in _all_input_names:
                    api_inputs[inp_name] = _wrap_value(resolved[1])
            else:
                api_inputs[inp_name] = [resolved[1], resolved[2]]

        if use_class_type and use_class_type in _FRONTEND_INJECTED_WIDGETS:
            injected = _FRONTEND_INJECTED_WIDGETS[use_class_type]
            for j, (widget_name, default_value) in enumerate(injected):
                if widget_name not in api_inputs:
                    wv_idx = _wv_consumed + j
                    if isinstance(widgets_values, list) and wv_idx < len(widgets_values):
                        val = widgets_values[wv_idx]
                        if val is None and isinstance(default_value, str):
                            val = default_value
                        api_inputs[widget_name] = val
                    elif use_class_type in _FRONTEND_OPTIONAL_INJECTED_WIDGETS:
                        continue
                    else:
                        api_inputs[widget_name] = default_value

        entry = {
            'class_type': use_class_type,
            'inputs': api_inputs,
        }
        title = node.get('title') or use_class_type
        entry['_meta'] = {'title': title}
        api_workflow[dto.exec_id] = entry

    for entry in api_workflow.values():
        inputs = entry['inputs']
        for key in list(inputs.keys()):
            val = inputs[key]
            if (
                isinstance(val, list)
                and len(val) == 2
                and isinstance(val[0], str)
                and val[0] not in api_workflow
            ):
                del inputs[key]

    from .workflow_rewrites import apply_to_api_workflow
    return apply_to_api_workflow(api_workflow)


def _get_node_class(node_mappings, class_type: str) -> Optional[type]:
    if hasattr(node_mappings, 'NODE_CLASS_MAPPINGS'):
        return node_mappings.NODE_CLASS_MAPPINGS.get(class_type)
    if isinstance(node_mappings, dict):
        return node_mappings.get(class_type)
    mappings = getattr(node_mappings, 'class_mappings', None)
    if mappings:
        return mappings.get(class_type)
    return None


def _get_input_types(class_def) -> Optional[dict]:
    try:
        if hasattr(class_def, "INPUT_TYPES"):
            result = class_def.INPUT_TYPES()
            if isinstance(result, dict):
                return result
    except Exception as exc:
        logger.debug(f"Failed to get INPUT_TYPES for {class_def}: {exc}")
    return None
