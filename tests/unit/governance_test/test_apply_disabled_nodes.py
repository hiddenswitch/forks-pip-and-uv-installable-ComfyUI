import copy
import logging
from pathlib import Path

import pytest

from comfy.app import governance
from comfy.cli_args import default_configuration



from comfy.app.node_replace_manager import NodeReplaceManager  # noqa: E402
from comfy.cmd import execution, folder_paths  # noqa: E402
from comfy.execution_context import context_configuration  # noqa: E402
from comfy.nodes.package import import_all_nodes_in_workspace  # noqa: E402
from comfy_api.latest import io  # noqa: E402


class TestNode:
    pass


@pytest.fixture(autouse=True)
def nodes(monkeypatch: pytest.MonkeyPatch, node_registry):
    """The node registry that execution, validation and NodeReplaceManager read, restored after the test."""
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset(), raising=False)
    return node_registry


def _prompt(class_type: str, with_meta: bool) -> dict:
    node = {"class_type": class_type, "inputs": {}}
    if with_meta:
        node["_meta"] = {"title": class_type}
    return {"1": node}


def _replacement(old_node_id: str = "LegacyNode", new_node_id: str = "CurrentNode") -> io.NodeReplace:
    return io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)


def _load_with_disabled_nodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str, module_source: str, disabled: set[str]):
    """Load a custom node module through the fork's loader, with a --disabled-nodes-config naming the given IDs."""
    custom_nodes_path = tmp_path / "custom_nodes"
    custom_nodes_path.mkdir()
    (custom_nodes_path / module_name).write_text(module_source, encoding="utf-8")
    config_path = tmp_path / "disabled_nodes.yaml"
    config_path.write_text("disabled_nodes:\n" + "".join(f"  - {node_id}\n" for node_id in sorted(disabled)), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(custom_nodes_path)] if name == "custom_nodes" else [])

    configuration = default_configuration()
    configuration.disable_all_custom_nodes = False
    configuration.disabled_nodes_config = str(config_path)
    with context_configuration(configuration):
        return import_all_nodes_in_workspace()


def test_apply_disabled_nodes_removes_class_and_display_name(nodes, caplog: pytest.LogCaptureFixture) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS["DisabledNode"] = TestNode
    nodes.NODE_DISPLAY_NAME_MAPPINGS["DisabledNode"] = "Disabled Node"

    # When
    with caplog.at_level(logging.INFO):
        governance.apply_disabled_nodes(nodes, {"DisabledNode"})

    # Then
    assert "DisabledNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "DisabledNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "Pruned 1 disabled node" in caplog.text


def test_apply_disabled_nodes_prunes_policy_nodes_without_config(nodes, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a signed policy disabled a node and no disabled-node config is given
    nodes.NODE_CLASS_MAPPINGS["PolicyNode"] = TestNode
    nodes.NODE_DISPLAY_NAME_MAPPINGS["PolicyNode"] = "Policy Node"
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset({"PolicyNode"}), raising=False)

    # When
    governance.apply_disabled_nodes(nodes, set())

    # Then the policy's node is pruned
    assert "PolicyNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "PolicyNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_apply_disabled_nodes_warns_once_for_all_missing_ids(nodes, caplog: pytest.LogCaptureFixture) -> None:
    # Given
    missing = {"MissingNodeA", "MissingNodeB"}

    # When
    with caplog.at_level(logging.WARNING):
        governance.apply_disabled_nodes(nodes, missing)

    # Then
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert all(node_id in warnings[0].getMessage() for node_id in missing)


@pytest.mark.parametrize("with_meta", [False, True])
def test_disabled_old_id_forwards_to_allowed_target(nodes, with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)

    # When
    governance.apply_disabled_nodes(nodes, {"LegacyNode"})
    manager.apply_replacements(prompt)

    # Then
    assert manager.has_replacement("LegacyNode") is True
    assert prompt["1"]["class_type"] == "CurrentNode"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_meta", [False, True])
async def test_disabled_target_is_not_applied_and_prompt_is_refused(nodes, with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS["CurrentNode"] = TestNode
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)

    # When
    governance.apply_disabled_nodes(nodes, {"CurrentNode"})
    manager.apply_replacements(prompt)
    valid = await execution.validate_prompt("prompt-id", prompt, None)

    # Then
    assert prompt["1"]["class_type"] == "LegacyNode"
    assert valid[0] is False
    assert valid[1]["type"] == "missing_node_type"
    assert valid[1]["extra_info"]["class_type"] == "LegacyNode"


@pytest.mark.asyncio
async def test_disabling_both_replacement_ends_refuses_prompt(nodes) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", False)

    # When
    governance.apply_disabled_nodes(nodes, {"LegacyNode", "CurrentNode"})
    manager.apply_replacements(prompt)
    valid = await execution.validate_prompt("prompt-id", prompt, None)

    # Then
    assert prompt["1"]["class_type"] == "LegacyNode"
    assert valid[0] is False
    assert valid[1]["type"] == "missing_node_type"


@pytest.mark.parametrize("with_meta", [False, True])
def test_replacement_behavior_is_unchanged_when_neither_end_is_disabled(nodes, with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)
    original_prompt = copy.deepcopy(prompt)

    # When
    governance.apply_disabled_nodes(nodes, set())
    manager.apply_replacements(prompt)

    # Then
    assert prompt == original_prompt


# Upstream prunes once and wraps nodes.load_custom_node so later loads skip the disabled IDs. The fork imports every
# node module through import_all_nodes_in_workspace, which reads --disabled-nodes-config and prunes the set on every
# load, so these load a custom node folder through it and check the disabled IDs never reach the registry.


def test_post_prune_v1_load_cannot_register_disabled_id(nodes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given
    module_source = (
        "class SealedV1Node:\n"
        "    pass\n\n"
        "class OtherV1Node:\n"
        "    pass\n\n"
        "NODE_CLASS_MAPPINGS = {'DisabledV1': SealedV1Node, 'OtherV1': OtherV1Node}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {'DisabledV1': 'Disabled V1', 'OtherV1': 'Other V1'}\n"
    )

    # When
    loaded = _load_with_disabled_nodes(monkeypatch, tmp_path, "sealed_v1_node.py", module_source, {"DisabledV1"})

    # Then
    assert "OtherV1" in loaded.NODE_CLASS_MAPPINGS
    assert "DisabledV1" not in loaded.NODE_CLASS_MAPPINGS
    assert "DisabledV1" not in loaded.NODE_DISPLAY_NAME_MAPPINGS
    assert "DisabledV1" not in nodes.NODE_CLASS_MAPPINGS


def test_post_prune_v3_load_cannot_register_disabled_schema_id(nodes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given
    module_source = (
        "from comfy_api.latest import ComfyExtension\n\n"
        "def _node(node_id, display_name):\n"
        "    class SealedV3Node:\n"
        "        @classmethod\n"
        "        def GET_SCHEMA(cls):\n"
        "            class Schema:\n"
        "                pass\n\n"
        "            Schema.node_id = node_id\n"
        "            Schema.display_name = display_name\n"
        "            return Schema()\n\n"
        "    return SealedV3Node\n\n\n"
        "class SealedExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [_node('DisabledV3', 'Disabled V3'), _node('OtherV3', 'Other V3')]\n\n\n"
        "async def comfy_entrypoint():\n"
        "    return SealedExtension()\n"
    )

    # When
    loaded = _load_with_disabled_nodes(monkeypatch, tmp_path, "sealed_v3_node.py", module_source, {"DisabledV3"})

    # Then
    assert "OtherV3" in loaded.NODE_CLASS_MAPPINGS
    assert "DisabledV3" not in loaded.NODE_CLASS_MAPPINGS
    assert "DisabledV3" not in loaded.NODE_DISPLAY_NAME_MAPPINGS
    assert "DisabledV3" not in nodes.NODE_CLASS_MAPPINGS


def test_seal_keeps_replacement_target_absent_after_v1_load(nodes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given
    module_source = (
        "class SealedTargetNode:\n"
        "    pass\n\n"
        "NODE_CLASS_MAPPINGS = {'DisabledTarget': SealedTargetNode}\n"
    )
    manager = NodeReplaceManager()
    manager.register(_replacement(new_node_id="DisabledTarget"))
    loaded = _load_with_disabled_nodes(monkeypatch, tmp_path, "sealed_target.py", module_source, {"DisabledTarget"})
    prompt = _prompt("LegacyNode", False)

    # When
    manager.apply_replacements(prompt)

    # Then
    assert "DisabledTarget" not in loaded.NODE_CLASS_MAPPINGS
    assert "DisabledTarget" not in nodes.NODE_CLASS_MAPPINGS
    assert prompt["1"]["class_type"] == "LegacyNode"
