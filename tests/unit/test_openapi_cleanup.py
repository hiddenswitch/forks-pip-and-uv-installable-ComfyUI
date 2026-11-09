from __future__ import annotations

from pathlib import Path

import yaml


def _root_spec() -> dict:
    return yaml.safe_load(Path("openapi.yaml").read_text(encoding="utf-8"))


def test_canonical_openapi_contains_fork_and_vanilla_routes():
    paths = _root_spec()["paths"]

    fork_routes = {
        "/api/assets",
        "/api/workflows",
        "/api/hub/workflows",
        "/api/auth/session",
        "/api/billing/balance",
        "/api/workspace/api-keys",
        "/api/secrets",
        "/api/tasks",
    }
    vanilla_routes = {
        "/prompt",
        "/api/prompt",
        "/queue",
        "/api/queue",
        "/history",
        "/api/history",
        "/object_info",
        "/api/object_info",
        "/api/v1/prompts",
        "/api/v1/prompts/{prompt_id}",
    }

    assert fork_routes <= set(paths)
    assert vanilla_routes <= set(paths)


def test_prompt_schemas_stay_permissive_for_custom_nodes():
    schemas = _root_spec()["components"]["schemas"]

    prompt_request = schemas["PromptRequest"]["properties"]["prompt"]
    assert prompt_request["type"] == "object"
    assert prompt_request["additionalProperties"] is True

    prompt_node_inputs = schemas["PromptNode"]["properties"]["inputs"]
    allowed_inputs = prompt_node_inputs["additionalProperties"]["anyOf"]
    assert {"type": "null"} in allowed_inputs
    assert {"type": "object", "additionalProperties": True} in allowed_inputs
    assert {"type": "array", "items": {}} in allowed_inputs


def test_generated_openapi_models_import_and_validate_core_payloads():
    from comfy.api.generated import models

    prompt = {"1": {"class_type": "CustomNode", "inputs": {"value": None}}}
    assert models.PromptRequest.model_validate({"prompt": prompt}).prompt == prompt
    assert models.QueueInfo.model_validate({"queue_running": [], "queue_pending": []})
    assert models.HistoryEntry.model_validate({"outputs": {"1": {"images": []}}})


def test_public_execution_and_node_outputs_validate_without_fork_only_fields():
    from comfy.api.generated import models
    from comfy.component_model.queue_types import ExecutionStatus

    status = ExecutionStatus('success', True, [('execution_success', {'prompt_id': 'id', 'timestamp': 123})])
    assert models.ExecutionStatusAsDict.model_validate(status.as_dict()).messages == status.messages
    output = models.OutputModel.model_validate({
        'images': [{'filename': 'image.png', 'subfolder': '', 'type': 'output'}],
        'text': ['hello'], 'animated': [False],
    })
    assert output.images[0].abs_path is None
    assert output.model_extra == {'text': ['hello'], 'animated': [False]}
