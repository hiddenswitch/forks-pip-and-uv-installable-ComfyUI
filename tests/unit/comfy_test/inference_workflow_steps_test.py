import json
from importlib.resources import files

import pytest


WORKFLOWS = sorted(files("tests.inference.workflows").glob("*.json"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.stem)
def test_inference_workflow_has_at_least_two_steps(path):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    for node_id, node in workflow.items():
        inputs = node["inputs"]
        for field in ("steps", "num_inference_steps", "inference_steps", "refine_steps"):
            value = inputs.get(field)
            if isinstance(value, int):
                assert value >= 2, (path.name, node_id, field, value)
        if node["class_type"] == "ManualSigmas":
            sigmas = [float(value) for value in inputs["sigmas"].split(",")]
            assert len(sigmas) >= 3, (path.name, node_id, sigmas)
