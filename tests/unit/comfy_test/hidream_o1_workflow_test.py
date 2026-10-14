import json
from importlib.resources import files

from comfy.api.components.schema.prompt import Prompt
from comfy.model_downloader import KNOWN_HUGGINGFACE_MODEL_REPOS


WORKFLOW_PATH = files("tests.inference.workflows").joinpath("hidream-o1-0.json")


def _load_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_hidream_o1_inference_workflow_is_api_and_two_steps():
    workflow = _load_workflow()

    Prompt.validate(workflow)

    assert "nodes" not in workflow
    ksamplers = [node for node in workflow.values() if node["class_type"] == "KSampler"]
    assert len(ksamplers) == 1
    assert ksamplers[0]["inputs"]["steps"] == 2


def test_hidream_o1_inference_workflow_uses_known_o1_repo():
    workflow = _load_workflow()
    loaders = [node for node in workflow.values() if node["class_type"] == "DiffusersLoader"]

    assert len(loaders) == 1
    assert loaders[0]["inputs"]["model_path"] == "HiDream-ai/HiDream-O1-Image-Dev"
    assert loaders[0]["inputs"]["model_path"] in KNOWN_HUGGINGFACE_MODEL_REPOS
    assert "HiDream-ai/HiDream-O1-Image" in KNOWN_HUGGINGFACE_MODEL_REPOS


def test_hidream_o1_inference_workflow_uses_o1_pixel_latent_and_decode():
    workflow = _load_workflow()
    class_types = {node["class_type"] for node in workflow.values()}

    assert "EmptyHiDreamO1LatentImage" in class_types
    assert "VAEDecode" in class_types
    assert "SaveImage" in class_types
