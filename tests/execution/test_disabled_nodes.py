from contextlib import contextmanager
import json
from pathlib import Path
import socket
import time
from typing import Iterator
import urllib.error
import urllib.request

import pytest

from comfy.cli_args import default_configuration
from ..conftest import comfy_background_server_from_config

# Each test starts a ComfyUI server process, so it runs in the isolated server-process group like the fixtures that do.
pytestmark = [pytest.mark.server_process, pytest.mark.xdist_group(name="server-process")]

def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def _post_prompt(base_url: str, prompt: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/prompt",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def _wait_for_history(base_url: str, prompt_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        history = _get_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.1)
    raise AssertionError(f"prompt {prompt_id} did not finish")


@contextmanager
def _running_server(tmp_path: Path, disabled_node: str) -> Iterator[str]:
    config_path = tmp_path / "disabled_nodes.yaml"
    config_path.write_text(f"disabled_nodes:\n  - {disabled_node}\n", encoding="utf-8")
    configuration = default_configuration()
    configuration.listen = "127.0.0.1"
    configuration.port = _unused_port()
    configuration.cpu = True
    configuration.cache_none = True
    configuration.disable_partner_nodes = True
    configuration.output_directory = str(tmp_path / "output")
    configuration.temp_directory = str(tmp_path)
    configuration.disabled_nodes_config = str(config_path)
    configuration.disable_all_custom_nodes = True

    server = comfy_background_server_from_config(configuration, server_module="tests.fast_server")
    next(server)
    try:
        yield f"http://{configuration.listen}:{configuration.port}"
    finally:
        server.close()


@pytest.mark.execution
def test_disabled_old_id_forwards_and_executes_allowed_target(tmp_path: Path) -> None:
    # Given
    prompt = {
        "source-a": {
            "class_type": "EmptyImage",
            "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0},
        },
        "source-b": {
            "class_type": "EmptyImage",
            "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0},
        },
        "legacy": {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["source-a", 0], "image2": ["source-b", 0]},
        },
        "output": {"class_type": "PreviewImage", "inputs": {"images": ["legacy", 0]}},
    }

    # When
    with _running_server(tmp_path, "ImageBatch") as base_url:
        object_info = _get_json(f"{base_url}/object_info/ImageBatch")
        status, response = _post_prompt(base_url, prompt)
        history = _wait_for_history(base_url, response["prompt_id"])

    # Then
    assert "ImageBatch" not in object_info
    assert status == 200
    assert history["status"]["status_str"] == "success"
    assert history["prompt"][2]["legacy"]["class_type"] == "BatchImagesNode"


@pytest.mark.execution
def test_replacement_pointing_to_disabled_target_is_refused(tmp_path: Path) -> None:
    # Given
    prompt = {"1": {"class_type": "ConditioningAverage ", "inputs": {}}}

    # When
    with _running_server(tmp_path, "ConditioningAverage") as base_url:
        object_info = _get_json(f"{base_url}/object_info/ConditioningAverage")
        status, response = _post_prompt(base_url, prompt)

    # Then
    assert "ConditioningAverage" not in object_info
    assert status == 400
    assert response["error"]["type"] == "missing_node_type"
    assert response["error"]["extra_info"]["class_type"] == "ConditioningAverage "


@pytest.mark.execution
def test_disabled_node_cannot_be_reached_through_expansion(tmp_path: Path) -> None:
    # Given
    prompt = {"1": {"class_type": "TestExpandsToDisabledNode", "inputs": {}}}

    # When
    with _running_server(tmp_path, "TestDisabledNode") as base_url:
        object_info = _get_json(f"{base_url}/object_info/TestExpandsToDisabledNode")
        disabled_info = _get_json(f"{base_url}/object_info/TestDisabledNode")
        status, response = _post_prompt(base_url, prompt)
        history = _wait_for_history(base_url, response["prompt_id"])

    # Then
    assert "TestExpandsToDisabledNode" in object_info
    assert "TestDisabledNode" not in disabled_info
    assert status == 200
    assert history["status"]["status_str"] == "error"
