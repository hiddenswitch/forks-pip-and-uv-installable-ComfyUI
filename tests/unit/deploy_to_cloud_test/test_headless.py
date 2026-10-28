import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from comfy.execution_context import current_execution_context
from comfy.cmd import execution, folder_paths, server
from comfy_api import feature_flags


@pytest.mark.parametrize("offline", [False, True])
async def test_cloud_discovery_does_not_require_headless_history(monkeypatch, offline):
    monkeypatch.setitem(current_execution_context().configuration, "offline", offline)
    prompt_server = server.PromptServer(asyncio.get_running_loop(), MagicMock(enabled=False))
    prompt_server.add_routes()
    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.get("/api/features")
        assert response.status == 200
        assert (await response.json())["deploy_to_cloud"] is not offline
    assert feature_flags.get_server_features()["deploy_to_cloud"] is not offline


@pytest.mark.parametrize("extra", [{}, {"client_id": "browser"}, {"extra_data": {"comfy_usage_source": "comfyui-frontend"}}])
async def test_prompt_submission_does_not_record_headless_use(monkeypatch, extra):
    async def validate_prompt(*args):
        return True, None, ["1"], {}

    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    prompt_server = server.PromptServer(asyncio.get_running_loop(), MagicMock(enabled=False))
    prompt_server.add_routes()
    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.post("/api/prompt", json={"prompt": {"1": {"class_type": "Test", "inputs": {}}}, **extra})
        assert response.status == 200
        assert len(prompt_server.prompt_queue.queue) == 1
    assert not (Path(folder_paths.get_user_directory()) / "used_headlessly").exists()
