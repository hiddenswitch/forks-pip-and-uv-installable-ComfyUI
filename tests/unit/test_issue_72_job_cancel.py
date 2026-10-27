import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from comfy import interruption
from comfy.cli_args import default_configuration
from comfy.cmd.execution import PromptQueue
from comfy.cmd.server import PromptServer
from comfy.component_model.queue_types import QueueItem, QueueTuple
from comfy.execution_context import context_configuration
from comfy_execution.jobs import (
    CANCEL_PENDING, CANCEL_RUNNING, CANCEL_TERMINAL, CANCEL_UNKNOWN,
    classify_job_for_cancel, get_job_create_times,
)


RUNNING = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
PENDING = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
FINISHED = "cccccccc-cccc-4ccc-cccc-cccccccccccc"
UNKNOWN = "dddddddd-dddd-4ddd-dddd-dddddddddddd"


def queue_item(prompt_id):
    return QueueItem(QueueTuple(0, prompt_id, {}, {"create_time": 1234}, []), None)


@pytest.mark.parametrize("shape", ["queue_item", "prompt_id", "id", "tuple"])
def test_issue_72_helpers_accept_real_and_serialized_items(shape):
    item = queue_item(RUNNING)
    if shape == "prompt_id":
        item = dict(item)
    elif shape == "id":
        item = {"id": RUNNING, "extra_data": item.extra_data}
    elif shape == "tuple":
        item = item.queue_tuple
    assert classify_job_for_cancel(RUNNING, [item], [], {}) == CANCEL_RUNNING
    assert classify_job_for_cancel(RUNNING, [], [item], {}) == CANCEL_PENDING
    assert classify_job_for_cancel(FINISHED, [item], [], {FINISHED: {}}) == CANCEL_TERMINAL
    assert classify_job_for_cancel(UNKNOWN, [item], [], {}) == CANCEL_UNKNOWN


@pytest.mark.parametrize("shape", ["queue_item", "dict", "tuple"])
@pytest.mark.parametrize("location", ["running", "queued", "history"])
def test_issue_72_creation_times_accept_real_and_serialized_items(shape, location):
    item = queue_item(RUNNING)
    if shape == "dict":
        item = dict(item)
    elif shape == "tuple":
        item = item.queue_tuple
    running, queued, history = [], [], {}
    if location == "history":
        history[RUNNING] = {"prompt": item}
    else:
        (running if location == "running" else queued).append(item)
    assert get_job_create_times([RUNNING, UNKNOWN], running, queued, history) == {RUNNING: 1234}


def test_issue_72_interrupt_checks_live_queue_under_mutex(monkeypatch):
    queue = PromptQueue(MagicMock())
    queue.put(queue_item(RUNNING))
    queue.get()
    called = []

    def interrupt():
        assert queue.mutex._is_owned()
        called.append(True)

    monkeypatch.setattr(interruption, "interrupt_current_processing", interrupt)
    assert queue.interrupt_if_running(RUNNING)
    queue.currently_running.clear()
    queue.currently_running[PENDING] = queue_item(PENDING)
    assert not queue.interrupt_if_running(RUNNING)
    assert called == [True]


@pytest.fixture
async def cancel_client(monkeypatch, tmp_path):
    config = default_configuration()
    config.cwd = str(tmp_path)
    config.disable_all_custom_nodes = True
    calls = []
    monkeypatch.setattr(interruption, "interrupt_current_processing", lambda: calls.append(True))
    with context_configuration(config):
        server = PromptServer(asyncio.get_running_loop(), MagicMock(enabled=False))
        queue = server.prompt_queue
        queue.put(queue_item(RUNNING))
        queue.get()
        queue.put(queue_item(PENDING))
        queue.history[FINISHED] = {"prompt": queue_item(FINISHED).queue_tuple, "outputs": {}}
        server.add_routes()
        async with TestClient(TestServer(server.app)) as client:
            yield client, queue, calls


@pytest.mark.parametrize("prompt_id,cancelled", [(RUNNING, True), (PENDING, True), (FINISHED, False), (UNKNOWN, False)])
async def test_issue_72_production_single_cancel(cancel_client, prompt_id, cancelled):
    client, queue, calls = cancel_client
    response = await client.post(f"/api/jobs/{prompt_id}/cancel")
    assert response.status == 200
    assert await response.json() == {"cancelled": cancelled}
    assert bool(calls) == (prompt_id == RUNNING)
    assert len(queue.queue) == (0 if prompt_id == PENDING else 1)
    if prompt_id == PENDING:
        response = await client.post(f"/api/jobs/{prompt_id}/cancel")
        assert response.status == 200
        assert await response.json() == {"cancelled": False}


async def test_issue_72_production_batch_cancel(cancel_client):
    client, queue, calls = cancel_client
    response = await client.post("/api/jobs/cancel", json={"job_ids": [RUNNING, PENDING, FINISHED, UNKNOWN]})
    assert response.status == 200
    assert await response.json() == {"cancelled": True}
    assert queue.queue == []
    assert calls == [True]
