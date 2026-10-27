from unittest.mock import Mock

import pytest
import torch

from comfy import model_management, model_prefetch
from comfy.interruption import InterruptProcessingException
from comfy.multigpu import MultiGPUThreadPool


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError, InterruptProcessingException])
def test_issue_71_worker_returns_error_and_accepts_more_work(monkeypatch, error_type):
    monkeypatch.setattr(model_management, "set_torch_device", lambda device: None)
    cleanup = Mock()
    monkeypatch.setattr(model_prefetch, "cleanup_malloc_graph", cleanup)
    device = torch.device("cuda:1")
    pool = MultiGPUThreadPool([device])
    error = error_type("worker failure")
    result_queue = pool._result_queues[device]
    get = result_queue.get
    monkeypatch.setattr(result_queue, "get", lambda: get(timeout=2))

    def fail():
        raise error

    try:
        pool.submit(device, fail)
        assert pool.get_result(device) == (None, error)
        pool.submit(device, lambda value: value + 1, 41)
        assert pool.get_result(device) == (42, None)
        assert pool._workers[0].is_alive()
    finally:
        pool.shutdown()
    assert not pool._workers[0].is_alive()
    cleanup.assert_called_once_with()
