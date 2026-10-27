import threading
from types import SimpleNamespace

import pytest
import torch

from comfy import ops


@pytest.fixture
def releases(monkeypatch):
    released = []
    monkeypatch.setattr(ops, "_DEFERRED_VBAR_UNPINS", [])
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, "vbar_unpin", released.append)
    return released


@pytest.mark.parametrize("block", [False, True])
def test_issue_70_concurrent_drains_release_each_allocation_once(releases, block):
    entered = threading.Event()
    second = threading.Event()
    release = threading.Event()
    errors = []

    def ready():
        if entered.is_set():
            second.set()
        entered.set()
        assert release.wait(5)
        return True

    event = SimpleNamespace(query=ready, synchronize=ready)
    ops._DEFERRED_VBAR_UNPINS.append((event, "allocation"))

    def drain():
        try:
            ops._drain_deferred_vbar_unpins(block=block)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=drain) for _ in range(2)]
    try:
        threads[0].start()
        assert entered.wait(5)
        threads[1].start()
        second.wait(0.5)
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    assert releases == ["allocation"]
    assert ops._DEFERRED_VBAR_UNPINS == []


def test_issue_70_enqueue_during_drain_is_not_lost(monkeypatch, releases):
    iterated = threading.Event()
    appended = threading.Event()
    release = threading.Event()
    errors = []

    class PausedList(list):
        def __iter__(self):
            yield from super().__iter__()
            iterated.set()
            assert release.wait(5)

    done = SimpleNamespace(query=lambda: True)
    pending = SimpleNamespace(record=lambda stream: None, query=lambda: False)
    items = PausedList([(done, "old")])
    monkeypatch.setattr(ops, "_DEFERRED_VBAR_UNPINS", items)
    monkeypatch.setattr(torch.cuda, "Event", lambda: pending)
    monkeypatch.setattr(ops.model_management, "current_stream", lambda device: None)

    def drain():
        try:
            ops._drain_deferred_vbar_unpins()
        except Exception as exc:
            errors.append(exc)

    def enqueue():
        try:
            ops._defer_vbar_unpin("new", torch.device("cuda:1"))
            appended.set()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=drain), threading.Thread(target=enqueue)]
    try:
        threads[0].start()
        assert iterated.wait(5)
        threads[1].start()
        appended.wait(0.5)
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    assert appended.is_set()
    assert releases == ["old"]
    assert items == [(pending, "new")]


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_non_cuda_unpin_is_immediate(releases, device):
    ops._defer_vbar_unpin("allocation", device)
    assert releases == ["allocation"]
    assert ops._DEFERRED_VBAR_UNPINS == []
