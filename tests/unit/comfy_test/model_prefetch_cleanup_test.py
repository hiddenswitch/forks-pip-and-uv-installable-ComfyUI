import pytest
import torch



import comfy.model_prefetch as model_prefetch  # noqa: E402


class MTPHead(torch.nn.Module):
    pass


@pytest.fixture(autouse=True)
def _reset_queues():
    model_prefetch.PREFETCH_QUEUES = []
    yield
    model_prefetch.PREFETCH_QUEUES = []


def _prefetched(module):
    module._prefetch = {"signature": None}
    return (None, (module, [module]))


@pytest.mark.parametrize("pending_unit", [tuple, list])
def test_cleanup_skips_unconsumed_grouped_unit(pending_unit):
    # Qwen3.5 MTP draft queue interrupted after the embed pop: the embed slot has
    # been pinned, the grouped (head, mtp) slot has not been reached yet.
    embed = torch.nn.Embedding(4, 4)
    head = torch.nn.Linear(4, 4)
    queue = [_prefetched(embed), pending_unit((head, MTPHead())), None]
    model_prefetch.PREFETCH_QUEUES = [queue]

    model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []
    assert embed._prefetch is None


def test_cleanup_resets_queues_when_entry_cleanup_fails(monkeypatch):
    def boom(module, comfy_modules):
        raise RuntimeError("unpin failed")

    monkeypatch.setattr(model_prefetch, "cleanup_prefetched_modules", boom)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(torch.nn.Linear(4, 4)), None]]

    with pytest.raises(RuntimeError):
        model_prefetch.cleanup_prefetch_queues()
    # A failed cleanup must not poison every later node on the worker.
    assert model_prefetch.PREFETCH_QUEUES == []
    model_prefetch.cleanup_prefetch_queues()


@pytest.mark.parametrize("pending_unit", [tuple, list])
def test_cleanup_skips_unreached_grouped_head(pending_unit):
    # Interrupted between popping the consumed slot and pinning the next one: the
    # head of the queue is still the raw (head, mtp) group, not a pinned slot.
    queue = [pending_unit((torch.nn.Linear(4, 4), MTPHead())), None]
    model_prefetch.PREFETCH_QUEUES = [queue]

    model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []


def test_cleanup_unpins_later_queues_when_an_earlier_one_fails(monkeypatch):
    real_cleanup = model_prefetch.cleanup_prefetched_modules
    failing = torch.nn.Linear(4, 4)
    later = torch.nn.Linear(4, 4)

    def cleanup(module, comfy_modules):
        if module is failing:
            raise RuntimeError("unpin failed")
        real_cleanup(module, comfy_modules)

    monkeypatch.setattr(model_prefetch, "cleanup_prefetched_modules", cleanup)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(failing), None], [_prefetched(later), None]]

    with pytest.raises(RuntimeError, match="unpin failed"):
        model_prefetch.cleanup_prefetch_queues()
    # The first failure is re-raised, but only after every other queue was cleaned up.
    assert later._prefetch is None
    assert model_prefetch.PREFETCH_QUEUES == []


def test_failed_unpin_still_clears_prefetch_markers(monkeypatch):
    # A module whose unpin raised must not keep _prefetch, or a later cast would skip
    # the fresh cast and reuse the stale transfer state. The rest of the group is still cleaned.
    calls = []

    def vbar_unpin(v):
        calls.append(v)
        if v == "first":
            raise RuntimeError("unpin failed")

    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_unpin", vbar_unpin)
    first = torch.nn.Linear(4, 4)
    second = torch.nn.Linear(4, 4)
    for module, v in ((first, "first"), (second, "second")):
        module._v = v
        module._prefetch = {"signature": "sig"}
    model_prefetch.PREFETCH_QUEUES = [[(None, (first, [first, second])), None]]

    with pytest.raises(RuntimeError, match="unpin failed"):
        model_prefetch.cleanup_prefetch_queues()
    assert calls == ["first", "second"]
    assert first._prefetch is None
    assert second._prefetch is None


def test_malloc_graph_cleanup_failure_still_cleans_queues(monkeypatch):
    # After a GPU failure the graph abort can raise too. Queues and _prefetch markers must not
    # survive into the next node, and the graph error still reaches the caller.
    def boom():
        raise RuntimeError("graph abort failed")

    monkeypatch.setattr(model_prefetch, "cleanup_malloc_graph", boom)
    module = torch.nn.Linear(4, 4)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(module), None]]

    with pytest.raises(RuntimeError, match="graph abort failed"):
        model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []
    assert module._prefetch is None


def test_sampler_preserves_inference_error_and_releases_models(monkeypatch):
    from unittest.mock import Mock
    from comfy import samplers

    patcher = Mock(load_device=torch.device("cpu"), model_options={})
    guider = samplers.CFGGuider(patcher)
    guider.conds = {}
    secondary = RuntimeError("graph abort failed")
    patcher.model.finish_execution.side_effect = secondary
    primary = ValueError("attention failed")
    guider.inner_sample = Mock(side_effect=primary)
    cleanup = Mock()
    monkeypatch.setattr(samplers.sampler_helpers, "prepare_sampling", lambda *args: (object(), {}, []))
    monkeypatch.setattr(samplers.sampler_helpers, "prepare_model_patcher_multigpu_clones", lambda *args: [])
    monkeypatch.setattr(samplers.sampler_helpers, "cleanup_models", cleanup)
    x = torch.zeros(1)
    with pytest.raises(ValueError) as error:
        guider.outer_sample(x, x, None, x)
    assert error.value is primary
    cleanup.assert_called_once()
    assert not hasattr(guider, "loaded_models")


def test_finish_execution_releases_deferred_weights_after_graph_failure(monkeypatch):
    from unittest.mock import Mock

    error = RuntimeError("graph abort failed")
    monkeypatch.setattr(model_prefetch, "cleanup_prefetch_queues", Mock(side_effect=error))
    finish = Mock()
    monkeypatch.setattr(model_prefetch.ops, "finish_weight_cast_execution", finish)
    with pytest.raises(RuntimeError) as result:
        model_prefetch.finish_model_execution()
    assert result.value is error
    finish.assert_called_once()


def test_malloc_graph_scope_reports_kernel_failure_when_abort_fails(monkeypatch):
    from unittest.mock import Mock

    primary = RuntimeError("kernel failed")
    monkeypatch.setattr(model_prefetch, "malloc_graph_begin", Mock())
    abort = Mock(side_effect=RuntimeError("graph abort failed"))
    monkeypatch.setattr(model_prefetch, "cleanup_malloc_graph", abort)
    with pytest.raises(RuntimeError) as result:
        with model_prefetch.malloc_graph_scope(torch.device("cpu")):
            raise primary
    assert result.value is primary
    abort.assert_called_once()


def test_native_abort_failure_discards_all_module_graphs(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    failed = torch.nn.Linear(1, 1)
    later = torch.nn.Linear(1, 1)
    first_reset = Mock(side_effect=RuntimeError("reset failed"))
    last_reset = Mock()
    failed._comfy_graph = {"graph": SimpleNamespace(reset=first_reset)}
    later._comfy_graph = {"graph": SimpleNamespace(reset=last_reset)}
    abort_error = RuntimeError("aimdo memory compile error")
    graph = SimpleNamespace(_comfy_active=True, abort=Mock(side_effect=abort_error),
                            _comfy_cuda_graph_modules=[failed, later])
    key = model_prefetch.threading.get_ident()
    monkeypatch.setitem(model_prefetch.MALLOC_GRAPHS, key, graph)

    with pytest.raises(RuntimeError) as result:
        model_prefetch.cleanup_malloc_graph()

    assert result.value is abort_error
    assert key not in model_prefetch.MALLOC_GRAPHS
    assert not hasattr(failed, "_comfy_graph")
    assert not hasattr(later, "_comfy_graph")
    first_reset.assert_called_once()
    last_reset.assert_called_once()
