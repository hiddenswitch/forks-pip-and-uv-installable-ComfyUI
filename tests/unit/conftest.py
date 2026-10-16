"""Unit-test session fixtures.

Pre-loads the real node system exactly once per pytest session so the
expensive ``import_all_nodes_in_workspace`` call happens at a predictable
point (before any test executes) and before any test has a chance to
mutate ``sys.modules`` or per-module ``NODE_CLASS_MAPPINGS``. Later
fixtures that need the full node set should reuse this snapshot instead
of calling ``import_all_nodes_in_workspace`` themselves.
"""
import os
import sys

import pytest


# This subprocess-heavy test has repeatedly destabilized Linux hosts by
# triggering system-wide Btrfs metadata lock stalls. Keep it disabled on Linux
# until the test can run without affecting the host filesystem.
if sys.platform.startswith("linux"):
    collect_ignore = ["test_custom_node_tensor_parallel.py"]


@pytest.fixture(scope="session")
def _preloaded_nodes():
    """Full snapshot of all importable nodes, frozen at session start.

    Copies NODE_CLASS_MAPPINGS into a standalone ExportedNodes so later
    ``import_all_nodes_in_workspace`` calls (which ``.clear()`` the shared
    ``_nodes_local.nodes`` instance) can't erase entries we already saw.
    Normally forces ``disable_all_custom_nodes=False`` in case an earlier
    test has left a Configuration on the current context that would skip
    custom node loading. Device-specific CI can set
    ``COMFYUI_UNIT_DISABLE_CUSTOM_NODES=1`` to isolate core unit tests from
    custom-node import side effects.
    """
    from comfy.cli_args import default_configuration
    from comfy.nodes.package import import_all_nodes_in_workspace
    from comfy.nodes.package_typing import ExportedNodes
    from comfy.execution_context import context_configuration

    cfg = default_configuration()
    disable_custom_nodes = os.environ.get("COMFYUI_UNIT_DISABLE_CUSTOM_NODES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cfg.disable_all_custom_nodes = disable_custom_nodes
    with context_configuration(cfg):
        live = import_all_nodes_in_workspace()

    snapshot = ExportedNodes()
    snapshot.NODE_CLASS_MAPPINGS.update(live.NODE_CLASS_MAPPINGS)
    snapshot.NODE_DISPLAY_NAME_MAPPINGS.update(live.NODE_DISPLAY_NAME_MAPPINGS)
    snapshot.EXTENSION_WEB_DIRS.update(live.EXTENSION_WEB_DIRS)
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def _eagerly_preload_nodes(_preloaded_nodes):
    """Force ``_preloaded_nodes`` to be constructed before any test runs.

    Session fixtures are otherwise lazy; we need this one eager because a
    handful of earlier tests in the sweep mutate ``sys.modules`` in ways
    that cause subsequent ``import_all_nodes_in_workspace`` calls to drop
    several hundred custom nodes.
    """
    yield _preloaded_nodes


@pytest.fixture
def cached_gpt2_hf_download(monkeypatch, tmp_path):
    """Provide a deterministic local stand-in for the public GPT-2 config."""
    from comfy import model_downloader

    cached = tmp_path / "hf-cache" / "config.json"
    cached.parent.mkdir()
    cached.write_text('{"model_type": "gpt2"}', encoding="utf-8")

    original_get_folder_paths = model_downloader.folder_paths.get_folder_paths
    monkeypatch.setattr(
        model_downloader.folder_paths,
        "get_folder_paths",
        lambda folder_name: [str(tmp_path)]
        if folder_name == "checkpoints"
        else original_get_folder_paths(folder_name),
    )
    monkeypatch.setattr(
        model_downloader,
        "_original_get_full_path",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(model_downloader, "hf_hub_download", fake_hf_hub_download)
    return cached, calls


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """The real file database: WAL mode, both engines and their hooks, in ``tmp_path``."""
    from comfy.app.database import db as db_module

    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.current_execution_context().configuration, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    db_module._init_file_db(db_module.get_database_url(), use_chain_hash=False)
    yield db_path
    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    db_module._db_lock.release(force=True)
