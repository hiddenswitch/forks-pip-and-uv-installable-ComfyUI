from __future__ import annotations

import importlib.resources
import re
from pathlib import Path


def _source_root() -> Path:
    return Path(importlib.resources.files("comfy")).parent


def test_maintained_docs_have_valid_local_links_and_no_deleted_pages():
    docs_root = _source_root() / "docs"
    deleted = {"lanpaint.md", "openapi_cleanup.md"}
    for document in docs_root.glob("*.md"):
        contents = document.read_text(encoding="utf-8")
        assert not any(name in contents for name in deleted)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", contents):
            target = target.split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (document.parent / target).exists(), (document, target)


def test_maintained_docs_use_uv_for_installation_examples():
    docs_root = _source_root() / "docs"
    for document in docs_root.glob("*.md"):
        contents = document.read_text(encoding="utf-8")
        assert not re.search(r"(?<!uv )(?<!python -m )\bpip install\b", contents), document


def test_distributed_docs_show_worker_and_frontend_startup():
    contents = (_source_root() / "docs" / "distributed.md").read_text(encoding="utf-8")
    assert "comfyui worker" in contents
    assert "comfyui serve" in contents
    assert "--distributed-queue-frontend" in contents
    assert "--distributed-queue-connection-uri" in contents
