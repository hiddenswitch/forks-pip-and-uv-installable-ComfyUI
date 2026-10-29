from __future__ import annotations

import importlib.resources
import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


def _source_root() -> Path:
    return Path(importlib.resources.files("comfy")).parent


def _maintained_documents():
    root = _source_root()
    # Vendor snapshots and licenses are source material, not locally maintained
    # instructions. Check their index and all maintained component READMEs.
    return sorted({
        *root.glob("*.md"),
        *(root / "docs").glob("*.md"),
        *(root / "comfy").rglob("README.md"),
        *(root / "ci").glob("*.md"),
        *(root / ".github" / "PULL_REQUEST_TEMPLATE").glob("*.md"),
        root / "docs" / "prompting_guides" / "README.md",
        root / "docs" / "prompting_guides" / "model_coverage.md",
        root / "docs" / "prompting_guides" / "flux3_prompting.md",
        root / "docs" / "prompting_guides" / "qwen_image_2_1_edit_prompting.md",
        root / "llms.txt",
    })


def _prose(contents):
    return re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", contents)


def _anchors(contents):
    prose = _prose(contents)
    anchors = set(re.findall(r'<[^>]+\b(?:id|name)=["\']([^"\']+)["\']', prose))
    counts = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", prose):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading).lower()
        slug = re.sub(r"[^\w\-\s]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


def _local_targets(contents):
    prose = _prose(contents)
    # Inline links also include Markdown images. Reference definitions cover
    # reference-style links; HTML href/src attributes cover embedded assets.
    yield from re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose)
    yield from re.findall(r"(?m)^\s*\[[^]]+\]:\s*(\S+)", prose)
    yield from re.findall(r'<[^>]+\b(?:href|src)=["\']([^"\']+)["\']', prose)


@pytest.mark.parametrize("document", _maintained_documents(), ids=lambda p: str(p.relative_to(_source_root())))
def test_maintained_docs_have_valid_local_links_and_no_deleted_pages(document):
    contents = document.read_text(encoding="utf-8")
    for target in _local_targets(contents):
        target = target.strip().split(' "', 1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        path = unquote(parsed.path)
        destination = document.parent / path if path else document
        assert destination.name not in {"lanpaint.md", "openapi_cleanup.md", "SPONSORS.md"}
        assert destination.exists(), (document, target)
        if parsed.fragment and destination.suffix in {".md", ".txt"}:
            assert unquote(parsed.fragment) in _anchors(destination.read_text(encoding="utf-8")), (document, target)


def test_maintained_docs_use_uv_for_installation_examples():
    for document in [*_source_root().glob("README.md"), *(_source_root() / "docs").glob("*.md"), _source_root() / "llms.txt"]:
        contents = document.read_text(encoding="utf-8")
        assert not re.search(r"(?<!uv )(?<!python -m )\bpip install\b", contents), document


def test_every_native_model_has_a_prompting_or_input_reference():
    root = _source_root()
    tree = ast.parse((root / "comfy" / "supported_models.py").read_text(encoding="utf-8"))
    registration = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "models" for target in node.targets)
    )
    assert isinstance(registration, (ast.List, ast.Tuple))
    registered = {node.id for node in registration.elts if isinstance(node, ast.Name)}
    assert len(registered) == len(registration.elts), "Review changed model registration structure"
    coverage = (root / "docs" / "prompting_guides" / "model_coverage.md").read_text(encoding="utf-8")
    documented = set()
    for row in coverage.splitlines():
        if row.startswith("|"):
            cells = row.split("|")
            names = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", cells[1]))
            if names & registered:
                assert re.search(r"\[[^]]+\]\([^)]+\)", cells[2]), row
                documented.update(names)
    assert registered <= documented, sorted(registered - documented)


def test_prompting_guide_index_lists_every_local_guide():
    directory = _source_root() / "docs" / "prompting_guides"
    targets = {urlsplit(target).path for target in _local_targets((directory / "README.md").read_text(encoding="utf-8"))}
    guides = {p.name for p in directory.iterdir() if p.suffix in {".md", ".json"} and p.name != "README.md"}
    assert guides <= targets, sorted(guides - targets)


def test_distributed_docs_show_worker_and_frontend_startup():
    contents = (_source_root() / "docs" / "deployment.md").read_text(encoding="utf-8")
    assert "comfyui worker" in contents
    assert "comfyui serve" in contents
    assert "--distributed-queue-frontend" in contents
    assert "--distributed-queue-connection-uri" in contents


def test_link_check_covers_images_references_and_duplicate_heading_anchors():
    sample = '# Intro\n## More `details`\n## More `details`\n<a id="custom"></a>\n'
    assert _anchors(sample) == {"intro", "more-details", "more-details-1", "custom"}
    assert list(_local_targets('![image](assets/a.png)\n[guide][ref]\n[ref]: guide.md#intro\n<img src="b.png">')) == ["assets/a.png", "guide.md#intro", "b.png"]
    assert list(_local_targets('```markdown\n[placeholder](missing.md)\n```')) == []


@pytest.mark.parametrize("relative_path", [
    "docs/api.md", "docs/embedded.md", "docs/configuration.md",
    "docs/custom_node_authoring.md", "llms.txt",
])
def test_integration_examples_have_valid_python_syntax(relative_path):
    contents = (_source_root() / relative_path).read_text(encoding="utf-8")
    examples = re.findall(r"(?ms)^```(?:python|py)\n(.*?)^```", contents)
    assert examples, relative_path
    for index, example in enumerate(examples):
        compile(example, f"{relative_path}:example-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_embedded_workflow_examples_build_valid_graphs():
    from comfy.api.components.schema.prompt import Prompt

    contents = (_source_root() / "docs" / "embedded.md").read_text(encoding="utf-8")
    examples = re.findall(r"(?ms)^```python\n(.*?)^```", contents)
    workflow_assignment = next(
        node
        for example in examples
        for node in ast.parse(example).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "WORKFLOW" for target in node.targets)
    )
    workflow = ast.literal_eval(workflow_assignment.value)
    graph_example = next(example for example in examples if "def build_graph(" in example)
    namespace = {}
    exec(compile(graph_example, "docs/embedded.md:GraphBuilder", "exec"), namespace)
    for graph in (workflow, namespace["prompt"]):
        validated = Prompt.validate(graph)
        assert len(validated) == 7
        assert {node.class_type for node in validated.values()} == {
            "CheckpointLoaderSimple", "EmptyLatentImage", "CLIPTextEncode",
            "KSampler", "VAEDecode", "SaveImage",
        }
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    assert len(value) == 2 and value[0] in graph and isinstance(value[1], int)
