"""Cross-validate Python workflow conversion against the real frontend JS.

Runs the compiled ComfyUI frontend in headless Chromium via Playwright.
For each template workflow, loads it via the frontend's ``app.loadGraphData()``,
calls ``app.graphToPrompt()`` for the authoritative output, and compares it
to the Python ``convert_ui_to_api()`` result.

Frontend outputs are cached on disk keyed by the frontend package version
so Playwright is only needed when the frontend changes.

Requires: ``pip install playwright && python -m playwright install chromium``
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
import traceback
from importlib.metadata import distributions, version as pkg_version
from importlib.resources import files
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

_STALE_SUBGRAPH_WIDGET_ORDER = (
    "saved subgraph proxy widget order disagrees with the current node schema",
)

_EXCLUDED_TEMPLATE_REASONS: dict[str, tuple[str, ...]] = {
    "api_flux2": ("frontend 1.54.7 drops connected dynamic Autogrow image inputs on load",),
    "template_image_speech_to_video": ("frontend 1.54.7 loses the rerouted subgraph prompt link and emits an empty string",),
    "basic_image_color_adjustment": ("frontend 1.54.7 throws DataCloneError while loading CURVE widgets",),
    "api_google_gemini_omni_flash_i2v": ("frontend 1.54.7 drops image inputs and routes an IMAGE link to aspect_ratio",),
    "video_minimax_h3_fun_controlnet_union": _STALE_SUBGRAPH_WIDGET_ORDER,
    "utility_sdpose_multi_person": _STALE_SUBGRAPH_WIDGET_ORDER,
    "utility_sdpose_multi_person_video": _STALE_SUBGRAPH_WIDGET_ORDER,
}

_EXCLUDED_TEMPLATE_IDS = frozenset(_EXCLUDED_TEMPLATE_REASONS)

# ---------------------------------------------------------------------------
# Guard: skip entire module if playwright is not installed
# ---------------------------------------------------------------------------
pw = pytest.importorskip("playwright")

from playwright.sync_api import sync_playwright  # noqa: E402

# ---------------------------------------------------------------------------
# Cache directory for frontend outputs
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(str(files("tests.unit").joinpath("playwright_cache")))


def _frontend_version() -> str:
    return pkg_version("comfyui-frontend-package")


def _templates_version() -> str:
    template_dists: list[tuple[str, str]] = []
    for dist in distributions():
        metadata = dist.metadata
        name = metadata.get("Name", "") if metadata is not None else ""
        if name == "comfyui-workflow-templates" or name.startswith("comfyui-workflow-templates-"):
            template_dists.append((name, dist.version))
    if not template_dists:
        return pkg_version("comfyui-workflow-templates")
    return "+".join(f"{name.removeprefix('comfyui-workflow-templates')}={version}" for name, version in sorted(template_dists))


def _cache_path(template_id: str) -> Path:
    return _CACHE_DIR / f"{_frontend_version()}+t{_templates_version()}" / f"{template_id}.json"


def _load_cached(template_id: str) -> dict | None:
    path = _cache_path(template_id)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "__frontend_error__" in data:
            return None
        return data
    return None


def _save_cached(template_id: str, output: dict) -> None:
    path = _cache_path(template_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"), sort_keys=True)


def invalidate_stale_cache() -> list[str]:
    """Delete cached frontend outputs that contain ``class_type: null`` nodes.

    Call this after adding new node implementations (e.g. in comfy_extras)
    so that the Playwright tests regenerate the frontend output with the
    updated ``/object_info`` response.

    Returns the list of deleted template IDs.
    """
    version_dir = _CACHE_DIR / f"{_frontend_version()}+t{_templates_version()}"
    if not version_dir.exists():
        return []
    deleted: list[str] = []
    for path in sorted(version_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if any(node.get("class_type") is None for node in data.values()):
            path.unlink()
            deleted.append(path.stem)
    return deleted


# ---------------------------------------------------------------------------
# Template discovery (runs at collection time)
# ---------------------------------------------------------------------------

def _load_template_workflow(template_id: str) -> dict | None:
    try:
        from comfyui_workflow_templates import get_asset_path, iter_templates
    except ImportError:
        return None
    for t in iter_templates():
        if t.template_id == template_id:
                json_assets = [a for a in t.assets if a.filename.endswith(".json")]
                if json_assets:
                    path = get_asset_path(t.template_id, json_assets[0].filename)
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
    return None


def _is_ui_workflow(data: dict) -> bool:
    return "nodes" in data and "links" in data


def _ui_template_ids() -> list[str]:
    """Discover template IDs where the JSON asset is a UI-format workflow."""
    try:
        from comfyui_workflow_templates import get_asset_path, iter_templates
    except ImportError:
        return []
    ids = []
    for t in iter_templates():
        json_assets = [a for a in t.assets if a.filename.endswith(".json")]
        if json_assets:
            path = get_asset_path(t.template_id, json_assets[0].filename)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if _is_ui_workflow(data) and t.template_id not in _EXCLUDED_TEMPLATE_IDS:
                ids.append(t.template_id)
    return ids


def _real_nodes_available() -> bool:
    try:
        from comfy.nodes.package import import_all_nodes_in_workspace  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Object-info generation (replicates server.py node_info logic)
# ---------------------------------------------------------------------------

def _build_object_info(nodes) -> dict:
    """Generate the full ``/object_info`` response dict from loaded nodes."""
    from comfy_api.internal import _ComfyNodeInternal

    out: dict[str, dict] = {}
    for node_class in nodes.NODE_CLASS_MAPPINGS:
        try:
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            if issubclass(obj_class, _ComfyNodeInternal):
                out[node_class] = obj_class.GET_NODE_INFO_V1()
                continue
            info: dict = {}
            info["input"] = obj_class.INPUT_TYPES()
            info["input_order"] = {
                key: list(value.keys())
                for key, value in obj_class.INPUT_TYPES().items()
            }
            info["is_input_list"] = getattr(obj_class, "INPUT_IS_LIST", False)
            _return_types = [
                "*" if isinstance(rt, list) and rt == [] else rt
                for rt in obj_class.RETURN_TYPES
            ]
            info["output"] = _return_types
            info["output_is_list"] = (
                obj_class.OUTPUT_IS_LIST
                if hasattr(obj_class, "OUTPUT_IS_LIST")
                else [False] * len(_return_types)
            )
            info["output_name"] = (
                obj_class.RETURN_NAMES
                if hasattr(obj_class, "RETURN_NAMES")
                else info["output"]
            )
            info["name"] = node_class
            info["display_name"] = (
                nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class]
                if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS
                else node_class
            )
            info["description"] = (
                obj_class.DESCRIPTION if hasattr(obj_class, "DESCRIPTION") else ""
            )
            info["python_module"] = getattr(
                obj_class, "RELATIVE_PYTHON_MODULE", "nodes"
            )
            info["category"] = "sd"
            info["output_node"] = bool(
                hasattr(obj_class, "OUTPUT_NODE") and obj_class.OUTPUT_NODE
            )
            if hasattr(obj_class, "CATEGORY"):
                info["category"] = obj_class.CATEGORY
            if hasattr(obj_class, "OUTPUT_TOOLTIPS"):
                info["output_tooltips"] = obj_class.OUTPUT_TOOLTIPS
            if getattr(obj_class, "DEPRECATED", False):
                info["deprecated"] = True
            if getattr(obj_class, "EXPERIMENTAL", False):
                info["experimental"] = True
            if getattr(obj_class, "DEV_ONLY", False):
                info["dev_only"] = True
            if hasattr(obj_class, "API_NODE"):
                info["api_node"] = obj_class.API_NODE
            info["search_aliases"] = getattr(obj_class, "SEARCH_ALIASES", [])
            out[node_class] = info
        except Exception:
            logger.warning("Failed to get node info for %s:\n%s", node_class, traceback.format_exc())
    return out


# ---------------------------------------------------------------------------
# Static file server (aiohttp)
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_static_dir() -> Path:
    import comfyui_frontend_package
    return Path(comfyui_frontend_package.__path__[0]) / "static"


def _start_static_server(port: int, object_info_json: str) -> asyncio.AbstractEventLoop:
    """Start an aiohttp server in a background thread, return the event loop."""
    import aiohttp
    from aiohttp import web

    async def _ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "feature_flags":
                    await ws.send_json({
                        "type": "feature_flags",
                        "data": {
                            "supports_preview_metadata": True,
                            "max_upload_size": 104857600,
                        },
                    })
                    await ws.send_json({
                        "type": "status",
                        "data": {
                            "status": {"exec_info": {"queue_remaining": 0}},
                        },
                    })
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break
        return ws

    static_dir = _get_static_dir()

    async def _index_handler(request):
        return web.FileResponse(static_dir / "index.html")

    app = web.Application()
    app.router.add_get("/", _index_handler)
    app.router.add_get("/ws", _ws_handler)
    app.router.add_static("/", static_dir)

    loop = asyncio.new_event_loop()
    _stop_event: asyncio.Event | None = None

    async def _run():
        nonlocal _stop_event
        _stop_event = asyncio.Event()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        await _stop_event.wait()
        await runner.cleanup()

    def _thread_target():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    # Wait until the server is actually listening
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    return loop, _stop_event


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _real_nodes(_preloaded_nodes):
    """Activate the preloaded node snapshot in the execution context."""
    from comfy.execution_context import context_add_custom_nodes
    with context_add_custom_nodes(_preloaded_nodes):
        yield _preloaded_nodes


@pytest.fixture(scope="session")
def _object_info_json(_real_nodes):
    """Generate the /object_info response as a JSON string."""
    info = _build_object_info(_real_nodes)
    return json.dumps(info)


@pytest.fixture(scope="session")
def _server_port():
    return _find_free_port()


@pytest.fixture(scope="session")
def _static_server(_server_port, _object_info_json):
    """Start the static file server for the session."""
    loop, stop_event = _start_static_server(_server_port, _object_info_json)
    yield _server_port
    if stop_event is not None:
        loop.call_soon_threadsafe(stop_event.set)


# ---------------------------------------------------------------------------
# Module-scoped page fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _app_page(_static_server, _object_info_json, _real_nodes):
    """Create a Playwright browser page while this test module is active.

    Playwright's synchronous API keeps its asyncio dispatcher running until
    the context manager exits.  A session-scoped fixture therefore leaves a
    running loop in the main thread when later pytest-asyncio tests start.
    """
    port = _static_server
    base_url = f"http://127.0.0.1:{port}"

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                yield None
                return
            raise
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Route API endpoints
        def _handle_object_info(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=_object_info_json,
            )

        def _handle_json_empty_dict(route):
            route.fulfill(status=200, content_type="application/json", body="{}")

        def _handle_json_empty_list(route):
            route.fulfill(status=200, content_type="application/json", body="[]")

        def _handle_prompt(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"exec_info": {"queue_remaining": 0}}',
            )

        def _handle_users(route):
            # Single-user mode response (skips user-select page)
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"storage": "server", "migrated": true}',
            )

        def _handle_system_stats(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"system": {"os": "linux"}, "devices": []}',
            )

        def _handle_userdata(route):
            route.fulfill(status=404, body="")

        def _handle_user_css(route):
            route.fulfill(status=200, content_type="text/css", body="")

        def _handle_features(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"supports_preview_metadata": true, "max_upload_size": 104857600}',
            )

        def _handle_upload_image(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"name":"temp_upload.png","subfolder":"threed","type":"temp"}',
            )

        def _handle_workflow_templates(route):
            route.fulfill(status=200, content_type="application/json", body="[]")

        def _handle_global_subgraphs(route):
            route.fulfill(status=200, content_type="application/json", body="{}")

        # Register route handlers (order matters - more specific first)
        page.route("**/object_info", _handle_object_info)
        page.route("**/settings/**", _handle_json_empty_dict)
        page.route("**/settings", _handle_json_empty_dict)
        page.route("**/embeddings", _handle_json_empty_list)
        page.route("**/extensions", _handle_json_empty_list)
        page.route("**/prompt", _handle_prompt)
        page.route("**/users", _handle_users)
        page.route("**/workflow_templates", _handle_workflow_templates)
        page.route("**/global_subgraphs", _handle_global_subgraphs)
        page.route("**/system_stats", _handle_system_stats)
        page.route("**/features", _handle_features)
        page.route("**/upload/image", _handle_upload_image)
        page.route("**/api/userdata/**", _handle_userdata)
        page.route("**/userdata/**", _handle_userdata)
        page.route("**/user.css", _handle_user_css)

        # Collect console errors for debugging
        console_errors = []
        page.on("console", lambda msg: (
            console_errors.append(f"[{msg.type}] {msg.text}")
            if msg.type == "error" else None
        ))

        # Navigate and wait for the app to initialize
        page.goto(base_url, wait_until="networkidle", timeout=60000)

        # Wait for the app graph to exist
        page.wait_for_function(
            """() => {
                try {
                    return !!(
                        window.comfyAPI &&
                        window.comfyAPI.app &&
                        window.comfyAPI.app.app &&
                        window.comfyAPI.app.app.graph &&
                        Object.keys(window.LiteGraph?.registered_node_types || {}).length > 100
                    );
                } catch(e) { return false; }
            }""",
            timeout=60000,
        )

        # Verify node types are registered
        node_count = page.evaluate("""() => {
            const app = window.comfyAPI.app.app;
            try {
                // LiteGraph stores registered node types
                return Object.keys(LiteGraph.registered_node_types || {}).length;
            } catch(e) {
                return -1;
            }
        }""")
        logger.info("Frontend registered %d node types", node_count)
        assert node_count > 100, (
            f"Expected >100 node types registered in frontend, got {node_count}. "
            f"Console errors: {console_errors[:5]}"
        )

        yield page

        context.close()
        browser.close()


# ---------------------------------------------------------------------------
# Normalization and comparison helpers
# ---------------------------------------------------------------------------

def _normalize_numeric(val):
    """Normalize JS numeric quirks: 512.0 → 512 when lossless."""
    if isinstance(val, float):
        int_val = int(val)
        if float(int_val) == val:
            return int_val
        # Round to avoid FP noise (e.g. 0.30000000000000004 → 0.3)
        return round(val, 10)
    return val


_UI_STATE_INPUTS: set[tuple[str, str]] = {
    ("LoadAudio", "audioUI"),
    ("SaveAudio", "audioUI"),
    ("PreviewAudio", "audioUI"),
    ("SaveAudioMP3", "audioUI"),
    ("SaveAudioOpus", "audioUI"),
    ("RecordAudio", "audio"),
    ("Preview3D", "image"),
    ("Preview3DAdvanced", "viewport_state"),
    ("Save3DAdvanced", "viewport_state"),
    ("PreviewGaussianSplat", "viewport_state"),
    ("PreviewPointCloud", "viewport_state"),
    ("SaveGaussianSplat", "viewport_state"),
    ("SavePointCloud", "viewport_state"),
    ("SaveGLB", "image"),
    ("Load3D", "image"),
    ("ImageCompare", "compare_view"),
    ("PreviewAny", "previewMode"),
}

def _find_missing_frontend_node_types(api_output: dict) -> list[str]:
    missing: list[str] = []
    for node_data in api_output.values():
        class_type = node_data.get("class_type")
        if class_type is None:
            title = node_data.get("_meta", {}).get("title")
            if isinstance(title, str) and title:
                missing.append(title)
            else:
                missing.append("<unknown>")
    return sorted(set(missing))


def _normalize_api_output(output: dict) -> dict:
    """Normalize an API output dict for comparison."""
    normalized = {}
    for node_id, node_data in output.items():
        node_id_str = str(node_id)
        class_type = node_data.get("class_type")
        entry = {
            "class_type": class_type,
            "inputs": {},
        }
        for key, val in node_data.get("inputs", {}).items():
            if key == "_meta":
                continue
            if (class_type, key) in _UI_STATE_INPUTS:
                continue
            if isinstance(val, list) and len(val) == 2:
                entry["inputs"][key] = [str(val[0]), int(val[1])]
            elif isinstance(val, dict) and "__value__" in val:
                entry["inputs"][key] = val
            else:
                entry["inputs"][key] = _normalize_numeric(val)
        normalized[node_id_str] = entry
    return normalized


def _compare_outputs(frontend: dict, python: dict) -> list[str]:
    """Compare two normalized API outputs, return list of mismatch descriptions."""
    mismatches = []

    frontend_ids = set(frontend.keys())
    python_ids = set(python.keys())

    missing_in_python = frontend_ids - python_ids
    extra_in_python = python_ids - frontend_ids

    if missing_in_python:
        mismatches.append(f"Nodes in frontend but not Python: {sorted(missing_in_python)}")
    if extra_in_python:
        mismatches.append(f"Nodes in Python but not frontend: {sorted(extra_in_python)}")

    for node_id in sorted(frontend_ids & python_ids):
        f_node = frontend[node_id]
        p_node = python[node_id]

        if f_node["class_type"] != p_node["class_type"]:
            mismatches.append(
                f"Node {node_id}: class_type mismatch: "
                f"frontend={f_node['class_type']!r} vs python={p_node['class_type']!r}"
            )
            continue

        f_inputs = f_node["inputs"]
        p_inputs = p_node["inputs"]

        f_keys = set(f_inputs.keys())
        p_keys = set(p_inputs.keys())

        missing_keys = f_keys - p_keys
        extra_keys = p_keys - f_keys

        if missing_keys:
            mismatches.append(
                f"Node {node_id} ({f_node['class_type']}): "
                f"inputs in frontend but not Python: {sorted(missing_keys)}"
            )
        if extra_keys:
            mismatches.append(
                f"Node {node_id} ({f_node['class_type']}): "
                f"inputs in Python but not frontend: {sorted(extra_keys)}"
            )

        for key in sorted(f_keys & p_keys):
            f_val = f_inputs[key]
            p_val = p_inputs[key]
            if f_val != p_val:
                mismatches.append(
                    f"Node {node_id} ({f_node['class_type']}).inputs[{key!r}]: "
                    f"frontend={f_val!r} vs python={p_val!r}"
                )

    return mismatches


def _format_mismatches(template_id: str, mismatches: list[str]) -> str:
    header = f"Template {template_id!r} has {len(mismatches)} mismatch(es):"
    details = "\n  ".join(mismatches[:20])
    if len(mismatches) > 20:
        details += f"\n  ... and {len(mismatches) - 20} more"
    return f"{header}\n  {details}"


def _get_frontend_output(
    template_id: str,
    workflow: dict,
    page,
    *,
    use_cache: bool = True,
) -> dict:
    """Get frontend output, using cache if available."""
    if use_cache:
        cached = _load_cached(template_id)
        if cached is not None:
            return cached

    if page is None:
        pytest.skip(
            "Playwright browsers not installed. Run: playwright install chromium"
        )

    def _evaluate_frontend_output() -> dict:
        # Frontend loadGraphData(clean=true) resets the canvas to rootGraph,
        # clears it, and awaits configuration of the supplied workflow. A
        # page reload is both redundant and racy: persisted-workflow startup
        # can finish afterward and replace the workflow under test.
        return page.evaluate(
            """async (wf) => {
                const app = window.comfyAPI.app.app;
                await app.loadGraphData(wf, true, true, null, {
                    showMissingNodesDialog: false,
                    showMissingModelsDialog: false,
                });
                const result = await app.graphToPrompt();
                return result.output;
            }""",
            workflow,
        )

    try:
        frontend_output = _evaluate_frontend_output()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if "InvalidLinkError" not in str(exc):
            raise
        logger.warning("Retrying frontend conversion for %s after InvalidLinkError", template_id)
        try:
            frontend_output = _evaluate_frontend_output()
        except Exception as exc2:
            if "InvalidLinkError" not in str(exc2):
                raise
            # Frontend cannot convert this workflow due to broken links.
            # Cache a sentinel so we don't re-run Playwright on every test.
            _save_cached(template_id, {"__frontend_error__": str(exc2)})
            return None

    _save_cached(template_id, frontend_output)
    return frontend_output


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _real_nodes_available(), reason="node system not available")
class TestFrontendParity:
    @pytest.mark.parametrize("template_id", _ui_template_ids())
    def test_convert_matches_frontend(self, template_id, _app_page, _real_nodes):
        from comfy.component_model.workflow_convert import convert_ui_to_api
        from comfy.execution_context import context_add_custom_nodes

        workflow = _load_template_workflow(template_id)
        if workflow is None:
            pytest.skip(f"template {template_id} not found")
        if not _is_ui_workflow(workflow):
            pytest.skip(f"{template_id} is not a UI workflow")

        # Frontend conversion (cached or via browser)
        used_cache = _load_cached(template_id) is not None
        frontend_output = _get_frontend_output(template_id, workflow, _app_page)
        if frontend_output is None:
            pytest.skip(f"{template_id}: frontend throws InvalidLinkError (broken workflow links)")

        # Python conversion (frontend-parity mode for comparison)
        with context_add_custom_nodes(_real_nodes):
            python_output = convert_ui_to_api(
                workflow,
                preserve_unknown_nodes=False,
                node_mappings=_real_nodes,
            )

        # Normalize and compare
        f = _normalize_api_output(frontend_output)
        p = _normalize_api_output(python_output)
        mismatches = _compare_outputs(f, p)
        if mismatches and used_cache and _app_page is not None:
            # A cache entry is only an optimization, never the parity
            # authority. Re-run the exact workflow through the pinned real
            # frontend before reporting a converter mismatch. This also
            # repairs entries accidentally populated with another template's
            # graph output.
            _cache_path(template_id).unlink(missing_ok=True)
            frontend_output = _get_frontend_output(
                template_id,
                workflow,
                _app_page,
                use_cache=False,
            )
            if frontend_output is None:
                pytest.skip(
                    f"{template_id}: frontend throws InvalidLinkError (broken workflow links)"
                )
            f = _normalize_api_output(frontend_output)
            mismatches = _compare_outputs(f, p)
        assert not mismatches, _format_mismatches(template_id, mismatches)
