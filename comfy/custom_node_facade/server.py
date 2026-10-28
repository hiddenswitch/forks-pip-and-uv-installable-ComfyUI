from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import html
import logging
import os

import aiohttp
from aiohttp import web

from ..component_model.configuration import Configuration
from ..vendor.appdirs import user_cache_dir
from .builder import (
    DEFAULT_CUDA_VARIANT,
    PYPI_PROXY_INDEX,
    PYPI_SDIST_REWRITE_FILENAME_INDEX,
    CachedWheel,
    FacadeCacheStore,
    FacadeWheelBuilder,
    is_index_variant,
)
from .triton_wheels import TritonWheelBuilder, prewarm_targets as triton_prewarm_targets
from .rocm_index import RocmSimpleIndexProxy
from .registry import FacadeRegistry, FacadeRegistryProtocol, SnapshotFacadeRegistry, SnapshotUnavailableError, canonicalize_project_name
from .storage import apply_fsspec_configuration

logger = logging.getLogger(__name__)

# How often the registry snapshot's source is checked for a new generation.
SNAPSHOT_REFRESH_INTERVAL_SECONDS = 30.0
# How often a local wheel cache directory is checked for a lost mount, and how
# long one check may take before it is reported as slow.
CACHE_HEALTH_INTERVAL_SECONDS = 30.0
CACHE_HEALTH_TIMEOUT_SECONDS = 30.0
_STREAM_CHUNK_BYTES = 1024 * 1024


def _simple_html(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><title>{html.escape(title)}</title></head><body>{body}</body></html>"


def create_facade_app(
    *,
    configuration: Configuration,
) -> web.Application:
    @web.middleware
    async def snapshot_errors(request: web.Request, handler):
        try:
            registry = request.app.get("facade_registry")
            if request.path.startswith("/simple") and isinstance(registry, SnapshotFacadeRegistry):
                await registry.list_projects()
            return await handler(request)
        except SnapshotUnavailableError as exc:
            raise web.HTTPServiceUnavailable(text=str(exc), headers={"Cache-Control": "no-store"}) from exc

    apply_fsspec_configuration(configuration)
    app = web.Application(middlewares=[snapshot_errors])
    app["facade_ready"] = False
    # Mutable state the background cache check updates after startup.
    app["facade_cache_state"] = {"error": None, "check": None}
    cache_prefix = configuration.pip_facade_cache_prefix
    if cache_prefix is None:
        cache_prefix = os.path.join(user_cache_dir(appname="comfyui"), "pip_facade")

    async def on_startup(application: web.Application) -> None:
        with tracer.start_as_current_span("Initialize Pip Facade Server") as span:
            span.set_attribute("facade.listen", configuration.listen)
            span.set_attribute("facade.port", configuration.port)
            span.set_attribute("facade.cache_prefix", str(cache_prefix))
            timeout = aiohttp.ClientTimeout(total=10 * 60.0, connect=60.0)
            session = aiohttp.ClientSession(timeout=timeout)
            registry: FacadeRegistryProtocol
            if configuration.pip_facade_snapshot_uri:
                registry = SnapshotFacadeRegistry(
                    snapshot_uri=configuration.pip_facade_snapshot_uri,
                    max_age_seconds=configuration.pip_facade_snapshot_max_age_seconds,
                )
                span.set_attribute("facade.snapshot_uri", configuration.pip_facade_snapshot_uri)
            else:
                registry = FacadeRegistry(
                    session,
                    base_url=configuration.pip_facade_registry_base_url,
                    only_known_nodes=configuration.pip_facade_only_known_nodes,
                )
                span.set_attribute("facade.registry_base_url", configuration.pip_facade_registry_base_url)
            cache_revision = getattr(configuration, "pip_facade_cache_revision", None)
            builder = FacadeWheelBuilder(
                session,
                registry,
                cache_prefix=cache_prefix,
                cache_revision=cache_revision,
            )
            application["facade_session"] = session
            application["facade_registry"] = registry
            application["facade_builder"] = builder
            application["facade_triton_builder"] = TritonWheelBuilder(session)
            application["facade_triton_cache"] = FacadeCacheStore(cache_prefix)
            application["facade_triton_locks"] = {}
            application["facade_pypi_rewrite_locks"] = {}
            application["facade_rocm_proxy"] = RocmSimpleIndexProxy()
            if isinstance(registry, SnapshotFacadeRegistry):
                # The snapshot loads in the background: the server answers
                # probes at once and reports ready when a generation is loaded.
                application["facade_snapshot_refresh"] = asyncio.create_task(
                    registry.run_refresh_loop(SNAPSHOT_REFRESH_INTERVAL_SECONDS)
                )
            else:
                with tracer.start_as_current_span("Warm Pip Facade Registry") as warmup_span:
                    projects = await registry.list_projects()
                    warmup_span.set_attribute("facade.project_count", len(projects))
            application["facade_cache_health"] = asyncio.create_task(_cache_health_loop(application))
            application["facade_ready"] = True
            # Build the triton / triton-windows wheels in the background so the
            # first request for a patched cu13x wheel is served from cache.
            # Readiness is already reported, so this does not block startup.
            application["facade_triton_prewarm"] = asyncio.create_task(_prewarm_triton(application))

    async def _prewarm_triton(application: web.Application) -> None:
        session: aiohttp.ClientSession = application["facade_session"]
        builder: TritonWheelBuilder = application["facade_triton_builder"]
        cache: FacadeCacheStore = application["facade_triton_cache"]
        built = 0
        try:
            async for served, cuda in triton_prewarm_targets(session):
                path = cache.custom_path("triton", cuda, served)
                if await asyncio.to_thread(cache.exists, path):
                    continue
                try:
                    data = await builder.build(served_filename=served, cuda=cuda)
                    await asyncio.to_thread(cache.write_bytes, path, data)
                    built += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("triton prewarm failed for %s/%s: %s", cuda, served, exc)
            logger.info("triton prewarm complete (%d wheel(s) built)", built)
        except asyncio.CancelledError:
            logger.debug("triton prewarm cancelled after %d wheel(s)", built)

    async def on_cleanup(application: web.Application) -> None:
        application["facade_ready"] = False
        for name in ("facade_triton_prewarm", "facade_snapshot_refresh", "facade_cache_health"):
            task: asyncio.Task | None = application.get(name)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        session: aiohttp.ClientSession | None = application.get("facade_session")
        if session is not None:
            await session.close()

    async def liveness(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "live": True, "ready": bool(app.get("facade_ready"))})

    async def readiness(_: web.Request) -> web.Response:
        # Answered from state the background tasks keep; never touches storage.
        ready = bool(app.get("facade_ready"))
        details = {}
        if ready and isinstance(app["facade_registry"], SnapshotFacadeRegistry):
            details = app["facade_registry"].health()
            ready = details["snapshot_error"] is None
        cache_error = app["facade_cache_state"]["error"]
        if cache_error is not None:
            details["cache_error"] = cache_error
            ready = False
        status = 200 if ready else 503
        return web.json_response({"ok": ready, "live": True, "ready": ready, **details}, status=status, headers={"Cache-Control": "no-store"})

    async def _build_index(cuda: str) -> web.Response:
        with tracer.start_as_current_span("Serve Pip Facade Index") as span:
            span.set_attribute("facade.cuda", cuda)
            registry: FacadeRegistryProtocol = app["facade_registry"]
            projects = await registry.list_projects()
            span.set_attribute("facade.project_count", len(projects))
            proxy_names = {
                name
                for name, proxy in PYPI_PROXY_INDEX.items()
                if proxy.supports_cuda(cuda)
            }
            names = sorted({project.canonical_name for project in projects} | proxy_names)
            prefix = f"/simple/{cuda}" if cuda != DEFAULT_CUDA_VARIANT else "/simple"
            links = "\n".join(
                f'<a href="{prefix}/{name}/">{html.escape(name)}</a><br/>'
                for name in names
            )
            return web.Response(text=_simple_html("Simple Index", links), content_type="text/html")

    async def index(_: web.Request) -> web.Response:
        return await _build_index(DEFAULT_CUDA_VARIANT)

    async def _build_project_page(request: web.Request, cuda: str) -> web.Response:
        with tracer.start_as_current_span("Serve Pip Facade Project Index") as span:
            span.set_attribute("facade.cuda", cuda)
            project_name = canonicalize_project_name(request.match_info["project"])
            proxy = PYPI_PROXY_INDEX.get(project_name)
            if proxy is not None:
                if not proxy.supports_cuda(cuda):
                    raise web.HTTPNotFound(text=f"{project_name} does not support CUDA variant: {cuda}")
                span.set_attribute("facade.proxy_project", project_name)
                session: aiohttp.ClientSession = app["facade_session"]
                body = await proxy.render_index(session, cuda)
                return web.Response(text=body, content_type="text/html")

            registry: FacadeRegistryProtocol = app["facade_registry"]
            builder: FacadeWheelBuilder = app["facade_builder"]
            project = await registry.get_project(request.match_info["project"])
            if project is None:
                raise web.HTTPNotFound(text="Unknown project")
            span.set_attribute("facade.project_name", project.canonical_name)

            # Serve the page under the name the client asked for: pip and uv
            # reject an index whose distributions are named differently from
            # the requested project, so alias requests get alias-named wheels.
            span.set_attribute("facade.distribution_name", project_name)
            versions = await registry.list_versions(project)
            span.set_attribute("facade.version_count", len(versions))
            links = "\n".join(
                f'<a href="/packages/{project_name}/{item.version}/'
                f'{builder.wheel_filename(project, item.version, project_name)}">'
                f'{html.escape(builder.wheel_filename(project, item.version, project_name))}</a><br/>'
                for item in versions
            )
            return web.Response(
                text=_simple_html(f"Simple Index for {project_name}", links),
                content_type="text/html",
            )

    async def package_download(request: web.Request) -> web.StreamResponse:
        with tracer.start_as_current_span("Serve Pip Facade Wheel") as span:
            registry: FacadeRegistryProtocol = app["facade_registry"]
            builder: FacadeWheelBuilder = app["facade_builder"]

            requested_name = canonicalize_project_name(request.match_info["project"])
            project = await registry.get_project(request.match_info["project"])
            if project is None:
                raise web.HTTPNotFound(text="Unknown project")
            span.set_attribute("facade.project_name", project.canonical_name)
            span.set_attribute("facade.distribution_name", requested_name)

            version = await registry.get_version(project, request.match_info["version"])
            if version is None:
                raise web.HTTPNotFound(text="Unknown version")
            span.set_attribute("facade.version", version.version)

            expected_name = builder.wheel_filename(project, version.version, requested_name)
            if request.match_info["filename"] != expected_name:
                raise web.HTTPNotFound(text="Filename does not match requested project/version")

            wheel = await builder.build_wheel(project, version, requested_name)
            span.set_attribute("facade.wheel_path", wheel.cache_path)
            return await _serve_cached(request, app["facade_triton_cache"], wheel, expected_name, "application/octet-stream")

    async def triton_package_download(request: web.Request) -> web.StreamResponse:
        with tracer.start_as_current_span("Serve Triton Wheel") as span:
            cuda = request.match_info["cuda"]
            filename = request.match_info["filename"]
            span.set_attribute("facade.cuda", cuda)
            span.set_attribute("facade.filename", filename)
            if not filename.endswith(".whl") or "/" in filename:
                raise web.HTTPNotFound(text="Not a wheel")
            if not (filename.startswith("triton-") or filename.startswith("triton_windows-")):
                raise web.HTTPNotFound(text="Unknown triton wheel")

            cache: FacadeCacheStore = app["facade_triton_cache"]
            builder: TritonWheelBuilder = app["facade_triton_builder"]
            path = cache.custom_path("triton", cuda, filename)
            locks: dict = app["facade_triton_locks"]
            lock = locks.setdefault((cuda, filename), asyncio.Lock())
            async with lock:
                if not await asyncio.to_thread(cache.exists, path):
                    data = await builder.build(served_filename=filename, cuda=cuda)
                    await asyncio.to_thread(cache.write_bytes, path, data)
                cached = cache.cached_wheel(path)
            return await _serve_cached(request, cache, cached, filename, "application/octet-stream")

    async def pypi_rewrite_package_download(request: web.Request) -> web.StreamResponse:
        filename = request.match_info["filename"]
        spec = PYPI_SDIST_REWRITE_FILENAME_INDEX.get(filename)
        if spec is None:
            raise web.HTTPNotFound(text="Unknown rewritten PyPI package")

        cache: FacadeCacheStore = app["facade_triton_cache"]
        path = cache.custom_path("pypi-rewrite", spec.name, filename)
        locks: dict = app["facade_pypi_rewrite_locks"]
        lock = locks.setdefault(filename, asyncio.Lock())
        async with lock:
            if not await asyncio.to_thread(cache.exists, path):
                session: aiohttp.ClientSession = app["facade_session"]
                await asyncio.to_thread(cache.write_bytes, path, await spec.build_sdist(session))
            cached = cache.cached_wheel(path)
        return await _serve_cached(request, cache, cached, filename, "application/gzip")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/livez", liveness)
    app.router.add_get("/readyz", readiness)
    app.router.add_get("/healthz", readiness)
    async def simple_one_segment(request: web.Request) -> web.Response:
        """Handles /simple/{segment}/ — either a CUDA variant index or a project page."""
        segment = request.match_info["segment"]
        if is_index_variant(segment):
            return await _build_index(segment)
        # Treat as project name with default CUDA
        request.match_info["project"] = segment
        return await _build_project_page(request, DEFAULT_CUDA_VARIANT)

    async def simple_two_segments(request: web.Request) -> web.Response:
        """Handles /simple/{first}/{second}/ — CUDA variant + project."""
        first = request.match_info["first"]
        if first == "rocm":
            proxy: RocmSimpleIndexProxy = app["facade_rocm_proxy"]
            body = await proxy.render_project(app["facade_session"], request.match_info["second"])
            return web.Response(text=body, content_type="text/html")
        if not is_index_variant(first):
            raise web.HTTPNotFound(text=f"Unsupported CUDA variant: {first}")
        request.match_info["project"] = request.match_info["second"]
        return await _build_project_page(request, first)

    app.router.add_get("/simple", index)
    app.router.add_get("/simple/", index)
    async def rocm_index(_: web.Request) -> web.Response:
        proxy: RocmSimpleIndexProxy = app["facade_rocm_proxy"]
        body = await proxy.render_root(app["facade_session"])
        return web.Response(text=body, content_type="text/html")

    app.router.add_get("/simple/rocm", rocm_index)
    app.router.add_get("/simple/rocm/", rocm_index)
    app.router.add_get("/simple/{first}/{second}/", simple_two_segments)
    app.router.add_get("/simple/{segment}/", simple_one_segment)
    app.router.add_get("/packages/triton/{cuda}/{filename}", triton_package_download)
    app.router.add_get("/packages/pypi-rewrite/{filename}", pypi_rewrite_package_download)
    app.router.add_get("/packages/{project}/{version}/{filename}", package_download)
    app.router.add_get("/{segment}/", simple_one_segment)
    return app


async def _serve_cached(
    request: web.Request,
    cache: FacadeCacheStore,
    cached: CachedWheel,
    filename: str,
    content_type: str,
) -> web.StreamResponse:
    """Send a cache entry, streaming it in chunks from a remote filesystem."""
    if cached.local_path is not None:
        return web.FileResponse(path=cached.local_path)
    handle = await asyncio.to_thread(cache.open_read, cached.cache_path)
    try:
        response = web.StreamResponse(
            headers={
                "Content-Type": content_type,
                "Content-Disposition": f'attachment; filename="{filename}"',
            }
        )
        size = getattr(handle, "size", None)
        if isinstance(size, int):
            response.content_length = size
        await response.prepare(request)
        while chunk := await asyncio.to_thread(handle.read, _STREAM_CHUNK_BYTES):
            await response.write(chunk)
        await response.write_eof()
        return response
    finally:
        await asyncio.to_thread(handle.close)


async def check_cache_health(app: web.Application) -> None:
    """Check a local wheel cache directory once and record the result for /readyz.

    A check that outlasts the timeout keeps running and is awaited again by the
    next call instead of starting another one on a hung mount.
    """
    cache: FacadeCacheStore = app["facade_triton_cache"]
    state = app["facade_cache_state"]
    check = state["check"]
    if check is None or check.done():
        check = asyncio.ensure_future(asyncio.to_thread(cache.check_local_storage))
        state["check"] = check
    try:
        await asyncio.wait_for(asyncio.shield(check), CACHE_HEALTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Wheel cache check has taken longer than %ss", CACHE_HEALTH_TIMEOUT_SECONDS)
        return
    except OSError as exc:
        state["error"] = str(exc)
        return
    state["error"] = None


async def _cache_health_loop(app: web.Application) -> None:
    while True:
        await check_cache_health(app)
        await asyncio.sleep(CACHE_HEALTH_INTERVAL_SECONDS)


async def run_facade_server(
    *,
    configuration: Configuration,
) -> None:
    with tracer.start_as_current_span("Run Pip Facade Server") as span:
        span.set_attribute("facade.listen", configuration.listen)
        span.set_attribute("facade.port", configuration.port)
        app = create_facade_app(configuration=configuration)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, configuration.listen, configuration.port)
        await site.start()
        logger.info("Serving custom-node pip facade on http://%s:%s/simple/", configuration.listen, configuration.port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()
