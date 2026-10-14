from __future__ import annotations

from asyncio import AbstractEventLoop
from asyncio import Future
from asyncio import Task
from enum import Enum
from io import BytesIO
from posixpath import join as urljoin
from typing import List
from typing import Optional
from typing import Union
from urllib.parse import quote
from urllib.parse import urlencode
import asyncio
import errno
import glob
import io
import ipaddress
import json
import logging
import mimetypes
import os
import socket
import struct
import sys
import traceback
import typing
import uuid

from PIL import Image
from PIL import ImageOps
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from can_ada import URL
from can_ada import parse as urlparse
from packaging import version
from typing_extensions import NamedTuple
import aiofiles
import aiohttp

from .. import __version__
from .. import interruption
from .. import model_management
from .. import node_helpers
from .. import utils
from ..api_server.routes.internal.internal_routes import InternalRoutes
from ..app.assets.event_log import emit
from ..app.asset_export import AssetExportManager
from ..app.assets.services.asset_management import get_export_file
from ..app.assets.services.asset_management import list_job_export_files
from ..app.assets.services.asset_management import resolve_hash_to_path
from ..app.custom_node_manager import CustomNodeManager
from ..app.frontend_management import FrontendManager
from ..app.lsp import LspManager
from ..app.model_manager import ModelFileManager
from ..app.subgraph_manager import SubgraphManager
from ..app.user_manager import UserManager
from ..comfy_api_env import get_environment_overrides
from ..deploy_environment import get_deploy_environment
from .latent_preview_image_encoding import encode_preview_image
from comfy_api import feature_flags
from comfy_execution.jobs import CANCEL_PENDING
from comfy_execution.jobs import CANCEL_RUNNING
from comfy_execution.jobs import JobStatus
from comfy_execution.jobs import cancel_job
from comfy_execution.jobs import get_all_jobs
from comfy_execution.jobs import get_job
from comfy_execution.jobs import get_job_create_times
from comfy_execution.jobs import validate_job_id

from . import execution
from . import folder_paths
from ..app.node_replace_manager import NodeReplaceManager
from ..app.assets.manager import AssetManager, default_asset_manager
from ..client.client_types import FileOutput
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.abstract_prompt_queue import AsyncAbstractPromptQueue
from ..component_model.encode_text_for_progress import encode_text_for_progress
from ..component_model.executor_types import ExecInfo
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.executor_types import PreviewImageWithMetadataMessage
from ..component_model.executor_types import QueueInfo
from ..component_model.executor_types import StatusMessage
from ..component_model.executor_types import UnencodedPreviewImageMessage
from ..component_model.file_output_path import file_output_path
from ..component_model.queue_types import BinaryEventTypes
from ..component_model.queue_types import ExecutionError
from ..component_model.queue_types import ExecutionStatus
from ..component_model.queue_types import ExtraData
from ..component_model.queue_types import HistoryEntry
from ..component_model.queue_types import QueueItem
from ..component_model.queue_types import QueueTuple
from ..component_model.queue_types import TaskInvocation
from ..component_model.workflow_convert import convert_ui_to_api
from ..component_model.workflow_convert import is_ui_workflow
from ..digest import digest
from ..execution_context import current_execution_context
from ..images import open_image
from ..middleware.cache_middleware import cache_control
from ..model_management import torch_version
from ..nodes.package_typing import ExportedNodes
from ..progress_types import PreviewImageMetadata
from ..app.assets.previews import generate_upload_preview
from ..app.assets.services.asset_management import get_preview_file_paths
from ..app.assets.services.preview_rules import preview_fields

logger = logging.getLogger(__name__)


class HeuristicPath(NamedTuple):
    filename_heuristic: str
    abs_path: str


# todo: what is this really trying to do?
LOADED_MODULE_DIRS = {}


def valid_workflow_metadata(json_data: dict) -> Optional[dict]:
    """The metadata a client may attach to a prompt's websocket messages.

    Returns None when absent, not a dict, or over 256 bytes serialised. The
    value is merged into outgoing messages, so it is validated here rather
    than at the point of use.
    """
    metadata = json_data.get("workflow_metadata")
    if isinstance(metadata, dict) and len(json.dumps(metadata)) <= 256:
        return metadata
    return None


def workflow_metadata_from_prompt(extra_data: dict) -> Optional[dict]:
    """Fallback for a client that sends a workflow but no explicit metadata.

    Reads the same id /api/jobs reports, so the websocket messages of a prompt
    carry it whether or not the client knows about workflow_metadata.
    """
    extra_pnginfo = extra_data.get("extra_pnginfo")
    if not isinstance(extra_pnginfo, dict):
        return None
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return None
    workflow_id = workflow.get("id")
    if isinstance(workflow_id, str) and workflow_id:
        return valid_workflow_metadata({"workflow_metadata": {"workflow_id": workflow_id}})
    return None


def _remove_sensitive_from_queue(queue: list) -> list:
    """Remove sensitive data (index 5) from queue item tuples."""
    items = []
    for item in queue:
        if isinstance(item, tuple) or isinstance(item, list) and len(item) >= 5:
            items.append(item[:5])
        else:
            items.append({
                **item,
                "sensitive": None,
            })
    return items


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError, BrokenPipeError, ConnectionError) as err:
        logger.warning("send error: {}".format(err))


def get_comfyui_version():
    return __version__


# Track deprecated paths that have been warned about to only warn once per file
_deprecated_paths_warned = set()


@web.middleware
async def deprecation_warning(request: web.Request, handler):
    """Middleware to warn about deprecated frontend API paths"""
    path = request.path

    if path.startswith("/scripts/ui") or path.startswith("/extensions/core/"):
        # Only warn once per unique file path
        if path not in _deprecated_paths_warned:
            _deprecated_paths_warned.add(path)
            logger.warning(
                f"[DEPRECATION WARNING] Detected import of deprecated legacy API: {path}. "
                f"This is likely caused by a custom node extension using outdated APIs. "
                f"Please update your extensions or contact the extension author for an updated version."
            )

    response: web.Response = await handler(request)
    return response


@web.middleware
async def compress_body(request: web.Request, handler):
    accept_encoding = request.headers.get("Accept-Encoding", "")
    response: web.Response = await handler(request)
    if not isinstance(response, web.Response):
        return response
    if response.content_type not in ["application/json", "text/plain"]:
        return response
    if response.body and "gzip" in accept_encoding:
        response.enable_compression()
    return response


@web.middleware
async def opentelemetry_middleware(request: web.Request, handler):
    """Middleware to extract and propagate OpenTelemetry context from request headers"""
    from opentelemetry import propagate
    from opentelemetry import context

    # Extract OpenTelemetry context from headers
    carrier = dict(request.headers)
    ctx = propagate.extract(carrier)

    # Attach context and execute handler
    token = context.attach(ctx)
    try:
        response = await handler(request)
        return response
    finally:
        context.detach(token)


def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS, PATCH'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, traceparent, tracestate'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware


def is_loopback(host):
    if host is None:
        return False
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
        else:
            return False
    except:
        pass

    loopback = False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in r:
                if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                    return loopback
                else:
                    loopback = True
        except socket.gaierror:
            pass

    return loopback


def create_origin_only_middleware():
    @web.middleware
    async def origin_only_middleware(request: web.Request, handler):
        if 'Sec-Fetch-Site' in request.headers:
            sec_fetch_site = request.headers['Sec-Fetch-Site']
            if sec_fetch_site == 'cross-site':
                return web.Response(status=403)
        # this code is used to prevent the case where a random website can queue comfy workflows by making a POST to 127.0.0.1 which browsers don't prevent for some dumb reason.
        # in that case the Host and Origin hostnames won't match
        # I know the proper fix would be to add a cookie but this should take care of the problem in the meantime
        if 'Host' in request.headers and 'Origin' in request.headers:
            origin_parsed = urlparse(request.headers['Origin'])
            host_parsed = urlparse("http://" + request.headers['Host'].lower())

            loopback = is_loopback(host_parsed.hostname)

            if origin_parsed.port and host_parsed.port:
                origin_domain = f"{origin_parsed.hostname}:{origin_parsed.port}"
                host_domain = f"{host_parsed.hostname}:{host_parsed.port}"
            else:
                origin_domain = origin_parsed.hostname
                host_domain = host_parsed.hostname

            if loopback and host_domain and origin_domain:
                if host_domain != origin_domain:
                    logger.warning("WARNING: request with non matching host and origin {} != {}, returning 403".format(host_domain, origin_domain))
                    return web.Response(status=403)

        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        return response

    return origin_only_middleware


def create_block_external_middleware():
    @web.middleware
    async def block_external_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self'; connect-src 'self' data: blob:; frame-src 'self'; object-src 'self';"
        return response

    return block_external_middleware


class PromptServer(ExecutorToClientProgress):
    instance: Optional['PromptServer'] = None

    def _get_server_features(self):
        features = feature_flags.get_server_features()
        features["assets"] = self.asset_manager.enabled
        return features

    def __init__(self, loop, asset_manager: "AssetManager | None" = None):
        server_args = current_execution_context().configuration
        # todo: this really needs to be set up differently, because sometimes the prompt server will not be initialized
        PromptServer.instance = self

        self.asset_manager = asset_manager if asset_manager is not None else default_asset_manager()
        self.user_manager = UserManager()
        self.model_file_manager = ModelFileManager()
        self.custom_node_manager = CustomNodeManager()
        self.subgraph_manager = SubgraphManager()
        self.node_replace_manager = NodeReplaceManager()
        self.internal_routes = InternalRoutes(self)
        # todo: this is probably read by custom nodes elsewhere
        self.supports: List[str] = ["custom_nodes_from_web"]
        self.prompt_queue: AbstractPromptQueue | AsyncAbstractPromptQueue | None = execution.PromptQueue(self)
        self.loop: AbstractEventLoop = loop
        self.messages: asyncio.Queue = asyncio.Queue()
        self.client_session: Optional[aiohttp.ClientSession] = None
        self.number: int = 0
        self.port: int = 8188
        self._external_address: Optional[str] = None
        self.background_tasks: dict[str, Task] = dict()

        middlewares = [opentelemetry_middleware, cache_control, deprecation_warning]
        if server_args.enable_compress_response_body:
            middlewares.append(compress_body)

        if server_args.enable_cors_header:
            middlewares.append(create_cors_middleware(server_args.enable_cors_header))
        else:
            middlewares.append(create_origin_only_middleware())

        if server_args.offline:
            middlewares.append(create_block_external_middleware())

        # Add manager middleware if available
        from ..manager_integration import get_middleware
        manager_middleware = get_middleware()
        if manager_middleware is not None:
            middlewares.append(manager_middleware)

        max_upload_size = round(server_args.max_upload_size * 1024 * 1024)
        self.app: web.Application = web.Application(client_max_size=max_upload_size,
                                                    handler_args={'max_field_size': 16380},
                                                    middlewares=middlewares)
        self.sockets = dict()
        self._sockets_metadata = dict()
        self.web_root = (
            FrontendManager.init_frontend(server_args.front_end_version)
            if server_args.front_end_root is None
            else server_args.front_end_root
        )
        logger.info(f"[Prompt Server] web root: {self.web_root}")
        self.asset_manager.register_routes(self.app, self.user_manager)
        self.asset_manager.set_event_sink(self.send_sync)

        def _job_create_times(prompt_ids):
            """Creation times of the given jobs, from the queue and history."""
            running, queued = self.prompt_queue.get_current_queue_volatile()
            history = {}
            for prompt_id in prompt_ids:
                history.update(
                    self.prompt_queue.get_history(prompt_id=prompt_id, map_function=lambda item: item)
                )
            return get_job_create_times(prompt_ids, running, queued, history)

        self.asset_export_manager = AssetExportManager(
            list_job_files=list_job_export_files,
            get_asset_file=get_export_file,
            get_job_create_times=_job_create_times,
            event_sink=self.send_sync,
            get_user_id=self.user_manager.get_request_user_id,
        )
        self.asset_export_manager.register_routes(self.app)
        if self.asset_manager.enabled:
            emit("assets.enabled", hashing_enabled=server_args.enable_asset_hashing)
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.last_prompt_id = None
        self.client_id = None
        self.workflow_metadata = {}

        self.on_prompt_handlers = []
        self.nodes: ExportedNodes = ExportedNodes()
        self.restart_requested: bool = False

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            # Store WebSocket for backward compatibility
            self.sockets[sid] = ws
            # Store metadata separately
            self.sockets_metadata[sid] = {"feature_flags": {}}

            try:
                # Send initial state to the new client
                await self.send("status", {"status": self.get_queue_info(), "sid": sid}, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", {"node": self.last_node_id}, sid)  # Flag to track if we've received the first message
                first_message = True
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logger.warning('ws connection closed with exception %s' % ws.exception())
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            # Check if first message is feature flags
                            if first_message and data.get("type") == "feature_flags":
                                # Store client feature flags
                                client_flags = data.get("data", {})
                                self.sockets_metadata[sid]["feature_flags"] = client_flags

                                # Send server feature flags in response
                                await self.send(
                                    "feature_flags",
                                    self._get_server_features(),
                                    sid,
                                )

                                logger.debug(
                                    f"Feature flags negotiated for client {sid}: {client_flags}"
                                )
                            first_message = False
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Invalid JSON received from client {sid}: {msg.data}"
                            )
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}")
            finally:
                self.sockets.pop(sid, None)
                self.sockets_metadata.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            response = web.FileResponse(os.path.join(self.web_root, "index.html"))
            response.headers['Cache-Control'] = 'no-store, must-revalidate'
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @routes.get("/embeddings")
        def get_embeddings(request):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/models")
        def list_model_types(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())

            return web.json_response(model_types)

        @routes.get("/models/{folder}")
        async def get_models(request):
            folder = request.match_info.get("folder", None)
            if folder not in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = folder_paths.get_filename_list(folder)
            return web.json_response(files)

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)
            extensions = list(map(lambda f: "/" + os.path.relpath(str(f), self.web_root).replace("\\", "/"), files))

            for name, extension_dir in self.nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(extension_dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + quote(name) + "/" + os.path.relpath(str(f), extension_dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type=None):
            type_dir = ""
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def compare_image_hash(filepath, image):
            hasher = node_helpers.hasher()

            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hasher()
                b = hasher()
                with open(filepath, "rb") as f:
                    a.update(f.read())
                    b.update(image.file.read())
                    image.file.seek(0)
                return a.hexdigest() == b.hexdigest()
            return False

        async def image_upload(post, image_save_function=None):
            resp, _, _ = await store_image_upload(post, image_save_function)
            return resp if isinstance(resp, web.Response) else web.json_response(resp)

        async def store_image_upload(post, image_save_function=None):
            """(response dict or error Response, registered asset view or None, file path)."""
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400), None, None

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400), None, None

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        if compare_image_hash(filepath, image):  # compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True
                            break
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:
                    if image_save_function is not None:
                        await image_save_function(image, post, filepath)
                    else:
                        async with aiofiles.open(filepath, mode="wb") as file:
                            await file.write(image.file.read())

                resp = {"name" : filename, "subfolder": subfolder, "type": image_upload_type}

                view = self.asset_manager.register_upload(
                    abs_path=filepath,
                    name=filename,
                    upload_type=image_upload_type,
                    subfolder=subfolder,
                    content_written=not image_is_duplicate,
                )
                if view is not None:
                    resp["asset"] = {
                        "id": view.asset.id,
                        "name": view.asset.name,
                        "asset_hash": view.asset_hash,
                        "size": view.size,
                        "mime_type": view.mime_type,
                        "tags": view.tags,
                    }

                return resp, view, filepath
            else:
                return web.Response(status=400), None, None

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            resp, view, filepath = await store_image_upload(post)
            if view is not None:
                preview_id = await generate_upload_preview(view.asset.id, filepath, view.asset.preview_id)
                live_previews = get_preview_file_paths([preview_id]) if preview_id else {}
                preview_id, preview_url = preview_fields(
                    view.asset.id, preview_id, view.mime_type, filepath, False, live_previews
                )
                if preview_url is not None:
                    resp["asset"]["preview_url"] = preview_url
                if preview_id is not None:
                    resp["asset"]["preview_id"] = preview_id
            return resp if isinstance(resp, web.Response) else web.json_response(resp)


        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            async def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil, 'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)

                        output_buffer = io.BytesIO()
                        original_pil.save(output_buffer, format='PNG', compress_level=4, pnginfo=metadata)
                        output_data = output_buffer.getvalue()
                    async with aiofiles.open(filepath, mode='wb') as file_dest:
                        await file_dest.write(output_data)

            return await image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            resolved_content_type = None
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                # todo: do we ever need annotated filenames support on this?

                if not filename:
                    return web.Response(status=400)

                # The frontend's LoadImage combo widget uses asset_hash values
                # (e.g. "blake3:...") as widget values. When litegraph renders the
                # node preview, it constructs /view?filename=<asset_hash>, so this
                # endpoint must resolve blake3 hashes to their on-disk file paths.
                if filename.startswith("blake3:"):
                    if not self.asset_manager.enabled:
                        return web.Response(status=404)
                    # Side-effect call: get_request_user_id raises KeyError for an unknown or
                    # system user in multi-user mode, which is what gates hash resolution.
                    # The returned id is deliberately unused (resolution is not owner-scoped).
                    self.user_manager.get_request_user_id(request)
                    result = resolve_hash_to_path(filename)
                    if result is None:
                        return web.Response(status=404)
                    file, filename, resolved_content_type = result.abs_path, result.download_name, result.content_type
                else:
                    type = request.rel_url.query.get("type", "output")
                    subfolder = request.rel_url.query["subfolder"] if "subfolder" in request.rel_url.query else None

                    try:
                        file = file_output_path(filename, type=type, subfolder=subfolder)
                    except FileNotFoundError:
                        return web.Response(status=404)
                    except PermissionError:
                        return web.Response(status=403)
                    except ValueError:
                        return web.Response(status=400)

                if os.path.isfile(file):
                    # todo: any image file we upload that browsers don't support, we should encode a preview
                    # todo: image handling has to be a little bit more standardized, sometimes we want a Pillow Image, sometimes
                    # we want something that will render to the user, sometimes we want tensors
                    if 'preview' in request.rel_url.query or file.endswith(".exr"):
                        with open_image(file) as img:
                            preview_info = request.rel_url.query.get("preview", "jpeg;90").split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        # Use the content type from asset resolution if available,
                        # otherwise guess from the filename.
                        content_type = (
                            resolved_content_type
                            or mimetypes.guess_type(filename)[0]
                            or 'application/octet-stream'
                        )

                        safe_filename = filename.replace("\\", "\\\\").replace('"', '\\"')
                        disposition = f'filename="{safe_filename}"'
                        headers = {"X-Content-Type-Options": "nosniff"}
                        sec_fetch_dest = request.headers.get("Sec-Fetch-Dest")
                        if folder_paths.is_dangerous_content_type(content_type):
                            headers["Vary"] = "Sec-Fetch-Dest"
                            headers["Cache-Control"] = "no-store"
                            if not folder_paths.renders_safely_as_image(content_type, sec_fetch_dest):
                                content_type = 'application/octet-stream'
                                disposition = f'attachment; filename="{safe_filename}"'

                        headers["Content-Disposition"] = disposition
                        headers["Content-Type"] = content_type
                        return web.FileResponse(file, headers=headers)
            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if "filename" not in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = utils.safetensors_header(safetensors_path, max_size=1024 * 1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if "__metadata__" not in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def system_stats(request):
            primary_device = model_management.get_torch_device()
            cpu_device = model_management.torch.device("cpu")
            ram_total = model_management.get_total_memory(cpu_device)
            ram_free = model_management.get_free_memory(cpu_device)
            required_frontend_version = FrontendManager.get_required_frontend_version()
            installed_templates_version = FrontendManager.get_installed_templates_version()
            required_templates_version = FrontendManager.get_required_templates_version()
            comfy_package_versions = getattr(FrontendManager, "get_comfy_package_versions", lambda: [])()

            # Report every torch device visible to multigpu, with the primary
            # device first so existing clients that read devices[0] keep working.
            torch_devices = model_management.get_all_torch_devices()
            if primary_device in torch_devices:
                torch_devices = [primary_device] + [d for d in torch_devices if d != primary_device]
            else:
                torch_devices = [primary_device] + list(torch_devices)

            device_entries = []
            for d in torch_devices:
                vram_total, torch_vram_total = model_management.get_total_memory(d, torch_total_too=True)
                vram_free, torch_vram_free = model_management.get_free_memory(d, torch_free_too=True)
                device_entries.append({
                    "name": model_management.get_torch_device_name(d),
                    "type": d.type,
                    "index": d.index,
                    "vram_total": vram_total,
                    "vram_free": vram_free,
                    "torch_vram_total": torch_vram_total,
                    "torch_vram_free": torch_vram_free,
                })

            system_stats = {
                "system": {
                    "os": sys.platform,
                    "ram_total": ram_total,
                    "ram_free": ram_free,
                    "comfyui_version": __version__,
                    "required_frontend_version": required_frontend_version,
                    "installed_templates_version": installed_templates_version,
                    "required_templates_version": required_templates_version,
                    "comfy_package_versions": comfy_package_versions,
                    "python_version": sys.version,
                    "pytorch_version": torch_version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded",
                    "deploy_environment": get_deploy_environment(),
                    "argv": sys.argv
                },
                "devices": device_entries
            }
            return web.json_response(system_stats)

        @routes.get("/features")
        async def get_features(request):
            features = self._get_server_features()
            overrides = get_environment_overrides()
            if overrides:
                features.update(overrides)
            return web.json_response(features)

        @routes.get("/default_workflow")
        async def get_default_workflow(request):
            return web.FileResponse(os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_workflow.json"), headers={"Cache-Control": "no-cache"})

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class_name):
            from .node_info import node_info as _node_info
            return _node_info(node_class_name, self.nodes.NODE_CLASS_MAPPINGS, self.nodes.NODE_DISPLAY_NAME_MAPPINGS)

        @routes.get("/object_info")
        async def get_object_info(request):
            # todo: what does this doozy do...
            if not server_args.disable_assets_autoscan:
                self.asset_manager.ensure_scan_started()
            out = {}
            for x in self.nodes.NODE_CLASS_MAPPINGS:
                try:
                    out[x] = node_info(x)
                except Exception:
                    logger.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                    logger.error(traceback.format_exc())
            return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in self.nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/api/jobs")
        async def get_jobs(request):
            """List all jobs with filtering, sorting, and pagination.

            Query parameters:
                status: Filter by status (comma-separated): pending, in_progress, completed, failed
                workflow_id: Filter by workflow ID
                sort_by: Sort field: created_at (default), execution_duration
                sort_order: Sort direction: asc, desc (default)
                limit: Max items to return (positive integer)
                offset: Items to skip (non-negative integer, default 0)
            """
            query = request.rel_url.query

            status_param = query.get('status')
            workflow_id = query.get('workflow_id')
            sort_by = query.get('sort_by', 'created_at').lower()
            sort_order = query.get('sort_order', 'desc').lower()

            status_filter = None
            if status_param:
                status_filter = [s.strip().lower() for s in status_param.split(',') if s.strip()]
                invalid_statuses = [s for s in status_filter if s not in JobStatus.ALL]
                if invalid_statuses:
                    return web.json_response(
                        {"error": f"Invalid status value(s): {', '.join(invalid_statuses)}. Valid values: {', '.join(JobStatus.ALL)}"},
                        status=400
                    )

            if sort_by not in {'created_at', 'execution_duration'}:
                return web.json_response(
                    {"error": "sort_by must be 'created_at' or 'execution_duration'"},
                    status=400
                )

            if sort_order not in {'asc', 'desc'}:
                return web.json_response(
                    {"error": "sort_order must be 'asc' or 'desc'"},
                    status=400
                )

            limit = None

            # If limit is provided, validate that it is a positive integer, else continue without a limit
            if 'limit' in query:
                try:
                    limit = int(query.get('limit'))
                    if limit <= 0:
                        return web.json_response(
                            {"error": "limit must be a positive integer"},
                            status=400
                        )
                except (ValueError, TypeError):
                    return web.json_response(
                        {"error": "limit must be an integer"},
                        status=400
                    )

            offset = 0
            if 'offset' in query:
                try:
                    offset = int(query.get('offset'))
                    if offset < 0:
                        offset = 0
                except (ValueError, TypeError):
                    return web.json_response(
                        {"error": "offset must be an integer"},
                        status=400
                    )

            running, queued = self.prompt_queue.get_current_queue_volatile()
            history = self.prompt_queue.get_history()

            running = _remove_sensitive_from_queue(running)
            queued = _remove_sensitive_from_queue(queued)

            jobs, total = get_all_jobs(
                running, queued, history,
                status_filter=status_filter,
                workflow_id=workflow_id,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset
            )

            has_more = (offset + len(jobs)) < total

            return web.json_response({
                'jobs': jobs,
                'pagination': {
                    'offset': offset,
                    'limit': limit,
                    'total': total,
                    'has_more': has_more
                }
            })

        @routes.get("/api/jobs/{job_id}")
        async def get_job_by_id(request):
            """Get a single job by ID."""
            job_id = request.match_info.get("job_id", None)
            if not job_id:
                return web.json_response(
                    {"error": "job_id is required"},
                    status=400
                )

            running, queued = self.prompt_queue.get_current_queue_volatile()
            history = self.prompt_queue.get_history(prompt_id=job_id)

            running = _remove_sensitive_from_queue(running)
            queued = _remove_sensitive_from_queue(queued)

            job = get_job(job_id, running, queued, history)
            if job is None:
                return web.json_response(
                    {"error": "Job not found"},
                    status=404
                )

            return web.json_response(job)

        def _cancel_job_by_id(job_id):
            """Cancel a single job by id using the queue's existing mechanics.

            Running jobs are interrupted (same mechanism as /interrupt); pending
            jobs are dequeued (same mechanism as /queue {"delete": [...]}).
            Already-finished or unknown ids are no-ops. State-agnostic.

            Returns True when a cancel was actually dispatched (running or
            pending job), False when the call was a no-op (terminal/unknown id).
            """
            running, queued = self.prompt_queue.get_current_queue()
            history = self.prompt_queue.get_history()

            def interrupt(prompt_id):
                logging.info(f"Cancelling running prompt {prompt_id}")
                # Atomic: only interrupts if the job is still the one running,
                # so a cancel can't land on a prompt that started in the gap
                # since the snapshot above. Returns whether it actually fired.
                return self.prompt_queue.interrupt_if_running(prompt_id)

            def dequeue(prompt_id):
                logging.info(f"Cancelling pending prompt {prompt_id}")
                return self.prompt_queue.delete_queue_item(lambda a: a[1] == prompt_id)

            classification = cancel_job(job_id, running, queued, history, interrupt, dequeue)
            return classification in (CANCEL_RUNNING, CANCEL_PENDING)

        @routes.post("/api/jobs/{job_id}/cancel")
        async def cancel_job_by_id(request):
            """Cancel a single job by id, regardless of state.

            Idempotent: cancelling a job that has already finished, or an id
            that is not known, returns 200 with {"cancelled": false} rather
            than an error.
            """
            job_id = request.match_info.get("job_id", None)
            if not job_id:
                return web.json_response(
                    {"error": "job_id is required"},
                    status=400
                )

            cancelled = _cancel_job_by_id(job_id)
            return web.json_response({"cancelled": cancelled})

        @routes.post("/api/jobs/cancel")
        async def cancel_jobs_batch(request):
            """Cancel a batch of jobs by id.

            Body: {"job_ids": ["<uuid>", ...]}

            Best-effort and idempotent: every well-formed id is cancelled if it
            is running or pending; ids that are already finished or unknown are
            no-ops, not errors. A batch of all no-ops still returns 200 with
            {"cancelled": false}. This matches the single-cancel endpoint and
            means "cancel all" still cancels the in-progress jobs even if some
            finished between the client's snapshot and the request. Malformed
            ids are still rejected up front with 400 (see below).
            """
            try:
                json_data = await request.json()
            except json.JSONDecodeError:
                return web.json_response(
                    {"error": "Request body must be valid JSON"},
                    status=400
                )

            job_ids = json_data.get("job_ids") if isinstance(json_data, dict) else None
            if not isinstance(job_ids, list):
                return web.json_response(
                    {"error": "job_ids must be a list"},
                    status=400
                )

            # Validate that every element is a well-formed job id before doing
            # anything else.  An unhashable element (e.g. a nested dict or list)
            # would cause a TypeError when used as a history dict key; a
            # non-string or non-UUID value is never a valid id.  Reject early
            # with 400 rather than letting the classify loop raise 500.
            invalid_ids = []
            for jid in job_ids:
                try:
                    validate_job_id(jid)
                except (ValueError, AttributeError):
                    invalid_ids.append(jid if isinstance(jid, str) else repr(jid))
            if invalid_ids:
                return web.json_response(
                    {"error": "job_ids contains invalid id(s)", "invalid_ids": invalid_ids},
                    status=400,
                )

            # Best-effort: cancel each id that is still running/pending; an id
            # that has finished or never existed is a no-op rather than a reason
            # to fail the whole batch.
            cancelled = False
            for jid in job_ids:
                if _cancel_job_by_id(jid):
                    cancelled = True

            return web.json_response({"cancelled": cancelled})

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)

            offset = request.rel_url.query.get("offset", None)
            if offset is not None:
                offset = int(offset)
            else:
                offset = -1

            return web.json_response(self.prompt_queue.get_history(max_items=max_items, offset=offset))

        @routes.get("/history/{prompt_id}")
        async def get_history_prompt_id(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue_volatile()
            queue_info['queue_running'] = _remove_sensitive_from_queue(current_queue[0])
            queue_info['queue_pending'] = _remove_sensitive_from_queue(current_queue[1])
            return web.json_response(queue_info)

        @routes.post("/prompt")
        async def post_prompt(request):
            json_data = await request.json()
            json_data = self.trigger_on_prompt(json_data)

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                client_prompt_id = json_data.get("prompt_id")
                if client_prompt_id is None:
                    # Absent or explicit null: the server mints the id.
                    prompt_id = str(uuid.uuid4())
                else:
                    try:
                        prompt_id = validate_job_id(client_prompt_id)
                    except ValueError:
                        error = {
                            "type": "invalid_prompt_id",
                            "message": "prompt_id must be a valid UUID",
                            "details": "prompt_id must be a UUID string in canonical lowercase hyphenated form; omit it to let the server generate one",
                            "extra_info": {}
                        }
                        return web.json_response({"error": error, "node_errors": {}}, status=400)

                partial_execution_targets = None
                if "partial_execution_targets" in json_data:
                    partial_execution_targets = json_data["partial_execution_targets"]

                # Strip ``__metadata_v1__`` (per-job Configuration overrides,
                # etc.) off the prompt BEFORE validate_prompt sees it —
                # validate_prompt treats every top-level key as a node and
                # would reject the metadata envelope with missing_node_type.
                from ..component_model.prompt_envelope import extract_metadata
                prompt, metadata = extract_metadata(prompt)

                self.node_replace_manager.apply_replacements(prompt)

                valid = await execution.validate_prompt(prompt_id, prompt, partial_execution_targets)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if metadata:
                    # Per-job envelope survives into the queue via extra_data
                    # so the prompt worker can read it on dequeue without
                    # touching the prompt dict itself.
                    extra_data["__metadata_v1__"] = metadata

                extra_data.pop("workflow_metadata", None)
                metadata = valid_workflow_metadata(json_data)
                if metadata is None:
                    metadata = workflow_metadata_from_prompt(extra_data)
                if metadata is not None:
                    extra_data["workflow_metadata"] = metadata

                if "comfy_usage_source" not in extra_data:
                    usage_source = request.headers.get("Comfy-Usage-Source")
                    if usage_source:
                        extra_data["comfy_usage_source"] = usage_source
                if valid[0]:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put(
                        QueueItem(queue_tuple=QueueTuple(number, prompt_id, prompt, extra_data, outputs_to_execute, None),
                                  completed=None))
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    logger.warning("invalid prompt: {}".format(valid[1]))
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                error = {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {}
                }
                return web.json_response({"error": error, "node_errors": {}}, status=400)

        @routes.post("/queue")
        async def post_queue(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            try:
                json_data = await request.json()
            except json.JSONDecodeError:
                json_data = {}

            # Check if a specific prompt_id was provided for targeted interruption
            prompt_id = json_data.get('prompt_id')
            if prompt_id:
                currently_running, _ = self.prompt_queue.get_current_queue()

                # Check if the prompt_id matches any currently running prompt
                should_interrupt = False
                for item in currently_running:
                    if item.prompt_id == prompt_id:
                        logger.debug(f"Interrupting prompt {prompt_id}")
                        should_interrupt = True
                        break

                if should_interrupt:
                    interruption.interrupt_current_processing()
                else:
                    logger.debug(f"Prompt {prompt_id} is not currently running, skipping interrupt")
            else:
                # No prompt_id provided, do a global interrupt
                logger.debug("Global interrupt (no prompt_id specified)")

                interruption.interrupt_current_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)

        @routes.post("/api/v1/reboot")
        async def post_api_v1_reboot(request: web.Request) -> web.Response:
            logger.info("Reboot requested via API")
            self.restart_requested = True

            # Schedule the shutdown *after* the response is sent.
            # Cancelling all running tasks causes asyncio.gather in
            # start_server to raise CancelledError, which triggers the
            # finally block (prompt worker shutdown) and then os.execv.
            async def _shutdown():
                await asyncio.sleep(0.5)
                for task in asyncio.all_tasks(self.loop):
                    if task is not asyncio.current_task():
                        task.cancel()

            self.loop.create_task(_shutdown())

            return web.json_response({"status": "restarting"})

        @routes.get("/api/v1/prompts/{prompt_id}")
        async def get_api_v1_prompts_prompt_id(request: web.Request) -> web.Response | web.FileResponse:
            prompt_id: str = request.match_info.get("prompt_id", "")
            if prompt_id == "":
                return web.json_response(status=404)

            history_items = self.prompt_queue.get_history(prompt_id)
            if len(history_items) == 0 or prompt_id not in history_items:
                # todo: this should really be moved to a stateful queue abstraction
                if prompt_id in self.background_tasks:
                    return web.json_response(status=204)
                else:
                    # todo: this should check a stateful queue abstraction
                    return web.json_response(status=404)
            elif prompt_id in history_items:
                history_entry = history_items[prompt_id]
                # Check if execution resulted in an error
                if "status" in history_entry:
                    status = history_entry["status"]
                    if isinstance(status, dict) and status.get("status_str") == "error":
                        # Return ExecutionStatusAsDict format with status 500, matching POST /api/v1/prompts behavior
                        return web.Response(
                            body=json.dumps(status),
                            status=500,
                            content_type="application/json"
                        )
                return web.json_response(history_entry["outputs"])
            else:
                return web.Response(status=404, reason="prompt not found in expected state")

        @routes.post("/api/v1/prompts")
        async def post_api_prompt(request: web.Request) -> web.Response | web.FileResponse:
            accept = request.headers.get("accept", "application/json")
            if accept == '*/*':
                accept = "application/json"
            content_type = request.headers.get("content-type", "application/json")
            preferences = request.headers.get("prefer", "") + request.query.get("prefer", "") + " " + content_type + " " + accept

            # handle media type parameters like "application/json+respond-async"
            if "+" in content_type:
                content_type = content_type.split("+")[0]
            if "+" in accept:
                accept = accept.split("+")[0]

            wait = "respond-async" not in preferences

            if accept not in ("application/json", "image/png"):
                return web.json_response(status=400, reason=f"invalid accept content type, expected application/json or image/png, got {accept}")

            # check if the queue is too long
            queue_size = self.prompt_queue.size()
            queue_too_busy_size = PromptServer.get_too_busy_queue_size()
            if queue_size > queue_too_busy_size:
                return web.json_response(status=429,
                                         reason=f"the queue has {queue_size} elements and {queue_too_busy_size} is the limit for this worker")
            # read the request
            prompt_dict: dict = {}
            if content_type == 'application/json':
                prompt_dict = await request.json()
            elif content_type == 'multipart/form-data':
                try:
                    reader = await request.multipart()
                    async for part in reader:
                        if part is None:
                            break
                        if part.headers[aiohttp.hdrs.CONTENT_TYPE] == 'application/json':
                            prompt_dict = await part.json()
                            if 'prompt' in prompt_dict:
                                prompt_dict = prompt_dict['prompt']
                        elif part.filename:
                            file_data = await part.read(decode=True)
                            # overwrite existing files
                            upload_dir = PromptServer.get_upload_dir()
                            async with aiofiles.open(os.path.join(upload_dir, part.filename), mode='wb') as file:
                                await file.write(file_data)
                except IOError as ioError:
                    return web.Response(status=507, reason=str(ioError))
                except MemoryError as memoryError:
                    return web.Response(status=507, reason=str(memoryError))
                except Exception as ex:
                    return web.Response(status=400, reason=str(ex))

            if len(prompt_dict) == 0:
                return web.Response(status=400, reason="no prompt was specified")

            if is_ui_workflow(prompt_dict):
                prompt_dict = convert_ui_to_api(prompt_dict)

            # Same envelope handling as /prompt: pull ``__metadata_v1__`` off
            # the dict before validate_prompt, forward it via ExtraData.
            from ..component_model.prompt_envelope import extract_metadata
            prompt_dict, metadata = extract_metadata(prompt_dict)

            content_digest = digest(prompt_dict)
            task_id = str(uuid.uuid4())
            valid = await execution.validate_prompt(task_id, prompt_dict)
            if not valid[0]:
                return web.Response(status=400, content_type="application/json", body=json.dumps(valid[1]))

            # convert a valid prompt to the queue tuple this expects
            number = self.number
            self.number += 1

            extra_data = ExtraData()
            if metadata:
                extra_data["__metadata_v1__"] = metadata

            result: TaskInvocation
            completed: Future[TaskInvocation | dict] = self.loop.create_future()
            # todo: actually implement idempotency keys
            # we would need some kind of more durable, distributed task queue
            # QueueItem deals with sensitive data uniformly now
            item = QueueItem(queue_tuple=QueueTuple(number, task_id, prompt_dict, extra_data, valid[2], None), completed=completed)

            try:
                if hasattr(self.prompt_queue, "put_async") or isinstance(self.prompt_queue, AsyncAbstractPromptQueue):
                    # this enables span propagation seamlessly
                    fut = self.prompt_queue.put_async(item)
                    if wait:
                        result = await fut
                        if result is None:
                            return web.Response(body="the queue is shutting down", status=503)
                    else:
                        return self._schedule_background_task_with_web_response(fut, task_id)
                else:
                    self.prompt_queue.put(item)
                    if wait:
                        await completed
                    else:
                        return self._schedule_background_task_with_web_response(completed, task_id)
                    task_invocation_or_dict: TaskInvocation | dict = completed.result()
                    if isinstance(task_invocation_or_dict, dict):
                        result = TaskInvocation(item_id=item.prompt_id, outputs=task_invocation_or_dict, status=ExecutionStatus("success", True, []))
                    else:
                        result = task_invocation_or_dict
            except ExecutionError as exec_exc:
                result = exec_exc.as_task_invocation()
            except Exception as ex:
                return web.Response(body=str(ex), status=500)

            if result.status is not None and result.status.status_str == "error":
                status_dict = result.status.as_dict(error_details=result.error_details)
                return web.Response(body=json.dumps(status_dict), status=500, content_type="application/json")
            # find images and read them
            output_images: List[FileOutput] = []
            for node_id, node in result.outputs.items():
                images: List[FileOutput] = []
                if 'images' in node:
                    images = node['images']
                    # todo: does this ever occur?
                elif (isinstance(node, dict)
                      and 'ui' in node and isinstance(node['ui'], dict)
                      and 'images' in node['ui']):
                    images = node['ui']['images']
                for image_tuple in images:
                    output_images.append(image_tuple)

            if len(output_images) > 0:
                main_image = output_images[0]
                filename = main_image["filename"]
                digest_headers_ = {
                    "Digest": f"SHA-256={content_digest}",
                    "Location": f"/api/v1/prompts/{task_id}"
                }
                urls_ = []
                if len(output_images) == 1:
                    digest_headers_.update({
                        "Content-Disposition": f"filename=\"{filename}\""
                    })

                for image_indv_ in output_images:
                    local_address = f"http://{self.address}:{self.port}"
                    external_address = self.external_address

                    for base in (local_address, external_address):
                        try:
                            url: URL = urlparse(urljoin(base, "view"))
                        except ValueError:
                            continue
                        url_search_dict: FileOutput = dict(image_indv_)
                        del url_search_dict["abs_path"]
                        if "name" in url_search_dict:
                            del url_search_dict["name"]
                        if url_search_dict["subfolder"] == "":
                            del url_search_dict["subfolder"]
                        url.search = f"?{urlencode(url_search_dict)}"
                        urls_.append(str(url))

                if accept == "application/json":
                    return web.Response(status=200,
                                        content_type="application/json",
                                        headers=digest_headers_,
                                        body=json.dumps({
                                            'urls': urls_,
                                            'outputs': result.outputs,
                                            'prompt_id': task_id,
                                        }))
                # todo: provide more ways to accept these, or support multiple file returns easily
                elif accept == "image/png" or accept == "image/jpeg":
                    return web.FileResponse(main_image["abs_path"],
                                            headers=digest_headers_)
                else:
                    return web.Response(status=500,
                                        reason="unreachable")
            else:
                return web.Response(status=204)

        @routes.get("/api/v1/prompts")
        async def get_api_prompt(_: web.Request) -> web.Response:
            history = self.prompt_queue.get_history()
            history_items = list(history.values())
            if len(history_items) == 0:
                return web.Response(status=404)

            last_history_item: HistoryEntry = history_items[-1]
            prompt = last_history_item['prompt'][2]
            return web.json_response(prompt, status=200)

    def _schedule_background_task_with_web_response(self, fut, task_id):
        if isinstance(fut, asyncio.Future):
            task = fut
        else:
            task = asyncio.create_task(fut, name=task_id)
        self.background_tasks[task_id] = task
        task.add_done_callback(lambda _: self.background_tasks.pop(task_id))
        # todo: type this from the OpenAPI spec
        return web.json_response({
            "prompt_id": task_id
        }, status=202, headers={
            "Location": f"api/v1/prompts/{task_id}",
            "Retry-After": "60"
        })

    @property
    def external_address(self):
        return self._external_address if self._external_address is not None else f"http://{'localhost' if self.address == '0.0.0.0' else self.address}:{self.port}"

    @external_address.setter
    def external_address(self, value):
        self._external_address = value

    @property
    def receive_all_progress_notifications(self) -> bool:
        return True

    async def setup(self):
        timeout = aiohttp.ClientTimeout(total=None)  # no timeout
        self.client_session = aiohttp.ClientSession(timeout=timeout)

    def add_routes(self):
        # a mitigation for vanilla comfyui custom nodes that are stateful and add routes to a global
        # prompt server instance. this is not a recommended pattern, but this mitigation is here to
        # support it
        from ..nodes.vanilla_node_importing import prompt_server_instance_routes
        for route in prompt_server_instance_routes.routes:
            self.routes.route(route.method, route.path)(route.func)
        prompt_server_instance_routes.clear()

        self.user_manager.add_routes(self.routes)
        self.model_file_manager.add_routes(self.routes)
        # todo: needs to use module directories
        self.custom_node_manager.add_routes(self.routes, self.app, {})
        self.subgraph_manager.add_routes(self.routes, LOADED_MODULE_DIRS.items())
        self.node_replace_manager.add_routes(self.routes)
        self.app.add_subapp('/internal', self.internal_routes.get_app())

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        api_routes = web.RouteTableDef()
        for route in self.routes:
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        self.app.add_routes(api_routes)
        self.app.add_routes(self.routes)

        # LSP WebSocket for eval node code intelligence
        self._lsp_manager = LspManager()
        self.app.router.add_get("/ws/lsp", self._lsp_manager.handle_websocket)

        # Add routes from web extensions.
        for name, extension_dir in self.nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([web.static('/extensions/' + name, extension_dir, follow_symlinks=True)])

        installed_templates_version = FrontendManager.get_installed_templates_version()
        use_legacy_templates = True
        if installed_templates_version:
            try:
                use_legacy_templates = (
                        version.parse(installed_templates_version)
                        < version.parse("0.3.0")
                )
            except Exception as exc:
                logger.warning(
                    "Unable to parse templates version '%s': %s",
                    installed_templates_version,
                    exc,
                )

        if use_legacy_templates:
            workflow_templates_path = FrontendManager.legacy_templates_path()
            if workflow_templates_path:
                self.app.add_routes([
                    web.static('/templates', workflow_templates_path)
                ])
        else:
            handler = FrontendManager.template_asset_handler()
            if handler:
                self.app.router.add_get("/templates/{path:.*}", handler)

        # Serve embedded documentation from the package
        embedded_docs_path = FrontendManager.embedded_docs_path()
        if embedded_docs_path:
            self.app.add_routes([
                web.static('/docs', embedded_docs_path)
            ])

        self.app.add_routes([
            web.static('/', self.web_root, follow_symlinks=True),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data: UnencodedPreviewImageMessage | PreviewImageWithMetadataMessage | bytes | bytearray | dict, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            # data is (preview_image, metadata)

            data: PreviewImageWithMetadataMessage
            preview_image, metadata = data
            if isinstance(preview_image, dict):
                # todo: this has to be fixed from transformers loader for previewing tokens in real time
                return
            await self.send_image_with_metadata(preview_image, metadata, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event: int | Enum | str, data: bytes | bytearray | typing.Sequence[int]):
        # todo: investigate what is propagating these spurious, string-repr'd previews
        if event == repr(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE):
            event = BinaryEventTypes.UNENCODED_PREVIEW_IMAGE.value
        elif event == repr(BinaryEventTypes.PREVIEW_IMAGE):
            event = BinaryEventTypes.PREVIEW_IMAGE.value
        elif isinstance(event, Enum):
            event: int = event.value
        elif not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data: UnencodedPreviewImageMessage, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        preview_bytes = encode_preview_image(image, image_type, max_size)
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_image_with_metadata(self, image_data: UnencodedPreviewImageMessage, metadata: Optional[PreviewImageMetadata] = None, sid=None):
        try:
            image_type = image_data[0]
            image = image_data[1]
            max_size = image_data[2]
        except Exception as exc_info:
            logger.warning(f"tried to send_image_with_metadata but an error occurred, aboring send, image_data={image_data} metadata={metadata} sid={sid}", exc_info=exc_info)
            return
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.Resampling.LANCZOS

            image = ImageOps.contain(image, (max_size, max_size), resampling)

        mimetype = "image/png" if image_type == "PNG" else "image/jpeg"

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["image_type"] = mimetype

        # Serialize metadata as JSON
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_json)

        # Prepare image data
        bytesIO = BytesIO()
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        image_bytes = bytesIO.getvalue()

        # Combine metadata and image
        combined_data = bytearray()
        combined_data.extend(struct.pack(">I", metadata_length))
        combined_data.extend(metadata_json)
        combined_data.extend(image_bytes)

        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA, combined_data, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data: dict, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        if self.workflow_metadata:
            if isinstance(data, dict) and "prompt_id" in data:
                data = {**self.workflow_metadata, **data}
            elif (
                event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA
                and isinstance(data, tuple)
                and len(data) == 2
                and isinstance(data[1], dict)
                and "prompt_id" in data[1]
            ):
                # Merged here rather than at publication: messages wait in the
                # queue, so by the time publish_loop sends a preview the next
                # prompt may already have replaced workflow_metadata, and the
                # frame would carry the wrong workflow.
                data = (data[0], {**self.workflow_metadata, **data[1]})

        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self, queue_remaining: Optional[int] = None):
        if queue_remaining is None:
            status = {"status": self.get_queue_info()}
        else:
            status = StatusMessage(status=QueueInfo(exec_info=ExecInfo(queue_remaining=queue_remaining)))
        self.send_sync("status", status)

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address: str | None, port: int | None, verbose=True, call_on_start=None):
        await self.start_multi_address([(address, port)], call_on_start=call_on_start, verbose=verbose)

    async def start_multi_address(self, addresses, call_on_start=None, verbose=True):
        address_print = "localhost"
        address: str = "127.0.0.1"
        port: int = 8188
        runner = web.AppRunner(self.app, access_log=None, keepalive_timeout=900)
        await runner.setup()

        server_args = current_execution_context().configuration
        if server_args.tls_keyfile or server_args.tls_certfile:
            logger.warning("Use caddy instead of aiohttp to serve https by setting up a reverse proxy. See README.md")
        if verbose:
            logging.info("Starting server\n")
            if server_args.debug_hang:
                logging.info(
                    f"{'-' * 80}\n"
                    "ComfyUI has been started in debug-hang mode. Run your workflow as normal up to\n"
                    "the point of the hang or freeze, then use ctrl-C in the cmd or controlling\n"
                    "terminal to dump the python backtraces for debugging. Please attach the extra\n"
                    "debug info to your bug report.\n"
                    f"{'-' * 80}"
                )

        def is_ipv4(address: str, *args):
            try:
                parsed = ipaddress.ip_address(address)
                return isinstance(parsed, ipaddress.IPv4Address)
            except:
                return False

        addresses = sorted(addresses, key=lambda address_tuple: is_ipv4(*address_tuple))
        for (address, port) in addresses:
            site = web.TCPSite(runner, address, port, backlog=PromptServer.get_too_busy_queue_size())
            try:
                await site.start()
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    logger.error(f"Port {port} is already in use on address {address}. Please close the other application or use a different port with --port.")
                    raise SystemExit(1)
                raise

            # preference for the ipv4 address achieved by sorting
            self.address = "localhost" if address == "0.0.0.0" else address
            self.port = port

            if address == '::' or address == "127.0.0.1" or address == "0.0.0.0":
                address_print = "localhost"
            elif ':' in address:
                address_print = "[{}]".format(address)
            else:
                address_print = address

        if verbose:
            logger.info(f"Server ready. To see the GUI go to: http://{address_print}:{port}")
        if call_on_start is not None:
            call_on_start("http", address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception:
                logger.warning("[ERROR] An error occurred during the on_prompt_handler processing")
                logger.warning(traceback.format_exc())

        return json_data

    @classmethod
    def get_upload_dir(cls) -> str:
        return folder_paths.get_input_directory()

    @classmethod
    def get_too_busy_queue_size(cls):
        return current_execution_context().configuration.max_queue_size

    def send_progress_text(
            self, text: Union[bytes, bytearray, str], node_id: str, sid=None
    ):
        message = encode_text_for_progress(node_id, text)

        self.send_sync(BinaryEventTypes.TEXT, message, sid)

    @property
    def sockets_metadata(self):
        return self._sockets_metadata

    @sockets_metadata.setter
    def sockets_metadata(self, value):
        self._sockets_metadata = value
