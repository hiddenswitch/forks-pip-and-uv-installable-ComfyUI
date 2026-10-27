import asyncio
import copy
import json
import logging
import os
import warnings
from typing import Optional, Literal

import typer

from .. import interruption
from ..cli_args_types import Configuration
from ..component_model.asyncio_files import stream_json_objects
from ..component_model.uris import is_uri
from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
from ..client.embedded_comfy_client import Comfy

logger = logging.getLogger(__name__)


def _ensure_api_format(obj: dict) -> dict:
    if not is_ui_workflow(obj):
        return obj
    logger.info("Converting UI workflow to API format")
    return convert_ui_to_api(obj)


def _unbypass_extra_loaders(ui: dict, kind: str, target_count: int) -> dict:
    """Un-bypass enough mode==4 loader nodes so total active >= *target_count*.

    Lets ``--image x --image y --image z`` extend a single-image workflow
    with bypassed optional inputs (Kontext / Kling / Luma I2V style) without
    requiring the user to manually flip mode flags.
    """
    from .workflow_params import (
        _IMAGE_INPUT_CLASSES, _VIDEO_INPUT_CLASSES, _AUDIO_INPUT_CLASSES,
    )
    classes = {
        "images": _IMAGE_INPUT_CLASSES,
        "videos": _VIDEO_INPUT_CLASSES,
        "audios": _AUDIO_INPUT_CLASSES,
    }[kind]

    nodes = ui.get("nodes") or []
    active = sum(1 for n in nodes if n.get("type") in classes and n.get("mode", 0) not in (2, 4))
    if active >= target_count:
        return ui
    needed = target_count - active

    import copy as _copy
    ui = _copy.deepcopy(ui)
    for node in ui.get("nodes") or []:
        if needed <= 0:
            break
        if node.get("type") in classes and node.get("mode") == 4:
            node["mode"] = 0
            needed -= 1
    return ui


def _apply_ui_pre_overrides(ui: dict, configuration: Configuration) -> dict:
    """Apply UI-side overrides that need to run before convert_ui_to_api."""
    if configuration.image:
        ui = _unbypass_extra_loaders(ui, "images", len(configuration.image))
    if configuration.video:
        ui = _unbypass_extra_loaders(ui, "videos", len(configuration.video))
    if configuration.audio:
        ui = _unbypass_extra_loaders(ui, "audios", len(configuration.audio))
    return ui


def _apply_sets(obj: dict, sets: list[str]) -> dict:
    import copy as _copy
    if not sets:
        return obj
    obj = _copy.deepcopy(obj)
    for item in sets:
        if "=" not in item:
            raise ValueError(f"Invalid --set format: {item!r} (expected key=value)")
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = obj
        for part in parts[:-1]:
            target = target[part]
        parsed = value
        if parsed.lower() == "true":
            parsed = True
        elif parsed.lower() == "false":
            parsed = False
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    pass
        target[parts[-1]] = parsed
    return obj


def _apply_overrides(obj: dict, configuration: Configuration) -> dict:
    # Simple value-set overrides funnel through workflow_params.apply_role:
    # one discover() pass tags every relevant widget, then each --flag writes
    # to whatever Params carry the matching role. The class-type tables that
    # used to live in prompt_utils.replace_* still drive role tagging via
    # workflow_params.class_type_roles + prompt_polarity, so behavior is
    # preserved.
    #
    # Media (--image/--video/--audio) needs a class_type rewrite for filesystem
    # loaders → URL loaders, and add_loras/enable_compile splice new nodes
    # into the graph; both stay bespoke.
    from ..component_model.prompt_utils import (
        replace_images, replace_videos, replace_audios,
        add_loras, enable_compile,
    )
    from .workflow_params import apply_role, discover

    if configuration.set:
        obj = _apply_sets(obj, configuration.set)

    role_overrides: list[tuple[str, object]] = [
        ("prompt", configuration.prompt),
        ("negative_prompt", configuration.negative_prompt),
        ("steps", configuration.steps),
        ("seed", configuration.seed),
        ("cfg", configuration.cfg),
        ("sampler", configuration.sampler),
        ("scheduler", configuration.scheduler),
        ("denoise", configuration.denoise),
        ("width", configuration.width),
        ("height", configuration.height),
        ("batch_size", configuration.batch_size),
        ("checkpoint", configuration.checkpoint),
        ("unet", configuration.diffusion_model),
    ]
    if any(value is not None for _, value in role_overrides):
        params = discover(obj)
        for role, value in role_overrides:
            if value is None:
                continue
            obj = apply_role(obj, role, value, params=params)

    if configuration.image is not None:
        obj = replace_images(obj, configuration.image)
    if configuration.video is not None:
        obj = replace_videos(obj, configuration.video)
    if configuration.audio is not None:
        obj = replace_audios(obj, configuration.audio)
    # LoRAs must splice in BEFORE --compile so the compiled graph captures
    # the LoRA patches. add_loras inserts right after the root loader
    # (earliest predecessor of the sampler); enable_compile wraps the
    # chain tail (latest predecessor of the sampler).
    if configuration.add_lora:
        obj = add_loras(obj, configuration.add_lora)
    if configuration.compile:
        obj = enable_compile(obj)
    return obj


_MAX_SEED = 0xffffffffffffffff


def _random_seed() -> int:
    import random
    return random.SystemRandom().randint(0, _MAX_SEED)


def _config_without_seed(configuration: Configuration) -> Configuration:
    cloned = Configuration(**dict(configuration))
    cloned.seed = None
    return cloned


def _quantity(configuration: Configuration) -> int:
    quantity = int(getattr(configuration, "quantity", 1) or 1)
    if quantity < 1:
        raise ValueError("--quantity must be at least 1")
    return quantity


def _apply_api_quantity(api: dict, index: int, bases: dict[tuple[str, str], int]) -> dict:
    if not bases:
        return api
    api = copy.deepcopy(api)
    for (node_id, field), base_seed in bases.items():
        try:
            api[node_id]["inputs"][field] = (int(base_seed) + index) % (_MAX_SEED + 1)
        except KeyError:
            continue
    return api


def _apply_explicit_seed(api: dict, seed_pairs: list[tuple[str, str]], seed: int, index: int) -> dict:
    """Make an explicit --seed reach every seed input of a converted UI workflow.

    The UI pass applies the seed to seed widgets it can see, honouring fixed / increment /
    decrement. Seeds it cannot reach (promoted out of a subgraph as a subgraph input or a
    proxy widget) and randomize controls leave other values; those take seed + index, as an
    API workflow's seeds do.
    """
    produced = {seed % (_MAX_SEED + 1), (seed + index) % (_MAX_SEED + 1), (seed - index) % (_MAX_SEED + 1)}
    for node_id, field in seed_pairs:
        value = api[node_id]["inputs"][field]
        if isinstance(value, list):  # linked to another node's output
            continue
        try:
            if int(value) in produced:
                continue
        except (TypeError, ValueError):
            pass
        api[node_id]["inputs"][field] = (seed + index) % (_MAX_SEED + 1)
    return api


def expand_workflow_quantity(obj: dict, configuration: Configuration) -> list[dict]:
    """Convert/apply overrides and expand a workflow object for ``--quantity``."""
    from ..component_model.prompt_utils import find_seed_nodes
    from ..component_model.workflow_convert import apply_ui_seed_quantity

    count = _quantity(configuration)
    if is_ui_workflow(obj):
        obj = _apply_ui_pre_overrides(obj, configuration)
        expanded: list[dict] = []
        no_seed_config = _config_without_seed(configuration)
        for index in range(count):
            ui = apply_ui_seed_quantity(
                obj,
                index,
                seed=configuration.seed,
                random_seed=_random_seed,
            )
            api = _apply_overrides(_ensure_api_format(ui), no_seed_config)
            if configuration.seed is not None:
                api = _apply_explicit_seed(api, find_seed_nodes(api), int(configuration.seed), index)
            expanded.append(api)
        return expanded

    api = _apply_overrides(obj, configuration)
    seed_pairs = find_seed_nodes(api)
    seed_bases: dict[tuple[str, str], int] = {}
    if seed_pairs and (count > 1 or configuration.seed is not None):
        random_base = _random_seed() if configuration.seed is None and count > 1 else None
        for node_id, field in seed_pairs:
            seed_bases[(node_id, field)] = (
                random_base if random_base is not None else int(api[node_id]["inputs"][field])
            )
    if not seed_bases:
        return [copy.deepcopy(api) for _ in range(count)]
    return [_apply_api_quantity(api, index, seed_bases) for index in range(count)]


def _resolve_workflow(workflow: str) -> str:
    if workflow == "-" or workflow.lstrip().startswith("{"):
        return workflow
    # Canonicalize known web URLs to their fsspec scheme (civitai://, hf://, ...).
    # The fsspec backends register via entry points (project.entry-points."fsspec.specs"
    # in pyproject.toml), so no explicit import is needed.
    from ..component_model.uri_rewrite import canonicalize_uri
    workflow = canonicalize_uri(workflow)
    if is_uri(workflow):
        return workflow
    if os.sep in workflow or os.path.exists(workflow) or workflow.lower().endswith((".json", ".png")):
        return workflow
    from ..cmd.workflow_templates import resolve_template
    return resolve_template(workflow)


async def run_workflows(workflows: list[str | Literal["-"]], configuration: Optional[Configuration] = None):
    if configuration is None:
        from ..cli_args import args
        configuration = args
    resolved = [_resolve_workflow(w) for w in workflows]
    show_progress = not getattr(configuration, "disable_progress", False) and os.isatty(2)
    try:
        async with Comfy(configuration=configuration) as comfy:
            for workflow in resolved:
                obj: dict
                async for obj in stream_json_objects(workflow):
                    if is_ui_workflow(obj):
                        logger.info("Converting UI workflow to API format")
                    for prompt in expand_workflow_quantity(obj, configuration):
                        try:
                            if show_progress:
                                res = await _run_with_progress(comfy, prompt)
                            else:
                                res = await comfy.queue_prompt_api(prompt)
                            typer.echo(json.dumps(res.outputs))
                        except asyncio.CancelledError:
                            interruption.interrupt_current_processing()
                            logger.info("Exiting gracefully.")
                            raise
    except asyncio.CancelledError:
        interruption.interrupt_current_processing()
        raise


async def _run_with_progress(comfy: Comfy, prompt: dict):
    """Execute a prompt with a rich progress bar on stderr."""
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    task = comfy.queue_with_progress(prompt)
    node_tasks: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=__import__("rich.console", fromlist=["Console"]).Console(stderr=True),
        transient=True,
    ) as progress:
        overall = progress.add_task("Running workflow", total=None)

        try:
            async for notification in task.progress():
                if notification.event == "progress":
                    data = notification.data
                    value = data.get("value", 0)
                    total = data.get("max", 100)
                    node_id = data.get("node")
                    if node_id and node_id not in node_tasks:
                        node_tasks[node_id] = progress.add_task(f"Node {node_id}", total=total)
                    if node_id and node_id in node_tasks:
                        progress.update(node_tasks[node_id], completed=value, total=total)
                elif notification.event == "execution_cached":
                    cached = notification.data.get("nodes", [])
                    for nid in cached:
                        if nid not in node_tasks:
                            node_tasks[nid] = progress.add_task(f"Node {nid} (cached)", total=1)
                        progress.update(node_tasks[nid], completed=1, total=1)
                elif notification.event == "executing":
                    node_id = notification.data.get("node")
                    if node_id:
                        progress.update(overall, description=f"Executing node {node_id}")
        except asyncio.CancelledError:
            interruption.interrupt_current_processing()
            raise

        progress.update(overall, description="Done", completed=1, total=1)

    return await task.get()


def entrypoint():
    warnings.warn(
        "comfyui-workflow is deprecated. Use: comfyui run-workflow",
        DeprecationWarning,
        stacklevel=1,
    )
    import sys
    from ..cmd.cli import app
    sys.argv = [sys.argv[0], "run-workflow"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    entrypoint()
