from __future__ import annotations

from types import SimpleNamespace

import click
import pytest
from typer.testing import CliRunner

import comfy.cmd.cli as cli_module
from comfy.cmd.cli import _RunWorkflowCommand
from comfy.entrypoints.workflow_params import Param


def _ctx() -> click.Context:
    command = click.Command(
        "run-workflow",
        params=[
            click.Option(["--all"], is_flag=True),
            click.Option(["--video"], multiple=True),
            click.Option(["--prompt"]),
            click.Option(["--set"], multiple=True),
        ],
    )
    return click.Context(command)


def _patch_params(monkeypatch: pytest.MonkeyPatch, params: list[Param]) -> None:
    monkeypatch.setattr(cli_module, "_discover_from_ref", lambda ref: params)


def test_rewrites_discovered_workflow_flag_to_set(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="266:300",
                class_type="PrimitiveInt",
                widget_name="value",
                value=81,
                type="INT",
                flag_name="num-frames-in-the-middle",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--video", "input.mp4", "--num-frames-in-the-middle", "17"],
        ctx=_ctx(),
    )

    assert args == [
        "workflow.json",
        "--video",
        "input.mp4",
        "--set",
        "266:300.inputs.value=17",
    ]


def test_rewrites_discovered_workflow_flag_equals_form(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="266:300",
                class_type="PrimitiveInt",
                widget_name="value",
                value=81,
                type="INT",
                flag_name="num-frames-in-the-middle",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--num-frames-in-the-middle=17"],
        ctx=_ctx(),
    )

    assert args == ["workflow.json", "--set", "266:300.inputs.value=17"]


def test_rewrites_discovered_boolean_flags(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="10",
                class_type="PrimitiveBoolean",
                widget_name="value",
                value=False,
                type="BOOLEAN",
                flag_name="enable-loop",
            )
        ],
    )

    assert _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--enable-loop"],
        ctx=_ctx(),
    ) == ["workflow.json", "--set", "10.inputs.value=true"]
    assert _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--no-enable-loop"],
        ctx=_ctx(),
    ) == ["workflow.json", "--set", "10.inputs.value=false"]


def test_rewrites_grouped_turbo_boolean_for_every_matching_branch(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="10",
                class_type="PrimitiveBoolean",
                widget_name="value",
                value=True,
                type="BOOLEAN",
                flag_name="turbo",
                targets=[("20", "value")],
            )
        ],
    )

    assert _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--no-turbo"],
        ctx=_ctx(),
    ) == [
        "workflow.json",
        "--set", "10.inputs.value=false",
        "--set", "20.inputs.value=false",
    ]


def test_builtin_option_collision_gets_x_prefixed_workflow_flag(monkeypatch):
    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="12",
                class_type="PrimitiveString",
                widget_name="value",
                value="old",
                type="STRING",
                flag_name="prompt",
            )
        ],
    )

    args = _RunWorkflowCommand._rewrite_workflow_param_args(
        ["workflow.json", "--prompt", "builtin", "--x-prompt", "workflow"],
        ctx=_ctx(),
    )

    assert args == [
        "workflow.json",
        "--prompt",
        "builtin",
        "--set",
        "12.inputs.value=workflow",
    ]


def test_workflow_flag_for_param_uses_x_prefix_on_collision():
    param = Param(
        node_id="12",
        class_type="PrimitiveString",
        widget_name="value",
        value="old",
        type="STRING",
        flag_name="prompt",
    )

    assert _RunWorkflowCommand._workflow_flag_for_param(param, ctx=_ctx()) == "--x-prompt"


def test_extract_workflow_ref_uses_click_option_arity():
    assert _RunWorkflowCommand._extract_workflow_ref(
        ["--all", "--video", "input.mp4", "workflow.json", "--help"],
        ctx=_ctx(),
    ) == "workflow.json"


def test_option_arity_supports_vendored_click_parameter_classes():
    ctx = SimpleNamespace(
        command=SimpleNamespace(
            params=[
                SimpleNamespace(name="workflows"),
                SimpleNamespace(is_flag=False, nargs=1, opts=["--seed"], secondary_opts=[]),
                SimpleNamespace(is_flag=True, nargs=1, opts=["--compile"], secondary_opts=["--no-compile"]),
            ]
        )
    )

    assert _RunWorkflowCommand._option_value_arity_by_name(ctx) == {
        "--seed": 1,
        "--compile": 0,
        "--no-compile": 0,
    }


def test_workflows_submit_rewrites_discovered_workflow_flag(monkeypatch):
    from comfy.cmd.cli import app, _register_sub_apps
    import comfy.cmd.sub_workflows as sub_workflows

    _register_sub_apps()

    _patch_params(
        monkeypatch,
        [
            Param(
                node_id="327",
                class_type="PrimitiveInt",
                widget_name="value",
                value=0,
                type="INT",
                flag_name="seed",
            )
        ],
    )

    captured = {}

    async def fake_submit_workflows(workflows, server, config, response_mode="sync"):
        captured["workflows"] = workflows
        captured["server"] = server
        captured["set"] = config.set
        captured["response_mode"] = response_mode

    monkeypatch.setattr(sub_workflows, "_submit_workflows", fake_submit_workflows)

    result = CliRunner().invoke(
        app,
        [
            "workflows",
            "submit",
            "workflow.json",
            "--server",
            "http://127.0.0.1:8199",
            "--x-seed",
            "2002",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["workflows"] == ["workflow.json"]
    assert captured["server"] == "http://127.0.0.1:8199"
    assert captured["set"] == ["327.inputs.value=2002"]
    assert captured["response_mode"] == "sync"
