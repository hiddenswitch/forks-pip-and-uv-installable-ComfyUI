"""Tests for the Typer CLI app (comfy.cmd.cli)."""
import json
import re
import sys

import pytest

import typer
from typer.testing import CliRunner
import comfy.cmd.cli as cli_module
from comfy.cmd.cli import app, _register_sub_apps

_register_sub_apps()

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "comfyui" in out.lower() or "ComfyUI" in out


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--listen" in out
    assert "--port" in out
    assert "--daemon" in out or "-d" in out


def test_serve_help_exposes_canonical_distributed_topology():
    result = runner.invoke(app, ["serve", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    out = _plain(result.output)
    for option in (
        "--rank",
        "--world-size",
        "--local-rank",
        "--local-world-size",
        "--master-addr",
        "--master-port",
        "--pipeline-parallel-size",
        "--hunyuan-expert-parallel-only",
        "--distributed-executor-backend",
    ):
        assert option in out


def test_hunyuan_expert_parallel_cli_configuration():
    command = typer.main.get_command(app).commands["serve"]
    with command.make_context("serve", ["--no-guess-settings", "--tensor-parallel-size", "2", "--hunyuan-expert-parallel-only"]) as context:
        configuration = cli_module._build_config(context.params)
    assert configuration.tensor_parallel_size == 2
    assert configuration.hunyuan_expert_parallel_only is True


def test_start_delegates_to_serve_daemon(monkeypatch):
    calls = []

    class Result:
        returncode = 17

    def fake_run(args, check=False):
        calls.append((args, check))
        return Result()

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_module.sys, "argv", ["/tmp/comfyui"])

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 17
    assert calls == [(["/tmp/comfyui", "serve", "--daemon", "--guess-settings"], False)]


def test_worker_help():
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "distributed-queue" in _plain(result.output)


def test_run_workflow_help():
    result = runner.invoke(app, ["run-workflow", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--prompt" in out
    assert "--seed" in out


def test_run_workflow_and_workflows_run_have_identical_params():
    root = typer.main.get_command(app)
    run_workflow_cmd = root.commands["run-workflow"]
    workflows_run_cmd = root.commands["workflows"].commands["run"]

    def param_signature(command):
        return [
            (
                param.name,
                tuple(getattr(param, "opts", ())),
                tuple(getattr(param, "secondary_opts", ())),
                getattr(param, "help", None),
                getattr(param, "default", None),
                getattr(param, "nargs", None),
                getattr(param, "multiple", None),
                type(param).__name__,
            )
            for param in command.params
        ]

    assert param_signature(workflows_run_cmd) == param_signature(run_workflow_cmd)


def test_create_directories_help():
    result = runner.invoke(app, ["create-directories", "--help"])
    assert result.exit_code == 0
    assert "--base-directory" in _plain(result.output)


def test_list_workflow_templates_help():
    result = runner.invoke(app, ["list-workflow-templates", "--help"])
    assert result.exit_code == 0
    assert "--format" in _plain(result.output)


def test_stop_help():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--server" in out or "--pid-file" in out


def test_logs_help():
    result = runner.invoke(app, ["logs", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--follow" in out or "-f" in out


def test_serve_pip_help():
    # Wider terminal so rich help doesn't wrap mid-flag — the long
    # ``--flag/--no-flag`` columns push other help text off an 80-col
    # render and the substring checks below become brittle.
    result = runner.invoke(app, ["serve-pip", "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--pip-facade-registry" in out or "registry API" in out
    assert "Read facade registry" in out or "snapshot URI" in out
    assert "--pip-facade-cache" in out or "cached" in out
    assert "--pip-facade-only-kno" in out or "Only expose nodes" in out


def test_snapshot_pip_registry_help():
    result = runner.invoke(app, ["snapshot-pip-registry", "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "pip-facade-registr" in out
    assert "pip-facade-snapshot" in out


def test_models_help():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ls" in out
    assert "find-local" in out
    assert "download" in out
    assert "paths" in out
    assert "available" not in out


def test_workflows_help():
    result = runner.invoke(app, ["workflows", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "run" in out or "submit" in out or "convert" in out or "show" in out


def test_nodes_help():
    result = runner.invoke(app, ["nodes", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ls" in out or "packages" in out or "info" in out


def test_jobs_help():
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "cancel" in out


def test_env_help():
    result = runner.invoke(app, ["env", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "info" in out or "check" in out or "packages" in out or "paths" in out


def test_workflows_run_help():
    result = runner.invoke(app, ["workflows", "run", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--dry-run" in out
    assert "--prompt" in out
    assert "--seed" in out
    assert "--cfg" in out
    assert "--sampler" in out
    assert "--width" in out
    assert "--set" in out


def test_workflows_run_workflow_specific_help_order():
    workflow = "tests/inference/workflows/z_image-0.json"
    result = runner.invoke(app, ["workflows", "run", workflow, "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    # Collapse all whitespace so narrower consoles (the Windows runner wraps
    # regardless of COLUMNS) can't split a marker phrase across lines.
    out = " ".join(_plain(result.output).split())

    explanation = out.index("Execute workflow(s) locally and exit.")
    workflow_params = out.index("Common parameters")
    default_params = out.index("Arguments")

    def _context(label: str, position: int) -> str:
        return f"{label}@{position}: ...{out[max(0, position - 120):position + 160]}..."

    diagnostics = "\n".join(
        (
            f"head: {out[:300]}",
            _context("explanation", explanation),
            _context("workflow_params", workflow_params),
            _context("default_params", default_params),
        )
    )
    assert explanation < workflow_params < default_params, diagnostics


def test_qwen_layered_help_keeps_colliding_cfg_and_steps_metadata():
    workflow = "tests/inference/workflows/qwen-image-layered-pipeline-0.json"
    result = runner.invoke(app, ["workflows", "run", workflow, "--help"], env={"COLUMNS": "220"})

    assert result.exit_code == 0
    out = " ".join(_plain(result.output).split())
    assert re.search(r"--x-cfg\s+FLOAT\s+2\.5", out)
    assert re.search(r"--x-steps\s+INT\s+20", out)
    assert "--x-cfg FLOAT 'randomize'" not in out
    assert "--x-steps INT 331728509923362" not in out


def test_run_workflow_accepts_comma_separated_cuda_devices(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "_run_workflow_cli", lambda config, **kwargs: calls.append(config))

    result = runner.invoke(app, [
        "workflows", "run", "workflow.json", "--cuda-device", "1,0",
    ])

    assert result.exit_code == 0, result.output
    assert calls[0].cuda_device == "1,0"


def test_workflows_run_workflow_specific_help_accepts_png_workflow(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    PngImagePlugin = pytest.importorskip("PIL.PngImagePlugin")
    workflow = {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 4,
                "cfg": 1.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        }
    }
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(workflow))
    path = tmp_path / "workflow.png"
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(path, pnginfo=metadata)

    result = runner.invoke(app, ["workflows", "run", str(path), "--help"], env={"COLUMNS": "220"})

    assert result.exit_code == 0
    out = _plain(result.output)
    assert "Common parameters" in out
    assert "--seed" in out
    assert "--steps" in out
    assert "--cfg" in out


def test_workflows_run_warns_about_unknown_args(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "_run_workflow_cli", lambda config, **kwargs: calls.append((config, kwargs)))

    result = runner.invoke(app, ["workflows", "run", "workflow.json", "--definitely-unknown"])

    assert result.exit_code == 0
    assert "[WARNING] Ignoring unknown CLI argument(s): --definitely-unknown" in _plain(result.stderr)
    assert len(calls) == 1
    assert calls[0][0].workflows == ["workflow.json"]


def test_workflows_run_dry_run_implies_all(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "_run_workflow_cli", lambda config, **kwargs: calls.append((config, kwargs)))

    result = runner.invoke(app, ["workflows", "run", "workflow.json", "--dry-run"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0].workflows == ["workflow.json"]
    assert calls[0][1] == {"all": True, "dry_run": True}


def test_workflows_convert_help():
    result = runner.invoke(app, ["workflows", "convert", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--output" in out or "-o" in out


def test_serve_has_new_override_opts():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--cfg" in out
    assert "--sampler" in out
    assert "--scheduler" in out
    assert "--width" in out
    assert "--height" in out
    assert "--batch-size" in out
    assert "--checkpoint" in out
    assert "--set" in out


def test_entrypoint_defaults_to_serve_with_no_subcommand(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve"]]


def test_entrypoint_defaults_to_serve_when_first_arg_is_option(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "--listen", "0.0.0.0"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve", "--listen", "0.0.0.0"]]


def test_entrypoint_expands_bare_listen(monkeypatch):
    """Bare --listen (no value) should be expanded to --listen 0.0.0.0,::"""
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "--listen"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve", "--listen", "0.0.0.0,::"]]


def test_entrypoint_does_not_expand_listen_with_value(monkeypatch):
    """--listen with an explicit value should be left alone."""
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "--listen", "192.168.1.1"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve", "--listen", "192.168.1.1"]]


def test_expand_bare_flags_at_end():
    from comfy.cmd.cli import _expand_bare_flags
    assert _expand_bare_flags(["prog", "--listen"]) == ["prog", "--listen", "0.0.0.0,::"]


def test_expand_bare_flags_before_another_flag():
    from comfy.cmd.cli import _expand_bare_flags
    assert _expand_bare_flags(["prog", "--listen", "--port", "8188"]) == ["prog", "--listen", "0.0.0.0,::", "--port", "8188"]


def test_expand_bare_flags_with_value():
    from comfy.cmd.cli import _expand_bare_flags
    assert _expand_bare_flags(["prog", "--listen", "1.2.3.4"]) == ["prog", "--listen", "1.2.3.4"]


def test_expand_bare_flags_enable_cors():
    from comfy.cmd.cli import _expand_bare_flags
    assert _expand_bare_flags(["prog", "--enable-cors-header"]) == ["prog", "--enable-cors-header", "*"]


def test_entrypoint_does_not_rewrite_unknown_verbs(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "definitely-not-a-command"])

    cli_module.entrypoint()

    assert called == [["comfyui", "definitely-not-a-command"]]


class TestParseListenAddress:
    def test_bare_ipv4(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0") == ("0.0.0.0", None)

    def test_ipv4_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0:8189") == ("0.0.0.0", 8189)

    def test_bare_ipv6(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("::") == ("::", None)

    def test_ipv6_loopback(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("::1") == ("::1", None)

    def test_bracketed_ipv6_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("[::]:8189") == ("::", 8189)

    def test_bracketed_ipv6_loopback_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("[::1]:8189") == ("::1", 8189)

    def test_hostname_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("localhost:9000") == ("localhost", 9000)

    def test_hostname_bare(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("localhost") == ("localhost", None)

    def test_comma_separated_not_split(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0,::") == ("0.0.0.0,::", None)


def test_workflows_run_workflow_specific_help_order_legacy_windows(monkeypatch):
    """Regression for the Windows runner: legacy consoles render rich panels
    with square box borders, which the help splitter didn't match, so the
    custom workflow panels printed after the generic Arguments/Options panels
    and the help-order test failed with the Common parameters panel at the
    end of the output."""
    import rich.console

    original_init = rich.console.Console.__init__

    def legacy_init(self, *args, **kwargs):
        kwargs.setdefault("legacy_windows", True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rich.console.Console, "__init__", legacy_init)

    workflow = "tests/inference/workflows/z_image-0.json"
    result = runner.invoke(app, ["workflows", "run", workflow, "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    out = " ".join(_plain(result.output).split())

    explanation = out.index("Execute workflow(s) locally and exit.")
    workflow_params = out.index("Common parameters")
    default_params = out.index("Arguments")

    assert explanation < workflow_params < default_params, out[:300]
