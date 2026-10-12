"""comfy-aimdo's native log level is its own setting, ERROR by default.

Its VRAM usage dump logs every page pinned at that moment at WARNING ("VBAR ...: Page 370
pin_count=2"), hundreds of lines per model while sampling, each through a Python logging callback.
Tied to --logging-level (INFO by default) the final evals of the spellsource trials printed them all.
"""
from __future__ import annotations

import pytest

from comfy.cli_args_types import Configuration
from comfy.cmd import cli


class _Control:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("set_log_"):
            return lambda: self.calls.append(name)
        raise AttributeError(name)


def test_the_default_aimdo_level_is_error_whatever_the_logging_level():
    config = Configuration()
    assert config.logging_level == "INFO"
    assert config.aimdo_logging_level == "ERROR"


@pytest.mark.parametrize("level,setter", [
    ("DEBUG", "set_log_debug"), ("INFO", "set_log_info"), ("WARNING", "set_log_warning"),
    ("ERROR", "set_log_error"), ("CRITICAL", "set_log_critical"),
])
def test_the_native_log_level_follows_the_aimdo_setting(level, setter):
    from comfy.memory_management import set_aimdo_log_level
    control = _Control()
    set_aimdo_log_level(control, level)
    assert control.calls == [setter]


def test_the_cli_sets_the_aimdo_level():
    names = [name for name, _, _ in cli._LOGGING_OPTS]
    assert "aimdo_logging_level" in names
    assert cli._build_config({"aimdo_logging_level": "WARNING"}).aimdo_logging_level == "WARNING"
