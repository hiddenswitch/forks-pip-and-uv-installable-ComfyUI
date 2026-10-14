import subprocess
import sys


_REPLAY = """
import importlib.abc
import os
import sys

class CheckEnvironment(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            assert os.environ.get("OPENCV_IO_ENABLE_OPENEXR") == "1"
            assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
        return None

sys.meta_path.insert(0, CheckEnvironment())
import comfy.cmd.main_pre
"""


def test_pre_import_setup_configures_environment_before_torch():
    subprocess.run([sys.executable, "-c", _REPLAY], check=True)
