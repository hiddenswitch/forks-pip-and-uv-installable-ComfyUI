import collections
import subprocess
from unittest.mock import patch, MagicMock, mock_open


from comfy.cli_args_types import Configuration, PerformanceFeature
from comfy.component_model.guess_settings import (
    apply_guess_settings,
    _total_ram_gb,
    _has_nvidia_gpu,
    _has_amd_gpu,
    _competing_gpu_processes,
)


def _config(**overrides) -> Configuration:
    return Configuration(**overrides)


class TestTotalRamGb:
    def test_psutil(self):
        mem = collections.namedtuple("svmem", ["total"])(total=16 * 1024 ** 3)
        with patch("psutil.virtual_memory", return_value=mem):
            assert abs(_total_ram_gb() - 16.0) < 0.1

    def test_fallback_proc_meminfo(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("builtins.open", mock_open(read_data="MemTotal:       16777216 kB\n")) as reader:
            assert _total_ram_gb() == 16.0
        reader.assert_called_once_with("/proc/meminfo", encoding="utf-8")

    def test_returns_zero_on_failure(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("builtins.open", side_effect=OSError):
            result = _total_ram_gb()
            assert result == 0.0


class TestGpuDetection:
    def test_nvidia_present(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            assert _has_nvidia_gpu() is True

    def test_nvidia_absent(self):
        with patch("shutil.which", return_value=None):
            assert _has_nvidia_gpu() is False

    def test_amd_rocm_smi(self):
        def _which(name):
            return "/usr/bin/rocm-smi" if name == "rocm-smi" else None
        with patch("shutil.which", side_effect=_which), \
             patch("os.path.exists", return_value=False):
            assert _has_amd_gpu() is True

    def test_amd_dev_kfd(self):
        with patch("shutil.which", return_value=None), \
             patch("os.path.exists", return_value=True):
            assert _has_amd_gpu() is True

    def test_amd_absent(self):
        with patch("shutil.which", return_value=None), \
             patch("os.path.exists", return_value=False):
            assert _has_amd_gpu() is False


class TestCompetingGpuProcesses:
    def test_no_nvidia_smi(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _competing_gpu_processes() == []

    def test_no_processes(self):
        result = MagicMock(returncode=0, stdout="\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_only_python(self):
        result = MagicMock(returncode=0, stdout="123, python3, 2048 MiB\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_only_nvidia(self):
        result = MagicMock(returncode=0, stdout="123, nvtop, 2048 MiB\n124, nvidia-smi, 2048 MiB\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_competing_processes(self):
        result = MagicMock(returncode=0, stdout="123, python3, 2048 MiB\n124, Discord, 1536 MiB\n125, firefox, 2048 MiB\n")
        with patch("subprocess.run", return_value=result):
            procs = _competing_gpu_processes()
            assert "Discord" in procs
            assert "firefox" in procs
            assert "python3" not in procs

    def test_ignores_small_desktop_helper_processes(self):
        result = MagicMock(
            returncode=0,
            stdout="123, gnome-remote-desktop-daemon, 32 MiB\n124, steamwebhelper, 128 MiB\n",
        )
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_ignores_small_processes(self):
        result = MagicMock(returncode=0, stdout="123, Discord, 256 MiB\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_current_process_family_is_not_competing(self):
        result = MagicMock(returncode=0, stdout="123, /usr/bin/comfyui, 2048 MiB\n124, /usr/bin/Discord, 2048 MiB\n")
        with patch("subprocess.run", return_value=result), \
             patch("comfy.component_model.guess_settings._current_process_family", return_value={123}):
            assert _competing_gpu_processes() == ["Discord"]

    def test_remote_desktop_process_is_not_competing(self):
        result = MagicMock(returncode=0, stdout="123, /usr/libexec/gnome-remote-desktop-daemon, 2048 MiB\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_windows_paths(self):
        result = MagicMock(
            returncode=0,
            stdout="123, C:\\Program Files\\Discord\\Discord.exe, 2048 MiB\n124, C:\\Python312\\python.exe, 2048 MiB\n",
        )
        with patch("subprocess.run", return_value=result):
            procs = _competing_gpu_processes()
            assert "Discord.exe" in procs
            assert "python.exe" not in procs

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            assert _competing_gpu_processes() == []


class TestGuessSettingsNvidia:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_cublas_ops_enabled(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps in cfg.fast

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_cublas_ops_added_to_existing_fast(self, *_mocks):
        cfg = _config()
        cfg.fast = [PerformanceFeature.Fp16Accumulation]
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps in cfg.fast
        assert PerformanceFeature.Fp16Accumulation in cfg.fast

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=["Discord", "firefox"])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_novram_when_competing_processes(self, *_mocks):
        cfg = _config()
        with patch("sys.platform", "linux"):
            apply_guess_settings(cfg)
        assert cfg.novram is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=["Discord"])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_novram_not_set_when_user_set_highvram(self, *_mocks):
        cfg = _config(highvram=True)
        apply_guess_settings(cfg)
        assert cfg.novram is False
        assert cfg.highvram is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_no_novram_when_no_competing_processes(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.novram is False


class TestGuessSettingsAmd:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_fp32_vae_enabled(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.fp32_vae is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_fp32_vae_not_set_when_user_set_fp16(self, *_mocks):
        cfg = _config(fp16_vae=True)
        apply_guess_settings(cfg)
        assert cfg.fp32_vae is False
        assert cfg.fp16_vae is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_no_cublas_on_amd(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps not in cfg.fast


class TestGuessSettingsRam:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=16.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_low_ram_disables_pinned(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_high_ram_keeps_pinned(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is False

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=31.9)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_boundary_ram(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=32.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_exactly_32gb(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is False


class TestGuessSettingsAttention:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_sage_attention_preferred(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "sageattention"):
            cfg = _config()
            with patch("sys.platform", "linux"):
                apply_guess_settings(cfg)
            assert cfg.use_sage_attention is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_xformers_fallback(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "xformers"):
            cfg = _config()
            apply_guess_settings(cfg)
            assert cfg.use_sage_attention is False
            assert cfg.disable_xformers is False

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_pytorch_attention_fallback(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.use_pytorch_cross_attention is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_user_set_attention_not_overridden(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "sageattention"):
            cfg = _config(use_flash_attention=True)
            apply_guess_settings(cfg)
            assert cfg.use_sage_attention is False
            assert cfg.use_flash_attention is True


class TestGuessSettingsMacOS:
    @patch("comfy.component_model.guess_settings.sys")
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=36.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_macos_sets_pytorch_cross_attention(self, _pkg, _procs, _ram, _amd, _nv, mock_sys):
        mock_sys.platform = "darwin"
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.use_pytorch_cross_attention is True

    @patch("comfy.component_model.guess_settings.sys")
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=36.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=True)
    def test_macos_ignores_sageattention(self, _pkg, _procs, _ram, _amd, _nv, mock_sys):
        mock_sys.platform = "darwin"
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.use_sage_attention is False
        assert cfg.use_pytorch_cross_attention is True

    @patch("comfy.component_model.guess_settings.sys")
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=36.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_macos_sets_gpu_only(self, _pkg, _procs, _ram, _amd, _nv, mock_sys):
        mock_sys.platform = "darwin"
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.gpu_only is True

    @patch("comfy.component_model.guess_settings.sys")
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=36.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_macos_respects_user_vram_mode(self, _pkg, _procs, _ram, _amd, _nv, mock_sys):
        mock_sys.platform = "darwin"
        cfg = _config(cpu=True)
        apply_guess_settings(cfg)
        assert cfg.gpu_only is False
        assert cfg.cpu is True

    @patch("comfy.component_model.guess_settings.sys")
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=36.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_macos_respects_user_attention(self, _pkg, _procs, _ram, _amd, _nv, mock_sys):
        mock_sys.platform = "darwin"
        cfg = _config(use_flash_attention=True)
        apply_guess_settings(cfg)
        assert cfg.use_pytorch_cross_attention is False
        assert cfg.use_flash_attention is True


class TestGuessSettingsCliArg:
    def test_default_is_true(self):
        """As of the move to guess-settings-by-default, ``Configuration()``
        with no CLI flags has ``guess_settings=True``. The inverse is
        ``--no-guess-settings`` or ``COMFYUI_GUESS_SETTINGS=0``."""
        from tests.unit.test_cli_args import _parse_test_args
        cfg = _parse_test_args([])
        assert cfg.guess_settings is True

    def test_flag_is_true(self):
        from tests.unit.test_cli_args import _parse_test_args
        cfg = _parse_test_args(["--guess-settings"])
        assert cfg.guess_settings is True

    # The ``--no-guess-settings`` inverse is wired via Typer's slash
    # syntax in comfy/cmd/cli.py; parsing is covered by
    # tests/unit/test_cli_no_flags.py::test_no_guess_settings_parses,
    # which exercises the real Typer app rather than the legacy test stub.

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_cli_args_configuration_applies_guess_settings(self, *_mocks):
        from comfy.cmd.setup import setup_guess_settings
        cfg = Configuration(guess_settings=True)
        setup_guess_settings(cfg)
        assert PerformanceFeature.CublasOps in cfg.fast


class TestGuessSettingsTensorParallel:
    """Identical NVIDIA GPUs default to tensor parallelism on Linux."""

    def _apply(self, cfg, names, environment=None, os_name="posix"):
        from comfy.component_model import guess_settings
        environment = {} if environment is None else environment
        with patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True), \
             patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False), \
             patch("comfy.component_model.guess_settings._total_ram_gb", return_value=128.0), \
             patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[]), \
             patch("comfy.component_model.guess_settings._nvidia_compute_caps", return_value=[(8, 6)] * len(names)), \
             patch("comfy.component_model.guess_settings._nvidia_gpu_names", return_value=list(names)), \
             patch("comfy.component_model.guess_settings._has_package", return_value=False), \
             patch("sys.platform", "linux"), \
             patch.object(guess_settings.os, "name", os_name), \
             patch.dict("os.environ", environment, clear=True):
            apply_guess_settings(cfg)
        return cfg

    def test_two_identical_gpus_default_to_tensor_parallel(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000", "NVIDIA RTX A5000"])
        assert cfg.tensor_parallel_size == 2
        assert cfg.pipeline_parallel_size is None

    def test_four_identical_gpus_use_all_four(self):
        cfg = self._apply(_config(), ["NVIDIA GeForce RTX 3090"] * 4)
        assert cfg.tensor_parallel_size == 4

    def test_three_identical_gpus_use_the_largest_power_of_two(self):
        cfg = self._apply(_config(), ["NVIDIA GeForce RTX 3090"] * 3)
        assert cfg.tensor_parallel_size == 2

    def test_mixed_gpus_keep_the_default(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"])
        assert cfg.tensor_parallel_size is None

    def test_single_gpu_keeps_the_default(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"])
        assert cfg.tensor_parallel_size is None

    def test_explicit_tensor_parallel_size_wins(self):
        cfg = self._apply(_config(tensor_parallel_size=1), ["NVIDIA RTX A5000"] * 2)
        assert cfg.tensor_parallel_size == 1

    def test_explicit_pipeline_parallel_size_wins(self):
        cfg = self._apply(_config(pipeline_parallel_size=2), ["NVIDIA RTX A5000"] * 2)
        assert cfg.tensor_parallel_size is None
        assert cfg.pipeline_parallel_size == 2

    def test_explicit_sequence_parallel_degree_wins(self):
        cfg = self._apply(_config(ulysses_degree=2), ["NVIDIA RTX A5000"] * 2)
        assert cfg.tensor_parallel_size is None

    def test_environment_tensor_parallel_size_wins(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"] * 4, {"COMFYUI_TENSOR_PARALLEL_SIZE": "1"})
        assert cfg.tensor_parallel_size is None

    def test_external_launcher_keeps_the_default(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"] * 2, {"RANK": "0", "WORLD_SIZE": "2"})
        assert cfg.tensor_parallel_size is None

    def test_cuda_device_selection_limits_the_group(self):
        cfg = self._apply(_config(cuda_device="0"), ["NVIDIA RTX A5000"] * 2)
        assert cfg.tensor_parallel_size is None
        cfg = self._apply(_config(cuda_device="1,2"), ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA RTX A5000"])
        assert cfg.tensor_parallel_size == 2

    def test_cuda_visible_devices_limits_the_group(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"] * 2, {"CUDA_VISIBLE_DEVICES": "1"})
        assert cfg.tensor_parallel_size is None
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"] * 2, {"CUDA_VISIBLE_DEVICES": "GPU-0123"})
        assert cfg.tensor_parallel_size is None

    def test_torch_device_selection_keeps_the_default(self):
        cfg = self._apply(_config(torch_device="cuda:0"), ["NVIDIA RTX A5000"] * 2)
        assert cfg.tensor_parallel_size is None

    def test_windows_keeps_the_default(self):
        cfg = self._apply(_config(), ["NVIDIA RTX A5000"] * 2, os_name="nt")
        assert cfg.tensor_parallel_size is None


class TestDefaultTensorParallelSize:
    def test_rule(self):
        from comfy.component_model.guess_settings import default_tensor_parallel_size
        assert default_tensor_parallel_size([]) == 1
        assert default_tensor_parallel_size(["a"]) == 1
        assert default_tensor_parallel_size(["a", "a"]) == 2
        assert default_tensor_parallel_size(["a", "a", "a"]) == 2
        assert default_tensor_parallel_size(["a"] * 8) == 8
        assert default_tensor_parallel_size(["a", "b"]) == 1


class TestNvidiaGpuNames:
    def test_parses_index_and_name(self):
        from comfy.component_model.guess_settings import _nvidia_gpu_names
        result = MagicMock(returncode=0, stdout="1, NVIDIA RTX A5000\n0, NVIDIA GeForce RTX 3090, extra\n")
        with patch("subprocess.run", return_value=result):
            assert _nvidia_gpu_names() == ["NVIDIA GeForce RTX 3090, extra", "NVIDIA RTX A5000"]

    def test_missing_nvidia_smi(self):
        from comfy.component_model.guess_settings import _nvidia_gpu_names
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _nvidia_gpu_names() == []
