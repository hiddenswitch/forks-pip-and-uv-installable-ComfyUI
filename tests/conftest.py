import faulthandler
import logging

logger = logging.getLogger(__name__)

import os
import pathlib
import pickle
import socket
import subprocess
import tempfile
from contextvars import ContextVar
from typing import List, Any, Generator

import pytest
import requests
import sys
import time
import fsspec

from tests.unit._subprocess_helpers import _DrainingProcess

pytest_plugins = ["tests.inference_benchmark"]

faulthandler.enable()

os.environ['OTEL_METRICS_EXPORTER'] = 'none'
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "True"
# fixes issues with running the testcontainers rabbitmqcontainer on Windows
os.environ["TC_HOST"] = "localhost"

from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration
from comfy.component_model.setup import setup_logging_filters
assert "pkg" in fsspec.available_protocols()

logging.getLogger("pika").setLevel(logging.CRITICAL + 1)
logging.getLogger("aio_pika").setLevel(logging.CRITICAL + 1)
setup_logging_filters()


_SERIAL_PROCESS_FIXTURES = frozenset(
    {
        "comfy_background_server",
        "comfy_url_and_proc",
        "frontend_backend_worker_with_rabbitmq",
        "manager_disabled_server",
        "manager_enabled_server",
        "manager_with_disabled_custom_nodes_server",
        "process_startup_timeout_seconds",
    }
)


def requires_serial_process_group(fixture_names) -> bool:
    """Return whether a test starts an isolated process or ComfyUI server."""
    return not _SERIAL_PROCESS_FIXTURES.isdisjoint(fixture_names)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    marker = pyfuncitem.get_closest_marker("ipython")
    if marker is None:
        return None

    # Notebook dependencies are only needed when executing a marked test.
    from tests.ipython_runner import run_in_ipython

    run_in_ipython(pyfuncitem, marker)
    return True


@pytest.fixture
def mock_user_directory():
    from comfy.component_model.folder_path_types import FolderNames
    from comfy.cmd.folder_paths import get_user_directory
    from comfy.execution_context import context_folder_names_and_paths
    """Create a temporary user directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fn = FolderNames(base_paths=[pathlib.Path(temp_dir)])
        with context_folder_names_and_paths(fn):
            yield get_user_directory()


@pytest.fixture(scope="function", autouse=False)
def has_gpu() -> bool:
    original_cpu_state = None
    try:
        from comfy import model_management
        original_cpu_state = model_management.cpu_state
    except ImportError:
        pass

    # mps
    has_gpu = False
    try:
        import torch
        has_gpu = torch.backends.mps.is_available() and torch.device("mps") is not None
        if has_gpu:
            # Probe: virtualized macOS runners (e.g. GitHub Actions) report
            # MPS as available but cannot actually allocate GPU memory.
            try:
                torch.tensor([1.0], device="mps")
            except RuntimeError:
                has_gpu = False
        if has_gpu:
            from comfy.model_management import CPUState
            model_management.cpu_state = CPUState.MPS
    except ImportError:
        pass

    if not has_gpu:
        # xpu
        try:
            import torch
            has_gpu = (
                hasattr(torch, "xpu")
                and torch.xpu.is_available()
                and torch.xpu.device_count() > 0
            )
            if has_gpu:
                torch.tensor([1.0], device="xpu")
        except (ImportError, AttributeError):
            has_gpu = False
        except RuntimeError:
            has_gpu = False

        if not has_gpu:
            # cuda
            try:
                import torch
                has_gpu = torch.device(torch.cuda.current_device()) is not None
            except:
                has_gpu = False

    if has_gpu:
        from comfy.model_management import CPUState
        if model_management.cpu_state != CPUState.MPS:
            model_management.cpu_state = CPUState.GPU if has_gpu else CPUState.CPU
    try:
        yield has_gpu
    finally:
        if original_cpu_state is not None:
            model_management.cpu_state = original_cpu_state


@pytest.fixture(scope="module", autouse=False, params=["ThreadPoolExecutor", "ProcessPoolExecutor"])
def frontend_backend_worker_with_rabbitmq(request, tmp_path_factory, num_workers: int = 1):
    from huggingface_hub import hf_hub_download
    from testcontainers.rabbitmq import RabbitMqContainer

    logging.getLogger("testcontainers.core.container").setLevel(logging.WARNING)
    logging.getLogger("testcontainers.core.waiting_utils").setLevel(logging.WARNING)

    hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors")
    hf_hub_download("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors")

    tmp_path = tmp_path_factory.mktemp("comfy_background_server")
    executor_factory = request.param
    processes_to_close: List[subprocess.Popen] = []

    with RabbitMqContainer("rabbitmq:4.0.5-management") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        # Check if OTEL endpoint is configured for integration testing
        otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

        env = os.environ.copy()
        if otel_endpoint:
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
            logger.info(f"Configuring services to export traces to: {otel_endpoint}")

        # NOTE: use --cwd=, not -w=, for the working directory argument.
        # Click does not treat '=' as a separator for short options: -w=/path
        # is parsed as cwd="=/path" (a relative path), not cwd="/path".
        # This caused a stray "=/" directory to be created in the project root.
        frontend_command = [
            "comfyui",
            "--listen=0.0.0.0",
            "--port=19001",
            "--cpu",
            "--disable-all-custom-nodes",
            "--distributed-queue-frontend",
            f"--cwd={str(tmp_path)}",
            f"--distributed-queue-connection-uri={connection_uri}",
        ]

        processes_to_close.append(subprocess.Popen(frontend_command, stdout=sys.stdout, stderr=sys.stderr, env=env))

        # Start multiple workers
        for i in range(num_workers):
            backend_command = [
                "comfyui-worker",
                "--disable-all-custom-nodes",
                f"--port={19002 + i}",
                f"--cwd={str(tmp_path)}",
                f"--distributed-queue-connection-uri={connection_uri}",
                f"--executor-factory={executor_factory}"
            ]
            processes_to_close.append(subprocess.Popen(backend_command, stdout=sys.stdout, stderr=sys.stderr, env=env))

        try:
            server_address = f"http://127.0.0.1:19001"
            start_time = time.time()
            connected = False
            while time.time() - start_time < server_startup_timeout_seconds():
                try:
                    response = requests.get(server_address)
                    if response.status_code == 200:
                        connected = True
                        break
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
            if not connected:
                raise RuntimeError("could not connect to frontend")
            yield server_address
        finally:
            for process in processes_to_close:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@pytest.fixture(scope="module", autouse=False)
def comfy_background_server(tmp_path_factory) -> Generator[tuple[Configuration, subprocess.Popen], Any, None]:
    tmp_path = tmp_path_factory.mktemp("comfy_background_server")
    # Start server

    configuration = default_configuration()
    configuration.listen = "localhost"
    configuration.output_directory = str(tmp_path)
    configuration.input_directory = str(tmp_path)

    yield from comfy_background_server_from_config(configuration)


def comfy_background_server_from_config(configuration: Configuration, server_module="tests.background_server"):
    with tempfile.NamedTemporaryFile(prefix="comfyui-test-server-", suffix=".pickle", delete=False) as config_file:
        pickle.dump(configuration, config_file)
        config_path = pathlib.Path(config_file.name)

    server_process = _DrainingProcess(
        [
            sys.executable,
            "-m",
            server_module,
            str(config_path),
        ]
    )

    success = False
    try:
        startup_timeout = server_startup_timeout_seconds()
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            return_code = server_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Background server exited during startup with code {return_code}\n"
                    f"Background server output:\n{server_process.tail()}"
                )
            try:
                with socket.create_connection((configuration.listen, configuration.port), timeout=1):
                    success = True
                    break
            except OSError:
                pass
            time.sleep(1)

        if not success:
            raise RuntimeError(
                f"Failed to start background server within {startup_timeout:g} seconds\n"
                f"Background server output:\n{server_process.tail()}"
            )
        yield configuration, server_process
    finally:
        server_process.shutdown()
        config_path.unlink(missing_ok=True)

    import torch
    torch.cuda.empty_cache()


def server_startup_timeout_seconds() -> float:
    """Return the server readiness budget used by subprocess integration tests."""
    value = float(os.environ.get("COMFYUI_TEST_SERVER_STARTUP_TIMEOUT", "60"))
    if value <= 0:
        raise ValueError("COMFYUI_TEST_SERVER_STARTUP_TIMEOUT must be positive")
    return value


@pytest.fixture(scope="session")
def process_startup_timeout_seconds() -> float:
    """Return the cold-start budget for tests that create a Python process."""
    return server_startup_timeout_seconds()


@pytest.fixture(scope="session")
def skip_timing_checks(pytestconfig):
    """Fixture that returns whether timing checks should be skipped."""
    # todo: in the LTS, we don't need to skip timing checks, everything just works
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    # Windows cold-imports Torch in every spawned process. Running several
    # server/process fixtures concurrently both wastes that work and can starve
    # an otherwise healthy server past its readiness budget. Mark them for an
    # isolated CI phase and keep them on one xdist worker in runs which do use
    # xdist.
    for item in items:
        if requires_serial_process_group(item.fixturenames):
            item.add_marker(pytest.mark.server_process)
            item.add_marker(pytest.mark.xdist_group(name="server-process"))

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            if item.module.__name__ == test_name:
                last_items.append(item)
                items.remove(item)

    items.extend(last_items)


@pytest.fixture(scope="module")
def vae():
    from comfy.nodes.base_nodes import VAELoader

    vae_file = "vae-ft-mse-840000-ema-pruned.safetensors"
    try:
        vae, = VAELoader().load_vae(vae_file)
    except FileNotFoundError:
        pytest.skip(f"{vae_file} not present on machine")
    return vae


@pytest.fixture(scope="module")
def clip():
    from comfy.nodes.base_nodes import CheckpointLoaderSimple

    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[1]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def model(clip):
    from comfy.nodes.base_nodes import CheckpointLoaderSimple
    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[0]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")


@pytest.fixture(scope="function", autouse=False)
def use_temporary_output_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_output_directory()
    folder_paths.set_output_directory(tmp_path)
    yield tmp_path
    folder_paths.set_output_directory(orig_dir)


@pytest.fixture(scope="function", autouse=False)
def use_temporary_input_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_input_directory()
    folder_paths.set_input_directory(tmp_path)
    yield tmp_path
    folder_paths.set_input_directory(orig_dir)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.sections = [
        (name, content) for name, content in report.sections
        if "stderr" not in name.lower()
    ]


current_test_name = ContextVar('current_test_name', default=None)


@pytest.fixture(autouse=True)
def set_test_name(request):
    token = current_test_name.set(request.node.name)
    yield
    current_test_name.reset(token)

def trigger_sync_seed_assets(session: requests.Session, base_url: str) -> None:
    """Force a fast sync/seed pass by calling the seed endpoint."""
    session.post(base_url + "/api/assets/seed", json={"roots": ["models", "input", "output"]}, timeout=30)
    time.sleep(0.2)


def get_asset_filename(asset_hash: str, extension: str) -> str:
    return asset_hash.removeprefix("blake3:") + extension
