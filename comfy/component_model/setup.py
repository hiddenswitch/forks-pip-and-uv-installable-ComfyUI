"""
Setup functions extracted from main_pre.py.

Called by Typer commands (cli.py) rather than running at import time.
"""
import ctypes
import faulthandler
import importlib.util
import logging
import os
import shutil
import signal
import sys
import warnings

from ..cli_args_types import Configuration
from ..distributed.config import resolve_distributed_configuration

logger = logging.getLogger(__name__)
_dumping_traceback = False


def setup_debug_hang(config: Configuration):
    """Enable upstream-style traceback dumps for hang debugging."""
    faulthandler.enable(all_threads=config.debug_hang)
    if not config.debug_hang:
        return

    def dump_traceback_on_sigint(signum, frame):
        del signum, frame
        global _dumping_traceback  # pylint: disable=global-statement
        if _dumping_traceback:
            raise KeyboardInterrupt
        _dumping_traceback = True
        # IPython patches enable(), but not dump_traceback(), to use this stream.
        faulthandler.dump_traceback(file=sys.__stderr__, all_threads=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, dump_traceback_on_sigint)


def setup_environment():
    """Set env vars before torch import."""
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    from .torch_cache import setup_torch_compile_cache_dirs
    setup_torch_compile_cache_dirs()
    if os.name == "nt":
        os.environ['MIMALLOC_PURGE_DELAY'] = '0'


def setup_warning_filters():
    if os.name == "nt":
        logging.getLogger("xformers").addFilter(
            lambda record: 'A matching Triton is not available' not in record.getMessage()
        )
    warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
    warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
    warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
    warnings.filterwarnings('ignore', category=FutureWarning, message=r'`torch\.cuda\.amp\.custom_fwd.*')
    warnings.filterwarnings("ignore", category=UserWarning, message="Please use the new API settings to control TF32 behavior.*")
    warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated, please import via timm.models", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated, please import via timm.layers", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Inheritance class _InstrumentedApplication from web.Application is discouraged", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Please import `gaussian_filter` from the `scipy.ndimage` namespace; the `scipy.ndimage.filters` namespace is deprecated", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support")
    warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version .* ONNX Runtime supports Windows 10 and above, only.")


def setup_logging_filters():
    log_msg_to_filter = "NOTE: Redirects are currently not supported in Windows or MacOs."
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
        lambda record: log_msg_to_filter not in record.getMessage()
    )
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger(__name__).addFilter(lambda record: "setup plugin" not in record.getMessage())
    logging.getLogger("asyncio").addFilter(lambda record: 'Using selector:' not in record.getMessage())
    logging.getLogger("requests_cache").setLevel(logging.ERROR)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def setup_logging(config: Configuration):
    from ..app import logger as app_logger
    app_logger.setup_logger(config.logging_level)


def setup_windows_multi_gpu_defaults(config: Configuration):
    """Apply upstream's safe Windows multi-GPU visibility policy before torch."""
    if os.name != "nt" or config.torch_device is not None:
        return

    from ..app import logger as app_logger
    from ..cmd import cuda_malloc

    cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_selection = config.cuda_device
    try:
        gpu_count = sum("NVIDIA" in name.upper() for name in cuda_malloc.get_gpu_names())
    except OSError:
        gpu_count = 0

    if gpu_count <= 1:
        return

    warning = None
    multiple_visible = False
    if device_selection is None and config.default_device is None and cuda_visibility is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        warning = "Multiple NVIDIA GPUs detected. ComfyUI will use GPU 0 only on Windows by default. To restore all GPUs, pass --cuda-device all --disable-pinned-memory."
    elif device_selection == "all":
        multiple_visible = cuda_visibility is None or "," in cuda_visibility
    elif device_selection is not None:
        multiple_visible = "," in device_selection
    elif config.default_device is not None:
        multiple_visible = True
    else:
        multiple_visible = "," in cuda_visibility

    if multiple_visible and not config.disable_pinned_memory:
        warning = "Multiple NVIDIA GPUs are visible on Windows with pinned memory enabled. Restart with --disable-pinned-memory to avoid CUDA host-transfer failures."

    if warning:
        app_logger.log_startup_warning(
            "\n".join((
                "_" * 72,
                "WARNING WARNING WARNING WARNING WARNING",
                "",
                warning,
                "_" * 72,
            ))
        )


def setup_cuda_devices(config: Configuration):
    if config.torch_device is not None:
        td = config.torch_device
        if td.startswith("cuda:"):
            cuda_idx = td.split(":", 1)[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_idx
            os.environ['HIP_VISIBLE_DEVICES'] = cuda_idx
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cuda_idx
        elif td.startswith("xpu:"):
            os.environ['ONEAPI_DEVICE_SELECTOR'] = f"level_zero:{td.split(':', 1)[1]}"
        elif td.startswith("npu:"):
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = td.split(":", 1)[1]
        logger.info("Set torch device to: %s", td)
    else:
        if config.default_device is not None and config.cuda_device != "all":
            default_dev = config.default_device
            devices = list(range(32))
            devices.remove(default_dev)
            devices.insert(0, default_dev)
            devices = ','.join(map(str, devices))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
            os.environ['HIP_VISIBLE_DEVICES'] = str(devices)

        if config.cuda_device == "all":
            logger.info("Set cuda devices to all")
        elif config.cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_device)
            os.environ['HIP_VISIBLE_DEVICES'] = str(config.cuda_device)
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(config.cuda_device)
            logger.info("Set cuda device to: %s", config.cuda_device)

    if config.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if config.oneapi_device_selector is not None and config.torch_device is None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = config.oneapi_device_selector
        logger.info("Set oneapi device selector to: %s", config.oneapi_device_selector)


def prepare_distributed_environment(config: Configuration):
    """Apply launcher identity before torch, nodes, or DynamicVRAM initialize."""
    distributed = resolve_distributed_configuration(config)
    local_process_peers = (
        distributed.tensor_parallel_size > 1
        or distributed.ulysses_degree * distributed.ring_degree > 1
        or (
            distributed.pipeline_parallel_size > 1
            and distributed.executor_backend in ("mp", "external_launcher")
        )
    )
    if distributed.externally_launched or local_process_peers:
        config.model_management_device_scope = "local"
    if not distributed.externally_launched:
        return distributed
    if distributed.rank != 0:
        config.disable_all_custom_nodes = True
    return distributed


def setup_distributed_runtime(distributed):
    from ..distributed.runtime import initialize_distributed_runtime
    return initialize_distributed_runtime(distributed)


def setup_distributed_device(distributed):
    from ..distributed.runtime import select_distributed_device
    select_distributed_device(distributed)


def setup_guess_settings(config: Configuration):
    if config.guess_settings:
        from .guess_settings import apply_guess_settings
        apply_guess_settings(config)


def setup_cuda_malloc(config: Configuration):
    from ..cmd import cuda_malloc
    cuda_malloc.configure(config)


_tracing_initialized = False
_tracing_provider = None
_tracing_export_endpoints = set()


def setup_tracing(config: Configuration):
    global _tracing_initialized
    if _tracing_initialized:
        _add_configured_trace_exporter(config)
        return False

    try:
        _setup_tracing_impl(config)
        _tracing_initialized = True
        return True
    except Exception:
        _tracing_initialized = False
        logger.debug("Failed to initialize OpenTelemetry tracing", exc_info=True)
        return False


def _add_configured_trace_exporter(config: Configuration):
    endpoint = (
        config.otel_exporter_otlp_endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if endpoint is None or endpoint in _tracing_export_endpoints or _tracing_provider is None:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    if endpoint.startswith("file://"):
        from .otel_file_exporter import FileSpanExporter
        exporter = FileSpanExporter(endpoint.removeprefix("file://"))
        processor = SimpleSpanProcessor(exporter)
    else:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
    _tracing_provider.add_span_processor(processor)
    _tracing_export_endpoints.add(endpoint)


def _setup_tracing_impl(config: Configuration):
    global _tracing_provider
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.semconv.attributes import service_attributes

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS
    from opentelemetry.instrumentation.aiohttp_server import AioHttpServerInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

    from ..tracing_compatibility import ProgressSpanSampler
    from ..tracing_compatibility import patch_spanbuilder_set_channel

    resource = Resource.create({
        service_attributes.SERVICE_NAME: config.otel_service_name,
        service_attributes.SERVICE_VERSION: config.otel_service_version,
    })

    sampler = ProgressSpanSampler()
    provider = TracerProvider(resource=resource, sampler=sampler)

    trace.set_tracer_provider(provider)
    active_provider = trace.get_tracer_provider()
    if active_provider is not provider:
        provider.shutdown()
        provider = active_provider

    _tracing_provider = provider
    _add_configured_trace_exporter(config)

    metrics_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if metrics_endpoint:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=metrics_endpoint),
            export_interval_millis=10000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

    patch_spanbuilder_set_channel()

    for instrumentor_cls in (AioPikaInstrumentor, AioHttpServerInstrumentor,
                             AioHttpClientInstrumentor, RequestsInstrumentor,
                             URLLib3Instrumentor):
        inst = instrumentor_cls()
        if not inst.is_instrumented_by_opentelemetry:
            inst.instrument()

    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))


def shutdown_tracing(timeout_millis: int = 30000):
    """Flush and close the configured tracer provider at CLI lifecycle exit."""
    global _tracing_provider
    provider = _tracing_provider
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis=timeout_millis)
    finally:
        provider.shutdown()
        _tracing_provider = None


def flush_tracing(timeout_millis: int = 30000):
    """Flush spans without ending the process-global tracing provider."""
    if _tracing_provider is not None:
        _tracing_provider.force_flush(timeout_millis=timeout_millis)


def setup_fsspec():
    import fsspec
    from . import package_filesystem
    fsspec.register_implementation(
        package_filesystem.PkgResourcesFileSystem.protocol,
        package_filesystem.PkgResourcesFileSystem,
    )


def fix_pytorch_240():
    """Fixes pytorch 2.4.0 libomp issue on Windows."""
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.submodule_search_locations is None:
        return
    for folder in torch_spec.submodule_search_locations:
        lib_folder = os.path.join(folder, "lib")
        test_file = os.path.join(lib_folder, "fbgemm.dll")
        dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
        if os.path.exists(dest):
            break

        try:
            with open(test_file, 'rb') as f:
                contents = f.read()
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                _ = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                logger.warning("Detected pytorch version with libomp issue, trying to patch")
                try:
                    shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
                except Exception as exc_info:
                    logger.error("While trying to patch a fix for torch 2.4.0, an error occurred", exc_info=exc_info)
        except Exception:
            pass


def setup_pre_torch(config: Configuration):
    """Must be called before torch import."""
    setup_environment()
    setup_debug_hang(config)
    setup_guess_settings(config)
    setup_windows_multi_gpu_defaults(config)
    setup_cuda_devices(config)
    setup_cuda_malloc(config)


def setup_post_torch(config: Configuration):
    # Tracing must be configured before imports such as model_management and
    # aimdo_integration touch functions decorated with the lazy main_pre tracer.
    # Otherwise that first access installs an exporter-less provider and the
    # CLI's configured endpoint arrives too late.
    setup_tracing(config)
    setup_warning_filters()
    setup_logging_filters()
    setup_logging(config)
    setup_fsspec()
    fix_pytorch_240()
    from .. import torchvision_compat  # noqa: F401
    # Activate dynamic VRAM (comfy-aimdo). Upstream activates this in root
    # main.py, but the fork's real startup runs through here, so the import
    # must live here or dynamic VRAM stays dormant. The module self-gates on
    # torch >= 2.8 and --disable-dynamic-vram, so this is a no-op when
    # unsupported or disabled.
    from .. import aimdo_integration  # noqa: F401
