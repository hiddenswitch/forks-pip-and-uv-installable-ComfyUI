import logging
import threading

import pytest
from pytest import fixture

from comfy_execution.graph_utils import GraphBuilder
from .common import ComfyClient, client_fixture


class _LogCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self.records.append(record)


class TestErrorLogging:
    client = fixture(client_fixture, scope="class", autouse=True, params=[
        {"extra_args": {}, "should_cache_results": True},
    ])

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    @fixture(autouse=True)
    def setup_comfy_logger(self):
        from comfy.app.logger import setup_logger
        setup_logger(log_level="DEBUG", capacity=300)

    @fixture
    def log_collector(self):
        collector = _LogCollector()
        collector.setLevel(logging.ERROR)
        root = logging.getLogger()
        root.addHandler(collector)
        yield collector
        root.removeHandler(collector)

    async def _run_error_workflow(self, client, builder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=256, width=256, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)
        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))
        with pytest.raises(Exception) as exc_info:
            await client.run(g)
        assert exc_info.value.args[0]["node_id"] == lazy_mix.id

    async def test_error_log_is_not_duplicated(self, client: ComfyClient, builder: GraphBuilder, log_collector: _LogCollector):
        await self._run_error_workflow(client, builder)

        node_error_records = [
            r for r in log_collector.records
            if "Node execution error" in r.message or "error occurred while executing" in r.message.lower()
        ]
        assert len(node_error_records) == 1, (
            f"Expected exactly 1 node error log, got {len(node_error_records)}:\n"
            + "\n---\n".join(r.message[:300] for r in node_error_records)
        )

    async def test_error_log_excludes_internal_frames(self, client: ComfyClient, builder: GraphBuilder, log_collector: _LogCollector):
        await self._run_error_workflow(client, builder)

        all_error_text = "\n".join(r.message for r in log_collector.records)

        assert all_error_text
        assert "asyncio/runners.py" not in all_error_text
        assert "concurrent/futures" not in all_error_text
        assert "threading.py" not in all_error_text
        assert "opentelemetry" not in all_error_text
        assert "embedded_comfy_client.py" not in all_error_text

    async def test_error_log_includes_node_context(self, client: ComfyClient, builder: GraphBuilder, log_collector: _LogCollector):
        await self._run_error_workflow(client, builder)

        all_error_text = "\n".join(r.message for r in log_collector.records)

        assert "TestLazyMixImages" in all_error_text
        assert "node_id=" in all_error_text

    async def test_error_log_includes_raising_frame(self, client: ComfyClient, builder: GraphBuilder, log_collector: _LogCollector):
        await self._run_error_workflow(client, builder)

        all_error_text = "\n".join(r.message for r in log_collector.records)

        assert "specific_tests.py" in all_error_text

    async def test_error_details_traceback_is_filtered(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=256, width=256, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)
        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        with pytest.raises(Exception) as exc_info:
            await client.run(g)
        error_data = exc_info.value.args[0]
        assert error_data["node_id"] == lazy_mix.id
        tb_text = "\n".join(error_data["traceback"])
        assert "specific_tests.py" in tb_text
        assert "embedded_comfy_client.py" not in tb_text
        assert "asyncio" not in tb_text.lower()
        assert "prompt_id" in error_data

    async def test_no_stack_trace_logger_duplication(self, client: ComfyClient, builder: GraphBuilder, log_collector: _LogCollector):
        await self._run_error_workflow(client, builder)

        assert log_collector.records
        for record in log_collector.records:
            assert record.stack_info is None, (
                f"Unexpected stack_info on record: {record.message[:200]}"
            )
