"""Benchmark command records must not persist inherited credentials."""
import ast
from pathlib import Path


def test_benchmark_records_only_its_public_environment_settings():
    source = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_custom_node_tp.py"
    tree = ast.parse(source.read_text())
    expressions = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                   for key, value in zip(node.keys, node.values)
                   if isinstance(key, ast.Constant) and key.value == "environment"]
    assert len(expressions) == 1
    environment = {
        "NCCL_DEBUG": "INFO", "HF_HUB_OFFLINE": "1",
        "HF_TOKEN": "private-token-sentinel", "HF_OTHER_CREDENTIAL": "private-sentinel",
        "NCCL_UNRELATED_SECRET": "private-sentinel", "OTHER_SECRET": "private-sentinel",
    }
    recorded = eval(compile(ast.Expression(expressions[0]), str(source), "eval"), {"env": environment})  # noqa: S307 -- Repository expression; synthetic environment only.
    assert recorded == {"NCCL_DEBUG": "INFO", "HF_HUB_OFFLINE": "1"}
