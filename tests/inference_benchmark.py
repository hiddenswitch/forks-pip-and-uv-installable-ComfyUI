"""Local second-iteration measurements for real workflow tests."""
from __future__ import annotations

import json
import copy
from contextlib import ExitStack
from functools import wraps
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def timed_iterations(iterable, synchronize, record, clock=time.perf_counter, begin_step=lambda: None):
    for index, value in enumerate(iterable):
        if index == 1:
            synchronize()
            begin_step()
            started = clock()
        yield value
        if index == 1:
            synchronize()
            record(clock() - started)


def install_benchmark(path):
    # Also used as the spawn initializer: records come from the process doing
    # inference, not from the parent waiting on its Future.
    from comfy.k_diffusion import sampling
    from comfy.extra_samplers import uni_pc
    from comfy.execution_context import comfyui_execution_context
    from comfy import model_management
    from comfy.cli_args import args
    import torch

    from comfy.ldm.modules import attention

    observed = set()
    patches = ExitStack()

    def observe(function, name):
        @wraps(function)
        def measured(*values, **options):
            observed.add(name)
            return function(*values, **options)
        return measured

    for name, attribute in (
        ('SageAttention', 'attention_sage'), ('SageAttention3', 'attention3_sage'),
        ('FlashAttention', 'attention_flash'), ('xFormers', 'attention_xformers'),
        ('PyTorch SDPA', 'attention_pytorch'), ('split', 'attention_split'),
        ('sub-quadratic', 'attention_sub_quad'),
        ('Comfy Kitchen INT8', 'attention_comfy_kitchen_int8'),
        ('Comfy Kitchen SOL', 'attention_comfy_kitchen_sol'),
    ):
        original_attention = getattr(attention, attribute)
        measured = observe(original_attention, name)
        # The public container adapter is a separate entry point.
        if hasattr(original_attention, 'container_function'):
            measured.container_function = observe(original_attention.container_function, name)
        patches.enter_context(patch.object(attention, attribute, measured))

    def record(seconds):
        enabled = [name for name, flag in (
            ('Comfy Kitchen INT8', args.use_ck_attention),
            ('SageAttention', args.use_sage_attention),
            ('FlashAttention', args.use_flash_attention),
            ('PyTorch SDPA', args.use_pytorch_cross_attention),
            ('split', args.use_split_cross_attention),
        ) if flag]
        device = model_management.get_torch_device()
        row = dict(node_id=comfyui_execution_context.get().node_id, second_step_seconds=seconds, iterations_per_second=1 / seconds,
                   hardware=model_management.get_torch_device_name(device),
                   torch=torch.__version__, attention=', '.join(sorted(observed)) or 'not observed', configured_attention=', '.join(enabled) or 'automatic',
                   tp=args.tensor_parallel_size or "auto", pp=args.pipeline_parallel_size or "auto")
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')

    def synchronize():
        if model_management.get_torch_device().type == 'mps':
            torch.mps.synchronize()
        else:
            model_management.synchronize()

    def measured_range(original):
        def iterate(*values, **options):
            return timed_iterations(original(*values, **options), synchronize, record,
                                    begin_step=observed.clear)
        return iterate

    for module in (sampling, uni_pc):
        patches.enter_context(patch.object(module, 'trange', measured_range(module.trange)))
    return patches


def benchmark_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def pytest_addoption(parser):
    parser.addoption('--inference-benchmark-json', help='Write real inference measurements as JSON')


def pytest_terminal_summary(terminalreporter):
    rows = []
    outcomes = {report.nodeid: report.outcome for reports in terminalreporter.stats.values()
                for report in reports if getattr(report, 'when', None) == 'call'}
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, 'when', None) != 'teardown':
                continue
            for name, value in report.user_properties:
                if name == 'inference_benchmark':
                    rows.append((report.nodeid, outcomes.get(report.nodeid, report.outcome), value))
    destination = terminalreporter.config.getoption('--inference-benchmark-json')
    if destination:
        Path(destination).write_text(json.dumps([
            dict(test=name, outcome=outcome, **row) for name, outcome, row in rows
        ], indent=2) + '\n', encoding='utf-8')
    if not rows:
        return
    terminalreporter.section('Inference benchmarks: second completed sampling iteration')
    terminalreporter.write_line('| Test / sampler node | Result | GPU / hardware | Requested TP / PP | Observed attention (configuration) | Step 2 (s) | it/s | Torch |')
    terminalreporter.write_line('|---|---|---|---|---|---:|---:|---|')
    for name, outcome, row in rows:
        terminalreporter.write_line(
            f"| {name} / {row['node_id']} | {outcome} | {row['hardware']} | {row['tp']} / {row['pp']} | "
            f"{row['attention']} ({row['configured_attention']}) | {row['second_step_seconds']:.6f} | "
            f"{row['iterations_per_second']:.6f} | {row['torch']} |")
    terminalreporter.write_line('Synchronized wall time; excludes the first iteration. Attention reports observed backend entry points; uninstrumented custom/distributed calls are not inferred from installed packages.')


@pytest.fixture
def inference_benchmark(request, tmp_path):
    path = tmp_path / 'sampling-steps.jsonl'
    with install_benchmark(path):
        yield
    for row in benchmark_rows(path):
        request.node.user_properties.append(('inference_benchmark', row))


def prepare_benchmark_prompt(prompt):
    prompt = copy.deepcopy(prompt)
    for node in prompt.values():
        inputs = node.get('inputs', {})
        for field in ('steps', 'num_inference_steps', 'inference_steps'):
            if field in inputs:
                value = inputs[field]
                inputs[field] = max(2, value) if isinstance(value, int) else 2
        if node.get('class_type') == 'KSamplerAdvanced':
            inputs['start_at_step'] = 0
            inputs['end_at_step'] = 2
    return prompt
