from tests.inference_benchmark import timed_iterations, benchmark_rows, prepare_benchmark_prompt


def test_second_iteration_includes_completed_device_work_only():
    events = []
    times = iter([10, 13])
    rows = []
    for value in timed_iterations(range(3), lambda: events.append('sync'), rows.append, lambda: next(times)):
        events.append(value)
    assert events == [0, 'sync', 1, 'sync', 2]
    assert rows == [3]


def test_incomplete_second_iteration_has_no_benchmark():
    rows = []
    iterator = timed_iterations(range(2), lambda: None, rows.append)
    next(iterator)
    next(iterator)
    iterator.close()
    assert rows == []


def test_single_step_does_not_invent_second_step():
    rows = []
    assert list(timed_iterations(range(1), lambda: None, rows.append)) == [0]
    assert rows == []


def test_missing_record_is_not_reported_as_zero(tmp_path):
    assert benchmark_rows(tmp_path / 'absent.jsonl') == []


def test_two_steps_cover_linked_and_model_specific_schedulers_without_changing_resolution():
    prompt = {'1': {'class_type': 'Ideogram4Scheduler', 'inputs': {'steps': ['2', 0]}},
              '2': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'resolution_steps': 1}},
              '3': {'class_type': 'KSamplerAdvanced', 'inputs': {'steps': 1, 'start_at_step': 10, 'end_at_step': 20}}}
    changed = prepare_benchmark_prompt(prompt)
    assert changed['1']['inputs']['steps'] == 2
    assert changed['2'] == prompt['2']
    assert changed['3']['inputs'] == {'steps': 2, 'start_at_step': 0, 'end_at_step': 2}
    assert prompt['1']['inputs']['steps'] == ['2', 0]


def test_existing_longer_sampling_is_preserved():
    prompt = {'1': {'class_type': 'KSampler', 'inputs': {'steps': 20}}}
    assert prepare_benchmark_prompt(prompt) == prompt
