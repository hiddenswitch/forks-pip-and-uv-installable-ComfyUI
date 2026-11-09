"""Execution events must survive history and RPC transport, not just WebSockets."""

import json
from dataclasses import asdict

import pytest

from comfy.component_model.queue_types import ExecutionError, ExecutionStatus, QueueItem, QueueTuple, public_queue_tuple
from comfy.distributed.distributed_types import RpcReply
from comfy.distributed.history import History
from comfy_execution.jobs import normalize_history_item


def test_rpc_preserves_error_events_and_public_history_excludes_inputs():
    status = ExecutionStatus('error', False, [
        ('execution_start', {'prompt_id': 'failed', 'timestamp': 100}),
        ('execution_error', {'prompt_id': 'failed', 'timestamp': 125,
                             'node_id': '2', 'node_type': 'LoadImage', 'executed': ['1'],
                             'exception_message': 'bad image', 'exception_type': 'ValueError',
                             'traceback': ['trace'], 'current_inputs': {'secret': ['private']}}),
    ])
    reply = RpcReply(**json.loads(json.dumps(asdict(RpcReply('failed', '', {}, status)))))
    invocation = reply.as_task_invocation()
    assert invocation.status.messages[1][1]['timestamp'] == 125
    assert 'bad image' in str(ExecutionError('failed', invocation.status))
    history = History()
    item = QueueItem(QueueTuple(0, 'failed', {}, {'token': 'private'}, [], {'api_key_comfy_org': 'private'}), None)
    history.put(item, {}, invocation.status)
    entry = history.copy('failed')['failed']
    assert 'private' not in json.dumps(entry)
    job = normalize_history_item('failed', entry)
    assert job['execution_start_time'] == 100
    assert job['execution_end_time'] == 125
    assert job['execution_error']['node_type'] == 'LoadImage'
    assert job['execution_error']['traceback'] == ['trace']
    # Sanitizing public history must not mutate the executor's diagnostics.
    assert status.messages[1][1]['current_inputs'] == {'secret': ['private']}


def test_history_pagination_unknown_ids_and_retention():
    history = History(maximum_history_size=4)
    for n in range(5):
        history.put(QueueItem(QueueTuple(n, str(n), {}, {}, [], {}), None), {}, ExecutionStatus('success', True, []))
    assert list(history.copy()) == ['1', '2', '3', '4']
    assert list(history.copy(max_items=2, offset=1)) == ['2', '3']
    assert list(history.copy(max_items=2)) == ['3', '4']
    assert history.copy('missing') == {}
    history.pop('missing')
    snapshot = history.copy('4')
    snapshot['4']['status']['messages'].append(('modified', {}))
    assert history.get_status('4').messages == []


@pytest.mark.parametrize('sensitive', [None, {}, {'other': 'value'}])
def test_queue_creation_and_public_representation(sensitive):
    extra_data = {'token': 'private', 'api_key_comfy_org': 'private', 'client_id': 'client'}
    item = QueueItem(QueueTuple(1, 'id', {}, extra_data, ['2'], sensitive), None)
    assert isinstance(item.extra_data['create_time'], int)
    assert item.sensitive['api_key_comfy_org'] == 'private'
    public = public_queue_tuple(item.queue_tuple)
    assert len(public) == 5
    assert public[3]['client_id'] == 'client'
    assert 'private' not in json.dumps(public)
    assert extra_data['api_key_comfy_org'] == 'private'


def test_interruption_stays_cancelled_in_jobs():
    history = History()
    history.put(QueueItem(QueueTuple(0, 'cancelled', {}, {}, []), None), {}, ExecutionStatus('error', False, [
        ('execution_start', {'prompt_id': 'cancelled', 'timestamp': 100}),
        ('execution_interrupted', {'prompt_id': 'cancelled', 'timestamp': 125, 'node_id': '1', 'node_type': 'KSampler', 'executed': []}),
    ]))
    job = normalize_history_item('cancelled', history.copy('cancelled')['cancelled'])
    assert job['status'] == 'cancelled'
    assert job['execution_end_time'] == 125
