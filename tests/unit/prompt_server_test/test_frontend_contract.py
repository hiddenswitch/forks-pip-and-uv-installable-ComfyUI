"""Frontend contracts exercised through the production CPU server and native nodes."""

import asyncio
import io
import socket
import uuid
from urllib.parse import quote

import aiohttp
import pytest
import requests
from PIL import Image

from comfy.cli_args import default_configuration
from tests.conftest import comfy_background_server_from_config

pytestmark = [pytest.mark.server_process, pytest.mark.xdist_group(name='server-process')]


@pytest.fixture(scope='module')
def api_server(tmp_path_factory):
    base = tmp_path_factory.mktemp('frontend-contract')
    configuration = default_configuration()
    configuration.cpu = True
    configuration.disable_all_custom_nodes = True
    configuration.disable_partner_nodes = True
    configuration.enable_assets = True
    configuration.base_directory = str(base)
    configuration.input_directory = str(base / 'input')
    configuration.output_directory = str(base / 'output')
    configuration.user_directory = str(base / 'user')
    configuration.database_url = f'sqlite:///{base}/user/comfyui.db'
    configuration.listen = '127.0.0.1'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        configuration.port = sock.getsockname()[1]
    for _ in comfy_background_server_from_config(configuration):
        yield f'http://127.0.0.1:{configuration.port}', base


@pytest.mark.parametrize('fail', [False, True])
async def test_execution_events_match_websocket_history_and_jobs(api_server, fail):
    base_url, base = api_server
    if fail:
        (base / 'input' / 'broken.png').write_bytes(b'not an image')
        prompt = {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'broken.png'}},
                  '2': {'class_type': 'PreviewImage', 'inputs': {'images': ['1', 0]}}}
    else:
        prompt = {'1': {'class_type': 'PrimitiveFloat', 'inputs': {'value': 1.0}},
                  '2': {'class_type': 'PreviewAny', 'inputs': {'source': ['1', 0]}}}
    client_id = str(uuid.uuid4())
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f'{base_url}/ws?clientId={client_id}') as ws:
            async with session.post(f'{base_url}/api/prompt', json={'prompt': prompt, 'client_id': client_id}) as response:
                assert response.status == 200, await response.text()
                prompt_id = (await response.json())['prompt_id']
            terminal = 'execution_error' if fail else 'execution_success'
            events = []
            async with asyncio.timeout(30):
                async for message in ws:
                    event = message.json()
                    if event['type'].startswith('execution_'):
                        events.append([event['type'], event['data']])
                    if event['type'] == terminal:
                        break
                while True:
                    async with session.get(f'{base_url}/api/history/{prompt_id}') as response:
                        history = await response.json()
                    if prompt_id in history:
                        break
                    await asyncio.sleep(.05)
            stored = history[prompt_id]['status']
            assert stored['completed'] is not fail
            assert stored['status_str'] == ('error' if fail else 'success')
            # Inputs may contain credentials/tensors; history retains the other event fields.
            public_events = [[name, {k: v for k, v in data.items() if k != 'current_inputs'}] for name, data in events]
            assert stored['messages'] == public_events
            assert all(isinstance(data['timestamp'], int) for _, data in events)
            async with session.get(f'{base_url}/api/jobs/{prompt_id}') as response:
                job = await response.json()
            assert job['status'] == ('failed' if fail else 'completed')
            assert job['create_time'] <= job['execution_start_time'] <= job['execution_end_time']
            assert job['execution_start_time'] == events[0][1]['timestamp']
            assert job['execution_end_time'] == events[-1][1]['timestamp']
            if fail:
                assert job['execution_error']['node_id'] == '1'
                assert job['execution_error']['node_type'] == 'LoadImage'
                assert job['execution_error']['traceback']
            else:
                assert job['outputs']['2']['text'] == ['1.0']
            async with session.get(f'{base_url}/api/jobs') as response:
                listing = await response.json()
            assert isinstance(listing['pagination']['limit'], int)
            assert any(item['id'] == prompt_id for item in listing['jobs'])


def test_settings_userdata_upload_and_view_contract(api_server):
    base_url, _ = api_server
    session = requests.Session()
    url = f'{base_url}/api'
    setting = 'Comfy.Test.FrontendContract'
    response = session.post(f'{url}/settings/{setting}', json={'enabled': True}, timeout=10)
    assert response.status_code == 200
    assert session.get(f'{url}/settings/{setting}', timeout=10).json() == {'enabled': True}
    assert session.get(f'{url}/settings', timeout=10).json()[setting] == {'enabled': True}
    name = 'workflows/contract space.json'
    path = f'{url}/userdata/{quote(name, safe="")}'
    workflow = {'nodes': [], 'links': [], 'version': .4}
    assert session.post(path, json=workflow, timeout=10).status_code == 200
    assert session.get(path, timeout=10).json() == workflow
    listing = session.get(f'{url}/userdata', params={'dir': 'workflows', 'recurse': 'true', 'full_info': 'true'}, timeout=10)
    assert listing.status_code == 200
    assert any(item['path'] == 'contract space.json' for item in listing.json())
    assert session.delete(path, timeout=10).status_code == 204
    assert session.get(path, timeout=10).status_code == 404
    image = io.BytesIO()
    Image.new('RGB', (3, 2), 'red').save(image, 'PNG')
    uploaded = session.post(f'{url}/upload/image', files={'image': ('contract.png', image.getvalue(), 'image/png')}, timeout=10)
    assert uploaded.status_code == 200
    ref = uploaded.json()
    assert ref['name'] == 'contract.png' and ref['type'] == 'input'
    viewed = session.get(f'{url}/view', params={'filename': ref['name'], 'subfolder': ref['subfolder'], 'type': ref['type']}, timeout=10)
    assert viewed.status_code == 200
    assert Image.open(io.BytesIO(viewed.content)).size == (3, 2)
    assert session.get(f'{url}/history/missing-id', timeout=10).json() == {}
    assert session.get(f'{url}/jobs/missing-id', timeout=10).status_code == 404
    for query in ['limit=abc', 'limit=0', 'offset=abc', 'sort_by=unknown', 'status=unknown']:
        assert session.get(f'{url}/jobs?{query}', timeout=10).status_code == 400


def test_native_node_metadata_has_frontend_types(api_server):
    base_url, _ = api_server
    response = requests.get(f'{base_url}/api/object_info', timeout=30)
    assert response.status_code == 200
    definitions = response.json()
    assert 'PrimitiveFloat' in definitions and 'LoadImage' in definitions
    for name, definition in definitions.items():
        assert isinstance(definition['display_name'], str), name
        for key in ('name', 'description', 'category', 'python_module'):
            assert isinstance(definition[key], str), (name, key)
        for key in ('search_aliases', 'output_tooltips', 'output_matchtypes'):
            if key in definition:
                assert isinstance(definition[key], list), (name, key)
                assert all(isinstance(value, str) for value in definition[key]), (name, key)
        assert all(isinstance(value, bool) for value in definition['output_is_list']), name
        for key in ('price_badge', 'essentials_category'):
            assert key not in definition or definition[key] is not None, (name, key)
    single = requests.get(f'{base_url}/api/object_info/PrimitiveFloat', timeout=10)
    assert single.json() == {'PrimitiveFloat': definitions['PrimitiveFloat']}


def test_v1_reports_real_execution_failure_and_serves_native_images(api_server):
    base_url, base = api_server
    (base / 'input' / 'v1-broken.png').write_bytes(b'not an image')
    prompt = {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'v1-broken.png'}},
              '2': {'class_type': 'PreviewImage', 'inputs': {'images': ['1', 0]}}}
    response = requests.post(f'{base_url}/api/v1/prompts', json=prompt, headers={'Accept': 'application/json'}, timeout=30)
    assert response.status_code == 500
    status = response.json()
    assert status['status_str'] == 'error'
    assert status['messages'][-1][0] == 'execution_error'
    Image.new('RGB', (3, 2), 'red').save(base / 'input' / 'v1-good.png')
    prompt['1']['inputs']['image'] = 'v1-good.png'
    for accept in ('application/json', 'image/png'):
        response = requests.post(f'{base_url}/api/v1/prompts', json=prompt, headers={'Accept': accept}, timeout=30)
        assert response.status_code == 200, response.text
        if accept == 'image/png':
            assert Image.open(io.BytesIO(response.content)).size == (3, 2)
        else:
            assert response.json()['urls']
            for url in response.json()['urls']:
                assert requests.get(url, timeout=10).status_code == 200
