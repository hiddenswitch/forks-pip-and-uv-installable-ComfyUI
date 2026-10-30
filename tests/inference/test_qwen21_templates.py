import json

import pytest
from PIL import Image
from comfyui_workflow_templates import get_asset_path, iter_templates

from comfy.cli_args import default_configuration
from comfy.component_model.workflow_convert import convert_ui_to_api
from comfy.execution_context import context_add_custom_nodes, context_configuration
from comfy.nodes.package import import_all_nodes_in_workspace
from .test_workflows import client  # noqa: F401


def _convert_template(template_id, nodes):
    template = next(t for t in iter_templates() if t.template_id == template_id)
    asset = next(a for a in template.assets if a.filename.endswith('.json'))
    with open(get_asset_path(template_id, asset.filename), encoding='utf-8') as stream:
        return convert_ui_to_api(json.load(stream), node_mappings=nodes)


def _ancestors(prompt, outputs):
    selected = {}

    def visit(node_id):
        if node_id in selected:
            return
        node = selected[node_id] = prompt[node_id]
        for value in node['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in prompt:
                visit(value[0])

    for node_id in outputs:
        visit(node_id)
    return selected


@pytest.mark.inference
@pytest.mark.requires_weights
@pytest.mark.requires_accelerator
@pytest.mark.asyncio
async def test_qwen21_templates_generate_and_edit_rgba(has_gpu, client):
    if not has_gpu:
        pytest.skip('requires gpu')
    configuration = default_configuration()
    configuration.disable_all_custom_nodes = True
    with context_configuration(configuration):
        nodes = import_all_nodes_in_workspace()
        with context_add_custom_nodes(nodes):
            generation = _convert_template('image_qwen_image_2_1_t2i', nodes)
            editing = _convert_template('image_qwen_image_2_1_image_edit', nodes)

    for prompt in (generation, editing):
        prompt['459:458']['inputs'].update(steps=2, seed=42)
        prompt['459:456']['inputs'].update(width=256, height=256)
        prompt['461'] = {'class_type': 'SaveImage', 'inputs': {
            'images': ['459:457', 0], 'filename_prefix': 'qwen21-template',
        }}
    generation['459:452']['inputs'].update(
        prompt='This is an RGBA format image with transparency. A red ceramic cup. The image has an alpha channel and a transparent background.',
        resolution=256,
    )
    editing['459:474']['inputs'].update(prompt='Make the cup blue', resolution=256)
    editing['459:474']['inputs'].pop('images.image_2')
    editing['459:474']['inputs']['images.image_1'] = ['generation:459:457', 0]

    combined = {}
    for node_id, node in _ancestors(generation, ['461']).items():
        inputs = {key: ['generation:' + value[0], value[1]]
                  if isinstance(value, list) and len(value) == 2 and value[0] in generation else value
                  for key, value in node['inputs'].items()}
        combined['generation:' + node_id] = {**node, 'inputs': inputs}
    combined.update(_ancestors(editing, ['461']))
    outputs = await client.queue_prompt(combined)
    for node_id in ('generation:461', '461'):
        with Image.open(outputs[node_id]['images'][0]['abs_path']) as image:
            assert image.mode == 'RGBA'
            assert image.size == (256, 256)
