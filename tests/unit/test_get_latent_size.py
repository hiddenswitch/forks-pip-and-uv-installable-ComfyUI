import pytest
import torch

from comfy.component_model.executor_types import ExecutorToClientProgress
from comfy.execution_context import context_execute_prompt
from comfy.progress_types import ProgressRegistryStub
from comfy_api.latest import IO
from comfy_extras.nodes.nodes_images import GetLatentSize
from comfy_extras.nodes.nodes_audio import GetAudioLatentSize


class _RecordingServer(ExecutorToClientProgress):
    def __init__(self):
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.texts = []

    @property
    def receive_all_progress_notifications(self):
        return True

    def send_sync(self, event, data, sid=None):
        pass

    def send_progress_text(self, text, node_id, sid=None):
        self.texts.append((text, node_id))


@pytest.mark.parametrize("node_class,shape,expected,text", [
    (GetLatentSize, (2, 16, 3, 8, 12), (12, 8, 3, 2), "width: 12, height: 8\n frames: 3, batch size: 2"),
    (GetLatentSize, (2, 16, 8, 12), (12, 8, 1, 2), "width: 12, height: 8\n frames: 1, batch size: 2"),
    (GetAudioLatentSize, (2, 64, 2, 12), (12, 2, 2), "length: 12, audio channels: 2\n batch size: 2"),
    (GetAudioLatentSize, (2, 64, 12), (12, 1, 2), "length: 12, audio channels: 1\n batch size: 2"),
])
def test_get_latent_size_reports_to_execution_server(node_class, shape, expected, text):
    server = _RecordingServer()
    node = node_class.PREPARE_CLASS_CLONE({"hidden_inputs": {IO.Hidden.unique_id: "7"}})
    with context_execute_prompt(server, "prompt", ProgressRegistryStub()):
        out = node.execute({"samples": torch.zeros(shape)})
    assert out.result == expected
    assert server.texts == [(text, "7")]
