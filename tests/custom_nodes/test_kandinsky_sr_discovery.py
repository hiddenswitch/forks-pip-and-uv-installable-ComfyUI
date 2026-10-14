import importlib
import json
from importlib.metadata import distribution
from pathlib import Path

import pytest

from comfy import model_downloader
from comfy.model_downloader_types import HuggingFile
from comfy.nodes.download_interception import patch_folder_paths_functions
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment


@pytest.mark.parametrize("missing_sr_folder", [False, True])
def test_sr_component_discovery_does_not_download_catalog_models(tmp_path, monkeypatch, missing_sr_folder):
    entry = next(ep for ep in distribution("kandinsky6-sr").entry_points if ep.group == "comfyui.custom_nodes").load()
    root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "kandinsky-6-sr"
    prepare_vanilla_environment()
    assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
    nodes = importlib.import_module(root.name + ".kandinsky6_vsr.nodes")
    checkpoint = tmp_path / "diffusion_pytorch_model.safetensors"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(json.dumps({"vae_type": nodes.VAE["name"]}))
    name = "sr/vae/diffusion_pytorch_model.safetensors"
    monkeypatch.setattr(model_downloader, "_get_known_models_for_folder_name", lambda folder: [HuggingFile(repo_id="unrelated/model", filename="weights.safetensors")])

    def resolve(folder, filename, **kwargs):
        assert filename == name, "SR attempted to resolve an unrelated catalog model"
        return str(checkpoint)

    monkeypatch.setattr(model_downloader, "get_or_download", resolve)

    def existing(folder):
        if folder == "diffusion_models":
            return [name]
        if missing_sr_folder:
            raise KeyError(folder)
        return []

    with patch_folder_paths_functions():
        monkeypatch.setattr(model_downloader, "_original_get_filename_list", existing)
        assert nodes._list_kvae_checkpoints() == [name]
