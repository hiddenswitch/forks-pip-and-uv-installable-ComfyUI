"""
Tests for URI handling in model_downloader.

Tests that hf:// URIs are properly detected and downloaded to local cache,
returning local file paths that work with downstream code.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import requests
from huggingface_hub import CacheNotFound
from huggingface_hub.utils import LocalEntryNotFoundError
from unittest.mock import patch

from comfy import model_downloader
from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import FolderPathsTuple
from comfy.component_model.uris import is_hf_uri, is_uri
from comfy.model_downloader import parse_hf_uri, get_or_download
from comfy.model_downloader_types import HuggingFile


class TestIsHfUri:
    """Tests for the is_hf_uri helper function."""

    @pytest.mark.parametrize("uri", [
        "hf://Comfy-Org/flux1-schnell/flux1-schnell-fp8.safetensors",
        "hf://gpt2/config.json",
        "hf://datasets/squad/train.json",
        "hf://spaces/gradio/hello_world/app.py",
    ])
    def test_valid_hf_uris(self, uri):
        """Test that valid hf:// URIs are detected."""
        assert is_hf_uri(uri) is True

    @pytest.mark.parametrize("path", [
        "https://example.com/model.safetensors",
        "s3://bucket/path/model.safetensors",
        "model.safetensors",
        "/path/to/model.safetensors",
        "relative/path/model.safetensors",
        "",
    ])
    def test_non_hf_uris(self, path):
        """Test that non-hf:// paths are not detected as HF URIs."""
        assert is_hf_uri(path) is False


class TestIsUri:
    """Tests for the is_uri helper function."""

    @pytest.mark.parametrize("uri", [
        "hf://gpt2/config.json",
        "https://example.com/model.safetensors",
        "http://example.com/model.safetensors",
        "s3://bucket/path/model.safetensors",
        "file:///path/to/file",
    ])
    def test_valid_uris(self, uri):
        """Test that valid URIs are detected."""
        assert is_uri(uri) is True

    @pytest.mark.parametrize("path", [
        "model.safetensors",
        "/path/to/model.safetensors",
        "relative/path/model.safetensors",
        "://missing-scheme",
        "",
    ])
    def test_non_uris(self, path):
        """Test that non-URIs are not detected."""
        assert is_uri(path) is False


class TestParseHfUri:
    """Tests for the parse_hf_uri function."""

    def test_parses_simple_repo(self):
        """Test parsing a simple repo/file URI."""
        uri = "hf://gpt2/config.json"
        hf_file = parse_hf_uri(uri)

        assert isinstance(hf_file, HuggingFile)
        assert hf_file.repo_id == "gpt2"
        assert hf_file.filename == "config.json"
        assert hf_file.repo_type == "model"

    def test_parses_org_repo(self):
        """Test parsing an org/repo/file URI."""
        uri = "hf://Comfy-Org/flux1-schnell/flux1-schnell-fp8.safetensors"
        hf_file = parse_hf_uri(uri)

        assert isinstance(hf_file, HuggingFile)
        assert hf_file.repo_id == "Comfy-Org/flux1-schnell"
        assert hf_file.filename == "flux1-schnell-fp8.safetensors"
        assert hf_file.repo_type == "model"

    def test_parses_nested_path(self):
        """Test parsing a URI with nested path."""
        uri = "hf://Comfy-Org/flux1-schnell/split_files/diffusion_models/flux1-schnell-fp8.safetensors"
        hf_file = parse_hf_uri(uri)

        assert hf_file.repo_id == "Comfy-Org/flux1-schnell"
        assert hf_file.filename == "split_files/diffusion_models/flux1-schnell-fp8.safetensors"

    def test_parses_a_repo_name_with_a_dot(self):
        """Repo names such as Qwen-Image-2.1 and FLUX.2-dev are not file names."""
        for uri, repo, filename in (
                ("hf://Comfy-Org/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors",
                 "Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_bf16.safetensors"),
                ("hf://black-forest-labs/FLUX.2-dev/flux2-dev.safetensors",
                 "black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors")):
            hf_file = parse_hf_uri(uri)
            assert (hf_file.repo_id, hf_file.filename) == (repo, filename)

    def test_parses_dataset_uri(self):
        """Test parsing a datasets URI."""
        uri = "hf://datasets/squad/data/train.json"
        hf_file = parse_hf_uri(uri)

        assert hf_file.repo_id == "squad/data"
        assert hf_file.filename == "train.json"
        assert hf_file.repo_type == "datasets"


class TestGetOrDownloadWithHfUri:
    """Tests for get_or_download handling of hf:// URIs."""

    def test_hf_uri_returns_local_path(self, cached_gpt2_hf_download):
        """Test that hf:// URIs return a local cached path string."""
        uri = "hf://gpt2/config.json"
        result = get_or_download("checkpoints", uri)

        assert isinstance(result, str)
        assert os.path.isfile(result)

    def test_hf_uri_file_is_usable(self, cached_gpt2_hf_download):
        """Test that the returned path can be used with standard file operations."""
        uri = "hf://gpt2/config.json"
        result = get_or_download("checkpoints", uri)

        with open(result, "r") as f:
            content = f.read()

        assert len(content) > 0
        data = json.loads(content)
        assert "model_type" in data

    def test_http_error_without_response_returns_none(self, monkeypatch, cached_gpt2_hf_download):
        """Connection failures do not always carry an HTTP response."""
        def fail_hf_hub_download(**kwargs):
            if kwargs.get("local_files_only"):
                raise LocalEntryNotFoundError("not cached")
            raise requests.exceptions.HTTPError("connection failed")

        monkeypatch.setattr(model_downloader, "hf_hub_download", fail_hf_hub_download)
        assert get_or_download("checkpoints", "hf://gpt2/config.json") is None

    @patch("comfy.model_downloader.folder_paths.get_full_path")
    def test_local_filename_uses_normal_lookup(self, mock_get_full_path):
        """Test that local filenames use normal path lookup."""
        import comfy.model_downloader as md
        mock_get_full_path.return_value = "/path/to/model.safetensors"

        saved = getattr(md, '_original_get_full_path', None)
        try:
            if hasattr(md, '_original_get_full_path'):
                delattr(md, '_original_get_full_path')
            result = get_or_download("checkpoints", "model.safetensors")
        finally:
            if saved is not None:
                md._original_get_full_path = saved

        assert result == "/path/to/model.safetensors"
        mock_get_full_path.assert_called_once()

    def test_existing_ideogram_plain_filename_is_found_before_download(self, tmp_path):
        models_dir = tmp_path / "diffusion_models"
        models_dir.mkdir()
        expected = models_dir / "ideogram4_fp8_scaled.safetensors"
        expected.write_bytes(b"already here")

        previous = folder_paths.folder_names_and_paths.get("diffusion_models")
        folder_paths.folder_names_and_paths["diffusion_models"] = FolderPathsTuple(
            "diffusion_models",
            [str(models_dir)],
            {".safetensors"},
        )
        known_file = HuggingFile(
            "Comfy-Org/Ideogram-4",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            save_with_filename="ideogram4_fp8_scaled.safetensors",
            alternate_filenames=("ideogram4_fp8_scaled.safetensors",),
        )
        try:
            with patch("comfy.model_downloader.hf_hub_download") as hf_hub_download:
                result = get_or_download(
                    "diffusion_models",
                    "ideogram4_fp8_scaled.safetensors",
                    known_files=[known_file],
                )
        finally:
            if previous is None:
                del folder_paths.folder_names_and_paths["diffusion_models"]
            else:
                folder_paths.folder_names_and_paths["diffusion_models"] = previous

        assert result == str(expected)
        hf_hub_download.assert_not_called()

    def test_offline_uses_hf_cache_without_linking(self, tmp_path):
        models_dir = tmp_path / "checkpoints"
        models_dir.mkdir()
        cached = tmp_path / "hf-cache" / "ideogram4_fp8_scaled.safetensors"
        cached.parent.mkdir()
        cached.write_bytes(b"cached")

        previous = folder_paths.folder_names_and_paths.get("checkpoints")
        folder_paths.folder_names_and_paths["checkpoints"] = FolderPathsTuple(
            "checkpoints",
            [str(models_dir)],
            {".safetensors"},
        )
        known_file = HuggingFile(
            "Comfy-Org/Ideogram-4",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            save_with_filename="ideogram4_fp8_scaled.safetensors",
            alternate_filenames=("ideogram4_fp8_scaled.safetensors",),
        )
        try:
            with patch("comfy.model_downloader.hf_hub_download", return_value=str(cached)) as hf_hub_download:
                result = get_or_download(
                    "checkpoints",
                    "ideogram4_fp8_scaled.safetensors",
                    known_files=[known_file],
                    offline=True,
                )
        finally:
            if previous is None:
                del folder_paths.folder_names_and_paths["checkpoints"]
            else:
                folder_paths.folder_names_and_paths["checkpoints"] = previous

        assert result == str(cached)
        assert not (models_dir / "ideogram4_fp8_scaled.safetensors").exists()
        hf_hub_download.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_is_not_hf_uri(self):
        """Test that empty string is not detected as HF URI."""
        assert is_hf_uri("") is False

    def test_hf_prefix_only(self):
        """Test that 'hf://' alone is detected."""
        assert is_hf_uri("hf://") is True


def test_huggingface_repo_list_with_no_cache(monkeypatch, tmp_path):
    """A clean machine without a Hugging Face cache still lists known repos."""
    def no_cache(cache_dir=None):
        raise CacheNotFound("cache is absent", cache_dir or tmp_path)

    original_get_folder_paths = model_downloader.folder_paths.get_folder_paths
    monkeypatch.setattr(model_downloader, "scan_cache_dir", no_cache)
    monkeypatch.setattr(
        model_downloader.folder_paths,
        "get_folder_paths",
        lambda folder_name: []
        if folder_name in {"huggingface", "huggingface_cache"}
        else original_get_folder_paths(folder_name),
    )

    assert set(model_downloader.get_huggingface_repo_list()) == model_downloader.KNOWN_HUGGINGFACE_MODEL_REPOS


@pytest.fixture
def local_downloads(tmp_path, monkeypatch):
    destination = tmp_path / "models"
    destination.mkdir()
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "checkpoints",
        FolderPathsTuple("checkpoints", [str(destination)], {".safetensors"}),
    )
    requested = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            if self.path == "/redirect.safetensors":
                self.send_response(302)
                self.send_header("Location", "/200.safetensors")
                self.end_headers()
                return
            status = int(Path(self.path).stem)
            body = b"tiny model bytes" if status == 200 else b"error response"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with requests.Session() as session, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        monkeypatch.setattr(model_downloader, "_session", session)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", destination, requested
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("filename", ["200.safetensors", "redirect.safetensors"])
def test_local_http_download_and_cache(local_downloads, filename):
    base, destination, requested = local_downloads
    result = get_or_download("checkpoints", f"{base}/{filename}")
    assert Path(result) == destination / filename
    assert Path(result).read_bytes() == b"tiny model bytes"
    requests_before = list(requested)
    assert get_or_download("checkpoints", f"{base}/{filename}") == result
    assert requested == requests_before


@pytest.mark.parametrize("status", [400, 401, 404, 500])
def test_local_http_errors_do_not_cache_error_pages(local_downloads, status):
    base, destination, requested = local_downloads
    filename = f"{status}.safetensors"
    with pytest.raises(requests.HTTPError) as error:
        get_or_download("checkpoints", f"{base}/{filename}")
    assert error.value.response.status_code == status
    assert requested == [f"/{filename}"]
    assert not (destination / filename).exists()


def test_offline_http_does_not_contact_server(local_downloads):
    base, destination, requested = local_downloads
    assert get_or_download("checkpoints", f"{base}/200.safetensors", offline=True) is None
    assert requested == []
    assert list(destination.iterdir()) == []


def test_file_uri_copies_local_bytes(local_downloads, tmp_path):
    _, destination, requested = local_downloads
    source = tmp_path / "local.safetensors"
    source.write_bytes(b"local tensor fixture")
    result = get_or_download("checkpoints", source.as_uri())
    assert Path(result) == destination / source.name
    assert Path(result).read_bytes() == source.read_bytes()
    assert requested == []
