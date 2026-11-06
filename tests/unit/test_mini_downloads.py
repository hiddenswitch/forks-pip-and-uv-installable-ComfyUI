"""Sparse transport must preserve native checkpoint metadata and required buffers."""
import httpx
import pytest
import torch
from huggingface_hub import constants, file_download
from safetensors import safe_open
from safetensors.torch import save_file

from tests.mini_downloads import miniature_downloads


@pytest.mark.parametrize('honor_range', [True, False])
def test_sparse_hub_download_preserves_buffers_and_restores_transport(tmp_path, monkeypatch, honor_range, request):
    source = tmp_path / 'source.safetensors'
    buffers = {'scale': torch.tensor(0.125), 'tokens': torch.arange(16, dtype=torch.int32)}
    save_file({'weight': torch.zeros(1024, 1024), **buffers}, str(source), metadata={'architecture': 'test'})
    size = source.stat().st_size
    transferred = []
    resolutions = []

    def serve(request):
        if request.url.host == 'checkpoint.test':
            assert request.headers['authorization'] == 'Bearer test-token'
            resolutions.append(request.url)
            return httpx.Response(302, headers={'Location': 'https://storage.test/weights'})
        assert 'authorization' not in request.headers
        start, end = map(int, request.headers['range'].removeprefix('bytes=').split('-'))
        if not honor_range:
            return httpx.Response(200)
        with source.open('rb') as f:
            f.seek(start)
            payload = f.read(end - start + 1)
        transferred.append(len(payload))
        return httpx.Response(206, content=payload, headers={'Content-Range': f'bytes {start}-{end}/{size}'})

    from huggingface_hub.utils import _http
    client = httpx.Client(transport=httpx.MockTransport(serve), follow_redirects=True)
    request.addfinalizer(client.close)
    monkeypatch.setattr(_http, 'get_session', lambda: client)
    original = file_download._download_to_tmp_and_move
    cache = constants.HF_HUB_CACHE
    destination = tmp_path / 'download.safetensors'
    with miniature_downloads(tmp_path):
        kwargs = dict(incomplete_path=tmp_path / 'incomplete', destination_path=destination,
                      url_to_download='https://checkpoint.test/model', headers={'authorization': 'Bearer test-token'},
                      expected_size=size, filename='model.safetensors', force_download=False, etag=None, xet_file_data=None)
        if not honor_range:
            with pytest.raises(ValueError, match='did not honor'):
                file_download._download_to_tmp_and_move(**kwargs)
            assert not destination.exists()
        else:
            file_download._download_to_tmp_and_move(**kwargs)
            assert destination.stat().st_size == size
            if hasattr(destination.stat(), 'st_blocks'):
                assert destination.stat().st_blocks * 512 < size // 10
            assert sum(transferred) < 4096
            with safe_open(destination, framework='pt') as reader:
                assert reader.metadata() == {'architecture': 'test'}
                assert reader.get_slice('weight').get_shape() == [1024, 1024]
                for key, value in buffers.items():
                    torch.testing.assert_close(reader.get_tensor(key), value)
            with pytest.raises(ValueError, match='outside disposable'):
                file_download._download_to_tmp_and_move(**{**kwargs, 'destination_path': tmp_path.parent / 'real-cache'})
    assert file_download._download_to_tmp_and_move is original
    assert constants.HF_HUB_CACHE == cache
    assert len(resolutions) == 1
    assert not list(tmp_path.glob('*.mini'))


@pytest.mark.parametrize('kind', ['gguf', 'pt'])
def test_sparse_archives_are_readable_by_native_loaders(tmp_path, kind):
    import gguf
    import numpy as np
    from tests.mini_downloads import sparse_gguf, sparse_torch_zip

    source, destination = tmp_path / ('source.' + kind), tmp_path / ('mini.' + kind)
    if kind == 'pt':
        torch.save({'weight': torch.ones(1024, 1024), 'buffer': torch.arange(16)}, source)
        copy = sparse_torch_zip
    else:
        writer = gguf.GGUFWriter(source, 'test')
        writer.add_name('metadata survives')
        writer.add_tensor('weight', np.ones((1024, 1024), dtype=np.float32))
        writer.add_tensor('buffer', np.arange(16, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        copy = sparse_gguf
    transferred = 0
    def read_range(offset, length):
        nonlocal transferred
        transferred += length
        with source.open('rb') as stream:
            stream.seek(offset)
            return stream.read(length)
    with destination.open('wb') as output:
        copy(read_range, output, source.stat().st_size)
    assert transferred < source.stat().st_size // 10
    if kind == 'pt':
        metadata = torch.load(destination, map_location='meta', weights_only=True)
        assert metadata['weight'].shape == (1024, 1024)
        assert metadata['weight'].is_meta
        torch.testing.assert_close(torch.load(destination, weights_only=True)['buffer'], torch.arange(16))
    else:
        reader = gguf.GGUFReader(destination)
        assert reader.get_field('general.name').contents() == 'metadata survives'
        tensors = {t.name: t for t in reader.tensors}
        assert list(tensors['weight'].shape) == [1024, 1024]
        np.testing.assert_array_equal(tensors['buffer'].data, np.arange(16, dtype=np.float32))
