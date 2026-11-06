"""Scoped, sparse checkpoint downloads for miniature checkpoint tests.

Sparse files retain the native header and small buffers. They are not usable
model checkpoints and must never be written into a user's model cache.
"""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import inspect
import io
import json
import os
from pathlib import Path
import struct
from unittest.mock import patch
import uuid
import zipfile

import httpx


def sparse_safetensors(read_range, output, size):
    """Copy metadata and small buffers, leaving large tensor payloads sparse."""
    prefix = read_range(0, 8)
    header_size, = struct.unpack('<Q', prefix)
    if not 0 < header_size <= min(size - 8, 64 * 1024**2):
        raise ValueError('Invalid or oversized safetensors header')
    raw = read_range(8, header_size)
    header = json.loads(raw)
    data_start = 8 + header_size
    output.write(prefix + raw)
    output.truncate(size)
    retained = 0
    ranges = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        start, end = info['data_offsets']
        if not 0 <= start <= end <= size - data_start:
            raise ValueError(f'Invalid safetensors offsets: {name}')
        integer = info['dtype'].startswith(('I', 'U')) or info['dtype'] == 'BOOL'
        if len(info['shape']) <= 1 and (integer or end - start <= 65536):
            retained += end - start
            if retained > 64 * 1024**2:
                raise ValueError('Miniature checkpoint buffers exceed 64 MiB')
            if end > start:
                ranges.append((data_start + start, data_start + end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] <= 4096:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    with ThreadPoolExecutor(max_workers=8) as executor:
        chunks = executor.map(lambda span: read_range(span[0], span[1] - span[0]), merged)
        for (start, _), data in zip(merged, chunks):
            output.seek(start)
            output.write(data)


class RangeFile(io.RawIOBase):
    """Seekable, bounded-block reader for native archive metadata readers."""
    def __init__(self, read_range, size):
        self.read_range, self.size, self.position = read_range, size, 0
        self.block_offset, self.block = -1, b''

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError('Negative checkpoint offset')
        return self.position

    def tell(self):
        return self.position

    def read(self, length=-1):
        length = min(self.size - self.position, length if length >= 0 else self.size)
        if length > 64 * 1024**2:
            raise ValueError('Checkpoint metadata exceeds 64 MiB')
        result = bytearray()
        while len(result) < length:
            offset = self.position // 65536 * 65536
            if offset != self.block_offset:
                self.block = self.read_range(offset, min(65536, self.size - offset))
                self.block_offset = offset
            start = self.position - offset
            data = self.block[start:start + length - len(result)]
            if not data:
                raise ValueError('Truncated checkpoint metadata')
            result.extend(data)
            self.position += len(data)
        return bytes(result)


def sparse_gguf(read_range, output, size):
    """Retain the GGUF directory, metadata and one-dimensional native buffers."""
    import gguf

    source = RangeFile(read_range, size)
    def number(kind):
        return struct.unpack('<' + kind, source.read(struct.calcsize(kind)))[0]
    def string():
        return source.read(number('Q'))
    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            element, count = number('I'), number('Q')
            if count > 16_000_000:
                raise ValueError('Oversized GGUF metadata array')
            for _ in range(count):
                value(element)
            return None
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        return number(formats[kind])

    if source.read(4) != b'GGUF' or number('I') not in (2, 3):
        raise ValueError('Unsupported GGUF header')
    tensors, fields = number('Q'), number('Q')
    if tensors + fields > 1_000_000:
        raise ValueError('Oversized GGUF directory')
    alignment = 32
    for _ in range(fields):
        key = string()
        item = value(number('I'))
        if key == b'general.alignment':
            alignment = item
    for _ in range(tensors):
        string()
        dimensions = number('I')
        if dimensions > 8:
            raise ValueError('Unsupported GGUF tensor rank')
        source.read(8 * dimensions + 4 + 8)
    if not isinstance(alignment, int) or not 0 < alignment <= 65536:
        raise ValueError('Invalid GGUF alignment')
    header_end = (source.tell() + alignment - 1) // alignment * alignment
    if header_end > 64 * 1024**2:
        raise ValueError('GGUF header exceeds 64 MiB')
    output.write(read_range(0, header_end))
    output.truncate(size)
    output.flush()
    reader = gguf.GGUFReader(output.name)
    retained = 0
    for tensor in reader.tensors:
        if len(tensor.shape) <= 1:
            retained += tensor.n_bytes
            if retained > 64 * 1024**2:
                raise ValueError('GGUF buffers exceed 64 MiB')
            output.seek(tensor.data_offset)
            output.write(read_range(tensor.data_offset, tensor.n_bytes))


def sparse_torch_zip(read_range, output, size):
    """Keep native torch ZIP records and small storages; omit large weight payloads."""
    source = RangeFile(read_range, size)
    with zipfile.ZipFile(source) as archive:
        omitted = []
        for entry in archive.infolist():
            if '/data/' not in entry.filename or entry.file_size <= 65536:
                continue
            if entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError('Compressed torch storages are unsupported')
            source.seek(entry.header_offset)
            header = source.read(30)
            name_size, extra_size = struct.unpack_from('<HH', header, 26)
            start = entry.header_offset + 30 + name_size + extra_size
            omitted.append((start, start + entry.compress_size))
        cursor, retained = 0, 0
        for start, end in sorted(omitted) + [(size, size)]:
            retained += start - cursor
            if retained > 64 * 1024**2:
                raise ValueError('Torch checkpoint metadata exceeds 64 MiB')
            output.seek(cursor)
            if start > cursor:
                output.write(read_range(cursor, start - cursor))
            cursor = end
        output.truncate(size)


@contextlib.contextmanager
def miniature_downloads(root):
    """Intercept Hub transfers only; keep resolution, authentication and locks native.

    All generated files must reside below the caller's disposable test directory.
    Unsupported large formats fail before transfer instead of filling the disk.
    """
    from huggingface_hub import constants, file_download

    root = Path(root).resolve()
    cache = root / 'mini_hub'
    original = file_download._download_to_tmp_and_move
    signature = inspect.signature(original)

    def download(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        destination = Path(values['destination_path'])
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f'Miniature download outside disposable test directory: {destination}')
        if destination.exists() and not values['force_download']:
            return
        size = values['expected_size']
        filename = values['filename'].lower()
        formats = {'.safetensors': sparse_safetensors, '.sft': sparse_safetensors, '.gguf': sparse_gguf,
                   '.pt': sparse_torch_zip, '.pth': sparse_torch_zip, '.bin': sparse_torch_zip, '.ckpt': sparse_torch_zip}
        describe = formats.get(Path(filename).suffix)
        if describe is None:
            if size is None or size > 64 * 1024**2:
                raise ValueError(f'Miniature download needs a descriptor reader for {filename} ({size} bytes)')
            return original(*args, **kwargs)
        if size is None:
            raise ValueError(f'Missing checkpoint size: {filename}')
        temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.mini')
        try:
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                def read_range(offset, length):
                    headers = {**values['headers'], 'Range': f'bytes={offset}-{offset + length - 1}'}
                    with client.stream('GET', values['url_to_download'], headers=headers) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{offset + length - 1}/{size}':
                            raise ValueError(f'Server did not honor checkpoint range for {filename}')
                        data = bytearray()
                        for chunk in response.iter_bytes():
                            data.extend(chunk)
                            if len(data) > length:
                                raise ValueError('Checkpoint range exceeded requested size')
                        if len(data) != length:
                            raise ValueError('Truncated checkpoint range')
                        return bytes(data)
                with temporary.open('wb') as output:
                    describe(read_range, output, size)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    with patch.object(constants, 'HF_HUB_CACHE', str(cache)), \
         patch.object(file_download, '_download_to_tmp_and_move', download):
        yield
