import pytest
import torch
from PIL import Image

from comfy_api.latest import _ui


@pytest.mark.parametrize("format", ["webp", "png"])
@pytest.mark.parametrize("frame_count, expected_ms", [(24, 1000), (241, 10042)])
def test_save_animation_preserves_24_fps_with_repeated_frames(tmp_path, monkeypatch, format, frame_count, expected_ms):
    monkeypatch.setattr(_ui.folder_paths, "get_output_directory", lambda: str(tmp_path))
    # Each source frame is held for two ticks, which the encoder can coalesce.
    values = (torch.arange(frame_count) // 2) % 2
    frames = values[:, None, None, None].expand(frame_count, 4, 4, 3).float()
    args = (frames, "clip", _ui.FolderType.output, None, 24)
    if format == "webp":
        result = _ui.ImageSaveHelper.save_animated_webp(*args, lossless=True, quality=80, method=0)
    else:
        result = _ui.ImageSaveHelper.save_animated_png(*args, compress_level=1)

    duration = 0
    with Image.open(tmp_path / result.filename) as image:
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
            duration += image.info["duration"]
    assert duration == expected_ms
