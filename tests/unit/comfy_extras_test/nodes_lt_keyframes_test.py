"""Unit tests for native LTXV generated-keyframe nodes and Freeze Latent.

They cover keyframe placement, conditioning metadata, guide conversion, and freeze-mask behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfy_extras.nodes import nodes_lt_keyframes as keyframes


def _zeros(shape):
    return torch.zeros(shape)


def _empty_121(channels=2):
    return {"samples": _zeros((1, channels, 16, 2, 1))}


def _empty_241(channels=2):
    return {"samples": _zeros((1, channels, 31, 2, 1))}


def _cond(**extra):
    return [({}, dict(extra))]


def _vae():
    return SimpleNamespace(downscale_index_formula=(8, 32, 32))


def _mask(shape, occupied):
    tensor = torch.ones(shape)
    for frame in occupied:
        tensor[:, :, frame] = 0.0
    return tensor


def _keyframe_idxs_at(starts, tokens_per_frame=1):
    times = []
    for start in starts:
        times.extend([start] * tokens_per_frame)
    n = len(times)
    coords = torch.zeros((1, 3, n, 2))
    for i, start in enumerate(times):
        coords[0, 0, i, 0] = float(start)
        coords[0, 0, i, 1] = float(start + 1)
        coords[0, 1, i, 1] = 1.0
        coords[0, 2, i, 1] = 1.0
    return coords



class TestPlacementHelpers:
    def test_detailing_positions_121_24(self):
        assert keyframes.detailing_positions(121, 24) == [24, 48, 72, 96, 120]
        assert keyframes.free_detailing_slots(121, 24, occupied=set()) == [24, 48, 72, 96, 120]
        assert keyframes.free_detailing_slots(241, 24, occupied={0, 48, 96, 144, 192, 240}) == [
            24, 72, 120, 168, 216
        ]

    def test_free_slots_skip_last_frame_when_occupied(self):
        assert keyframes.free_detailing_slots(121, 24, occupied={120}) == [24, 48, 72, 96]

    def test_free_slots_rejects_when_every_candidate_is_occupied(self):
        with pytest.raises(ValueError, match="already has an image keyframe"):
            keyframes.free_detailing_slots(121, 24, occupied={24, 48, 72, 96, 120})

    def test_scale_frame_indices_temporal_x2(self):
        assert keyframes.scale_frame_indices([24, 48, 72, 96, 120], 121, 241) == [
            48, 96, 144, 192, 240
        ]
        with pytest.raises(ValueError, match="from a 1-frame"):
            keyframes.scale_frame_indices([0], 1, 241)
        with pytest.raises(ValueError, match="onto a 1-frame"):
            keyframes.scale_frame_indices([24], 121, 1)

    def test_scale_frame_indices_rejects_collapsed_duplicates(self):
        with pytest.raises(ValueError, match="collapsed"):
            keyframes.scale_frame_indices([0, 1], 121, 3)

    def test_detailing_positions_keeps_last_skips_zero(self):
        positions = keyframes.detailing_positions(121, 24.0)
        assert positions[0] != 0
        assert positions[-1] == 120

    def test_detailing_positions_rejects_nonpositive_interval(self):
        with pytest.raises(ValueError, match="interval_frames"):
            keyframes.detailing_positions(121, 0)

    def test_detailing_positions_rejects_one_frame_canvas(self):
        with pytest.raises(ValueError, match="no pixel frames"):
            keyframes.detailing_positions(1, 24)
        with pytest.raises(ValueError, match="no pixel frames"):
            keyframes.free_detailing_slots(1, 24, occupied=set())

    def test_keyframes_from_video_stacking_shape(self):
        samples = torch.arange(1 * 2 * 4 * 2 * 1, dtype=torch.float32).reshape(1, 2, 4, 2, 1)
        stacked = keyframes.keyframes_from_video(samples, [8, 16, 24], temporal_scale=8)
        assert stacked.shape == (1, 2, 3, 2, 1)
        assert torch.equal(stacked[:, :, 0:1], samples[:, :, 1:2])
        assert torch.equal(stacked[:, :, 1:2], samples[:, :, 2:3])
        assert torch.equal(stacked[:, :, 2:3], samples[:, :, 3:4])

    def test_keyframes_from_video_rejects_non_video_and_bad_scale(self):
        with pytest.raises(ValueError, match="plain 5D video latent"):
            keyframes.keyframes_from_video([0], [8], 8)
        with pytest.raises(ValueError, match="temporal_scale"):
            keyframes.keyframes_from_video(_zeros((1, 2, 4, 2, 1)), [8], 0)
        with pytest.raises(ValueError, match="no frames to copy"):
            keyframes.keyframes_from_video(_zeros((1, 2, 0, 2, 1)), [8], 8)

    def test_nearest_latent_index_clamps(self):
        assert keyframes.nearest_latent_index(0, 8, 4) == 0
        assert keyframes.nearest_latent_index(8, 8, 4) == 1
        assert keyframes.nearest_latent_index(999, 8, 4) == 3

    def test_should_copy_nearest_video_frames(self):
        assert keyframes.should_copy_nearest_video_frames(31, 5, False, False) is True
        assert keyframes.should_copy_nearest_video_frames(5, 5, False, False) is False
        assert keyframes.should_copy_nearest_video_frames(4, 5, False, False) is False
        assert keyframes.should_copy_nearest_video_frames(31, 5, True, False) is False
        assert keyframes.should_copy_nearest_video_frames(31, None, False, False) is False
        assert keyframes.should_copy_nearest_video_frames(1, 5, False, True) is False

    def test_parse_frame_index_list_validates_count_range_and_duplicates(self):
        assert keyframes._parse_frame_index_list(
            "24, 48", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
        ) == [24, 48]
        assert keyframes._parse_frame_index_list(
            "24 48", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
        ) == [24, 48]
        with pytest.raises(ValueError, match="lists 1"):
            keyframes._parse_frame_index_list(
                "24", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
            )
        with pytest.raises(ValueError, match="same pixel frame"):
            keyframes._parse_frame_index_list(
                "24,24", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
            )
        with pytest.raises(ValueError, match="must lie between"):
            keyframes._parse_frame_index_list(
                "0,24", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
            )
        with pytest.raises(ValueError, match="could not parse"):
            keyframes._parse_frame_index_list(
                "24,abc", "frame_indices", 2, 1, 120, "num_keyframes is 2", "to space them"
            )

    def test_parse_frame_index_list_allows_omitted_count(self):
        assert keyframes._parse_frame_index_list(
            "24,48,72", "frame_indices", None, 1, 120, "unused", "to auto-place"
        ) == [24, 48, 72]

    def test_parse_frame_index_list_rejects_empty_separator_only(self):
        with pytest.raises(ValueError, match="is empty"):
            keyframes._parse_frame_index_list(
                ",", "frame_indices", None, 1, 120, "unused", "to place them from interval_frames"
            )
        with pytest.raises(ValueError, match="is empty"):
            keyframes._parse_frame_index_list(
                " , , ", "frame_indices", None, 1, 120, "unused", "to place them from interval_frames"
            )

    def test_add_parse_frame_indices_allows_last_frame(self):
        assert keyframes.LTXVAddGeneratedKeyframes.parse_frame_indices("24,120", 121) == [24, 120]
        with pytest.raises(ValueError, match="no pixel frames"):
            keyframes.LTXVAddGeneratedKeyframes.parse_frame_indices("1", 1)

    def test_occupied_from_nonzero_samples_without_mask(self):
        samples = _zeros((1, 2, 16, 2, 1))
        samples[0, 0, 0, 0, 0] = 1.0
        taken = keyframes.occupied_pixel_frames({"samples": samples}, 8, 121)
        assert 0 in taken
        assert 120 not in taken

    def test_occupied_prefers_noise_mask_over_nonzero_samples(self):
        samples = _zeros((1, 2, 16, 2, 1))
        samples[0, 0, 0, 0, 0] = 1.0
        latent = {"samples": samples, "noise_mask": _mask((1, 1, 16, 1, 1), occupied=set())}
        assert keyframes.occupied_pixel_frames(latent, 8, 121) == set()

    def test_occupied_ignores_appended_guide_frames(self):
        latent = {
            "samples": _zeros((1, 2, 21, 2, 1)),
            "noise_mask": _mask((1, 1, 21, 1, 1), occupied={0, 16, 17, 18, 19, 20}),
        }
        taken = keyframes.occupied_pixel_frames(latent, 8, 121, video_latent_frames=16)
        assert taken == {0}

    def test_pixel_frames_from_keyframe_idxs_uses_start_not_exclusive_end(self):
        idxs = _keyframe_idxs_at([24])
        assert idxs[0, 0, :, 0].tolist() == [24.0]
        assert idxs[0, 0, :, 1].tolist() == [25.0]
        assert keyframes.pixel_frames_from_keyframe_idxs(idxs) == {24}
        assert keyframes.pixel_frames_from_keyframe_idxs(None) == set()

    def test_pixel_frames_from_keyframe_idxs_rejects_malformed(self):
        with pytest.raises((TypeError, AttributeError, IndexError, ValueError)):
            keyframes.pixel_frames_from_keyframe_idxs("not-a-tensor")
        with pytest.raises((TypeError, ValueError)):
            keyframes._as_int_set(object())


class TestNativeSchemas:
    def test_generated_keyframe_nodes_use_ltxv_conditioning_category(self):
        for cls, node_id, display_name in (
            (
                keyframes.LTXVAddGeneratedKeyframes,
                "LTXVAddGeneratedKeyframes",
                "LTXV Add Generated Keyframes",
            ),
            (
                keyframes.LTXVSeparateGeneratedKeyframes,
                "LTXVSeparateGeneratedKeyframes",
                "LTXV Separate Generated Keyframes",
            ),
            (
                keyframes.LTXVGeneratedKeyframesToGuides,
                "LTXVGeneratedKeyframesToGuides",
                "LTXV Generated Keyframes to Guides",
            ),
        ):
            schema = cls.define_schema()
            assert schema.node_id == node_id
            assert schema.display_name == display_name
            assert schema.category == "model/conditioning/ltxv"
            assert "dfr" in schema.search_aliases

    def test_freeze_latent_uses_ltxv_latent_category(self):
        schema = keyframes.LTXVFreezeLatent.define_schema()
        assert schema.node_id == "LTXVFreezeLatent"
        assert schema.display_name == "LTXV Freeze Latent"
        assert schema.category == "model/latent/ltxv"


class TestAddGeneratedKeyframes:
    def test_rejects_non_video_latent(self):
        with pytest.raises(ValueError, match="plain video latent"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(), _cond(), _vae(), {"samples": torch.zeros(1, 2, 16, 2)}
            )

    def test_execute_rejects_separator_only_frame_indices(self):
        with pytest.raises(ValueError, match="is empty"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(), _cond(), _vae(), _empty_121(), frame_indices=","
            )

    def test_execute_rejects_one_frame_canvas(self):
        with pytest.raises(ValueError, match="no pixel frames"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(), _cond(), _vae(), {"samples": _zeros((1, 2, 1, 2, 1))}
            )

    def test_rejects_rescaled_or_noncontiguous_existing_keyframes(self):
        latent = _empty_121()
        with pytest.raises(ValueError, match="rescaled"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(
                    generated_keyframes={
                        "tokens_per_frame": 99,
                        "first_latent_frame": 16,
                        "num_keyframes": 0,
                    }
                ),
                _cond(),
                _vae(),
                latent,
            )
        with pytest.raises(ValueError, match="contiguous"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(
                    generated_keyframes={
                        "tokens_per_frame": 2,
                        "first_latent_frame": 10,
                        "num_keyframes": 3,
                    }
                ),
                _cond(),
                _vae(),
                latent,
            )

    def test_execute_appends_zero_keyframes_on_t(self):
        _positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), _empty_121()
        )
        assert out["samples"].shape == (1, 2, 21, 2, 1)
        assert out["noise_mask"].shape[2] == 21
        assert torch.all(out["noise_mask"][:, :, 16:21] == 1.0)

    def test_execute_copies_nearest_frames_from_longer_video(self):
        video = {"samples": torch.arange(1 * 2 * 16 * 2 * 1, dtype=torch.float32).reshape(1, 2, 16, 2, 1)}
        _positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(),
            _cond(),
            _vae(),
            _empty_121(),
            frame_indices="24,48,72,96,120",
            keyframes=video,
        )
        assert out["samples"].shape[2] == 21
        stacked = out["samples"][:, :, 16:21]
        source = video["samples"]
        assert torch.equal(stacked[:, :, 0:1], source[:, :, 3:4])
        assert torch.equal(stacked[:, :, 4:5], source[:, :, 15:16])

    def test_execute_keeps_stacked_keyframes_when_t_equals_count(self):
        stacked = {"samples": torch.arange(1 * 2 * 5 * 2 * 1, dtype=torch.float32).reshape(1, 2, 5, 2, 1)}
        _positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(),
            _cond(),
            _vae(),
            _empty_121(),
            frame_indices="24,48,72,96,120",
            keyframes=stacked,
        )
        assert out["samples"].shape[2] == 21
        assert torch.equal(out["samples"][:, :, 16:21], stacked["samples"])

    def test_execute_reshapes_batched_single_frame_keyframes(self):
        batched = {"samples": _zeros((5, 2, 1, 2, 1))}
        _positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(),
            _cond(),
            _vae(),
            _empty_121(),
            frame_indices="24,48,72,96,120",
            keyframes=batched,
        )
        assert out["samples"].shape[2] == 21

    def test_execute_records_density_slots_and_canvas_length(self):
        positive, _negative, _out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), _empty_121()
        )
        record = positive[0][1]["generated_keyframes"]
        assert record["frame_indices"] == [24, 48, 72, 96, 120]
        assert record["num_pixel_frames"] == 121
        assert record["num_keyframes"] == 5
        assert record["first_latent_frame"] == 16
        assert record["guide_entry_index"] == 0
        entries = positive[0][1]["guide_attention_entries"]
        assert len(entries) == 1
        assert entries[0]["pre_filter_count"] == 5 * 2 * 1
        assert entries[0]["latent_shape"] == [5, 2, 1]

    def test_execute_copies_from_video_using_auto_slots(self):
        video = {"samples": torch.arange(1 * 2 * 16 * 2 * 1, dtype=torch.float32).reshape(1, 2, 16, 2, 1)}
        _positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), _empty_121(), keyframes=video
        )
        stacked = out["samples"][:, :, 16:21]
        source = video["samples"]
        assert torch.equal(stacked[:, :, 0:1], source[:, :, 3:4])
        assert torch.equal(stacked[:, :, 4:5], source[:, :, 15:16])

    def test_execute_replaces_stacked_tokens_on_current_canvas(self):
        stacked = {
            "samples": torch.arange(1 * 2 * 5 * 2 * 1, dtype=torch.float32).reshape(1, 2, 5, 2, 1),
            "generated_keyframe_indices": [24, 48, 72, 96, 120],
            "generated_keyframe_num_frames": 121,
        }
        positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), _empty_121(), keyframes=stacked
        )
        assert torch.equal(out["samples"][:, :, 16:21], stacked["samples"])
        assert positive[0][1]["generated_keyframes"]["frame_indices"] == [24, 48, 72, 96, 120]

    def test_execute_skips_i2v_last_frame_noise_mask(self):
        latent = {
            "samples": _zeros((1, 2, 16, 2, 1)),
            "noise_mask": _mask((1, 1, 16, 1, 1), occupied={15}),
        }
        positive, _negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), latent
        )
        indices = positive[0][1]["generated_keyframes"]["frame_indices"]
        assert 120 not in indices
        assert indices == [24, 48, 72, 96]
        assert out["samples"].shape[2] == 20

    def test_execute_replaces_stacked_tokens_on_longer_canvas(self):
        stacked = {
            "samples": _zeros((1, 2, 5, 2, 1)),
            "generated_keyframe_indices": [24, 48, 72, 96, 120],
            "generated_keyframe_num_frames": 121,
        }
        latent = _empty_241()
        latent["noise_mask"] = _mask((1, 1, 31, 1, 1), occupied={0})
        positive, _negative, _out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), latent, keyframes=stacked
        )
        indices = positive[0][1]["generated_keyframes"]["frame_indices"]
        assert indices != [24, 48, 72, 96, 120]
        assert indices == [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

    def test_execute_skips_existing_guide_keyframe_idxs(self):
        latent = {
            "samples": _zeros((1, 2, 36, 2, 1)),
            "noise_mask": _mask((1, 1, 36, 1, 1), occupied={0}),
        }
        idxs = _keyframe_idxs_at([48, 96, 144, 192, 240], tokens_per_frame=2)
        positive, _negative, _out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(keyframe_idxs=idxs), _cond(keyframe_idxs=idxs), _vae(), latent
        )
        assert positive[0][1]["generated_keyframes"]["frame_indices"] == [24, 72, 120, 168, 216]

    def test_execute_rejects_occupied_manual_indices(self):
        latent = {
            "samples": _zeros((1, 2, 16, 2, 1)),
            "noise_mask": _mask((1, 1, 16, 1, 1), occupied={15}),
        }
        with pytest.raises(ValueError, match="reuses pixel frame"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(), _cond(), _vae(), latent, frame_indices="24,120"
            )

    def test_execute_rejects_wrong_spatial_size_keyframes(self):
        stacked = {"samples": _zeros((1, 2, 5, 4, 4))}
        with pytest.raises(ValueError, match="whole latent frames"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(),
                _cond(),
                _vae(),
                _empty_121(),
                frame_indices="24,48,72,96,120",
                keyframes=stacked,
            )

    def test_execute_rejects_too_many_stacked_keyframes(self):
        stacked = {
            "samples": _zeros((1, 2, 6, 2, 1)),
            "generated_keyframe_indices": [24, 48, 72, 96, 120, 8],
        }
        with pytest.raises(ValueError, match="only 5 free slot"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(),
                _cond(),
                _vae(),
                _empty_121(),
                frame_indices="24,48,72,96,120",
                keyframes=stacked,
            )

    def test_execute_rejects_non_5d_keyframes(self):
        with pytest.raises(ValueError, match="5 dimensional"):
            keyframes.LTXVAddGeneratedKeyframes.execute(
                _cond(),
                _cond(),
                _vae(),
                _empty_121(),
                frame_indices="24",
                keyframes={"samples": torch.zeros(1, 2, 1, 2)},
            )

    def test_execute_grows_existing_generated_block(self):
        latent = _empty_121()
        positive, negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            _cond(), _cond(), _vae(), latent, frame_indices="24,48"
        )
        positive, negative, out = keyframes.LTXVAddGeneratedKeyframes.execute(
            positive, negative, _vae(), out, frame_indices="72,96"
        )
        record = positive[0][1]["generated_keyframes"]
        assert record["frame_indices"] == [24, 48, 72, 96]
        assert record["num_keyframes"] == 4
        assert record["first_latent_frame"] == 16
        assert out["samples"].shape[2] == 20
        entries = positive[0][1]["guide_attention_entries"]
        assert len(entries) == 1
        assert entries[0]["pre_filter_count"] == 4 * 2 * 1
        assert entries[0]["latent_shape"] == [4, 2, 1]


class TestSeparateGeneratedKeyframes:
    def test_requires_generated_keyframes(self):
        with pytest.raises(ValueError, match="no generated keyframes"):
            keyframes.LTXVSeparateGeneratedKeyframes.execute(_cond(), _cond(), _empty_121())

    def test_execute_peels_keyframes_and_indices(self):
        video = _zeros((1, 2, 16, 2, 1))
        keys = torch.arange(1, 1 + 1 * 2 * 5 * 2 * 1, dtype=torch.float32).reshape(1, 2, 5, 2, 1)
        samples = torch.cat([video, keys], dim=2)
        record = {
            "first_latent_frame": 16,
            "num_keyframes": 5,
            "frame_indices": [24, 48, 72, 96, 120],
            "num_pixel_frames": 121,
            "guide_entry_index": 0,
            "tokens_per_frame": 2,
        }
        positive, negative, latent, peeled = keyframes.LTXVSeparateGeneratedKeyframes.execute(
            _cond(
                generated_keyframes=record,
                guide_attention_entries=[{"keep": False}, {"keep": True}],
            ),
            _cond(generated_keyframes=record),
            {"samples": samples},
        )
        assert latent["samples"].shape == (1, 2, 16, 2, 1)
        assert peeled["samples"].shape == (1, 2, 5, 2, 1)
        assert peeled["generated_keyframe_indices"] == [24, 48, 72, 96, 120]
        assert peeled["generated_keyframe_num_frames"] == 121
        assert torch.equal(peeled["samples"], keys)
        assert positive[0][1]["generated_keyframes"] is None
        assert positive[0][1]["guide_attention_entries"] == [{"keep": True}]
        assert negative[0][1]["generated_keyframes"] is None

    def test_execute_keyframes_to_batch(self):
        samples = _zeros((1, 2, 18, 2, 1))
        record = {
            "first_latent_frame": 16,
            "num_keyframes": 2,
            "frame_indices": [24, 48],
            "guide_entry_index": 0,
            "tokens_per_frame": 2,
        }
        _p, _n, _latent, peeled = keyframes.LTXVSeparateGeneratedKeyframes.execute(
            _cond(generated_keyframes=record),
            _cond(generated_keyframes=record),
            {"samples": samples},
            keyframes_to_batch=True,
        )
        assert peeled["samples"].shape == (2, 2, 1, 2, 1)

    def test_rejects_token_mismatch_and_short_latent(self):
        record = {
            "first_latent_frame": 16,
            "num_keyframes": 5,
            "frame_indices": [24, 48, 72, 96, 120],
            "guide_entry_index": 0,
            "tokens_per_frame": 99,
        }
        with pytest.raises(ValueError, match="rescaled"):
            keyframes.LTXVSeparateGeneratedKeyframes.execute(
                _cond(generated_keyframes=record),
                _cond(generated_keyframes=record),
                _empty_121(),
            )
        record = dict(record)
        record["tokens_per_frame"] = 2
        with pytest.raises(ValueError, match="only has"):
            keyframes.LTXVSeparateGeneratedKeyframes.execute(
                _cond(generated_keyframes=record),
                _cond(generated_keyframes=record),
                _empty_121(),
            )

    def test_strip_guide_entry(self):
        remaining = keyframes.LTXVSeparateGeneratedKeyframes.strip_guide_entry(
            [({}, {"guide_attention_entries": [{"a": 1}, {"b": 2}]})], 0
        )
        assert remaining == [{"b": 2}]
        empty = keyframes.LTXVSeparateGeneratedKeyframes.strip_guide_entry(
            [({}, {"guide_attention_entries": [{"a": 1}]})], 0
        )
        assert empty is None
        with pytest.raises(ValueError, match="recorded guide entry"):
            keyframes.LTXVSeparateGeneratedKeyframes.strip_guide_entry(
                [({}, {"guide_attention_entries": [{"a": 1}]})], 5
            )

    def test_rejects_non_video_latent(self):
        record = {
            "first_latent_frame": 0,
            "num_keyframes": 1,
            "frame_indices": [24],
            "guide_entry_index": 0,
            "tokens_per_frame": 2,
        }
        with pytest.raises(ValueError, match="plain video latent"):
            keyframes.LTXVSeparateGeneratedKeyframes.execute(
                _cond(generated_keyframes=record),
                _cond(generated_keyframes=record),
                {"samples": torch.zeros(1, 2, 16, 2)},
            )


class TestGeneratedKeyframesToGuides:
    def test_requires_recorded_indices(self):
        with pytest.raises(ValueError, match="does not carry generated keyframe positions"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(), _cond(), _vae(), _empty_121(128), {"samples": _zeros((1, 128, 5, 2, 1))}, 1.0
            )

    def test_rejects_unseparated_conditioning(self):
        with pytest.raises(ValueError, match="still carries generated keyframes"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(generated_keyframes={"num_keyframes": 1}),
                _cond(),
                _vae(),
                _empty_121(128),
                {"samples": _zeros((1, 128, 1, 2, 1)), "generated_keyframe_indices": [24]},
                1.0,
            )

    def test_rejects_non_video_and_batched_canvas(self):
        kf = {"samples": _zeros((1, 128, 1, 2, 1)), "generated_keyframe_indices": [24]}
        with pytest.raises(ValueError, match="plain video latent"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(), _cond(), _vae(), {"samples": torch.zeros(1, 128, 16, 2)}, kf, 1.0
            )
        with pytest.raises(ValueError, match="batch size of 1"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(), _cond(), _vae(), {"samples": _zeros((2, 128, 16, 2, 1))}, kf, 1.0
            )

    def test_pins_same_size_keyframes_via_append(self, mocker):
        spy = mocker.spy(keyframes.LTXVAddGuide, 'append_keyframe')
        kf = {
            "samples": _zeros((1, 128, 2, 2, 1)),
            "generated_keyframe_indices": [24, 48],
            "generated_keyframe_num_frames": 121,
        }
        positive, negative, out = keyframes.LTXVGeneratedKeyframesToGuides.execute(
            _cond(), _cond(), _vae(), _empty_121(128), kf, 1.0
        )
        assert out["samples"].shape[2] == 18
        assert torch.all(out["noise_mask"][:, :, 16:] == 0.0)
        assert [call.args[2] for call in spy.call_args_list] == [24, 48]
        entries = positive[0][1]["guide_attention_entries"]
        assert len(entries) == 2

    def test_scales_indices_after_temporal_x2(self, mocker):
        spy = mocker.spy(keyframes.LTXVAddGuide, 'append_keyframe')
        kf = {
            "samples": _zeros((1, 128, 2, 2, 1)),
            "generated_keyframe_indices": [24, 120],
            "generated_keyframe_num_frames": 121,
        }
        keyframes.LTXVGeneratedKeyframesToGuides.execute(
            _cond(), _cond(), _vae(), _empty_241(128), kf, 1.0
        )
        assert [call.args[2] for call in spy.call_args_list] == [48, 240]

    def test_override_frame_indices(self, mocker):
        spy = mocker.spy(keyframes.LTXVAddGuide, 'append_keyframe')
        kf = {
            "samples": _zeros((1, 128, 2, 2, 1)),
            "generated_keyframe_indices": [24, 48],
            "generated_keyframe_num_frames": 121,
        }
        keyframes.LTXVGeneratedKeyframesToGuides.execute(
            _cond(), _cond(), _vae(), _empty_121(128), kf, 0.5, override_frame_indices="32,96"
        )
        assert [call.args[2] for call in spy.call_args_list] == [32, 96]
        assert all(call.args[6] == 0.5 for call in spy.call_args_list)

    def test_resize_path_decodes_and_calls_add_guide(self, mocker):
        spy = mocker.spy(keyframes.LTXVAddGuide, 'execute')
        vae = _vae()
        decoded = []

        def decode(samples):
            decoded.append(tuple(samples.shape))
            return torch.zeros((samples.shape[0], 8, 8, 3))

        vae.decode = decode
        vae.encode = lambda images: torch.zeros((1, 128, 1, 2, 1))
        kf = {
            "samples": _zeros((1, 128, 2, 4, 4)),
            "generated_keyframe_indices": [24, 48],
            "generated_keyframe_num_frames": 121,
        }
        _p, _n, out = keyframes.LTXVGeneratedKeyframesToGuides.execute(
            _cond(), _cond(), vae, _empty_121(128), kf, 1.0
        )
        assert decoded == [(2, 128, 1, 4, 4)]
        assert spy.call_count == 2
        assert out["samples"].shape[2] == 18

    def test_rejects_count_mismatch(self):
        kf = {
            "samples": _zeros((1, 128, 2, 2, 1)),
            "generated_keyframe_indices": [24],
            "generated_keyframe_num_frames": 121,
        }
        with pytest.raises(ValueError, match="2 keyframes for 1 recorded"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(), _cond(), _vae(), _empty_121(128), kf, 1.0
            )

    def test_override_rejects_separator_only_indices(self):
        kf = {
            "samples": _zeros((1, 128, 2, 2, 1)),
            "generated_keyframe_indices": [24, 48],
            "generated_keyframe_num_frames": 121,
        }
        with pytest.raises(ValueError, match="is empty"):
            keyframes.LTXVGeneratedKeyframesToGuides.execute(
                _cond(), _cond(), _vae(), _empty_121(128), kf, 1.0, override_frame_indices=","
            )


class TestFreezeLatent:
    def test_video_and_audio_masks(self):
        video = keyframes.LTXVFreezeLatent.execute({"samples": _zeros((2, 4, 8, 3, 5))})[0]
        assert video["noise_mask"].shape == (2, 1, 8, 1, 1)
        assert video["noise_mask"].device.type == "cpu"
        assert torch.all(video["noise_mask"] == 0)
        audio = keyframes.LTXVFreezeLatent.execute({"samples": _zeros((1, 8, 16, 4))})[0]
        assert audio["noise_mask"].shape == (1, 1, 16, 1)
        assert torch.all(audio["noise_mask"] == 0)

    def test_preserves_extra_latent_keys(self):
        out = keyframes.LTXVFreezeLatent.execute(
            {"samples": _zeros((1, 4, 8, 2, 2)), "downscale_ratio_spacial": 32}
        )[0]
        assert out["downscale_ratio_spacial"] == 32

    def test_rejects_av_and_wrong_rank(self):
        with pytest.raises(ValueError, match="plain tensor"):
            keyframes.LTXVFreezeLatent.execute({"samples": [0.0]})
        with pytest.raises(ValueError, match="4D audio or 5D video"):
            keyframes.LTXVFreezeLatent.execute({"samples": _zeros((1, 2, 3))})


class TestKeyframeCoords:
    def test_single_pixel_span_at_requested_index(self):
        latent = torch.zeros((1, 4, 1, 2, 2))
        coords = keyframes.LTXVAddGeneratedKeyframes.keyframe_coords(latent, 24, (8, 32, 32))
        assert coords.shape[0] == 1
        assert coords.shape[1] == 3
        assert coords.shape[-1] == 2
        starts = coords[0, 0, :, 0]
        ends = coords[0, 0, :, 1]
        assert torch.all(starts == 24)
        assert torch.all(ends == 25)


def test_extension_registers_all_four_nodes():
    import asyncio

    ext = asyncio.run(keyframes.comfy_entrypoint())
    names = [cls.__name__ for cls in asyncio.run(ext.get_node_list())]
    assert names == [
        "LTXVAddGeneratedKeyframes",
        "LTXVSeparateGeneratedKeyframes",
        "LTXVGeneratedKeyframesToGuides",
        "LTXVFreezeLatent",
    ]
