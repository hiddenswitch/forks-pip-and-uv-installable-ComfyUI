import json

from scripts.benchmark_custom_node_tp import add_trace_timings, decode_timings


def test_decode_timings_ignore_progress_and_count_completed_sequences():
    assert decode_timings("rewriting: 8 tokens, 4s, 0.5s/token") == {
        "autoregressive_generated_tokens": None, "autoregressive_decode_seconds_rounded": None,
    }
    assert decode_timings("rewriting: 8 tokens, 4s\nrewriting: 273 tokens in 127s (0.47s/token)\n"
                          "rewriting: 10 tokens in 3s (0.30s/token)") == {
        "autoregressive_generated_tokens": 283, "autoregressive_decode_seconds_rounded": 130,
    }


def test_trace_timings_separate_text_diffusion_and_prefix_preparation(tmp_path):
    directory = tmp_path / "tp2"
    directory.mkdir()
    spans = [
        {"name": "Execute Node", "start_time_unix_nano": 1, "duration_ms": 9000,
         "attributes": {"class_type": "HunyuanImage3TextEncode"}},
        {"name": "Execute Node", "start_time_unix_nano": 11, "duration_ms": 1000,
         "attributes": {"class_type": "HunyuanImage3ModelLoader"}},
        {"name": "Execute Node", "start_time_unix_nano": 12, "duration_ms": 4000,
         "attributes": {"class_type": "HunyuanImage3TextEncode"}},
        {"name": "Execute Node", "start_time_unix_nano": 14, "duration_ms": 3000,
         "attributes": {"class_type": "KSampler"}},
        {"name": "Sampler Invoke", "start_time_unix_nano": 15, "duration_ms": 2500},
    ]
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(span) for span in spans))
    rows = [{"tp": 2, "start_time_unix_nano": 10, "end_time_unix_nano": 20}]
    add_trace_timings(rows, tmp_path)
    assert rows[0]["text_encode_node_seconds"] == 4
    assert rows[0]["model_loader_seconds"] == 1
    assert rows[0]["generator_node_seconds"] == 3
    assert rows[0]["sampler_seconds"] == 2.5
