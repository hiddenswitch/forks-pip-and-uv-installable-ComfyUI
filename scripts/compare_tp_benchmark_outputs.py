"""Compare saved TP1/TP2 sample outputs without using container or file hashes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image


def differences(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Output shapes differ: {left.shape} / {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "identical": bool(np.array_equal(left, right)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height,nb_frames,avg_frame_rate,duration,sample_rate,channels",
        "-of", "json", str(path),
    ]))["streams"]


def video_comparison(left, right):
    streams = [probe(path) for path in (left, right)]
    video = [next(stream for stream in group if stream["codec_type"] == "video") for group in streams]
    if video[0] != video[1]:
        raise ValueError(f"Video dimensions, frame counts or timing differ: {video}")
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-i", str(left), "-i", str(right),
        "-lavfi", "[0:v][1:v]ssim", "-an", "-f", "null", "-",
    ], capture_output=True, text=True, check=True)
    scores = re.findall(r"SSIM .* All:([0-9.]+)", result.stderr)
    if len(scores) != 1:
        raise ValueError("Expected one decoded-video SSIM result")
    report = {"video": video[0], "decoded_video_ssim": float(scores[0])}
    audio = [[stream for stream in group if stream["codec_type"] == "audio"] for group in streams]
    if audio[0] != audio[1]:
        raise ValueError(f"Audio formats or durations differ: {audio}")
    if audio[0]:
        samples = [np.frombuffer(subprocess.check_output([
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
            "-map", "0:a:0", "-f", "f32le", "-",
        ]), dtype="<f4") for path in (left, right)]
        report["audio"] = dict(audio[0][0], **differences(*samples))
    return report


def saved_outputs(history, directory):
    outputs = {}
    for node_id, output in history["outputs"].items():
        for index, item in enumerate(output.get("images", [])):
            if item.get("type") == "output":
                outputs[f"{node_id}:{index}"] = directory / item.get("subfolder", "") / item["filename"]
    if not outputs:
        raise ValueError("History has no saved image/video outputs")
    return outputs


def compare(directory, seeds=(42, 43, 44, 45)):
    results = []
    for seed in seeds:
        outputs = []
        for size in (1, 2):
            phase = directory / f"tp{size}"
            history = json.loads((phase / f"seed{seed}.history.json").read_text())
            if history["status"]["status_str"] != "success":
                raise ValueError(f"TP{size} seed {seed} did not succeed")
            outputs.append(saved_outputs(history, phase))
        if outputs[0].keys() != outputs[1].keys():
            raise ValueError(f"Saved output nodes differ for seed {seed}")
        for key in outputs[0]:
            left, right = [output[key] for output in outputs]
            if left.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                with Image.open(left) as a, Image.open(right) as b:
                    metrics = differences(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB")))
            else:
                metrics = video_comparison(left, right)
            results.append({"seed": seed, "output": key, "tp1": str(left), "tp2": str(right), **metrics})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.directory, args.seeds), indent=2) + "\n")
