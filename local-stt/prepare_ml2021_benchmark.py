#!/usr/bin/env python3
"""Create a small, fixed Mandarin-English benchmark from ML2021_ASR_ST.

Only selected test clips are downloaded. The generated directory is ignored by
git so source audio and benchmark results never become extension assets.
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np
from datasets import load_dataset


def write_wav(path: Path, audio: object) -> None:
    samples = audio.get_all_samples()
    data = samples.data.numpy()
    if data.ndim == 2:
        data = data.mean(axis=0)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(samples.sample_rate)
        output.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-data/ml2021-mixed-8"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--scan-limit", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_dataset("ky552/ML2021_ASR_ST", split="test", streaming=True)
    clips = []
    for index, row in enumerate(rows):
        if index >= args.scan_limit or len(clips) >= args.limit:
            break
        reference = row["transcription"]
        # Keep actual intra-utterance code-switching, not merely a translated
        # English field alongside all-Chinese speech.
        if not re.search(r"[A-Za-z]", reference):
            continue
        clip_id = f"ml2021-test-{index:04d}"
        audio_path = args.output / f"{clip_id}.wav"
        write_wav(audio_path, row["audio"])
        clips.append({"id": clip_id, "audio": str(audio_path.resolve()), "reference": reference})
        print(f"saved {clip_id}: {reference}")
    if len(clips) < args.limit:
        raise RuntimeError(f"found only {len(clips)} mixed-language clips in first {args.scan_limit} rows")
    manifest = {"source": "ky552/ML2021_ASR_ST test split", "clips": clips}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
