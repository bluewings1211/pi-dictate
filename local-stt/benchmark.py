#!/usr/bin/env python3
"""Reproducible local benchmark for Qwen and Breeze ASR adapters."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from backends import create_transcriber


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=("qwen-mlx", "breeze"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    transcriber = create_transcriber(args.backend, args.model)
    rows = []
    for clip in manifest["clips"]:
        started = time.perf_counter()
        text = transcriber.transcribe(Path(clip["audio"]))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        reference = normalize(clip["reference"])
        hypothesis = normalize(text)
        errors = edit_distance(reference, hypothesis)
        rows.append(
            {
                "id": clip["id"],
                "reference": clip["reference"],
                "transcript": text,
                "latency_ms": elapsed_ms,
                "reference_characters": len(reference),
                "errors": errors,
                "cer": errors / max(1, len(reference)),
            }
        )
        print(f"{clip['id']}: CER={rows[-1]['cer']:.3%}, latency={elapsed_ms:.1f} ms")
    total_chars = sum(row["reference_characters"] for row in rows)
    total_errors = sum(row["errors"] for row in rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model": transcriber.name,
        "python": sys.version,
        "machine": platform.platform(),
        "summary": {"clips": len(rows), "cer": total_errors / max(1, total_chars)},
        "clips": rows,
    }
    output = args.output or Path(f"benchmark-{args.backend}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {output}; aggregate CER={report['summary']['cer']:.3%}")


if __name__ == "__main__":
    main()
