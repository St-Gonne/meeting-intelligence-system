#!/usr/bin/env python3
"""Private exact-array OpenAI Whisper fallback for guarded MLX failures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise SystemExit("fallback output already exists")
    import numpy as np
    import whisper

    samples = np.load(args.samples, allow_pickle=False)
    if samples.dtype != np.float32 or samples.ndim != 1:
        raise SystemExit("fallback requires a mono float32 array")
    model = whisper.load_model(
        "large-v3", device="cpu", download_root=str(args.model_cache)
    )
    result = model.transcribe(
        samples,
        language=None,
        task="transcribe",
        verbose=False,
        fp16=False,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
