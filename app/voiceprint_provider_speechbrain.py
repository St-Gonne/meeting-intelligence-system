#!/usr/bin/env python3
"""Isolated SpeechBrain ECAPA provider for the Gate 4 bake-off only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from voiceprint_provider_pyannote import ProviderError, _ordinary_file, decode_mono_16k


def _ordinary_directory(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ProviderError(f"{label} is missing or unsafe")
    resolved = lexical.resolve(strict=True)
    if any(item.is_symlink() for item in resolved.rglob("*")):
        raise ProviderError(f"{label} contains a symlink")
    return resolved


def embed(model_asset: Path, audio: Path, *, decoder: Callable[[Path], Sequence[float]] = decode_mono_16k) -> tuple[float, ...]:
    model_path = _ordinary_directory(model_asset, "model asset")
    audio_path = _ordinary_file(audio, "audio")
    samples = decoder(audio_path)
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        waveform = torch.tensor(samples, dtype=torch.float32).reshape(1, -1)
        model = EncoderClassifier.from_hparams(source=str(model_path), run_opts={"device": "cpu"})
        result = model.encode_batch(waveform)
        vector = tuple(float(value) for value in result.detach().cpu().reshape(-1))
    except Exception as exc:
        raise ProviderError("embedding inference failed") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ProviderError("embedding vector is invalid")
    return vector


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 4 SpeechBrain ECAPA provider")
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        vector = embed(args.model_asset, args.audio)
    except ProviderError:
        print("VOICEPRINT_PROVIDER_REJECTED=ProviderError", file=__import__("sys").stderr)
        return 2
    print(json.dumps({"embedding": vector}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
