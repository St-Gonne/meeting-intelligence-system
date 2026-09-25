#!/usr/bin/env python3
"""Isolated pyannote embedding provider for the Gate 4 bake-off only."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence


class ProviderError(RuntimeError):
    pass


def _ordinary_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ProviderError(f"{label} is missing or unsafe")
    return lexical.resolve(strict=True)


def decode_mono_16k(path: Path, *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> tuple[float, ...]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProviderError("ffmpeg is unavailable")
    completed = runner(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not completed.stdout or len(completed.stdout) % 4:
        raise ProviderError("audio decode failed")
    import array

    values = array.array("f")
    values.frombytes(completed.stdout)
    if len(values) < 8000 or not all(math.isfinite(value) for value in values):
        raise ProviderError("decoded audio is too short or invalid")
    return tuple(values)


def embed(model_asset: Path, audio: Path, *, decoder: Callable[[Path], Sequence[float]] = decode_mono_16k) -> tuple[float, ...]:
    model_path = _ordinary_file(model_asset, "model asset")
    audio_path = _ordinary_file(audio, "audio")
    samples = decoder(audio_path)
    try:
        import torch
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        waveform = torch.tensor(samples, dtype=torch.float32).reshape(1, 1, -1)
        model = PretrainedSpeakerEmbedding(str(model_path))
        result = model(waveform)
    except Exception as exc:
        raise ProviderError("embedding inference failed") from exc
    if getattr(result, "shape", None) is None or len(result.shape) != 2 or result.shape[0] != 1:
        raise ProviderError("embedding shape is invalid")
    vector = tuple(float(value) for value in result[0])
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ProviderError("embedding vector is invalid")
    return vector


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 4 pyannote embedding provider")
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
