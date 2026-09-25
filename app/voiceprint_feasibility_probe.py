#!/usr/bin/env python3
"""Default-off Gate 1 probe for an explicit local speaker embedding sample.

This module is intentionally standalone. It is not imported by MeetingIntel
processing and writes only a shape/interface diagnostic, never embedding data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL = "pyannote/embedding"
REPORT_FILENAME = "voiceprint_feasibility.json"
INTERFACE_NAME = "pyannote.audio.pipelines.SpeakerEmbedding"
REPOSITORY_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProbeResult:
    status: str
    interface: str
    runtime_version: str | None
    model: str
    embedding_shape: tuple[int, ...] | None = None
    embedding_dtype: str | None = None
    error_type: str | None = None

    def report(self) -> dict[str, Any]:
        # Deliberately omit the audio path, output path, token state, and all
        # embedding values. This report is safe feasibility metadata only.
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": self.status,
            "interface": self.interface,
            "model": self.model,
            "runtime_version": self.runtime_version,
            "embedding_shape": list(self.embedding_shape)
            if self.embedding_shape is not None
            else None,
            "embedding_dtype": self.embedding_dtype,
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


def _outside_repository(path: Path) -> bool:
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return True
    return False


def validate_audio_path(raw_path: Path) -> Path:
    candidate = raw_path.expanduser()
    if candidate.is_symlink():
        raise ValueError("audio path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("audio path was not found") from exc
    if not resolved.is_file():
        raise ValueError("audio path must be a regular file")
    if not _outside_repository(resolved):
        raise ValueError("audio path must be outside the repository")
    return resolved


def validate_output_dir(raw_path: Path) -> Path:
    candidate = raw_path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("output directory must not be a symlink")
    resolved = candidate.resolve()
    if not _outside_repository(resolved):
        raise ValueError("output directory must be outside the repository")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output path must be a directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_cache_dir(raw_path: Path | None) -> Path | None:
    if raw_path is None:
        return None
    candidate = raw_path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("cache directory must not be a symlink")
    resolved = candidate.resolve()
    if not _outside_repository(resolved):
        raise ValueError("cache directory must be outside the repository")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("cache path must be a directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _runtime_version() -> str | None:
    try:
        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_embedding_factory(
    model: str, token: str | None, cache_dir: Path | None
) -> Callable[[str], Any]:
    from pyannote.audio.pipelines import SpeakerEmbedding

    pipeline = SpeakerEmbedding(
        embedding=model,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return pipeline


def _embedding_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("embedding result has no shape")
    try:
        normalized = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError("embedding result shape is invalid") from exc
    if len(normalized) != 2 or normalized[0] != 1 or normalized[1] <= 0:
        raise ValueError("embedding result must have shape (1, dimension)")
    return normalized


def write_report(output_dir: Path, result: ProbeResult) -> Path:
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(result.report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def run_probe(
    audio_path: Path,
    output_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    token_env: str = "HF_TOKEN",
    cache_dir: Path | None = None,
    embedding_factory: Callable[[str, str | None, Path | None], Callable[[str], Any]]
    | None = None,
    runtime_version: str | None = None,
) -> tuple[ProbeResult, Path]:
    audio = validate_audio_path(audio_path)
    output = validate_output_dir(output_dir)
    cache = validate_cache_dir(cache_dir)
    token = os.environ.get(token_env) or None
    factory = embedding_factory or _default_embedding_factory
    version = runtime_version if runtime_version is not None else _runtime_version()

    try:
        embed = factory(model, token, cache)
        result = embed(str(audio))
        shape = _embedding_shape(result)
        dtype = getattr(result, "dtype", None)
        probe_result = ProbeResult(
            status="success",
            interface=INTERFACE_NAME,
            runtime_version=version,
            model=model,
            embedding_shape=shape,
            embedding_dtype=str(dtype) if dtype is not None else None,
        )
    except Exception as exc:
        probe_result = ProbeResult(
            status="failure",
            interface=INTERFACE_NAME,
            runtime_version=version,
            model=model,
            error_type=type(exc).__name__,
        )
    return probe_result, write_report(output, probe_result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 1 local speaker-embedding feasibility probe (default-off)."
    )
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="explicit disposable directory for shape-only diagnostics",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional disposable model cache directory; never a repository path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result, report_path = run_probe(
            args.audio_path,
            args.output_dir,
            model=args.model,
            token_env=args.token_env,
            cache_dir=args.cache_dir,
        )
    except ValueError as exc:
        print(f"PROBE_REJECTED reason={type(exc).__name__}")
        return 2

    print(f"PROBE_STATUS={result.status}")
    print(f"PROBE_INTERFACE={result.interface}")
    print(f"PROBE_RUNTIME_VERSION={result.runtime_version or 'unavailable'}")
    if result.embedding_shape is not None:
        print(f"PROBE_EMBEDDING_SHAPE={result.embedding_shape}")
    if result.error_type is not None:
        print(f"PROBE_ERROR_TYPE={result.error_type}")
    print(f"PROBE_REPORT_WRITTEN={report_path.name}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
