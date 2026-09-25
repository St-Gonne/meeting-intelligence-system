"""Project-scoped WhisperMLX launch adapter; never edits the installed package.

Bound the reusable MLX allocator cache and record content-free backend language
evidence before WhisperMLX's alignment writer can overwrite its language field.
The decoding defaults and language-selection behaviour remain upstream-owned.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from transcription_quality import assess_backend_result

CACHE_LIMIT_BYTES = 512 * 1024 * 1024


def guarded_backend(transcribe, memory, observations):
    def run(*args, **kwargs):
        try:
            result = transcribe(*args, **kwargs)
            quality = assess_backend_result(result)
            observations.append({
                "requested_language": kwargs.get("language"),
                "decoded_language": result.get("language"),
                "quality": quality,
                "peak_mlx_bytes": int(memory.get_peak_memory()),
            })
            if not quality["accepted"]:
                raise RuntimeError("transcription_quality:repetition_loop; review source audio")
            return result
        finally:
            memory.clear_cache()
    return run


def main() -> int:
    # This entrypoint is called only with the helper's explicit output directory.
    try:
        output = Path(sys.argv[sys.argv.index("--output_dir") + 1])
    except (ValueError, IndexError):
        raise SystemExit("MeetingIntel runtime requires --output_dir")
    if not output.is_dir():
        raise SystemExit("MeetingIntel runtime output directory must already exist")
    import mlx.core as mx
    import mlx_whisper
    from whispermlx.__main__ import cli

    observations = []
    original = mlx_whisper.transcribe
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mlx_whisper.transcribe = guarded_backend(original, mx, observations)
    succeeded = False
    try:
        cli()
        succeeded = True
        return 0
    finally:
        mlx_whisper.transcribe = original
        report = {
            "schema_version": 1,
            "cache_limit_bytes": CACHE_LIMIT_BYTES,
            "cache_cleared_after_each_backend_call": True,
            "completed": succeeded,
            "chunks": observations,
        }
        path = output / "transcription_runtime.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
