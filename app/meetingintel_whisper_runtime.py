"""Project-scoped WhisperMLX launch adapter; never edits the installed package.

Bound the reusable MLX allocator cache and record content-free backend language
evidence before WhisperMLX's alignment writer can overwrite its language field.
Pass WhisperMLX's intended independent-window policy to the MLX decoder; the
installed wrapper otherwise drops it. Other decoding and language controls stay
upstream-owned.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from transcription_quality import assess_backend_result
from meetingintel_progress import worker_activity
from meetingintel_speaker_turns import project_speaker_turns, preserve_overlap_uncertainty
import math

CACHE_LIMIT_BYTES = 512 * 1024 * 1024


def guarded_speaker_assignment(assign):
    def run(*args, **kwargs):
        diarization = args[0] if args else kwargs["diarize_df"]
        assigned = assign(*args, **kwargs)
        return project_speaker_turns(preserve_overlap_uncertainty(
            assigned, diarization.to_dict("records")))
    return run


def guarded_backend(transcribe, memory, observations):
    def run(*args, **kwargs):
        try:
            # WhisperMLX constructs this policy in transcribe.py but its MLX
            # adapter forwards only initial_prompt from asr_options. MLX's own
            # default is True, which upstream documents as prone to loops.
            kwargs["condition_on_previous_text"] = False
            result = transcribe(*args, **kwargs)
            quality = assess_backend_result(result)
            temperatures = sorted({float(segment["temperature"]) for segment in result.get("segments", [])
                                   if type(segment.get("temperature")) in {int, float}
                                   and math.isfinite(segment["temperature"])})
            observations.append({
                "chunk_index": len(observations) + 1,
                "requested_language": kwargs.get("language"),
                "decoded_language": result.get("language"),
                "condition_on_previous_text": kwargs["condition_on_previous_text"],
                "observed_decode_temperatures": temperatures,
                "quality": quality,
                "peak_mlx_bytes": int(memory.get_peak_memory()),
            })
            worker_activity("transcription")
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
    import importlib
    transcribe_module = importlib.import_module("whispermlx.transcribe")

    observations = []
    original = mlx_whisper.transcribe
    original_assign = transcribe_module.assign_word_speakers
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mlx_whisper.transcribe = guarded_backend(original, mx, observations)
    transcribe_module.assign_word_speakers = guarded_speaker_assignment(original_assign)
    succeeded = False
    try:
        cli()
        succeeded = True
        return 0
    finally:
        mlx_whisper.transcribe = original
        transcribe_module.assign_word_speakers = original_assign
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
