"""Isolated multilingual transcription candidate; not production discovery.

The default uses the backend's standard automatic language detection afresh for
each 30-second VAD chunk, then retains each decoded language for alignment.
The earlier forced English/Hindi selector remains available only to reproduce
the failed experiment; it is not the default repair.
Private output must be a new directory outside the repository and source tree.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import importlib.metadata
import json
import os
import subprocess
from meetingintel_gpu import child_options
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from meetingintel_whisper_runtime import CACHE_LIMIT_BYTES
from transcription_quality import assess_backend_result, assess_segments

ASR_CHECKPOINT_SCHEMA_VERSION = 2
ASR_OPTIONS = {
    "language": None, "task": "transcribe", "verbose": False,
    "initial_prompt": None, "word_timestamps": False,
}


def admission_assessment(report: dict) -> dict[str, object]:
    """Missing or unbound review evidence is unknown, never accepted."""
    assessment = report.get("asr_content_review")
    if not isinstance(assessment, dict):
        return {"status": "not_assessed", "eligible": False}
    bound = (
        assessment.get("source_sha256") == report.get("source_sha256")
        and assessment.get("asr_backend") == report.get("asr_backend")
        and assessment.get("model_revision") == report.get("model_revision")
    )
    accepted = (
        assessment.get("status") == "accepted"
        and bound
        and not assessment.get("unresolved_indices")
    )
    return {
        "status": "accepted" if accepted else ("failed" if assessment.get("status") == "failed" else "not_assessed"),
        "eligible": accepted,
        "provenance_bound": bound,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def callable_sha256(function: Callable) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str]:
    names = ("whispermlx", "mlx-whisper", "mlx", "torch", "pyannote.audio")
    return {name: importlib.metadata.version(name) for name in names}


def model_asset_provenance(model_identifier: str) -> dict[str, object]:
    if "/" not in model_identifier:
        raise ValueError("candidate:model_identity_not_reproducible")
    cache = Path.home() / ".cache/huggingface/hub" / (
        "models--" + model_identifier.replace("/", "--")
    )
    revision = (cache / "refs/main").read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    assets = []
    for path in sorted(p for p in snapshot.iterdir() if p.is_file()):
        resolved = path.resolve(strict=True)
        assets.append({"name": path.name, "size": resolved.stat().st_size,
                       "blob": resolved.name, "sha256": sha256_file(resolved) if path.name == "config.json" else None})
    if not assets:
        raise ValueError("candidate:model_assets_missing")
    return {"identifier": model_identifier, "revision": revision, "assets": assets}


def vad_bounds_evidence(vad_segments, waveform) -> dict[str, object]:
    bounds = [{"start": float(s["start"]), "end": float(s["end"]),
               "first_sample": int(float(s["start"]) * 16000),
               "last_sample": int(float(s["end"]) * 16000)} for s in vad_segments]
    encoded = json.dumps(bounds, sort_keys=True, separators=(",", ":")).encode()
    return {"count": len(bounds), "bounds": bounds,
            "bounds_sha256": hashlib.sha256(encoded).hexdigest(),
            "waveform": {"sample_rate": 16000, "dtype": str(waveform.dtype),
                         "sample_count": len(waveform),
                         "sha256": hashlib.sha256(memoryview(waveform).cast("B")).hexdigest()}}


def fallback_provenance(reference_python: Path | None,
                        reference_model_cache: Path | None) -> dict[str, object]:
    if reference_python is None or reference_model_cache is None:
        return {"enabled": False}
    script = Path(__file__).with_name("meetingintel_openai_whisper_fallback.py")
    checkpoint = reference_model_cache / "large-v3.pt"
    version = subprocess.run(
        [str(reference_python), "-c", "import importlib.metadata as m; print(m.version('openai-whisper'))"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"enabled": True, "implementation_sha256": sha256_file(script),
            "openai_whisper_version": version, "model": "large-v3",
            "model_sha256": sha256_file(checkpoint), "device": "cpu", "fp16": False,
            "options": {"language": None, "task": "transcribe", "verbose": False}}


def validate_asr_checkpoint(checkpoint, source_sha256, raw_bytes, backend_bytes,
                            waveform, expected_model, expected_chunk_seconds,
                            expected_language_policy, expected_fallback,
                            expected_vad_runtime_sha256=None):
    if checkpoint.get("schema_version") != ASR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("candidate:asr_checkpoint_schema_mismatch")
    expected = {
        "source_sha256": source_sha256,
        "model": expected_model,
        "chunk_seconds": expected_chunk_seconds,
        "language_policy": expected_language_policy,
        "runtime_versions": package_versions(),
        "decoder_options": ASR_OPTIONS,
        "transcribe_vad_chunks_sha256": callable_sha256(transcribe_vad_chunks),
        "quality_guard_sha256": callable_sha256(assess_backend_result),
        "fallback": expected_fallback,
    }
    if expected_vad_runtime_sha256 is not None:
        expected["vad_runtime_sha256"] = expected_vad_runtime_sha256
    expected["vad_options"] = {"chunk_size": expected_chunk_seconds}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate:asr_checkpoint_provenance_mismatch")
    if sha256_file(Path(checkpoint["source_audio"])) != source_sha256:
        raise ValueError("candidate:asr_checkpoint_source_mismatch")
    if hashlib.sha256(raw_bytes).hexdigest() != checkpoint.get("raw_sha256") or hashlib.sha256(backend_bytes).hexdigest() != checkpoint.get("backend_chunks_sha256"):
        raise ValueError("candidate:asr_checkpoint_hash_mismatch")
    vad = checkpoint.get("vad", {}); chunks = json.loads(backend_bytes)
    if vad.get("count") != len(chunks) or checkpoint.get("raw_segment_count") != len(json.loads(raw_bytes).get("segments", [])):
        raise ValueError("candidate:asr_checkpoint_count_mismatch")
    current_wave = {"sample_rate": 16000, "dtype": str(waveform.dtype), "sample_count": len(waveform),
                    "sha256": hashlib.sha256(memoryview(waveform).cast("B")).hexdigest()}
    if vad.get("waveform") != current_wave:
        raise ValueError("candidate:asr_checkpoint_waveform_mismatch")
    bounds = vad.get("bounds", [])
    if len(bounds) != len(chunks) or any(float(b["start"]) != float(c["start"]) or float(b["end"]) != float(c["end"]) or b["first_sample"] != int(float(c["start"])*16000) or b["last_sample"] != int(float(c["end"])*16000) for b,c in zip(bounds,chunks)):
        raise ValueError("candidate:asr_checkpoint_bounds_mismatch")
    return True


def choose_bilingual_language(probabilities):
    if any(language not in probabilities for language in ("en", "hi")):
        raise ValueError("candidate:missing_language_probabilities")
    return max(("en", "hi"), key=lambda language: probabilities[language])


def requested_backend_language(language_policy, probabilities=None):
    """Return the language override, keeping standard auto truly unforced."""
    if language_policy in {"standard_auto", "unrestricted"}:
        return None
    if language_policy == "bilingual":
        if probabilities is None:
            raise ValueError("candidate:missing_language_probabilities")
        return choose_bilingual_language(probabilities)
    raise ValueError("candidate:unsupported_language_policy")


def attach_languages(segments, observations):
    if len(segments) != len(observations):
        raise ValueError("candidate:chunk_language_count_mismatch")
    return [dict(segment, language=observation["decoded_language"])
            for segment, observation in zip(segments, observations)]


def merge_aligned(results):
    segments = sorted([s for r in results for s in r["segments"]], key=lambda s: (s["start"], s["end"]))
    words = [w for s in segments for w in s.get("words", [])]
    languages = sorted({s["language"] for s in segments})
    return {"segments": segments, "word_segments": words,
            "language": languages[0] if len(languages) == 1 else "mixed",
            "languages": languages}


def alignment_preserves_text(raw, aligned):
    def compact(segments):
        return "".join("".join(s["text"].split()) for s in segments)
    return compact(raw["segments"]) == compact(aligned["segments"])


def alignment_languages_for(segments):
    """Require alignment assets only for chunks containing spoken text."""
    return {
        segment["language"] for segment in segments
        if segment.get("text", "").strip()
    }


def prepare_output(audio: Path, output: Path, allow_precreated: bool = False):
    source = audio.resolve(strict=True)
    if not source.is_file():
        raise ValueError("candidate:source_not_file")
    destination = output.parent.resolve(strict=True) / output.name
    for protected in (Path(__file__).resolve().parent, source.parent):
        if destination.is_relative_to(protected):
            raise ValueError("candidate:protected_output_root")
    if allow_precreated:
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("candidate:precreated_output_not_empty")
        destination.chmod(0o700)
    else:
        destination.mkdir(mode=0o700, exist_ok=False)
    return source, destination


def transcribe_vad_chunks(
    waveform,
    vad_segments,
    backend: Callable,
    model_path: str,
    language_policy: str,
    clear_cache: Callable,
    on_chunk: Callable | None = None,
    fallback_backend: Callable | None = None,
):
    """Decode real VAD arrays independently and retain content-free provenance."""
    segments = []
    observations = []
    for index, vad_segment in enumerate(vad_segments):
        start = float(vad_segment["start"])
        end = float(vad_segment["end"])
        first = int(start * 16000)
        last = int(end * 16000)
        audio_chunk = waveform[first:last]
        requested_language = requested_backend_language(language_policy)
        try:
            result = backend(
                audio_chunk,
                path_or_hf_repo=model_path,
                language=requested_language,
                task="transcribe",
                verbose=False,
                initial_prompt=None,
                word_timestamps=False,
            )
            quality = assess_backend_result(result)
            primary_quality = quality
            fallback_used = False
            if not quality["accepted"] and fallback_backend is not None:
                result = fallback_backend(audio_chunk, index)
                quality = assess_backend_result(result)
                fallback_used = True
            observation = {
                "index": index,
                "start": start,
                "end": end,
                "sample_count": len(audio_chunk),
                "requested_language": requested_language,
                "decoded_language": result["language"],
                "quality": quality,
                "primary_quality": primary_quality,
                "fallback_used": fallback_used,
            }
            observations.append(observation)
            segments.append({
                "text": result.get("text", "").strip(),
                "start": round(start, 3),
                "end": round(end, 3),
                "language": result["language"],
            })
            if on_chunk is not None:
                on_chunk(observation, result)
            if not quality["accepted"]:
                raise ValueError("candidate:repetition_requires_review")
        finally:
            clear_cache()
    languages = {segment["language"] for segment in segments}
    return {
        "segments": segments,
        "language": next(iter(languages)) if len(languages) == 1 else "mixed",
    }, observations


def run(
    audio: Path,
    output: Path,
    chunk_seconds: int = 30,
    language_policy: str = "standard_auto",
    min_speakers: int = 2,
    max_speakers: int = 2,
    allow_precreated_output: bool = False,
    alignment_model_dir: Path | None = None,
    reference_python: Path | None = None,
    reference_model_cache: Path | None = None,
    resume_asr_dir: Path | None = None,
    asr_backend: str = "whispermlx",
    qwen_python: Path | None = None,
    qwen_model_dir: Path | None = None,
):
    if chunk_seconds not in (10, 15, 20, 30):
        raise ValueError("candidate:unsupported_chunk_size")
    if language_policy not in ("standard_auto", "bilingual", "unrestricted"):
        raise ValueError("candidate:unsupported_language_policy")
    if min_speakers < 1 or max_speakers < min_speakers:
        raise ValueError("candidate:invalid_speaker_limits")
    if (reference_python is None) != (reference_model_cache is None):
        raise ValueError("candidate:incomplete_reference_fallback")
    if reference_python is not None and resume_asr_dir is None:
        raise ValueError("candidate:sampled_reference_fallback_not_admitted")
    if asr_backend not in {"whispermlx", "qwen3-asr-1.7b-contiguous"}:
        raise ValueError("candidate:unsupported_asr_backend")
    if asr_backend == "qwen3-asr-1.7b-contiguous":
        if qwen_python is None or qwen_model_dir is None:
            raise ValueError("candidate:qwen_runtime_required")
        if reference_python is not None:
            raise ValueError("candidate:qwen_reference_fallback_unsupported")
    elif qwen_python is not None or qwen_model_dir is not None:
        raise ValueError("candidate:qwen_options_require_qwen_backend")
    source, destination = prepare_output(audio, output, allow_precreated_output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    started = time.monotonic()
    report = {"schema_version": 1, "status": "running", "source_audio": str(source),
              "candidate_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "model": "large-v3" if asr_backend == "whispermlx" else "Qwen/Qwen3-ASR-1.7B-hf",
              "asr_backend": asr_backend, "chunk_seconds": chunk_seconds if asr_backend == "whispermlx" else 60,
              "language_policy": language_policy, "task": "transcribe",
              "min_speakers": min_speakers, "max_speakers": max_speakers,
              "cache_limit_bytes": CACHE_LIMIT_BYTES, "phases": {}, "chunks": []}

    def save(name, data):
        temporary = destination / (name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.chmod(0o600)
        temporary.replace(destination / name)

    save("candidate_manifest.json", report)
    try:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        from whispermlx import asr
        from whispermlx.audio import load_audio
        from whispermlx.alignment import (
            DEFAULT_ALIGN_MODELS_HF,
            DEFAULT_ALIGN_MODELS_TORCH,
            align,
            load_align_model,
        )
        from whispermlx.diarize import DiarizationPipeline, assign_word_speakers
        from whispermlx.utils import get_writer
        import torch

        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        backend_chunks = []
        model_evidence = model_asset_provenance("mlx-community/whisper-large-v3-mlx") if asr_backend == "whispermlx" else None
        fallback_evidence = fallback_provenance(reference_python, reference_model_cache)

        phase = time.monotonic()
        waveform = load_audio(str(source))
        if asr_backend == "qwen3-asr-1.7b-contiguous":
            command = [
                str(qwen_python), str(Path(__file__).with_name("meetingintel_qwen_asr_runtime.py")),
                "--model-dir", str(qwen_model_dir), "--audio", str(source),
                "--output", str(destination),
            ]
            if resume_asr_dir is not None:
                command.extend(["--resume-dir", str(resume_asr_dir)])
            environment = os.environ.copy()
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
            completed = subprocess.run(command, env=environment, check=False, **child_options())
            if completed.returncode != 0:
                raise ValueError("candidate:qwen_asr_failed")
            qwen_manifest = json.loads((destination / "qwen_asr_manifest.json").read_text(encoding="utf-8"))
            raw = json.loads((destination / "raw_transcription.json").read_text(encoding="utf-8"))
            backend_chunks = json.loads((destination / "backend_chunks.json").read_text(encoding="utf-8"))
            if any(
                not c.get("empty_resolution")
                and not assess_backend_result({"text": c.get("text", ""), "segments": []})["accepted"]
                for c in backend_chunks
            ):
                raise ValueError("candidate:repetition_requires_review")
            report["chunks"] = [
                {
                    **{k: c[k] for k in ("index", "start", "end", "sample_count", "sample_sha256", "language", "output_token_count", "max_new_tokens", "ended_with_eos", "stop_reason")},
                    "empty_resolution": c.get("empty_resolution"),
                    "language_resolution": c.get("language_resolution", "model"),
                }
                for c in backend_chunks
            ]
            report["qwen_asr"] = qwen_manifest
            report["model_revision"] = qwen_manifest["model"]["revision"]
            report["asr_content_review"] = {"status": "not_assessed"}
        elif resume_asr_dir is not None:
            checkpoint_dir = resume_asr_dir.resolve(strict=True)
            checkpoint = json.loads(
                (checkpoint_dir / "asr_checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            raw_bytes = (checkpoint_dir / "raw_transcription.json").read_bytes()
            backend_bytes = (checkpoint_dir / "backend_chunks.json").read_bytes()
            validate_asr_checkpoint(
                checkpoint, report["source_sha256"], raw_bytes, backend_bytes,
                waveform, model_evidence, chunk_seconds, language_policy,
                fallback_evidence, sha256_file(Path(asr.__file__)),
            )
            raw = json.loads(raw_bytes)
            report["asr_checkpoint"] = {"path": str(checkpoint_dir),
                                        "raw_sha256": checkpoint["raw_sha256"],
                                        "validated": True}
            report["asr_content_review"] = checkpoint.get(
                "content_review", {"status": "not_recorded"}
            )
        else:
            pipeline = asr.load_model("large-v3", device="mps", language=None,
                                      vad_options={"chunk_size": chunk_seconds}, local_files_only=True)
            try:
                if hasattr(pipeline.vad_model, "preprocess_audio"):
                    vad_waveform = pipeline.vad_model.preprocess_audio(waveform)
                    merge_chunks = pipeline.vad_model.merge_chunks
                else:
                    vad_waveform = asr.Pyannote.preprocess_audio(waveform)
                    merge_chunks = asr.Pyannote.merge_chunks
                vad_segments = pipeline.vad_model(
                    {"waveform": vad_waveform, "sample_rate": 16000}
                )
                vad_segments = merge_chunks(
                    vad_segments, chunk_seconds,
                    onset=pipeline._vad_params["vad_onset"],
                    offset=pipeline._vad_params["vad_offset"],
                )
                vad_evidence = vad_bounds_evidence(vad_segments, waveform)

                def record_chunk(observation, result):
                    observation["peak_mlx_bytes"] = int(mx.get_peak_memory())
                    report["chunks"].append(observation)
                    backend_chunks.append({**observation, "text": result.get("text", "")})
                    save("backend_chunks.json", backend_chunks)
                    save("candidate_manifest.json", report)

                def official_fallback(audio_chunk, index):
                    if reference_python is None or reference_model_cache is None:
                        raise ValueError("candidate:repetition_requires_review")
                    import numpy as np
                    samples = destination / f"fallback-{index:04d}-samples.npy"
                    raw_output = destination / f"fallback-{index:04d}-raw.json"
                    np.save(samples, np.asarray(audio_chunk, dtype=np.float32), allow_pickle=False)
                    samples.chmod(0o600)
                    command = [
                        str(reference_python),
                        str(Path(__file__).with_name("meetingintel_openai_whisper_fallback.py")),
                        "--samples", str(samples), "--output", str(raw_output),
                        "--model-cache", str(reference_model_cache),
                    ]
                    completed = subprocess.run(command, check=False, **child_options())
                    if completed.returncode != 0 or not raw_output.is_file():
                        raise ValueError("candidate:reference_fallback_failed")
                    return json.loads(raw_output.read_text(encoding="utf-8"))

                raw, _ = transcribe_vad_chunks(
                    waveform, vad_segments, mlx_whisper.transcribe,
                    pipeline.model_path, language_policy, mx.clear_cache,
                    record_chunk, official_fallback if reference_python else None,
                )
            finally:
                del pipeline
                ModelHolder.model = None
                ModelHolder.model_path = None
                gc.collect()
                mx.clear_cache()
                torch.mps.empty_cache()
        save("raw_transcription.json", raw)
        report["phases"]["transcription_seconds"] = round(time.monotonic() - phase, 2)
        if resume_asr_dir is None and asr_backend == "whispermlx":
            raw_path = destination / "raw_transcription.json"
            backend_path = destination / "backend_chunks.json"
            checkpoint = {
                "schema_version": ASR_CHECKPOINT_SCHEMA_VERSION,
                "source_audio": str(source), "source_sha256": report["source_sha256"],
                "model": model_evidence, "chunk_seconds": chunk_seconds,
                "language_policy": language_policy, "decoder_options": ASR_OPTIONS,
                "runtime_versions": package_versions(),
                "transcribe_vad_chunks_sha256": callable_sha256(transcribe_vad_chunks),
                "quality_guard_sha256": callable_sha256(assess_backend_result),
                "vad_runtime_sha256": sha256_file(Path(asr.__file__)),
                "vad_options": {"chunk_size": chunk_seconds}, "vad": vad_evidence,
                "fallback": fallback_evidence,
                "raw_sha256": sha256_file(raw_path),
                "backend_chunks_sha256": sha256_file(backend_path),
                "raw_segment_count": len(raw.get("segments", [])),
                "transcription_seconds": report["phases"]["transcription_seconds"],
            }
            save("asr_checkpoint_manifest.json", checkpoint)
            report["asr_checkpoint_written"] = str(destination / "asr_checkpoint_manifest.json")
        report["language_counts"] = dict(Counter(s["language"] for s in raw["segments"]))
        report["quality"] = assess_segments(raw["segments"])
        save("candidate_manifest.json", report)
        if not report["quality"]["accepted"]:
            raise ValueError("candidate:repetition_requires_review")
        supported_alignment_languages = (
            set(DEFAULT_ALIGN_MODELS_TORCH) | set(DEFAULT_ALIGN_MODELS_HF)
        )
        alignment_languages = alignment_languages_for(raw["segments"])
        unsupported = alignment_languages - supported_alignment_languages
        report["non_english_hindi_languages"] = sorted(
            set(report["language_counts"]) - {"en", "hi"}
        )
        if unsupported:
            report["unsupported_alignment_languages"] = sorted(unsupported)
            raise ValueError("candidate:unsupported_alignment_language")

        phase = time.monotonic()
        results = []
        for language in sorted(alignment_languages):
            subset = [s for s in raw["segments"]
                      if s["language"] == language and s.get("text", "").strip()]
            # The private supplemental cache is for newly encountered languages;
            # retain the established English/Hindi caches and model provenance.
            model_dir = (
                str(alignment_model_dir)
                if alignment_model_dir and language not in {"en", "hi"}
                else None
            )
            model, metadata = load_align_model(
                language,
                "mps",
                model_dir=model_dir,
                model_cache_only=True,
            )
            try:
                result = align(subset, model, metadata, waveform, "mps")
                for segment in result["segments"]:
                    segment["language"] = language
                results.append(result)
            finally:
                del model
                gc.collect()
                torch.mps.empty_cache()
        aligned = merge_aligned(results)
        save("aligned_transcription.json", aligned)
        report["alignment_text_preserved"] = alignment_preserves_text(raw, aligned)
        if not report["alignment_text_preserved"]:
            raise ValueError("candidate:alignment_changed_text")
        report["phases"]["alignment_seconds"] = round(time.monotonic() - phase, 2)
        save("candidate_manifest.json", report)

        phase = time.monotonic()
        diarizer = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1",
                                      token=os.environ.get("HF_TOKEN"), device="mps")
        turns = diarizer(
            str(source), min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = assign_word_speakers(turns, aligned)
        report["phases"]["diarization_seconds"] = round(time.monotonic() - phase, 2)
        report["anonymous_speakers"] = sorted({s.get("speaker", "unknown") for s in result["segments"]})
        report["unaligned_words"] = sum("start" not in w or "end" not in w for w in result.get("word_segments", []))
        report["word_count"] = len(result.get("word_segments", []))
        writer = get_writer("all", str(destination))
        writer(result, str(source), {"highlight_words": False, "max_line_count": None, "max_line_width": None})
        for path in destination.iterdir():
            if path.is_file():
                path.chmod(0o600)
        admission = admission_assessment(report)
        report["quality_assessment"] = admission
        report["quality_admission_eligible"] = admission["eligible"]
        report["status"] = "ready_for_review" if admission["eligible"] else "ready_for_review_not_assessed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_category"] = str(exc) if str(exc).startswith("candidate:") else "provider_failure_see_private_log"
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save("candidate_manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=int, choices=(10, 15, 20, 30), default=30)
    parser.add_argument(
        "--language-policy",
        choices=("standard_auto", "bilingual", "unrestricted"),
        default="standard_auto",
    )
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--alignment-model-dir", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-model-cache", type=Path)
    parser.add_argument("--resume-asr-dir", type=Path)
    parser.add_argument("--asr-backend", choices=("whispermlx", "qwen3-asr-1.7b-contiguous"), default="whispermlx")
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--qwen-model-dir", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    run(
        args.audio,
        args.output,
        args.chunk_seconds,
        args.language_policy,
        args.min_speakers,
        args.max_speakers,
        alignment_model_dir=args.alignment_model_dir,
        reference_python=args.reference_python,
        reference_model_cache=args.reference_model_cache,
        resume_asr_dir=args.resume_asr_dir,
        asr_backend=args.asr_backend,
        qwen_python=args.qwen_python,
        qwen_model_dir=args.qwen_model_dir,
    )


if __name__ == "__main__":
    main()
