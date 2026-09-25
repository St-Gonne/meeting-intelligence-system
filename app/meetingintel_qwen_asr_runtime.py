#!/usr/bin/env python3
"""Offline Qwen3-ASR adapter for fixed contiguous MeetingIntel windows."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FFMPEG = Path(str(public_path('home/homebrew/bin/ffmpeg')))
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 60
MIN_WINDOW_SECONDS = 30
MAX_NEW_TOKENS = 1_024
LANGUAGE_CODES = {"English": "en", "Hindi": "hi"}


def split_for_missing_eos(sample_start: int, sample_end: int) -> list[tuple[int, int, int]]:
    """Bisect once to the admitted minimum; never accept truncated output."""
    if sample_end - sample_start <= MIN_WINDOW_SECONDS * SAMPLE_RATE:
        raise ValueError("qwen_runtime:generation_not_terminated_at_minimum_window")
    midpoint = sample_start + (sample_end - sample_start) // 2
    return [(sample_start, midpoint, sample_start), (midpoint, sample_end, sample_start)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def model_provenance(model_dir: Path) -> dict[str, object]:
    required = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")
    assets = []
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"qwen_runtime:missing_model_asset:{name}")
        target = path.resolve()
        assets.append({"name": name, "size": target.stat().st_size, "sha256": file_sha256(target)})
    return {"identifier": "Qwen/Qwen3-ASR-1.7B-hf", "revision": model_dir.name, "assets": assets}


def dependency_report(model_dir: Path) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    try:
        import torch
        import transformers
        transformers_version = transformers.__version__
        torch_version = torch.__version__
        mps_built = bool(torch.backends.mps.is_built())
        mps_available = bool(torch.backends.mps.is_available())
    except Exception as exc:
        failures.append({"prerequisite": "python_runtime", "category": type(exc).__name__})
        transformers_version = None
        torch_version = None
        mps_built = False
        mps_available = False
    interpreter = Path(sys.executable)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        failures.append({"prerequisite": "python_interpreter", "category": "not_executable"})
    if not FFMPEG.is_file() or not os.access(FFMPEG, os.X_OK):
        failures.append({"prerequisite": "ffmpeg", "category": "missing_or_not_executable"})
    if not mps_built:
        failures.append({"prerequisite": "mps", "category": "torch_not_built_with_mps"})
    elif not mps_available:
        failures.append({"prerequisite": "mps", "category": "unavailable_in_current_process"})
    try:
        model = model_provenance(model_dir)
    except (OSError, ValueError) as exc:
        failures.append({"prerequisite": "model_assets", "category": str(exc)})
        model = None
    return {
        "status": "ready" if not failures else "failed",
        "failed_prerequisites": failures,
        "python": str(interpreter.resolve()),
        "selected_interpreter": str(interpreter),
        "transformers": transformers_version,
        "torch": torch_version,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "ffmpeg": str(FFMPEG),
        "ffmpeg_executable": FFMPEG.is_file() and os.access(FFMPEG, os.X_OK),
        "model": model,
        "policy": {"window_seconds": WINDOW_SECONDS, "minimum_window_seconds": MIN_WINDOW_SECONDS,
                   "split_on_missing_eos": True, "max_new_tokens": MAX_NEW_TOKENS,
                   "automatic_language": True, "do_sample": False},
    }


def transcribe_waveform(
    waveform: np.ndarray,
    output: Path,
    processor: object,
    model: object,
    torch_module: object,
    initial_chunks: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Generate admitted chunks, saving content-free diagnostics on failure."""
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    chunks: list[dict[str, object]] = list(initial_chunks or [])
    window_samples = WINDOW_SECONDS * SAMPLE_RATE
    resume_start = int(chunks[-1]["sample_end"]) if chunks else 0
    pending = [(sample_start, min(waveform.size, sample_start + window_samples), None)
               for sample_start in range(resume_start, waveform.size, window_samples)]
    while pending:
        sample_start, sample_end, parent_start = pending.pop(0)
        samples = waveform[sample_start:sample_end]
        inputs = processor.apply_transcription_request(audio=samples).to("mps", torch_module.float16)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch_module.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                return_dict_in_generate=True,
            )
        ids = generated.sequences[:, prompt_tokens:]
        token_ids = ids[0].detach().cpu().tolist()
        ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
        if not ended_with_eos:
            try:
                pending[0:0] = split_for_missing_eos(sample_start, sample_end)
                continue
            except ValueError as exc:
                atomic_json(output / "qwen_asr_failure.json", {
                    "schema_version": 1, "status": "failed",
                    "failure_category": "generation_not_terminated_at_minimum_window",
                    "chunk_index": len(chunks), "sample_start": sample_start,
                    "sample_end": sample_end, "start": sample_start / SAMPLE_RATE,
                    "end": sample_end / SAMPLE_RATE,
                })
                raise ValueError("qwen_runtime:generation_not_terminated_at_minimum_window") from exc
        try:
            decoded = processor.decode(ids, return_format="parsed")
            parsed = decoded[0] if isinstance(decoded, list) and decoded else None
        except Exception as exc:
            parsed = None
            parse_category = type(exc).__name__
        else:
            parse_category = "empty_or_invalid_parsed_output"
        if not isinstance(parsed, dict):
            atomic_json(output / "qwen_asr_failure.json", {
                "schema_version": 1, "status": "failed", "failure_category": parse_category,
                "chunk_index": len(chunks), "sample_start": sample_start, "sample_end": sample_end,
                "start": sample_start / SAMPLE_RATE, "end": sample_end / SAMPLE_RATE,
            })
            raise ValueError("qwen_runtime:invalid_parsed_output")
        language_name = parsed.get("language")
        text = parsed.get("transcription")
        empty_resolution = None
        if not isinstance(text, str) or not text.strip():
            if sample_end - sample_start > MIN_WINDOW_SECONDS * SAMPLE_RATE:
                pending[0:0] = split_for_missing_eos(sample_start, sample_end)
                continue
            text = ""
            language_name = "English"
            empty_resolution = "no_transcription_at_minimum_window"
        language_resolution = "model"
        if language_name not in LANGUAGE_CODES:
            if sample_end - sample_start > MIN_WINDOW_SECONDS * SAMPLE_RATE:
                pending[0:0] = split_for_missing_eos(sample_start, sample_end)
                continue
            language_code = "hi" if any("\u0900" <= char <= "\u097f" for char in text) else "en"
            language_resolution = "script_fallback"
        else:
            language_code = LANGUAGE_CODES[language_name]
        start, end = sample_start / SAMPLE_RATE, sample_end / SAMPLE_RATE
        record = {
            "index": len(chunks), "start": start, "end": end,
            "parent_start": parent_start / SAMPLE_RATE if parent_start is not None else None,
            "sample_start": sample_start, "sample_end": sample_end,
            "sample_count": sample_end - sample_start,
            "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "language": language_code, "detected_language": language_name,
            "language_resolution": language_resolution,
            "empty_resolution": empty_resolution,
            "text": text, "output_token_count": len(token_ids),
            "max_new_tokens": MAX_NEW_TOKENS, "ended_with_eos": True,
            "stop_reason": "eos",
        }
        chunks.append(record)
        atomic_json(output / "backend_chunks.json", chunks)
    return chunks


def load_resume_chunks(waveform: np.ndarray, resume_dir: Path) -> list[dict[str, object]]:
    """Validate and reuse a contiguous private Qwen prefix for the exact audio."""
    root = resume_dir.expanduser().resolve(strict=True)
    payload = json.loads((root / "backend_chunks.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("qwen_runtime:resume_chunks_missing")
    expected_start = 0
    validated: list[dict[str, object]] = []
    for index, chunk in enumerate(payload):
        if not isinstance(chunk, dict):
            raise ValueError("qwen_runtime:resume_chunk_malformed")
        sample_start, sample_end = chunk.get("sample_start"), chunk.get("sample_end")
        if (
            chunk.get("index") != index
            or not isinstance(sample_start, int)
            or not isinstance(sample_end, int)
            or sample_start != expected_start
            or sample_end <= sample_start
            or sample_end > waveform.size
            or chunk.get("sample_count") != sample_end - sample_start
            or chunk.get("start") != sample_start / SAMPLE_RATE
            or chunk.get("end") != sample_end / SAMPLE_RATE
            or chunk.get("language") not in {"en", "hi"}
            or not isinstance(chunk.get("text"), str)
            or (
                not str(chunk.get("text")).strip()
                and chunk.get("empty_resolution") != "no_transcription_at_minimum_window"
            )
            or chunk.get("ended_with_eos") is not True
        ):
            raise ValueError("qwen_runtime:resume_chunk_malformed")
        samples = waveform[sample_start:sample_end]
        if chunk.get("sample_sha256") != hashlib.sha256(samples.tobytes()).hexdigest():
            raise ValueError("qwen_runtime:resume_audio_mismatch")
        validated.append(dict(chunk))
        expected_start = sample_end
    return validated


def decode_audio(source: Path) -> np.ndarray:
    completed = subprocess.run([
        str(FFMPEG), "-nostdin", "-threads", "0", "-i", str(source),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-",
    ], check=True, capture_output=True)
    return np.frombuffer(completed.stdout, np.int16).flatten().astype(np.float32) / 32768.0


def run(
    source: Path,
    output: Path,
    model_dir: Path,
    resume_dir: Path | None = None,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    source = source.resolve(strict=True)
    output = output.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    for owned in ("qwen_asr_failure.json", "qwen_asr_manifest.json",
                  "raw_transcription.json", "backend_chunks.json"):
        (output / owned).unlink(missing_ok=True)
    provenance = dependency_report(model_dir)
    if provenance["status"] != "ready":
        atomic_json(output / "qwen_asr_failure.json", {
            "schema_version": 1, "status": "failed", "failure_category": "prerequisite_failed",
            "failed_prerequisites": provenance["failed_prerequisites"],
        })
        raise ValueError("qwen_runtime:prerequisite_failed")
    waveform = decode_audio(source)
    waveform_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
    initial_chunks = load_resume_chunks(waveform, resume_dir) if resume_dir else []
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float16
    ).to("mps").eval()
    started = time.monotonic()
    chunks = transcribe_waveform(
        waveform, output, processor, model, torch, initial_chunks
    )
    segments = [{"start": c["start"], "end": c["end"], "text": c["text"],
                 "language": c["language"]} for c in chunks]
    raw = {"text": " ".join(x["text"] for x in segments if x["text"]), "segments": segments}
    atomic_json(output / "raw_transcription.json", raw)
    manifest = {
        "schema_version": 1, "status": "success", "source_sha256": file_sha256(source),
        "waveform": {"sample_rate": SAMPLE_RATE, "dtype": str(waveform.dtype),
                     "sample_count": int(waveform.size), "sha256": waveform_hash},
        "model": provenance["model"], "runtime": {"python": provenance["python"],
          "transformers": provenance["transformers"], "torch": provenance["torch"]},
        "policy": provenance["policy"], "chunk_count": len(chunks),
        "all_generations_ended_with_eos": True,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "raw_sha256": file_sha256(output / "raw_transcription.json"),
        "backend_chunks_sha256": file_sha256(output / "backend_chunks.json"),
        "resumed_chunk_count": len(initial_chunks),
    }
    atomic_json(output / "qwen_asr_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.check:
        if args.audio or args.output or args.resume_dir:
            parser.error("--check does not accept --audio, --output, or --resume-dir")
        report = dependency_report(args.model_dir)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ready" else 1
    if args.audio is None or args.output is None:
        parser.error("transcription requires --audio and --output")
    run(args.audio, args.output, args.model_dir, args.resume_dir)
    return 0


if __name__ == "__main__":
    from meetingintel_gpu import supervised_main
    raise SystemExit(supervised_main(main))
