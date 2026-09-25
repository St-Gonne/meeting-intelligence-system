#!/usr/bin/env python3
"""Operator-controlled WhisperMLX diarization for explicit meeting audio."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


MEETILY_ROOT = Path(str(public_path('home/Movies/meetily-recordings')))
OUTPUT_ROOT = Path(str(public_path('work/diarization')))
TRIMMED_AUDIO_ROOT = Path(str(public_path('work/trimmed_audio')))
WHISPERMLX_BIN = Path(str(public_path('home/whispermlx-env/bin/whispermlx')))
WHISPERMLX_RUNTIME = Path(__file__).with_name("meetingintel_whisper_runtime.py")
FFMPEG_BIN = Path(str(public_path('home/homebrew/bin/ffmpeg')))
MODEL = "large-v3"
QWEN_PYTHON = Path(str(public_path('work/transcription-runtime/qwen-env/bin/python')))
QWEN_MODEL_DIR = Path(str(public_path('work/transcription-runtime/cache/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3')))
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
CANONICAL_OUTPUT_EXTENSIONS = ("srt", "txt", "tsv", "json", "vtt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated WhisperMLX diarization on selected Meetily folders.",
        epilog=(
            "Date selection uses the final YYYY-MM-DD token in each folder name. "
            f"Outputs are written below {OUTPUT_ROOT}."
        ),
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--meeting-folder",
        type=Path,
        help="Exact Meetily meeting folder path.",
    )
    selector.add_argument(
        "--today",
        action="store_true",
        help="Select folders whose final date token is today's local date.",
    )
    selector.add_argument(
        "--date-from",
        type=parse_iso_date,
        metavar="YYYY-MM-DD",
        help="Start of an inclusive folder-date range; requires --date-to.",
    )
    selector.add_argument(
        "--audio-path",
        type=Path,
        help="Explicit validated source audio path for non-Meetily admission.",
    )
    parser.add_argument("--source-folder", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument(
        "--language", choices=("en", "hi", "auto"), default="auto",
        help="Primary decoding language override for isolated review (default: auto).",
    )
    parser.add_argument(
        "--transcription-mode",
        choices=("legacy", "repaired-standard-auto", "qwen3-asr-1.7b-contiguous"),
        default="legacy",
        help="Explicit isolated repaired mode; legacy remains the default.",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        help="New exact private output directory for repaired-standard-auto mode.",
    )
    parser.add_argument(
        "--alignment-model-dir",
        type=Path,
        help="Optional private alignment cache for repaired mode.",
    )
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-model-cache", type=Path)
    parser.add_argument("--resume-asr-dir", type=Path)
    parser.add_argument(
        "--date-to",
        type=parse_iso_date,
        metavar="YYYY-MM-DD",
        help="End of an inclusive folder-date range; requires --date-from.",
    )
    parser.add_argument("--min-speakers", type=positive_int, default=2)
    parser.add_argument("--max-speakers", type=positive_int, default=2)
    parser.add_argument(
        "--audio-override",
        type=Path,
        help="Use a pre-trimmed audio file under the MeetingIntel test trim root.",
    )
    args = parser.parse_args()

    if (args.date_from is None) != (args.date_to is None):
        parser.error("--date-from and --date-to must be provided together")
    if args.date_to is not None and args.date_from > args.date_to:
        parser.error("--date-from cannot be later than --date-to")
    if args.min_speakers > args.max_speakers:
        parser.error("--min-speakers cannot be greater than --max-speakers")
    if args.audio_override is not None and args.meeting_folder is None:
        parser.error("--audio-override requires --meeting-folder")
    isolated_mode = args.transcription_mode in {"repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}
    if isolated_mode != (
        args.private_output is not None
    ):
        parser.error(
            "--transcription-mode repaired-standard-auto requires --private-output, "
            "and --private-output is not accepted by legacy mode"
        )
    if isolated_mode and args.language != "auto":
        parser.error("isolated transcription modes require --language auto")
    if args.alignment_model_dir is not None and not isolated_mode:
        parser.error("--alignment-model-dir requires an isolated transcription mode")
    if (args.reference_python is None) != (args.reference_model_cache is None):
        parser.error("reference fallback requires both --reference-python and --reference-model-cache")
    if args.reference_python is not None and args.transcription_mode != "repaired-standard-auto":
        parser.error("reference fallback requires repaired-standard-auto")
    if args.reference_python is not None and args.resume_asr_dir is None:
        parser.error("sampled reference fallback is not admitted for new ASR; references are accepted only to validate an existing checkpoint")
    if args.resume_asr_dir is not None and args.transcription_mode not in {
        "repaired-standard-auto", "qwen3-asr-1.7b-contiguous"
    }:
        parser.error("--resume-asr-dir requires an isolated transcription mode")
    if args.resume_asr_dir is not None and not args.resume_asr_dir.expanduser().is_dir():
        parser.error("--resume-asr-dir must be an existing checkpoint directory")
    explicit = args.audio_path is not None
    if explicit != (args.source_folder is not None) or explicit != bool(args.source_id):
        parser.error("--audio-path requires --source-folder and --source-id")
    if args.source_id is not None and re.fullmatch(r"[0-9a-f]{64}", args.source_id) is None:
        parser.error("--source-id must be a lowercase 64-character SHA-256 identity")
    return args


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(2)


def available_meeting_folders() -> list[Path]:
    if not MEETILY_ROOT.is_dir():
        fail(f"Meetily recordings folder was not found: {MEETILY_ROOT}")
    return sorted(
        folder
        for folder in MEETILY_ROOT.iterdir()
        if folder.is_dir() and (folder / "audio.mp4").is_file()
    )


def folder_date(folder: Path) -> date | None:
    matches = DATE_PATTERN.findall(folder.name)
    if not matches:
        return None
    try:
        return date.fromisoformat(matches[-1])
    except ValueError:
        return None


def select_folders(args: argparse.Namespace) -> list[Path]:
    available = available_meeting_folders()
    root = MEETILY_ROOT.resolve()

    if args.meeting_folder is not None:
        selected = args.meeting_folder.expanduser().resolve()
        if selected.parent != root:
            fail(f"--meeting-folder must be a direct child of {MEETILY_ROOT}")
        if selected not in [folder.resolve() for folder in available]:
            fail(f"folder does not exist or does not contain audio.mp4: {selected}")
        return [selected]

    start = end = date.today() if args.today else None
    if args.date_from is not None:
        start, end = args.date_from, args.date_to

    assert start is not None and end is not None
    return [
        folder
        for folder in available
        if (meeting_date := folder_date(folder)) is not None
        and start <= meeting_date <= end
    ]


def create_output_folder(source_folder: Path, source_label: str | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate = OUTPUT_ROOT / (source_label or source_folder.name)
    if not candidate.exists():
        candidate.mkdir()
        return candidate

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    counter = 0
    while True:
        suffix = timestamp if counter == 0 else f"{timestamp}_{counter}"
        candidate = OUTPUT_ROOT / f"{source_label or source_folder.name}__{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            counter += 1


def validate_audio_override(raw_path: Path) -> Path:
    audio_path = raw_path.expanduser().resolve()
    trim_root = TRIMMED_AUDIO_ROOT.resolve()
    try:
        audio_path.relative_to(trim_root)
    except ValueError as exc:
        fail(f"audio override must be inside {TRIMMED_AUDIO_ROOT}: {audio_path}")
    if not audio_path.is_file():
        fail(f"audio override was not found: {audio_path}")
    if audio_path.stat().st_size <= 0:
        fail(f"audio override is empty: {audio_path}")
    return audio_path


def validate_explicit_audio(raw_path: Path) -> Path:
    try:
        audio_path = raw_path.expanduser().resolve(strict=True)
    except OSError:
        fail(f"explicit audio path was not found: {raw_path.expanduser()}")
    if not audio_path.is_file() or not stat.S_ISREG(audio_path.stat().st_mode):
        fail(f"explicit audio path is not a regular file: {audio_path}")
    return audio_path


def normalize_output_files(
    output_folder: Path,
    source_audio: Path,
) -> list[dict[str, str]]:
    copies: list[dict[str, str]] = []
    preferred_stem = source_audio.stem
    for extension in CANONICAL_OUTPUT_EXTENSIONS:
        canonical = output_folder / f"audio.{extension}"
        if canonical.exists():
            continue

        preferred = output_folder / f"{preferred_stem}.{extension}"
        if preferred.is_file() and preferred != canonical:
            source = preferred
        else:
            candidates = sorted(
                path
                for path in output_folder.glob(f"*.{extension}")
                if path != canonical and path.is_file()
                and path.name not in {"transcription_runtime.json", "run_manifest.json"}
            )
            if not candidates:
                continue
            if len(candidates) > 1:
                rendered = ", ".join(str(path) for path in candidates)
                raise RuntimeError(
                    f"ambiguous WhisperMLX .{extension} outputs: {rendered}"
                )
            source = candidates[0]

        shutil.copy2(source, canonical)
        copies.append(
            {
                "source": str(source),
                "canonical": str(canonical),
            }
        )

    if not (output_folder / "audio.srt").is_file() and not (
        output_folder / "audio.txt"
    ).is_file():
        raise RuntimeError(
            "WhisperMLX completed but canonical audio.srt/audio.txt could not be created"
        )
    return copies


def write_manifest(output_folder: Path, manifest: dict[str, object]) -> None:
    manifest_path = output_folder / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_diarization(
    source_folder: Path,
    min_speakers: int,
    max_speakers: int,
    hf_token: str,
    audio_override: Path | None = None,
    source_mode: str = "meetily",
    source_identity: str | None = None,
    language: str = "auto",
    transcription_mode: str = "legacy",
    private_output: Path | None = None,
    alignment_model_dir: Path | None = None,
    reference_python: Path | None = None,
    reference_model_cache: Path | None = None,
    resume_asr_dir: Path | None = None,
) -> bool:
    if language not in {"en", "hi", "auto"}:
        raise ValueError("unsupported transcription language")
    original_source_audio = source_folder / "audio.mp4"
    source_audio = audio_override or original_source_audio
    source_label = (
        f"audio_first_{source_identity}"
        if source_mode == "audio_first" and source_identity
        else None
    )
    if transcription_mode not in {"legacy", "repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}:
        raise ValueError("unsupported transcription mode")
    if reference_python is not None and resume_asr_dir is None:
        raise ValueError("sampled reference fallback is not admitted for new ASR")
    if transcription_mode in {"repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}:
        if private_output is None:
            raise ValueError("repaired mode requires private output")
        output_folder = private_output.expanduser().resolve()
        if output_folder.exists():
            raise ValueError("private output must not already exist")
        for protected in (Path(__file__).resolve().parent, source_audio.parent.resolve()):
            if output_folder.is_relative_to(protected):
                raise ValueError("private output must be outside protected roots")
        output_folder.parent.resolve(strict=True)
        output_folder.mkdir(mode=0o700)
    else:
        if private_output is not None:
            raise ValueError("legacy mode does not accept private output")
        output_folder = create_output_folder(source_folder, source_label)
    started_at = datetime.now().astimezone().isoformat()
    command = [
        str(WHISPERMLX_BIN.with_name("python")),
        str(WHISPERMLX_RUNTIME),
        str(source_audio),
        "--model",
        MODEL,
        "--diarize",
        "--min_speakers",
        str(min_speakers),
        "--max_speakers",
        str(max_speakers),
        "--output_dir",
        str(output_folder),
        "--output_format",
        "all",
    ]
    if language != "auto":
        command.extend(["--language", language])
    command.extend(["--task", "transcribe"])
    if transcription_mode in {"repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}:
        command = [
            str(WHISPERMLX_BIN.with_name("python")),
            str(Path(__file__).resolve()),
            "--audio-path", str(source_audio),
            "--source-folder", str(source_folder),
            "--source-id", str(source_identity),
            "--min-speakers", str(min_speakers),
            "--max-speakers", str(max_speakers),
            "--transcription-mode", transcription_mode,
            "--private-output", str(output_folder),
        ]
        if alignment_model_dir is not None:
            command.extend(["--alignment-model-dir", str(alignment_model_dir)])
        if reference_python is not None and reference_model_cache is not None:
            command.extend([
                "--reference-python", str(reference_python),
                "--reference-model-cache", str(reference_model_cache),
            ])
        if resume_asr_dir is not None:
            command.extend(["--resume-asr-dir", str(resume_asr_dir)])

    manifest: dict[str, object] = {
        "source_folder": str(source_folder),
        "source_audio": str(source_audio),
        "original_source_audio": str(original_source_audio),
        "audio_override_used": audio_override is not None,
        "source_mode": source_mode,
        "source_identity": source_identity,
        "output_folder": str(output_folder),
        "model": MODEL if transcription_mode != "qwen3-asr-1.7b-contiguous" else "Qwen/Qwen3-ASR-1.7B-hf",
        "transcription_language": language,
        "transcription_task": "transcribe",
        "runtime_adapter": (
            "multilingual_transcription_candidate_v1"
            if transcription_mode in {"repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}
            else "meetingintel_whisper_runtime_v1"
        ),
        "transcription_mode": transcription_mode,
        "chunk_seconds": 60 if transcription_mode == "qwen3-asr-1.7b-contiguous" else (30 if transcription_mode == "repaired-standard-auto" else None),
        "resume_asr_dir": str(resume_asr_dir) if resume_asr_dir else None,
        "asr_resumed": resume_asr_dir is not None,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "command_used": shlex.join(command),
        "hf_auth_source": "HF_TOKEN environment",
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
        "output_normalization": {
            "applied": False,
            "preferred_source_stem": source_audio.stem,
            "copies": [],
        },
    }

    print(f"Running: {source_audio}")
    print(f"Output:  {output_folder}")
    try:
        if transcription_mode in {"repaired-standard-auto", "qwen3-asr-1.7b-contiguous"}:
            from multilingual_transcription_candidate import run as run_multilingual

            candidate_report = run_multilingual(
                source_audio,
                output_folder,
                chunk_seconds=30,
                language_policy="standard_auto",
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                allow_precreated_output=True,
                alignment_model_dir=alignment_model_dir,
                reference_python=reference_python,
                reference_model_cache=reference_model_cache,
                resume_asr_dir=resume_asr_dir,
                asr_backend=("qwen3-asr-1.7b-contiguous" if transcription_mode == "qwen3-asr-1.7b-contiguous" else "whispermlx"),
                qwen_python=QWEN_PYTHON if transcription_mode == "qwen3-asr-1.7b-contiguous" else None,
                qwen_model_dir=QWEN_MODEL_DIR if transcription_mode == "qwen3-asr-1.7b-contiguous" else None,
            )
            manifest["candidate_stage_provenance"] = {
                "candidate_code_sha256": candidate_report.get("candidate_code_sha256"),
                "asr_checkpoint": candidate_report.get("asr_checkpoint"),
                "asr_checkpoint_written": candidate_report.get("asr_checkpoint_written"),
                "phase_seconds": candidate_report.get("phases", {}),
                "qwen_asr": candidate_report.get("qwen_asr"),
                "quality_assessment": candidate_report.get("quality_assessment"),
            }
            result = subprocess.CompletedProcess(command, 0)
        else:
            environment = os.environ.copy()
            environment["PATH"] = f"{FFMPEG_BIN.parent}:{environment.get('PATH', '')}"
            result = subprocess.run(command, env=environment, check=False)
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["exit_code"] = result.returncode
        if result.returncode == 0:
            try:
                normalized_copies = normalize_output_files(
                    output_folder,
                    source_audio,
                )
                manifest["output_normalization"] = {
                    "applied": bool(normalized_copies),
                    "preferred_source_stem": source_audio.stem,
                    "copies": normalized_copies,
                }
                for copied in normalized_copies:
                    print(
                        "Normalized output: "
                        f"{copied['source']} -> {copied['canonical']}"
                    )
                manifest["status"] = "success"
                print(f"Success: {source_folder.name}")
                success = True
            except (OSError, RuntimeError) as exc:
                manifest["status"] = "failure"
                manifest["error"] = f"Output normalization failed: {exc}"
                print(f"Failure: {source_folder.name} ({exc})")
                success = False
        else:
            manifest["status"] = "failure"
            manifest["error"] = f"WhisperMLX exited with status {result.returncode}"
            print(f"Failure: {source_folder.name} (exit {result.returncode})")
            success = False
    except (OSError, RuntimeError, ValueError) as exc:
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["status"] = "failure"
        manifest["error"] = str(exc)
        print(f"Failure: {source_folder.name} ({exc})")
        success = False

    write_manifest(output_folder, manifest)
    print(f"Manifest: {output_folder / 'run_manifest.json'}")
    return success


def validate_environment(transcription_mode: str = "legacy") -> str:
    if not WHISPERMLX_BIN.is_file():
        fail(f"WhisperMLX binary was not found: {WHISPERMLX_BIN}")
    if not FFMPEG_BIN.is_file():
        fail(f"ffmpeg was not found: {FFMPEG_BIN}")
    if transcription_mode == "qwen3-asr-1.7b-contiguous":
        if not QWEN_PYTHON.is_file() or not QWEN_MODEL_DIR.is_dir():
            fail("stable Qwen transcription runtime is incomplete")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token and transcription_mode == "legacy":
        fail(
            "HF_TOKEN is not set. Set it in your environment before running "
            "this helper; do not place the token in the command itself."
        )
    return hf_token


def main() -> int:
    args = parse_args()
    hf_token = validate_environment(args.transcription_mode)
    if args.audio_path is not None:
        try:
            source_folder = args.source_folder.expanduser().resolve(strict=True)
        except OSError:
            fail(f"explicit source folder was not found: {args.source_folder}")
        if not source_folder.is_dir():
            fail(f"explicit source folder is not a directory: {source_folder}")
        source_audio = validate_explicit_audio(args.audio_path)
        print("Selected 1 explicit audio source:")
        print(f"  - {source_folder}")
        success = run_diarization(
            source_folder,
            args.min_speakers,
            args.max_speakers,
            hf_token,
            source_audio,
            source_mode="audio_first",
            source_identity=args.source_id,
            language=args.language,
            transcription_mode=args.transcription_mode,
            private_output=args.private_output,
            alignment_model_dir=args.alignment_model_dir,
            reference_python=args.reference_python,
            reference_model_cache=args.reference_model_cache,
            resume_asr_dir=args.resume_asr_dir,
        )
        print(f"\nFinished: {1 if success else 0} succeeded, {0 if success else 1} failed.")
        return 0 if success else 1

    selected = select_folders(args)
    audio_override = (
        validate_audio_override(args.audio_override)
        if args.audio_override is not None
        else None
    )
    if not selected:
        print("No Meetily folders with audio.mp4 matched the selection.")
        return 0

    print(f"Selected {len(selected)} folder(s):")
    for folder in selected:
        print(f"  - {folder}")

    failures = 0
    for index, folder in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {folder.name}")
        if not run_diarization(
            folder,
            args.min_speakers,
            args.max_speakers,
            hf_token,
            audio_override,
            language=args.language,
            transcription_mode=args.transcription_mode,
            private_output=args.private_output,
            alignment_model_dir=args.alignment_model_dir,
            reference_python=args.reference_python,
            reference_model_cache=args.reference_model_cache,
            resume_asr_dir=args.resume_asr_dir,
        ):
            failures += 1

    print(f"\nFinished: {len(selected) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    from meetingintel_gpu import supervised_main
    raise SystemExit(supervised_main(main))
