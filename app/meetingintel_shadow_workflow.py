#!/usr/bin/env python3
"""Operator-friendly orchestration for the diarized MeetingIntel shadow path."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import fcntl
import json
import math
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(str(public_path('project')))
MEETILY_ROOT = Path(str(public_path('home/Movies/meetily-recordings')))
DIARIZATION_ROOT = Path(str(public_path('work/diarization')))
SHADOW_OUTPUT_ROOT = Path(
    str(public_path('work/layer2_diarized_shadow'))
)
TRIMMED_AUDIO_ROOT = Path(str(public_path('work/trimmed_audio')))
DIARIZATION_SCRIPT = PROJECT_ROOT / "whispermlx_diarization_helper.py"
SHADOW_SCRIPT = PROJECT_ROOT / "diarized_layer2_shadow.py"
FFMPEG_BIN = Path(str(public_path('home/homebrew/bin/ffmpeg')))
FFPROBE_BIN = Path(str(public_path('home/homebrew/bin/ffprobe')))
LOCK_PATH = Path("/tmp/meetingintel_shadow_workflow.lock")


class WorkflowError(RuntimeError):
    """An expected, operator-actionable workflow failure."""


@dataclass
class WorkflowResult:
    diarization_folder: Path
    layer2_path: str
    layer3_path: str
    comparison_path: str
    manifest_path: Path
    warnings: list[str]
    trim_status: str
    stage_timings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrimResult:
    audio_path: Path
    manifest_path: Path
    details: dict[str, Any]


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def positive_minutes(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive number of minutes") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive number of minutes")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manual diarized MeetingIntel shadow workflow."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--meeting-folder",
        type=Path,
        help="Exact Meetily meeting folder; runs diarization first.",
    )
    inputs.add_argument(
        "--diarization-folder",
        type=Path,
        help="Exact existing diarization output folder; skips diarization.",
    )
    parser.add_argument(
        "--with-layer3",
        action="store_true",
        help="Generate the optional shadow Layer 3 brief after Layer 2.",
    )
    parser.add_argument("--min-speakers", type=positive_int, default=2)
    parser.add_argument("--max-speakers", type=positive_int, default=2)
    parser.add_argument(
        "--trim-after-minutes",
        type=positive_minutes,
        metavar="N",
        help="Keep only the first N minutes before diarization (meeting-folder only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without subprocesses or writes.",
    )
    parser.add_argument(
        "--speaker-map",
        action="append",
        default=[],
        metavar="SPEAKER_XX=NAME|unknown",
        help=(
            "Operator-confirmed speaker mapping. Repeat for each supplied label; "
            "the diarized runner prompts only for missing material labels."
        ),
    )
    args = parser.parse_args(argv)
    if args.min_speakers > args.max_speakers:
        parser.error("--min-speakers cannot be greater than --max-speakers")
    if args.trim_after_minutes is not None and args.diarization_folder is not None:
        parser.error(
            "trimming existing diarization is unsupported; rerun from the original "
            "Meetily folder"
        )
    return args


def validate_exact_child(raw_path: Path, root: Path, description: str) -> Path:
    path = raw_path.expanduser().resolve()
    if path.parent != root.resolve():
        raise WorkflowError(f"{description} must be a direct child of {root}: {path}")
    if not path.is_dir():
        raise WorkflowError(f"{description} was not found: {path}")
    return path


def validate_meeting_folder(raw_path: Path) -> Path:
    folder = validate_exact_child(raw_path, MEETILY_ROOT, "meeting folder")
    if not (folder / "audio.mp4").is_file():
        raise WorkflowError(f"meeting folder does not contain audio.mp4: {folder}")
    return folder


def validate_diarization_folder(raw_path: Path) -> Path:
    folder = validate_exact_child(
        raw_path,
        DIARIZATION_ROOT,
        "diarization folder",
    )
    if not (folder / "audio.srt").is_file() and not (folder / "audio.txt").is_file():
        raise WorkflowError(
            f"diarization folder contains neither audio.srt nor audio.txt: {folder}"
        )
    return folder


def build_diarization_command(
    meeting_folder: Path,
    min_speakers: int,
    max_speakers: int,
    audio_override: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(DIARIZATION_SCRIPT),
        "--meeting-folder",
        str(meeting_folder),
        "--min-speakers",
        str(min_speakers),
        "--max-speakers",
        str(max_speakers),
    ]
    if audio_override is not None:
        command.extend(["--audio-override", str(audio_override)])
    return command


def build_shadow_command(
    diarization_folder: Path,
    with_layer3: bool,
    speaker_map: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SHADOW_SCRIPT),
        "--diarization-folder",
        str(diarization_folder),
    ]
    if with_layer3:
        command.append("--with-layer3")
    for mapping in speaker_map or []:
        command.extend(["--speaker-map", mapping])
    return command


def format_number(value: float) -> str:
    return f"{value:g}"


def build_ffprobe_command(source_audio: Path) -> list[str]:
    return [
        str(FFPROBE_BIN),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_audio),
    ]


def build_ffmpeg_command(
    source_audio: Path,
    trimmed_audio: Path,
    trim_seconds: float,
) -> list[str]:
    return [
        str(FFMPEG_BIN),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(source_audio),
        "-t",
        format_number(trim_seconds),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(trimmed_audio),
    ]


def probe_source_duration(source_audio: Path) -> float:
    if not FFPROBE_BIN.is_file():
        raise WorkflowError(f"ffprobe was not found: {FFPROBE_BIN}")
    command = build_ffprobe_command(source_audio)
    print(f"\n[Trim probe] {shlex.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise WorkflowError(f"ffprobe could not start: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise WorkflowError(f"ffprobe failed: {detail}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise WorkflowError(
            f"ffprobe returned an invalid source duration: {result.stdout!r}"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise WorkflowError(f"ffprobe returned an invalid source duration: {duration}")
    return duration


def create_trim_folder(meeting_folder: Path) -> Path:
    TRIMMED_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", meeting_folder.name).strip("_")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    base = TRIMMED_AUDIO_ROOT / f"{safe_name or 'meeting'}__trim_{timestamp}"
    candidate = base
    counter = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = Path(f"{base}_{counter}")
            counter += 1


def validate_trimmed_audio(path: Path) -> None:
    if not path.is_file():
        raise WorkflowError(f"ffmpeg did not create trimmed audio: {path}")
    if path.stat().st_size <= 0:
        raise WorkflowError(f"ffmpeg created an empty trimmed audio file: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def prepare_trim(meeting_folder: Path, trim_minutes: float) -> TrimResult:
    if not math.isfinite(trim_minutes) or trim_minutes <= 0:
        raise WorkflowError("trim duration must be a positive number of minutes")
    source_audio = meeting_folder / "audio.mp4"
    source_duration = probe_source_duration(source_audio)
    trim_seconds = trim_minutes * 60
    if trim_seconds >= source_duration:
        raise WorkflowError(
            f"trim cutoff ({format_number(trim_minutes)} minutes) must be shorter "
            f"than source duration ({source_duration / 60:.2f} minutes)"
        )
    if not FFMPEG_BIN.is_file():
        raise WorkflowError(f"ffmpeg was not found: {FFMPEG_BIN}")

    trim_folder = create_trim_folder(meeting_folder)
    trimmed_audio = trim_folder / "audio_trimmed.mp4"
    trim_manifest_path = trim_folder / "trim_manifest.json"
    ffmpeg_command = build_ffmpeg_command(
        source_audio,
        trimmed_audio,
        trim_seconds,
    )
    run_child(ffmpeg_command, "Trim audio")
    validate_trimmed_audio(trimmed_audio)

    details: dict[str, Any] = {
        "applied": True,
        "mode": "after_minutes",
        "trim_after_minutes": trim_minutes,
        "trim_after_seconds": trim_seconds,
        "source_duration_seconds": source_duration,
        "original_audio": str(source_audio),
        "trimmed_audio": str(trimmed_audio),
        "ffprobe_command": shlex.join(build_ffprobe_command(source_audio)),
        "ffmpeg_command": shlex.join(ffmpeg_command),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    write_json(trim_manifest_path, {"workflow_trim": details})
    return TrimResult(
        audio_path=trimmed_audio,
        manifest_path=trim_manifest_path,
        details=details,
    )


def snapshot_manifests(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.glob("*/run_manifest.json")}


def discover_new_manifest(root: Path, before: set[Path], stage: str) -> Path:
    new_manifests = sorted(snapshot_manifests(root) - before)
    if not new_manifests:
        raise WorkflowError(f"{stage} created no new run_manifest.json")
    if len(new_manifests) > 1:
        rendered = "\n  - ".join(str(path) for path in new_manifests)
        raise WorkflowError(
            f"{stage} output discovery is ambiguous; found "
            f"{len(new_manifests)} new manifests:\n  - {rendered}"
        )
    return new_manifests[0]


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"could not read child manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"child manifest is not a JSON object: {path}")
    return payload


def record_trim_details(
    manifest_path: Path,
    manifest: dict[str, Any],
    trim_details: dict[str, Any],
) -> None:
    manifest["workflow_trim"] = trim_details
    write_json(manifest_path, manifest)


def inherited_trim_details(diarization_folder: Path) -> dict[str, Any]:
    manifest_path = diarization_folder / "run_manifest.json"
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        details = manifest.get("workflow_trim")
        if isinstance(details, dict):
            return details
    return {"applied": False, "mode": "none"}


def require_manifest_path(
    manifest: dict[str, Any],
    field: str,
    expected: Path,
    stage: str,
) -> None:
    raw_value = manifest.get(field)
    if not isinstance(raw_value, str) or Path(raw_value).expanduser().resolve() != expected:
        raise WorkflowError(
            f"{stage} manifest field {field!r} does not match the exact input"
        )


def run_child(command: list[str], stage: str) -> None:
    print(f"\n[{stage}] {shlex.join(command)}", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise WorkflowError(f"{stage} failed with exit code {result.returncode}")


@contextmanager
def workflow_lock() -> Iterator[None]:
    handle = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(
                "another MeetingIntel shadow workflow is already running"
            ) from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def collect_warnings(manifest: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in (
        "speaker_attribution_warnings",
        "layer3_attribution_warnings",
    ):
        values = manifest.get(field, [])
        if isinstance(values, list):
            warnings.extend(str(value) for value in values)
    return warnings


def stage_entry(status: str, duration_seconds: float | None = None) -> dict[str, Any]:
    return {"status": status, "duration_seconds": duration_seconds}


def runner_stage_entry(manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    timings = manifest.get("stage_timings")
    if isinstance(timings, dict) and isinstance(timings.get(stage), dict):
        entry = timings[stage]
        return stage_entry(
            str(entry.get("status", "unknown")),
            (
                float(entry["duration_seconds"])
                if isinstance(entry.get("duration_seconds"), (int, float))
                else None
            ),
        )
    return stage_entry("unavailable")


def build_workflow_stage_timings(
    trim: dict[str, Any],
    diarization: dict[str, Any],
    shadow_manifest: dict[str, Any],
    workflow_elapsed_seconds: float,
) -> dict[str, dict[str, Any]]:
    speaker_confirmation = runner_stage_entry(
        shadow_manifest, "speaker_confirmation"
    )
    layer2 = runner_stage_entry(shadow_manifest, "layer2")
    layer3 = runner_stage_entry(shadow_manifest, "layer3")
    measured_compute_stages = (trim, diarization, layer2, layer3)
    compute_total = sum(
        float(entry["duration_seconds"])
        for entry in measured_compute_stages
        if isinstance(entry.get("duration_seconds"), (int, float))
    )
    return {
        "trim": trim,
        "diarization": diarization,
        "speaker_confirmation": speaker_confirmation,
        "layer2": layer2,
        "layer3": layer3,
        "compute_stage_total": stage_entry("measured", compute_total),
        "workflow_elapsed": stage_entry("measured", workflow_elapsed_seconds),
    }


def format_stage_duration(entry: dict[str, Any]) -> str:
    duration = entry.get("duration_seconds")
    if isinstance(duration, (int, float)):
        return f"{float(duration):.1f}s"
    return str(entry.get("status", "unavailable"))


def result_from_manifest(
    diarization_folder: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    with_layer3: bool,
    trim_details: dict[str, Any],
) -> WorkflowResult:
    layer3 = manifest.get("layer3_path") if with_layer3 else "not requested"
    if with_layer3 and manifest.get("layer3_status") != "success":
        layer3 = f"not created (status: {manifest.get('layer3_status', 'unknown')})"
    return WorkflowResult(
        diarization_folder=diarization_folder,
        layer2_path=str(manifest.get("layer2_output") or "not created"),
        layer3_path=str(layer3 or "not created"),
        comparison_path=str(manifest.get("comparison_output") or "not created"),
        manifest_path=manifest_path,
        warnings=collect_warnings(manifest),
        trim_status=format_trim_status(trim_details),
        stage_timings=(
            manifest.get("workflow_stage_timings")
            if isinstance(manifest.get("workflow_stage_timings"), dict)
            else {}
        ),
    )


def format_trim_status(trim_details: dict[str, Any]) -> str:
    if not trim_details.get("applied"):
        return "not applied"
    minutes = float(trim_details["trim_after_minutes"])
    artifact = trim_details.get("trimmed_audio", "unknown artifact")
    return f"applied after {format_number(minutes)} minutes — {artifact}"


def print_final_outputs(result: WorkflowResult) -> None:
    print("\nSPEAKER STRUCTURE / FORMAT CHECKS")
    print("Lint only — does not verify factual speaker attribution.")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("None")
    if result.stage_timings:
        print("\nSTAGE TIMING")
        labels = (
            ("trim", "Trim"),
            ("diarization", "Diarization"),
            ("speaker_confirmation", "Speaker confirmation"),
            ("layer2", "Layer 2"),
            ("layer3", "Layer 3"),
            ("compute_stage_total", "Compute/stage total"),
            ("workflow_elapsed", "Workflow elapsed"),
        )
        for key, label in labels:
            entry = result.stage_timings.get(key, stage_entry("unavailable"))
            print(f"{label}: {format_stage_duration(entry)}")
    print("\nFINAL OUTPUTS")
    print(f"Diarization: {result.diarization_folder}")
    print(f"Trim: {result.trim_status}")
    print(f"Layer 2: {result.layer2_path}")
    print(f"Layer 3: {result.layer3_path}")
    print(f"Comparison: {result.comparison_path}")
    print(f"Manifest: {result.manifest_path}")
    print("\nSTATUS")
    print("Shadow only. Production output, ledger, and daily brief were not touched.")


def print_dry_run(args: argparse.Namespace, input_folder: Path) -> None:
    print("DRY RUN: no subprocesses will run and no files will be written.")
    if args.meeting_folder is not None:
        audio_override = None
        if args.trim_after_minutes is not None:
            source_audio = input_folder / "audio.mp4"
            trim_seconds = args.trim_after_minutes * 60
            planned_audio = (
                TRIMMED_AUDIO_ROOT
                / "<timestamped-trim-folder>"
                / "audio_trimmed.mp4"
            )
            print(
                f"ffprobe command: {shlex.join(build_ffprobe_command(source_audio))}"
            )
            print(
                "ffmpeg command: "
                f"{shlex.join(build_ffmpeg_command(source_audio, planned_audio, trim_seconds))}"
            )
            audio_override = planned_audio
        diarization_command = build_diarization_command(
            input_folder,
            args.min_speakers,
            args.max_speakers,
            audio_override,
        )
        print(f"Diarization command: {shlex.join(diarization_command)}")
        print(
            "Shadow command: "
            f"{sys.executable} {SHADOW_SCRIPT} --diarization-folder "
            "<exact new folder from child manifest>"
            f"{' --with-layer3' if args.with_layer3 else ''}"
            + "".join(
                f" --speaker-map {shlex.quote(mapping)}"
                for mapping in args.speaker_map
            )
        )
        diarization_display = "planned; exact path will come from child manifest"
        trim_display = (
            f"planned after {format_number(args.trim_after_minutes)} minutes "
            "(dry-run; not applied)"
            if args.trim_after_minutes is not None
            else "not applied"
        )
    else:
        shadow_command = build_shadow_command(
            input_folder, args.with_layer3, args.speaker_map
        )
        print(f"Shadow command: {shlex.join(shadow_command)}")
        diarization_display = str(input_folder)
        trim_display = "not applied"
    print("\nFINAL OUTPUTS")
    print(f"Diarization: {diarization_display}")
    print(f"Trim: {trim_display}")
    print("Layer 2: planned")
    print(f"Layer 3: {'planned' if args.with_layer3 else 'not requested'}")
    print("Comparison: planned")
    print("Manifest: planned")
    print("Speaker confirmation: planned in the diarized runner; dry-run will not prompt")
    print("Stage timing: planned; no durations are measured in dry-run")
    print("\nSTATUS")
    print("Shadow only. Production output, ledger, and daily brief were not touched.")


def run_workflow(args: argparse.Namespace) -> WorkflowResult | None:
    if args.meeting_folder is not None:
        input_folder = validate_meeting_folder(args.meeting_folder)
    else:
        input_folder = validate_diarization_folder(args.diarization_folder)

    if args.dry_run:
        print_dry_run(args, input_folder)
        return None

    workflow_started = time.monotonic()
    with workflow_lock():
        trim_details: dict[str, Any] = {"applied": False, "mode": "none"}
        trim_timing = stage_entry("not_applied")
        diarization_timing = stage_entry("reused")
        if args.meeting_folder is not None:
            audio_override = None
            if args.trim_after_minutes is not None:
                trim_started = time.monotonic()
                trim_result = prepare_trim(input_folder, args.trim_after_minutes)
                trim_timing = stage_entry(
                    "success", time.monotonic() - trim_started
                )
                audio_override = trim_result.audio_path
                trim_details = trim_result.details
            before_diarization = snapshot_manifests(DIARIZATION_ROOT)
            diarization_started = time.monotonic()
            run_child(
                build_diarization_command(
                    input_folder,
                    args.min_speakers,
                    args.max_speakers,
                    audio_override,
                ),
                "Diarization",
            )
            diarization_timing = stage_entry(
                "success", time.monotonic() - diarization_started
            )
            diarization_manifest_path = discover_new_manifest(
                DIARIZATION_ROOT,
                before_diarization,
                "Diarization",
            )
            diarization_manifest = load_manifest(diarization_manifest_path)
            require_manifest_path(
                diarization_manifest,
                "source_folder",
                input_folder,
                "Diarization",
            )
            if audio_override is not None:
                require_manifest_path(
                    diarization_manifest,
                    "source_audio",
                    audio_override,
                    "Diarization",
                )
            record_trim_details(
                diarization_manifest_path,
                diarization_manifest,
                trim_details,
            )
            diarization_folder = diarization_manifest_path.parent
        else:
            diarization_folder = input_folder
            trim_details = inherited_trim_details(diarization_folder)
            if trim_details.get("applied"):
                trim_timing = stage_entry("reused")

        before_shadow = snapshot_manifests(SHADOW_OUTPUT_ROOT)
        run_child(
            build_shadow_command(
                diarization_folder,
                args.with_layer3,
                getattr(args, "speaker_map", []),
            ),
            "Layer 2/3 shadow",
        )
        shadow_manifest_path = discover_new_manifest(
            SHADOW_OUTPUT_ROOT,
            before_shadow,
            "Layer 2/3 shadow",
        )
        shadow_manifest = load_manifest(shadow_manifest_path)
        require_manifest_path(
            shadow_manifest,
            "diarization_folder",
            diarization_folder,
            "Layer 2/3 shadow",
        )
        workflow_stage_timings = build_workflow_stage_timings(
            trim_timing,
            diarization_timing,
            shadow_manifest,
            time.monotonic() - workflow_started,
        )
        shadow_manifest["workflow_stage_timings"] = workflow_stage_timings
        record_trim_details(
            shadow_manifest_path,
            shadow_manifest,
            trim_details,
        )
        result = result_from_manifest(
            diarization_folder,
            shadow_manifest_path,
            shadow_manifest,
            args.with_layer3,
            trim_details,
        )

    print_final_outputs(result)
    return result


def main() -> int:
    try:
        args = parse_args()
        run_workflow(args)
        return 0
    except WorkflowError as exc:
        print(f"\nWORKFLOW ERROR: {exc}", file=sys.stderr)
        print(
            "Shadow only. Production output, ledger, and daily brief were not touched.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
