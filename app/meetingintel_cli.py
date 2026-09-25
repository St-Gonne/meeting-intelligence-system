#!/usr/bin/env python3
"""Small operator command layer for MeetingIntel production processing."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, MutableMapping, Sequence

from meetingintel_runtime import exclusive_operation, RuntimeSafetyError, source_annotation
import meetingintel_observe as observe
from meetingintel_help import help_text
import meetingintel_pipeline as pipeline
import laptop_capture_guard
from audio_first_meeting_source import (
    AudioFirstMeetingSource,
    SourceValidationError,
    discover_audio_first_sources,
    load_audio_first_source,
)
import laptop_capture_ingest as capture_ingest
from laptop_capture_ingest import (
    NormalizeConfig,
    normalize_capture,
    terminal_source_loss_recovery_eligible,
)
from laptop_capture_meeting_source import (
    LaptopMeetingSource,
    LaptopSourceValidationError,
    discover_laptop_meeting_sources,
    load_laptop_meeting_source,
)
from laptop_capture_contract import audit_capture_session, load_manifest
from meetingintel_inventory import format_inventory, inventory as build_inventory
import phone_recording_ingest as phone_ingest
import meetingintel_inbox as inbox
import phone_recording_review as phone_review
from phone_drive_fetch import (
    FetchConfig,
    classify_fetch_failure,
    fetch_recordings,
    format_scheduler_report,
    install_daily_schedule,
    print_fetch_status,
    remove_daily_schedule,
    scheduler_diagnose,
    scheduler_status,
    run_scheduler_probe,
    write_scheduler_receipt,
)
from ollama_endpoint import (
    OllamaEndpointError,
    OllamaUnreachable,
    preflight_ollama,
    resolve_ollama_endpoint,
)
from ollama_service import (
    MANAGED_ENDPOINT,
    OllamaServiceError,
    OllamaServiceManager,
    ServiceConfigInvalid,
    ServiceConflict,
    ServiceNotInstalled,
    ServiceRecoveryFailed,
    format_status,
)


PROJECT_ROOT = Path(str(public_path('project')))
MEETINGS_ROOT = pipeline.DEFAULT_MEETINGS_ROOT
AUDIO_FIRST_ROOT = public_path("project/ingest/phone")
LAPTOP_INGEST_ROOT = public_path("project/ingest/laptop")
LEDGER_PATH = pipeline.DEFAULT_LEDGER_PATH
OUTPUT_DIR = pipeline.DEFAULT_OUTPUT_DIR
SECRET_PATH = public_path("project/.secrets/meetingintel.env")
CHECKPOINT_ROOT = public_path("project/checkpoints")
LAPTOP_RECORDINGS_ROOT = public_path("project/ingest/captures")
OLLAMA_SERVICE_MANAGER = OllamaServiceManager()
VOICE_COMMAND_PYTHON = Path(str(public_path('home/whispermlx-env/bin/python')))
VOICE_COMMAND_SCRIPT = PROJECT_ROOT / "meetingintel_voice_cli.py"
SCOPES = ("today", "last", "new", "all")
PHONE_TRANSCRIPTION_OPTION = "--phone-transcription-backend"
CHEAT_SHEET_TEXT = help_text()



@dataclass(frozen=True)
class SourceCandidate:
    fingerprint: str
    created_at: datetime
    source_kind: str
    display_name: str


@dataclass(frozen=True)
class Discovery:
    candidates: tuple[SourceCandidate, ...]
    warnings: tuple[str, ...]


def parse_scope(argv: Sequence[str]) -> str:
    parser = argparse.ArgumentParser(prog="mi", description="Run MeetingIntel")
    parser.add_argument("scope", nargs="?", choices=SCOPES, default="today")
    parser.add_argument("--audio-first", action="store_true")
    return parser.parse_args(list(argv)).scope


def extract_phone_transcription_backend(argv: Sequence[str]) -> tuple[list[str], str]:
    """Remove one global phone-ASR selector without changing read-only dispatch."""
    command = list(argv)
    values: list[str] = []
    cleaned: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == PHONE_TRANSCRIPTION_OPTION:
            if index + 1 >= len(command):
                raise ValueError(f"{PHONE_TRANSCRIPTION_OPTION} requires legacy or qwen")
            values.append(command[index + 1]); index += 2; continue
        if token.startswith(PHONE_TRANSCRIPTION_OPTION + "="):
            values.append(token.split("=", 1)[1]); index += 1; continue
        cleaned.append(token); index += 1
    if len(values) > 1:
        raise ValueError("phone transcription backend may be selected only once")
    backend = values[0] if values else pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT
    if backend not in pipeline.PHONE_TRANSCRIPTION_BACKENDS:
        raise ValueError("phone transcription backend must be legacy or qwen")
    return cleaned, backend


def _candidate_sort_key(candidate: SourceCandidate) -> tuple[datetime, str, str]:
    return candidate.created_at, candidate.source_kind, candidate.fingerprint


def discover_candidates(
    meetings_root: Path = MEETINGS_ROOT,
    audio_first_root: Path | None = None,
    audio_first_only: bool = False,
) -> Discovery:
    candidates: list[SourceCandidate] = []
    warnings: list[str] = []
    if audio_first_only and audio_first_root is None:
        raise ValueError("audio-first-only mode requires an audio-first root")
    if not audio_first_only:
        for folder in pipeline.list_meeting_dirs(meetings_root):
            if not (folder / "metadata.json").is_file() or not (folder / "transcripts.json").is_file():
                continue
            try:
                metadata, _transcripts, transcript_hash, transcript_text, created_at = (
                    pipeline.parse_meeting_folder(folder)
                )
                if metadata.get("status") != "completed" or not transcript_text:
                    continue
                fingerprint = pipeline.build_fingerprint(
                    folder.name, metadata["created_at"], transcript_hash
                )
                candidates.append(
                    SourceCandidate(
                        fingerprint,
                        created_at,
                        "meetily",
                        metadata.get("meeting_name") or folder.name,
                    )
                )
            except Exception as exc:
                warnings.append(f"Meetily source rejected: {folder}: {type(exc).__name__}")

    if audio_first_root is not None and audio_first_root.exists():
        try:
            discoveries = discover_audio_first_sources(audio_first_root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            warnings.append(f"Phone source discovery failed: {exc}")
        else:
            for discovery in discoveries:
                if not discovery.valid or discovery.source is None:
                    warnings.append(
                        f"Phone source rejected: {discovery.source_folder}: {discovery.error}"
                    )
                    continue
                source = pipeline.production_source_from_audio_first(discovery.source)
                candidates.append(
                    SourceCandidate(
                        source.fingerprint,
                        source.created_at,
                        source.source_kind,
                        source.display_name,
                    )
                )
    return Discovery(tuple(sorted(candidates, key=_candidate_sort_key)), tuple(warnings))


def select_candidates(
    scope: str,
    candidates: Iterable[SourceCandidate],
    processed_fingerprints: set[str],
    today_ist: datetime,
) -> tuple[SourceCandidate, ...]:
    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    if scope == "today":
        return tuple(candidate for candidate in ordered if candidate.created_at.astimezone(pipeline.IST).date() == today_ist.astimezone(pipeline.IST).date())
    if scope == "last":
        return ordered[-1:] if ordered else ()
    if scope == "new":
        return tuple(candidate for candidate in ordered if candidate.fingerprint not in processed_fingerprints)
    if scope == "all":
        return ordered
    raise ValueError(f"unsupported scope: {scope}")


def load_secret_environment(
    path: Path = SECRET_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    target = os.environ if environ is None else environ
    if not path.is_file():
        return bool(target.get("HF_TOKEN", "").strip())
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in target:
            target[key] = value
    return bool(target.get("HF_TOKEN", "").strip())


def create_all_checkpoint(
    ledger_path: Path,
    output_dir: Path,
    selected: Sequence[SourceCandidate],
    clock: Callable[[], datetime],
    checkpoint_root: Path = CHECKPOINT_ROOT,
) -> Path:
    ledger_bytes = ledger_path.read_bytes() if ledger_path.is_file() else b""
    ledger_hash = hashlib.sha256(ledger_bytes).hexdigest()
    timestamp = clock().astimezone(pipeline.IST).strftime("%Y%m%dT%H%M%S%z")
    final = checkpoint_root / f"mi_all_pre_{timestamp}_{ledger_hash[:12]}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=checkpoint_root)
    )
    try:
        if ledger_path.is_file():
            shutil.copy2(ledger_path, temporary / "processed_ledger.json")
        if output_dir.is_dir():
            shutil.copytree(output_dir, temporary / "output")
        manifest = {
            "created_at": clock().astimezone(pipeline.IST).isoformat(),
            "ledger_sha256": ledger_hash,
            "selected_fingerprints": [candidate.fingerprint for candidate in selected],
            "selected_count": len(selected),
        }
        (temporary / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final


def _pipeline_args(
    selected: Sequence[SourceCandidate],
    *,
    refresh: bool,
    brief_date: str,
    audio_first_only: bool = False,
    ollama_url: str | None = None,
    explicit_meeting_folder: Path | None = None,
    explicit_audio_first_source_folders: Sequence[Path] | None = None,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> argparse.Namespace:
    return argparse.Namespace(
        meetings_root=MEETINGS_ROOT,
        audio_first_root=AUDIO_FIRST_ROOT,
        audio_first_only=audio_first_only,
        ledger_path=LEDGER_PATH,
        output_dir=OUTPUT_DIR,
        prompt_path=pipeline.DEFAULT_PROMPT_PATH,
        layer2_prompt_path=pipeline.DEFAULT_LAYER2_PROMPT_PATH,
        ollama_url=ollama_url or pipeline.DEFAULT_OLLAMA_URL,
        model=pipeline.DEFAULT_MODEL,
        watch=False,
        poll_seconds=60,
        dry_run=False,
        refresh_existing=refresh,
        brief_date=brief_date,
        selected_fingerprint=[candidate.fingerprint for candidate in selected],
        explicit_meeting_folder=explicit_meeting_folder,
        explicit_audio_first_source_folders=tuple(explicit_audio_first_source_folders or ()),
        phone_transcription_backend=phone_transcription_backend,
    )


def _print_plan(scope: str, selected: Sequence[SourceCandidate], processed: set[str]) -> None:
    print("MeetingIntel")
    print(f"Scope: {scope}")
    print(f"Selected: {len(selected)} meetings")
    for candidate in selected:
        status = "already done" if candidate.fingerprint in processed else "ready"
        created = candidate.created_at.astimezone(pipeline.IST).strftime("%Y-%m-%d %H:%M")
        print(f"  {created} | {candidate.source_kind} | {candidate.display_name} | {status}")


@exclusive_operation("processing")
@observe.production_attempt
def _run_production(
    selected: Sequence[SourceCandidate],
    *,
    scope: str,
    processed: set[str],
    clock: Callable[[], datetime],
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]],
    ollama_manager: OllamaServiceManager | None,
    audio_first_only: bool,
    refresh_existing: bool = False,
    explicit_meeting_folder: Path | None = None,
    explicit_audio_first_source_folders: Sequence[Path] | None = None,
    suppress_private_paths: bool = False,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> int:
    if os.environ.get("MI_ORIGIN") != "gui":
        for candidate in selected:
            observe.emit("action.confirmed", source_id=candidate.fingerprint, stage="processing", outcome="confirmed", details={"action": "process"})
    already_done = sum(candidate.fingerprint in processed for candidate in selected)
    to_run = tuple(
        candidate
        for candidate in selected
        if refresh_existing or candidate.fingerprint not in processed
    )
    if not to_run:
        print("Processed: 0")
        print(f"Already done: {already_done}")
        print("Failed: 0")
        return 0
    if any(candidate.source_kind == "phone_recording" for candidate in to_run):
        print(f"Phone transcription backend: {phone_transcription_backend}")
    if any(candidate.source_kind == "laptop_capture" for candidate in to_run):
        print("Laptop transcription backend: legacy (Whisper)")

    if not load_secret_environment():
        print(
            f"ERROR: HF_TOKEN is unavailable. Add it to {SECRET_PATH} before processing.",
            file=sys.stderr,
        )
        print("Processed: 0")
        print(f"Already done: {already_done}")
        print(f"Failed: {len(to_run)}")
        return 1

    try:
        ollama_url = resolve_ollama_endpoint()
        manager = ollama_manager or OLLAMA_SERVICE_MANAGER
        try:
            preflight_ollama(ollama_url)
        except OllamaUnreachable as unreachable:
            try:
                manager.recover(ollama_url)
            except OllamaServiceError as recovery_error:
                print(f"Ollama unreachable: {unreachable}", file=sys.stderr)
                print(
                    f"Managed recovery was not completed: {recovery_error}. "
                    "Run `mi ollama status` or `mi ollama restart`.",
                    file=sys.stderr,
                )
                print("Processed: 0")
                print(f"Already done: {already_done}")
                print(f"Failed: {len(to_run)}")
                return 1
            print("Recovered the MeetingIntel-managed local Ollama service.")
    except OllamaEndpointError as exc:
        print(pipeline.describe_ollama_error(exc), file=sys.stderr)
        print("Processed: 0")
        print(f"Already done: {already_done}")
        print(f"Failed: {len(to_run)}")
        return 1

    if scope == "all":
        checkpoint = create_all_checkpoint(LEDGER_PATH, OUTPUT_DIR, selected, clock)
        print(f"Checkpoint: {checkpoint}")

    try:
        processed_count, brief_path = pipeline_runner(
            _pipeline_args(
                selected,
                refresh=refresh_existing,
                brief_date=clock().astimezone(pipeline.IST).date().isoformat(),
                audio_first_only=audio_first_only,
                ollama_url=ollama_url,
                explicit_meeting_folder=explicit_meeting_folder,
                explicit_audio_first_source_folders=explicit_audio_first_source_folders,
                phone_transcription_backend=phone_transcription_backend,
            )
        )
    except pipeline.ProductionPersistenceError as exc:
        print(f"Pipeline persistence error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    failed = max(0, len(to_run) - processed_count)
    if failed and not processed_count:
        print("No new meeting was completed; the daily brief contains no result from this attempt.")
    elif suppress_private_paths:
        print("Brief written: local daily brief updated")
    else:
        print(f"Brief written: {brief_path}")
    print(f"Processed: {processed_count}")
    print(f"Already done: {already_done}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


def _run_exact_phone_sources(
    sources: Sequence[AudioFirstMeetingSource],
    *,
    clock: Callable[[], datetime],
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]],
    ollama_manager: OllamaServiceManager | None,
    suppress_private_paths: bool = False,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> int:
    if len(sources) == 0:
        raise ValueError("exact phone processing requires at least one source")
    production_sources = tuple(
        pipeline.production_source_from_audio_first(source) for source in sources
    )
    if len({source.fingerprint for source in production_sources}) != len(production_sources):
        raise ValueError("exact phone sources must be unique")
    selected = tuple(
        SourceCandidate(
            source.fingerprint,
            source.created_at,
            source.source_kind,
            source.display_name,
        )
        for source in production_sources
    )
    ledger = pipeline.load_ledger(LEDGER_PATH)
    processed = set(ledger.get("records", {}))
    _print_plan("exact phone set", selected, processed)
    return _run_production(
        selected,
        scope="exact phone set",
        processed=processed,
        clock=clock,
        pipeline_runner=pipeline_runner,
        ollama_manager=ollama_manager,
        audio_first_only=True,
        explicit_audio_first_source_folders=tuple(source.source_folder for source in sources),
        suppress_private_paths=suppress_private_paths,
        phone_transcription_backend=phone_transcription_backend,
    )


def _matching_phone_review(
    item: inbox.InboxItem,
    review_sets: Sequence[phone_review.ReviewSet],
) -> phone_review.ReviewSet | None:
    if item.source_kind != "phone":
        return None
    if not item.identity:
        return None
    matches = [
        review
        for review in review_sets
        if review.segments
        and review.group_key
        and review.group_key == item.identity
        and review.segments[0].created_at == item.created_at
        and (not item.segment_count or len(review.segments) == item.segment_count)
    ]
    return matches[0] if len(matches) == 1 else None


def _exact_direct_child(candidate: Path, root: Path, label: str) -> Path:
    """Validate one selected local folder without following an unsafe path."""

    lexical_root = root.expanduser().absolute()
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError(f"{label} root is unavailable or unsafe")
    lexical_candidate = candidate.expanduser().absolute()
    if lexical_candidate.is_symlink():
        raise ValueError(f"selected {label} is a symlink")
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved_candidate = lexical_candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"selected {label} is unavailable") from exc
    if resolved_candidate.parent != resolved_root:
        raise ValueError(f"selected {label} is outside the configured root")
    return resolved_candidate


def _phone_source_identity_from_metadata(metadata: dict[str, object]) -> str | None:
    source_group = metadata.get("source_group")
    if isinstance(source_group, dict) and isinstance(
        source_group.get("group_fingerprint_sha256"), str
    ):
        return source_group["group_fingerprint_sha256"]
    values = (
        metadata.get("source_kind"),
        metadata.get("source_path"),
        metadata.get("created_at"),
        metadata.get("source_fingerprint_sha256"),
    )
    if not all(isinstance(value, str) and value for value in values):
        return None
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def _load_exact_phone_source(
    item: inbox.InboxItem,
    *,
    ingest_root: Path | None = None,
) -> AudioFirstMeetingSource:
    """Resolve one normalized phone item from its ingest ledger only."""

    if item.source_kind != "phone" or not item.identity:
        raise ValueError("selected phone item has no stable identity")
    root = (ingest_root or AUDIO_FIRST_ROOT).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("phone ingest root is unavailable or unsafe")
    ledger = phone_ingest.load_ledger(root / phone_ingest.LEDGER_FILENAME)
    records = ledger.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("phone ingest ledger is malformed")

    targets: list[Path] = []
    for record in records.values():
        if not isinstance(record, dict) or record.get("status") != "normalized":
            continue
        target_text = record.get("target_folder")
        if not isinstance(target_text, str) or not target_text:
            continue
        target = _exact_direct_child(Path(target_text), root, "phone source")
        metadata_path = target / phone_ingest.METADATA_FILENAME
        if metadata_path.is_symlink() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("selected phone metadata is unreadable") from exc
        if not isinstance(metadata, dict):
            raise ValueError("selected phone metadata is malformed")
        source_identity = _phone_source_identity_from_metadata(metadata)
        if source_identity is None:
            raise ValueError("selected phone metadata has no stable identity")
        if pipeline.build_audio_first_fingerprint(source_identity) == item.identity:
            targets.append(target)

    if len(targets) != 1:
        raise ValueError("selected phone source could not be revalidated uniquely")
    try:
        source = load_audio_first_source(targets[0])
    except (OSError, SourceValidationError, ValueError) as exc:
        raise ValueError("selected phone source failed exact validation") from exc
    production_source = pipeline.production_source_from_audio_first(source)
    if production_source.fingerprint != item.identity:
        raise ValueError("selected phone source identity changed")
    return source


def _load_exact_laptop_source(
    item: inbox.InboxItem,
    *,
    ingest_root: Path | None = None,
    capture_root: Path | None = None,
) -> LaptopMeetingSource:
    if item.source_kind != "laptop" or not item.identity:
        raise ValueError("selected laptop item has no stable identity")
    root = (ingest_root or LAPTOP_INGEST_ROOT).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("laptop ingest root is unavailable or unsafe")
    folder_name = (
        f"laptop_{item.created_at.astimezone(pipeline.IST):%Y-%m-%d_%H-%M-%S}_"
        f"{item.identity[:16]}"
    )
    folder = _exact_direct_child(root / folder_name, root, "laptop source")
    try:
        source = load_laptop_meeting_source(
            folder, capture_root=capture_root or LAPTOP_RECORDINGS_ROOT
        )
    except (OSError, LaptopSourceValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("selected laptop source failed exact validation") from exc
    if source.source_identity != item.identity:
        raise ValueError("selected laptop source identity changed")
    annotation = source_annotation(source.source_identity)
    if annotation and annotation.get("completeness") == "incomplete":
        raise ValueError("Recording incomplete: the meeting continued after capture stopped. Original media is preserved; full-meeting processing is blocked.")
    return source


def _find_exact_capture_session(
    item: inbox.InboxItem,
    *,
    recordings_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    if item.source_kind != "laptop" or not item.identity:
        raise ValueError("selected capture has no stable identity")
    root = (recordings_root or LAPTOP_RECORDINGS_ROOT).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("laptop recordings root is unavailable or unsafe")
    matches: list[tuple[Path, dict[str, object]]] = []
    for session in root.iterdir():
        if session.is_symlink() or not session.is_dir():
            continue
        manifest_path = session / capture_ingest.CAPTURE_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("selected capture evidence is unreadable") from exc
        if manifest.get("capture_id") == item.identity:
            matches.append((session, manifest))
    if len(matches) != 1:
        raise ValueError("selected capture could not be revalidated uniquely")
    return matches[0]


def _confirm_exact_action(
    label: str,
    *,
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
) -> bool:
    if not terminal_isatty():
        print(f"{label}: not started (interactive confirmation required)")
        print("Run mi inbox again to return to the menu.")
        return False
    print("")
    print(f"Process exactly this {label} now?")
    print("  [y] Yes")
    print("  [n] Not now")
    answer = phone_review._prompt(input_func, "Choice: ")
    if answer == "y":
        return True
    print(f"{label.capitalize()} not processed. Nothing changed.")
    return False


def _run_exact_phone_item(
    item: inbox.InboxItem,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None,
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]],
    ollama_manager: OllamaServiceManager | None,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> int:
    source = _load_exact_phone_source(item)
    if not _confirm_exact_action(
        "phone meeting", input_func=input_func, terminal_isatty=terminal_isatty
    ):
        return 0
    runner = process_runner or (
        lambda sources: _run_exact_phone_sources(
            sources,
            clock=clock,
            pipeline_runner=pipeline_runner,
            ollama_manager=ollama_manager,
            suppress_private_paths=True,
            phone_transcription_backend=phone_transcription_backend,
        )
    )
    return runner((source,))


def _run_exact_meetily_item(
    item: inbox.InboxItem,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]],
    ollama_manager: OllamaServiceManager | None,
) -> int:
    if item.source_kind != "meetily" or not item.identity:
        raise ValueError("selected Meetily item has no stable identity")
    folder = _exact_direct_child(Path(item.identity), MEETINGS_ROOT, "Meetily source")
    for filename in ("metadata.json", "transcripts.json"):
        path = folder / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("selected Meetily source is incomplete or unsafe")
    metadata, _transcripts, transcript_hash, transcript_text, created_at = pipeline.parse_meeting_folder(folder)
    if metadata.get("status") != "completed" or not transcript_text:
        raise ValueError("selected Meetily source is no longer ready")
    fingerprint = pipeline.build_fingerprint(folder.name, metadata["created_at"], transcript_hash)
    candidate = SourceCandidate(fingerprint, created_at, "meetily", "Meetily recording")
    if not _confirm_exact_action(
        "Meetily recording", input_func=input_func, terminal_isatty=terminal_isatty
    ):
        return 0
    ledger = pipeline.load_ledger(LEDGER_PATH)
    processed = set(ledger.get("records", {}))
    _print_plan("exact Meetily recording", (candidate,), processed)
    return _run_production(
        (candidate,),
        scope="exact Meetily recording",
        processed=processed,
        clock=clock,
        pipeline_runner=pipeline_runner,
        ollama_manager=ollama_manager,
        audio_first_only=False,
        explicit_meeting_folder=folder,
        suppress_private_paths=True,
    )


def _run_exact_laptop_item(
    item: inbox.InboxItem,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    ollama_manager: OllamaServiceManager | None,
) -> int:
    source = _load_exact_laptop_source(item)
    return _offer_exact_laptop_processing(
        source,
        clock=clock,
        input_func=input_func,
        terminal_isatty=terminal_isatty,
        ollama_manager=ollama_manager,
        later="mi inbox",
        suppress_private_paths=True,
    )


def _normalize_exact_laptop_item(
    item: inbox.InboxItem,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    ollama_manager: OllamaServiceManager | None,
) -> int:
    session_root, manifest = _find_exact_capture_session(item)
    audit = audit_capture_session(session_root, manifest)
    status = manifest.get("status")
    if status == "complete" and audit.accepted and audit.merge_ready:
        if not _confirm_exact_action(
            "laptop capture", input_func=input_func, terminal_isatty=terminal_isatty
        ):
            return 0
        normalized = normalize_capture(
            NormalizeConfig(
                session_root=session_root,
                capture_root=LAPTOP_RECORDINGS_ROOT,
                ingest_root=LAPTOP_INGEST_ROOT,
            )
        )
    elif status == "interrupted" and terminal_source_loss_recovery_eligible(manifest, audit):
        if not terminal_isatty():
            print("Recovery not started (interactive confirmation required)")
            print("Run mi inbox again to return to the menu.")
            return 0
        print("This laptop capture has terminal source-loss evidence.")
        print("Recover only if the meeting ended before the source was lost.")
        decision = phone_review._prompt(
            input_func,
            "Type RECOVER to preserve and normalize finalized audio: ",
        )
        if decision != "RECOVER":
            print("Recovery cancelled. Capture remains unchanged.")
            return 0
        normalized = normalize_capture(
            NormalizeConfig(
                session_root=session_root,
                capture_root=LAPTOP_RECORDINGS_ROOT,
                ingest_root=LAPTOP_INGEST_ROOT,
                allow_terminal_source_loss_recovery=True,
                operator_attested_meeting_ended_before_source_loss=True,
            )
        )
    else:
        print("The selected laptop capture is not currently eligible for normalization.")
        print("Details remain read-only. Run mi recordings diagnose last for evidence.")
        return 0
    source = load_laptop_meeting_source(
        normalized.target_root, capture_root=LAPTOP_RECORDINGS_ROOT
    )
    print("Normalized one laptop recording. Original capture files were retained.")
    return _offer_exact_laptop_processing(
        source,
        clock=clock,
        input_func=input_func,
        terminal_isatty=terminal_isatty,
        ollama_manager=ollama_manager,
        later="mi inbox",
        suppress_private_paths=True,
    )


def _run_selected_inbox_item(
    item: inbox.InboxItem,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    review_collector: Callable[..., Sequence[phone_review.ReviewSet]],
    review_runner: Callable[..., int],
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None,
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]],
    ollama_manager: OllamaServiceManager | None,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> int:
    if item.category not in {"needs_decision", "ready"}:
        print("That item is read-only in this view; nothing changed.")
        return 0
    if item.source_kind == "phone":
        if item.category == "ready" and (
            item.status == "phone normalized and ready"
            or item.status.startswith("phone processing stopped")
        ):
            return _run_exact_phone_item(
                item,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                process_runner=process_runner,
                pipeline_runner=pipeline_runner,
                ollama_manager=ollama_manager,
                phone_transcription_backend=phone_transcription_backend,
            )
        try:
            selected = _matching_phone_review(item, tuple(review_collector()))
        except (OSError, ValueError, phone_review.ReviewError) as exc:
            raise ValueError(f"phone review could not be revalidated: {type(exc).__name__}") from exc
        if selected is None:
            raise ValueError("the selected phone recording could not be revalidated safely")
        runner = process_runner or (
            lambda sources: _run_exact_phone_sources(
                sources,
                clock=clock,
                pipeline_runner=pipeline_runner,
                ollama_manager=ollama_manager,
                suppress_private_paths=True,
                phone_transcription_backend=phone_transcription_backend,
            )
        )
        return review_runner(
            selected,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            process_runner=runner,
        )
    if item.source_kind == "laptop":
        if item.status == "laptop normalized and ready":
            return _run_exact_laptop_item(
                item,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                ollama_manager=ollama_manager,
            )
        if item.status in {
            "laptop validated; waiting for normalization",
            "laptop interrupted; recovery eligible",
        }:
            return _normalize_exact_laptop_item(
                item,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                ollama_manager=ollama_manager,
            )
        print("The selected laptop item needs specialist capture diagnostics first.")
        print("Run mi recordings diagnose last; no changes were made.")
        return 0
    if item.source_kind == "meetily" and item.status == "meetily ready":
        return _run_exact_meetily_item(
            item,
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            pipeline_runner=pipeline_runner,
            ollama_manager=ollama_manager,
        )
    if item.status == "phone fetch failed":
        print("Phone delivery needs attention; no processing was started.")
        print("Run mi sync status for the safe local failure summary.")
        return 0
    print("The selected item is not currently eligible for an exact action.")
    return 0


def run_inbox(
    argv: Sequence[str] = (),
    *,
    clock: Callable[[], datetime] = pipeline.now_ist,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
    inbox_builder: Callable[..., inbox.InboxReport] | None = None,
    review_collector: Callable[..., Sequence[phone_review.ReviewSet]] | None = None,
    review_runner: Callable[..., int] | None = None,
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None = None,
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]] = pipeline.process_meetings,
    ollama_manager: OllamaServiceManager | None = None,
    phone_transcription_backend: str = pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT,
) -> int:
    if argv:
        print("Usage: mi inbox", file=sys.stderr)
        return 2
    inbox_builder = inbox_builder or inbox.build_inbox
    review_collector = review_collector or phone_review.collect_review_sets
    review_runner = review_runner or phone_review.run_selected_review
    report = inbox_builder(clock=clock)
    print(inbox.format_inbox(report))
    if not terminal_isatty() or not report.items:
        return 0

    numbered_items = report.numbered_items
    for offered in numbered_items:
        sid = observe.meetily_id(offered.identity) if offered.source_kind == "meetily" else (pipeline.build_laptop_capture_fingerprint(offered.identity) if offered.source_kind == "laptop" and len(offered.identity) == 64 else offered.identity)
        if len(sid) == 64:
            observe.emit("action.offered", source_id=sid, details={"action": "process" if offered.category == "ready" else "details"})
    if not numbered_items:
        return 0
    latest_action = report.current_action_item
    for _ in range(phone_review.MAX_INPUT_ATTEMPTS):
        choice = phone_review._prompt(
            input_func,
            "\nWhat would you like to do?\n"
            "  [number] Select that exact item\n"
            "  [d] Show details and older items\n"
            "  [q] Quit\nChoice: ",
        )
        if choice is None or choice in {"", "q"}:
            return 0
        if choice == "d":
            print(inbox.format_inbox(report, details=True))
            continue
        legacy_prepare = choice == "p"
        if choice in {"p", "r"}:
            selected_item = latest_action
        elif choice.isdigit() and 1 <= int(choice) <= len(numbered_items):
            selected_item = numbered_items[int(choice) - 1]
        else:
            selected_item = None
        if selected_item is None:
            print("Invalid choice; no default action will be selected.")
            continue
        if legacy_prepare:
            confirmation = phone_review._prompt(
                input_func,
                "Prepare only this selected phone recording for normalization? "
                "It will not be processed. Type y to continue: ",
            )
            if confirmation != "y":
                print("Left unchanged.")
                return 0
        try:
            if legacy_prepare and selected_item.source_kind != "phone":
                raise ValueError("legacy phone alias cannot select a non-phone item")
            result = _run_selected_inbox_item(
                selected_item,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                review_collector=review_collector,
                review_runner=review_runner,
                process_runner=process_runner,
                pipeline_runner=pipeline_runner,
                ollama_manager=ollama_manager,
                phone_transcription_backend=phone_transcription_backend,
            )
        except (OSError, ValueError, phone_review.ReviewError) as exc:
            print(f"Inbox action stopped safely: {exc}", file=sys.stderr)
            print("Run mi inbox again to return to the menu.")
            return 1
        if result:
            print("Run mi inbox again to return to the menu.")
        return result
    return 0


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = pipeline.now_ist,
    input_func: Callable[[str], str] = input,
    pipeline_runner: Callable[[argparse.Namespace], tuple[int, Path]] = pipeline.process_meetings,
    ollama_manager: OllamaServiceManager | None = None,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    try:
        command, phone_transcription_backend = extract_phone_transcription_backend(command)
    except ValueError as exc:
        print(f"Usage error: {exc}", file=sys.stderr)
        return 2
    if command[:1] in (["help"], ["--help"], ["-h"]):
        try:
            if len(command) > 2:
                raise ValueError("Use mi help [phone|recordings|recovery|advanced|setup|all]")
            print(help_text(command[1] if len(command) == 2 else None))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if command[:1] == ["ui"]:
        from meetingintel_ui import run_cli as run_ui
        return run_ui(command[1:])
    if command[:1] == ["learning"]:
        from meetingintel_learning import run_cli as run_learning
        return run_learning(command[1:])
    if command[:1] == ["ollama"]:
        return run_ollama(command[1:], manager=ollama_manager)
    if command[:1] == ["voice"]:
        return subprocess.run(
            [str(VOICE_COMMAND_PYTHON), str(VOICE_COMMAND_SCRIPT), *command[1:]],
            check=False,
        ).returncode
    if command[:1] == ["sync"]:
        return run_sync(command[1:])
    if command[:1] == ["record"]:
        return run_record(
            command[1:],
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            ollama_manager=ollama_manager,
        )
    if command[:1] == ["recordings"]:
        return run_recordings(
            command[1:],
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            ollama_manager=ollama_manager,
        )
    if command[:1] == ["inbox"]:
        return run_inbox(
            command[1:],
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            pipeline_runner=pipeline_runner,
            ollama_manager=ollama_manager,
            phone_transcription_backend=phone_transcription_backend,
        )
    if command[:2] == ["phone", "review"]:
        try:
            scope, dry_run = phone_review.parse_review_args(command[2:])
            return phone_review.run_review(
                scope,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                dry_run=dry_run,
                process_runner=lambda sources: _run_exact_phone_sources(
                    sources,
                    clock=clock,
                    pipeline_runner=pipeline_runner,
                    ollama_manager=ollama_manager,
                    phone_transcription_backend=phone_transcription_backend,
                ),
            )
        except (OSError, ValueError, phone_review.ReviewError) as exc:
            print(f"Phone review error: {exc}", file=sys.stderr)
            return 1
    audio_first = "--audio-first" in command
    inventory_tokens = [token for token in command if token != "--audio-first"]
    inventory_scope: str | None = None
    if inventory_tokens == ["list"]:
        inventory_scope = "today"
    elif (
        len(inventory_tokens) == 2
        and inventory_tokens[1] == "list"
        and inventory_tokens[0] in SCOPES
    ):
        inventory_scope = inventory_tokens[0]
    if inventory_scope is not None:
        report = build_inventory(inventory_scope, audio_first=audio_first, clock=clock)
        print(format_inventory(report))
        return 0
    scope = parse_scope(command)
    discovery = discover_candidates(
        audio_first_root=AUDIO_FIRST_ROOT if audio_first else None,
        audio_first_only=audio_first,
    )
    if audio_first:
        discovery = Discovery(
            tuple(candidate for candidate in discovery.candidates if candidate.source_kind == "phone_recording"),
            discovery.warnings,
        )
    for warning in discovery.warnings:
        print(warning, file=sys.stderr)
    ledger = pipeline.load_ledger(LEDGER_PATH)
    processed = set(ledger.get("records", {}))
    selected = select_candidates(scope, discovery.candidates, processed, clock())
    _print_plan(scope, selected, processed)
    if not selected:
        print("No Meetily meetings selected for this legacy processing lane.")
        print("Next: mi inbox")
        return 0

    if scope == "all":
        print("WARNING: mi all will reprocess every valid source meeting.")
        print(f"Source candidates: {len(selected)}")
        if input_func("Type ALL to continue: ") != "ALL":
            print("Aborted safely. Nothing was reprocessed.")
            return 0

    return _run_production(
        selected,
        scope=scope,
        processed=processed,
        clock=clock,
        pipeline_runner=pipeline_runner,
        ollama_manager=ollama_manager,
        audio_first_only=audio_first,
        refresh_existing=scope == "all",
        phone_transcription_backend=phone_transcription_backend,
    )


def prepare_recording_session_root(
    *,
    recordings_root: Path = LAPTOP_RECORDINGS_ROOT,
    clock: Callable[[], datetime] = pipeline.now_ist,
) -> Path:
    if recordings_root.is_symlink():
        raise laptop_capture_guard.GuardError(
            "recordings root must not be a symlink"
        )
    recordings_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(recordings_root, 0o700)
    session_name = clock().strftime("Meeting %Y-%m-%d_%H-%M-%S")
    session_root = recordings_root / session_name
    if session_root.exists():
        raise laptop_capture_guard.GuardError(
            f"recording session already exists: {session_root}"
        )
    return session_root


def run_record(
    argv: Sequence[str],
    *,
    clock: Callable[[], datetime] = pipeline.now_ist,
    recordings_root: Path = LAPTOP_RECORDINGS_ROOT,
    ingest_root: Path = LAPTOP_INGEST_ROOT,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
    ollama_manager: OllamaServiceManager | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="mi record",
        description="Record one guarded laptop meeting; Control-C stops.",
    )
    parser.parse_args(list(argv))
    try:
        session_root = prepare_recording_session_root(
            recordings_root=recordings_root,
            clock=clock,
        )
        print(f"Recording folder: {session_root}")
        result = laptop_capture_guard.run_guard(
            laptop_capture_guard.GuardConfig(
                session_root=session_root,
                allowed_root=recordings_root,
            )
        )
    except (laptop_capture_guard.GuardError, OSError) as error:
        print(f"Recorder error: {error}", file=sys.stderr)
        return 2
    if result == 0:
        print(f"Saved recording: {session_root}")
        try:
            normalized = normalize_capture(
                NormalizeConfig(
                    session_root=session_root,
                    capture_root=recordings_root,
                    ingest_root=ingest_root,
                )
            )
            source = load_laptop_meeting_source(
                normalized.target_root, capture_root=recordings_root
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            print("Recording was preserved, but normalization failed.", file=sys.stderr)
            print(f"Normalizer error: {error}", file=sys.stderr)
            return 1
        print(
            f"Normalized: 1 laptop meeting ({normalized.segment_count} capture segment"
            f"{'s' if normalized.segment_count != 1 else ''})"
        )
        print("Original capture files: retained")
        return _offer_exact_laptop_processing(
            source,
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            ollama_manager=ollama_manager,
        )

    try:
        failed_manifest = load_manifest(session_root / "capture_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError):
        failed_manifest = None
    if failed_manifest is not None and not failed_manifest.get("segments"):
        reason = failed_manifest.get("stop_reason")
        if isinstance(reason, str) and reason.startswith("preflight failed:"):
            print("RECORDING DID NOT START — NO AUDIO WAS SAVED BY THIS ATTEMPT.", file=sys.stderr)
        else:
            print("NO RECORDED SEGMENTS WERE CONFIRMED FOR THIS ATTEMPT.", file=sys.stderr)
        if reason == "preflight failed: battery_critical":
            print("Connect power, then run `mi record` again and wait for `Capture guard active`.",
                  file=sys.stderr)
    print(f"Recording interrupted; evidence at: {session_root}")
    print("Normalization and processing were not started.")
    print("Later conversation may be missing. Check `mi recordings diagnose last` or open `mi ui`.")
    print("If the meeting is still continuing, keep the lid open and run `mi record` again for a new capture.")
    return result


def _offer_exact_laptop_processing(
    source: LaptopMeetingSource,
    *,
    clock: Callable[[], datetime],
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    ollama_manager: OllamaServiceManager | None,
    later: str | None = None,
    suppress_private_paths: bool = False,
) -> int:
    recovery = later or f"mi recordings process {source.source_identity[:12]}"
    if not terminal_isatty():
        print("Production processing: not started (interactive confirmation required)")
        print(f"Later: {recovery}")
        return 0
    print("")
    print("Process exactly this meeting now?")
    print("  [y] Yes")
    print("  [n] Not now")
    try:
        decision = input_func("Choice: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNot processed. Normalized source remains available.")
        print(f"Later: {recovery}")
        return 0
    if decision != "y":
        print("Not processed. Normalized source remains available.")
        print(f"Later: {recovery}")
        return 0
    return _run_exact_laptop_source(
        source,
        clock=clock,
        ollama_manager=ollama_manager,
        suppress_private_paths=suppress_private_paths,
    )


def _run_exact_laptop_source(
    source: LaptopMeetingSource,
    *,
    clock: Callable[[], datetime],
    ollama_manager: OllamaServiceManager | None,
    suppress_private_paths: bool = False,
) -> int:
    production_source = pipeline.production_source_from_laptop_capture(source)
    candidate = SourceCandidate(
        production_source.fingerprint,
        production_source.created_at,
        production_source.source_kind,
        production_source.display_name,
    )
    ledger = pipeline.load_ledger(LEDGER_PATH)
    processed = set(ledger.get("records", {}))
    _print_plan("exact laptop recording", (candidate,), processed)
    return _run_production(
        (candidate,),
        scope="exact laptop recording",
        processed=processed,
        clock=clock,
        pipeline_runner=lambda args: pipeline.process_explicit_laptop_sources(
            args, (production_source,)
        ),
        ollama_manager=ollama_manager,
        audio_first_only=True,
        explicit_audio_first_source_folders=(source.source_folder,),
        suppress_private_paths=suppress_private_paths,
    )


def _laptop_sources(
    ingest_root: Path,
    recordings_root: Path,
) -> tuple[tuple[LaptopMeetingSource, ...], tuple[str, ...]]:
    discoveries = discover_laptop_meeting_sources(
        ingest_root, capture_root=recordings_root
    )
    sources: list[LaptopMeetingSource] = []
    warnings: list[str] = []
    for discovery in discoveries:
        if discovery.source is None:
            warnings.append(
                f"Laptop source rejected: {discovery.source_folder.name}: "
                f"{discovery.error}"
            )
        else:
            sources.append(discovery.source)
    return (
        tuple(sorted(sources, key=lambda source: (source.created_at, source.source_identity))),
        tuple(warnings),
    )


def _select_laptop_source(
    sources: Sequence[LaptopMeetingSource],
    selector: str,
) -> LaptopMeetingSource:
    if not sources:
        raise ValueError("no validated laptop recordings are available")
    if selector == "last":
        return sources[-1]
    if len(selector) < 8 or any(character not in "0123456789abcdef" for character in selector):
        raise ValueError("recording selector must be `last` or at least 8 hex characters")
    matches = [
        source for source in sources if source.source_identity.startswith(selector)
    ]
    if not matches:
        raise ValueError("recording selector did not match a validated source")
    if len(matches) != 1:
        raise ValueError("recording selector is ambiguous; use more characters")
    return matches[0]


def _latest_capture_session(recordings_root: Path) -> Path:
    if recordings_root.is_symlink() or not recordings_root.is_dir():
        raise ValueError("recordings root is unavailable or unsafe")
    candidates: list[tuple[datetime, Path]] = []
    for folder in recordings_root.iterdir():
        if folder.name.startswith(".") or folder.is_symlink() or not folder.is_dir():
            continue
        manifest_path = folder / "capture_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest(manifest_path)
            started_text = manifest.get("started_at")
            if not isinstance(started_text, str):
                continue
            started_at = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
            if started_at.tzinfo is None:
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((started_at, folder))
    if not candidates:
        raise ValueError("no capture manifest is available")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def _print_capture_diagnostic(session_root: Path) -> bool:
    manifest_path = session_root / "capture_manifest.json"
    manifest = load_manifest(manifest_path)
    audit = audit_capture_session(session_root, manifest)
    started = manifest.get("started_at", "unknown")
    ended = manifest.get("ended_at", "unknown")
    segments = manifest.get("segments")
    raw_segments = segments if isinstance(segments, list) else []
    total_duration = sum(
        float(item.get("duration_seconds", 0))
        for item in raw_segments
        if isinstance(item, dict)
        and isinstance(item.get("duration_seconds"), (int, float))
    )
    print("LAPTOP_CAPTURE_DIAGNOSTIC")
    print(f"  session={session_root.name}")
    print(f"  status={manifest.get('status', 'unknown')}")
    print(f"  stop_reason={manifest.get('stop_reason', 'unknown')}")
    print(f"  started_at={started}")
    print(f"  ended_at={ended}")
    print(
        f"  finalized_segments={len(audit.finalized_segments)} "
        f"duration={pipeline.format_duration(total_duration)}"
    )
    cumulative = 0.0
    for position, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict) or not isinstance(
            item.get("duration_seconds"), (int, float)
        ):
            continue
        duration = float(item["duration_seconds"])
        cumulative += duration
        print(
            f"  chunk={position} duration={pipeline.format_duration(duration)} "
            f"cumulative_end={pipeline.format_duration(cumulative)}"
        )
    print(
        f"  accepted={str(audit.accepted).lower()} "
        f"merge_ready={str(audit.merge_ready).lower()} "
        f"recovery_ready={str(audit.recovery_ready).lower()}"
    )
    if audit.findings:
        for finding in audit.findings:
            print(f"  finding={finding.severity}:{finding.code}")
    else:
        print("  findings=none")
    terminal_recovery = terminal_source_loss_recovery_eligible(manifest, audit)
    if manifest.get("status") == "complete" and audit.merge_ready:
        print("  next=mi recordings normalize-last")
    elif terminal_recovery:
        print(
            "  next=mi recordings recover-last "
            "(only if the meeting ended before lid close/source loss)"
        )
    else:
        print("  next=operator review required; do not normalize")
    return terminal_recovery


def run_recordings(
    argv: Sequence[str],
    *,
    clock: Callable[[], datetime] = pipeline.now_ist,
    recordings_root: Path = LAPTOP_RECORDINGS_ROOT,
    ingest_root: Path = LAPTOP_INGEST_ROOT,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
    ollama_manager: OllamaServiceManager | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="mi recordings",
        description="Review or resume guarded laptop recordings",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list")
    process_parser = subparsers.add_parser("process")
    process_parser.add_argument("selector")
    subparsers.add_parser("normalize-last")
    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("selector", nargs="?", default="last", choices=("last",))
    recover_parser = subparsers.add_parser("recover-last")
    recover_parser.add_argument(
        "--keep-segments",
        type=int,
        help="recover only the first N finalized chunks and preserve the rest as excluded tail",
    )
    args = parser.parse_args(list(argv))

    try:
        if args.action == "diagnose":
            session_root = _latest_capture_session(recordings_root)
            _print_capture_diagnostic(session_root)
            return 0

        if args.action == "recover-last":
            session_root = _latest_capture_session(recordings_root)
            eligible = _print_capture_diagnostic(session_root)
            if not eligible:
                raise ValueError(
                    "latest capture is not eligible for terminal source-loss recovery"
                )
            if not terminal_isatty():
                raise ValueError("recovery requires an interactive terminal")
            manifest = load_manifest(session_root / "capture_manifest.json")
            raw_segments = manifest.get("segments")
            segment_count = len(raw_segments) if isinstance(raw_segments, list) else 0
            keep_count = args.keep_segments
            if keep_count is not None and not (1 <= keep_count <= segment_count):
                raise ValueError(
                    f"--keep-segments must be between 1 and {segment_count}"
                )
            print("")
            print(
                "Recover only if the meeting ended before the lid was closed "
                "or the source was lost."
            )
            if keep_count is None:
                print(f"Recovery will keep all {segment_count} finalized chunks.")
                if segment_count > 1:
                    print(
                        "If recording continued after the meeting, cancel and rerun with "
                        "--keep-segments N."
                    )
            else:
                durations = [
                    float(item.get("duration_seconds", 0))
                    for item in raw_segments
                    if isinstance(item, dict)
                ]
                retained_duration = sum(durations[:keep_count])
                excluded_duration = sum(durations[keep_count:])
                print(
                    f"Recovery will keep chunks 1-{keep_count} "
                    f"({pipeline.format_duration(retained_duration)}) and exclude "
                    f"{segment_count - keep_count} tail chunks "
                    f"({pipeline.format_duration(excluded_duration)})."
                )
                print("Every original capture chunk will remain unchanged.")
            try:
                decision = input_func("Type RECOVER to preserve and normalize finalized audio: ")
            except (EOFError, KeyboardInterrupt):
                print("\nRecovery cancelled. Capture remains unchanged.")
                return 0
            if decision != "RECOVER":
                print("Recovery cancelled. Capture remains unchanged.")
                return 0
            normalized = normalize_capture(
                NormalizeConfig(
                    session_root=session_root,
                    capture_root=recordings_root,
                    ingest_root=ingest_root,
                    allow_terminal_source_loss_recovery=True,
                    operator_attested_meeting_ended_before_source_loss=True,
                    keep_finalized_segment_count=keep_count,
                )
            )
            source = load_laptop_meeting_source(
                normalized.target_root, capture_root=recordings_root
            )
            print(
                f"Recovered and normalized: {source.created_at:%Y-%m-%d %H:%M} IST | "
                f"id={source.source_identity[:12]} | segments={source.source_segment_count}"
            )
            print("Interrupted manifest and original capture files: retained")
            return _offer_exact_laptop_processing(
                source,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                ollama_manager=ollama_manager,
            )

        if args.action == "normalize-last":
            session_root = _latest_capture_session(recordings_root)
            normalized = normalize_capture(
                NormalizeConfig(
                    session_root=session_root,
                    capture_root=recordings_root,
                    ingest_root=ingest_root,
                )
            )
            source = load_laptop_meeting_source(
                normalized.target_root, capture_root=recordings_root
            )
            print(
                f"Normalized: {source.created_at:%Y-%m-%d %H:%M} IST | "
                f"id={source.source_identity[:12]} | segments={source.source_segment_count}"
            )
            print("Original capture files: retained")
            return _offer_exact_laptop_processing(
                source,
                clock=clock,
                input_func=input_func,
                terminal_isatty=terminal_isatty,
                ollama_manager=ollama_manager,
            )

        sources, warnings = _laptop_sources(ingest_root, recordings_root)
        for warning in warnings:
            print(warning, file=sys.stderr)
        if args.action == "list":
            ledger = pipeline.load_ledger(LEDGER_PATH)
            processed = set(ledger.get("records", {}))
            print(
                f"LAPTOP_RECORDINGS total={len(sources)} rejected={len(warnings)}"
            )
            if not sources:
                print("No validated laptop recordings.")
                return 0
            for source in sources:
                fingerprint = pipeline.build_laptop_capture_fingerprint(
                    source.source_identity
                )
                status = "processed" if fingerprint in processed else "ready"
                print(
                    f"  {source.created_at:%Y-%m-%d %H:%M} IST | "
                    f"id={source.source_identity[:12]} | "
                    f"segments={source.source_segment_count} | {status}"
                )
            return 0

        source = _select_laptop_source(sources, args.selector)
        print(
            f"Selected: {source.created_at:%Y-%m-%d %H:%M} IST | "
            f"id={source.source_identity[:12]} | segments={source.source_segment_count}"
        )
        return _offer_exact_laptop_processing(
            source,
            clock=clock,
            input_func=input_func,
            terminal_isatty=terminal_isatty,
            ollama_manager=ollama_manager,
        )
    except (
        LaptopSourceValidationError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Laptop recordings error: {error}", file=sys.stderr)
        return 1


def run_ollama(
    argv: Sequence[str],
    *,
    manager: OllamaServiceManager | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="mi ollama")
    parser.add_argument(
        "action",
        choices=("status", "install", "start", "restart", "stop", "remove"),
    )
    args = parser.parse_args(list(argv))
    service = manager or OLLAMA_SERVICE_MANAGER
    try:
        if args.action == "status":
            try:
                endpoint = resolve_ollama_endpoint()
            except OllamaEndpointError as exc:
                print(f"Ollama endpoint invalid: {exc}", file=sys.stderr)
                return 1
            status = service.status(endpoint)
            print(format_status(status))
            return 0 if status.healthy else 1
        if args.action == "install":
            service.install()
            print("MeetingIntel Ollama LaunchAgent installed and loaded.")
        elif args.action == "start":
            service.start()
            print("MeetingIntel Ollama LaunchAgent started.")
        elif args.action == "restart":
            service.restart()
            print("MeetingIntel Ollama LaunchAgent restarted.")
        elif args.action == "stop":
            service.stop()
            print("MeetingIntel Ollama LaunchAgent stopped.")
        elif args.action == "remove":
            service.remove()
            print("MeetingIntel Ollama LaunchAgent removed.")
        return 0
    except ServiceNotInstalled as exc:
        print(f"Ollama service not installed: {exc}", file=sys.stderr)
    except ServiceConfigInvalid as exc:
        print(f"Ollama managed service configuration invalid: {exc}", file=sys.stderr)
    except ServiceConflict as exc:
        print(
            f"Ollama listener conflict: {exc}. Quit or disable Ollama.app background startup; do not reuse another port.",
            file=sys.stderr,
        )
    except ServiceRecoveryFailed as exc:
        print(f"Ollama recovery failed: {exc}. Run `mi ollama status`.", file=sys.stderr)
    except OllamaServiceError as exc:
        print(f"Ollama service error: {exc}", file=sys.stderr)
    return 1


def run_sync(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mi sync",
        description="Fetch finalized phone recordings from the configured Google Drive folder.",
    )
    if list(argv[:1]) == ["status"]:
        if len(argv) != 1:
            parser.error("sync status does not accept fetch options")
        try:
            print_fetch_status(FetchConfig().ledger_path)
        except Exception as exc:
            print(f"Phone fetch status error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    if list(argv[:1]) == ["scheduler"]:
        if len(argv) != 2 or argv[1] not in {"install", "remove", "status", "diagnose", "probe", "probe-command"}:
            parser.error("use mi sync scheduler install|remove|status|diagnose|probe")
        action = argv[1]
        try:
            if action == "install":
                path = install_daily_schedule(7, 0)
                print(f"Phone fetch scheduler installed: {path}")
            elif action == "remove":
                remove_daily_schedule()
                print("Phone fetch scheduler removed.")
            elif action == "probe":
                return run_scheduler_probe()
            elif action == "probe-command":
                # Called only by the managed wrapper after it has established
                # the LaunchAgent -> wrapper -> interpreter chain.  It must
                # not enter Drive, fetch, audio, or production code.
                return 0
            else:
                state = scheduler_status() if action == "status" else scheduler_diagnose()
                print(format_scheduler_report(state))
                return 0 if state["healthy"] else 1
        except Exception as exc:
            print(f"Phone fetch scheduler error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--destination-root", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--install-daily", action="store_true")
    schedule.add_argument("--remove-daily", action="store_true")
    parser.add_argument("--daily-hour", type=int, default=7)
    parser.add_argument("--daily-minute", type=int, default=0)
    args = parser.parse_args(list(argv))
    if args.install_daily:
        try:
            path = install_daily_schedule(args.daily_hour, args.daily_minute)
        except Exception as exc:
            print(f"Phone fetch scheduler error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"Daily phone fetch scheduled: {path}")
        return 0
    if args.remove_daily:
        try:
            remove_daily_schedule()
        except Exception as exc:
            print(f"Phone fetch scheduler error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("Daily phone fetch schedule removed.")
        return 0
    defaults = FetchConfig()
    config = FetchConfig(
        remote_root=args.remote_root or defaults.remote_root,
        destination_root=args.destination_root or defaults.destination_root,
        ledger_path=args.ledger_path or defaults.ledger_path,
        dry_run=args.dry_run,
    )
    try:
        summary = fetch_recordings(config)
    except Exception as exc:
        if os.environ.get("MEETINGINTEL_PHONE_FETCH_SCHEDULED") == "1":
            try:
                write_scheduler_receipt(success=False, category="fetch_setup_failure", exit_code=1)
            except Exception:
                pass
        print(f"Phone fetch error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if os.environ.get("MEETINGINTEL_PHONE_FETCH_SCHEDULED") == "1":
        if summary.already_running:
            receipt_success, receipt_category = False, "lock_conflict"
        elif summary.failed:
            failure_category = classify_fetch_failure(config.ledger_path)
            receipt_success, receipt_category = False, f"fetch_failure_{failure_category}"
            print(
                "PHONE_FETCH_FAILURE "
                f"category={failure_category} failed={summary.failed} "
                "detail=see_private_fetch_ledger",
                file=sys.stderr,
            )
        elif summary.requires_operator_decision:
            receipt_success, receipt_category = False, "operator_decision_required"
        else:
            receipt_success, receipt_category = True, "success"
        try:
            write_scheduler_receipt(
                success=receipt_success,
                category=receipt_category,
                exit_code=summary.exit_code,
            )
        except Exception as exc:
            print(f"Phone fetch receipt error: {type(exc).__name__}", file=sys.stderr)
            return 1
    observe.emit("phone.failed" if summary.failed else "phone.delivered", stage="delivery",
                 outcome="failure" if summary.failed else "success", details={"count": summary.failed})
    return summary.exit_code


def main() -> int:
    os.environ.setdefault("MI_ORIGIN", "cli")
    try:
        return run_cli()
    except RuntimeSafetyError as exc:
        print(f"MeetingIntel stopped safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
