"""Synthetic contract checks for a future laptop recording lane.

This module is deliberately isolated from MeetingIntel production discovery.
It models the durability boundary that any recorder (OBS, Meetily, or another
engine) must satisfy before its audio can be normalized or processed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SOURCE_KIND = "laptop_capture"
VALID_STATUSES = frozenset({"recording", "complete", "interrupted"})
VALID_SEGMENT_STATES = frozenset({"active", "finalized"})
VALID_SOURCE_EVENT_TYPES = frozenset(
    {"source_lost", "source_recovered", "sleep", "wake"}
)

FIRST_OUTPUT_DEADLINE_SECONDS = 10.0
OUTPUT_STALL_SECONDS = 20.0
AUDIO_CALLBACK_STALL_SECONDS = 10.0
LOW_BATTERY_WARNING_PERCENT = 15
CRITICAL_BATTERY_PERCENT = 5
LOW_DISK_WARNING_BYTES = 1024 * 1024 * 1024
CRITICAL_DISK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class MediaProbe:
    decodable: bool
    has_audio: bool
    duration_seconds: float


@dataclass(frozen=True)
class SessionAudit:
    findings: tuple[Finding, ...]
    finalized_segments: tuple[Path, ...]
    merge_ready: bool
    recovery_ready: bool

    @property
    def accepted(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)

    def codes(self) -> set[str]:
        return {item.code for item in self.findings}


@dataclass(frozen=True)
class PreflightSnapshot:
    websocket_reachable: bool = True
    websocket_authenticated: bool = True
    capture_permission: bool = True
    microphone_permission: bool = True
    output_writable: bool = True
    output_path_matches: bool = True
    disk_free_bytes: int = 10 * LOW_DISK_WARNING_BYTES
    expected_sources: tuple[str, ...] = ("microphone", "system_audio")
    sources_reporting_frames: tuple[str, ...] = ("microphone", "system_audio")


@dataclass(frozen=True)
class WatchdogSnapshot:
    recording_reported: bool = True
    recorder_process_alive: bool = True
    output_exists: bool = True
    elapsed_seconds: float = 30.0
    seconds_since_output_growth: float = 1.0
    seconds_since_audio_callback: float = 1.0
    expected_sources: tuple[str, ...] = ("microphone", "system_audio")
    sources_reporting_frames: tuple[str, ...] = ("microphone", "system_audio")
    sleep_detected: bool = False
    on_ac_power: bool = True
    battery_percent: int = 100
    disk_free_bytes: int = 10 * LOW_DISK_WARNING_BYTES


@dataclass(frozen=True)
class ControlAction:
    code: str
    action: str
    message: str


@dataclass(frozen=True)
class CleanupGate:
    session_status: str
    canonical_exists: bool
    canonical_decodable: bool
    canonical_hash_matches: bool
    operator_confirmed: bool


def _finding(code: str, message: str, severity: str = "error") -> Finding:
    return Finding(code=code, severity=severity, message=message)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained_regular_file(root: Path, relative_path: object) -> tuple[Path | None, str | None]:
    if not isinstance(relative_path, str) or not relative_path:
        return None, "segment_path_invalid"
    candidate = root / relative_path
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError):
        return None, "segment_missing_or_outside_session"
    if candidate.is_symlink() or not resolved_candidate.is_file():
        return None, "segment_not_regular_file"
    return resolved_candidate, None


def ffprobe_media(path: Path, ffprobe: str = "ffprobe") -> MediaProbe:
    """Probe one synthetic or real media file without reading meeting content."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return MediaProbe(False, False, 0.0)
    if result.returncode != 0:
        return MediaProbe(False, False, 0.0)
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
    except (TypeError, ValueError, json.JSONDecodeError):
        return MediaProbe(False, False, 0.0)
    raw_duration = payload.get("format", {}).get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0
    has_audio = any(
        isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
        for stream in streams
    )
    if has_audio and duration <= 0:
        duration = _ffprobe_audio_packet_duration(path, ffprobe)
    return MediaProbe(True, has_audio, duration)


def _ffprobe_audio_packet_duration(path: Path, ffprobe: str) -> float:
    """Derive duration when a valid short/interrupted MKV has no format duration."""

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "packet=pts_time,duration_time",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    if result.returncode != 0:
        return 0.0

    final_timestamp = 0.0
    for line in result.stdout.splitlines():
        fields = line.split(",")
        try:
            timestamp = float(fields[0])
            packet_duration = float(fields[1]) if len(fields) > 1 else 0.0
        except (IndexError, ValueError):
            continue
        final_timestamp = max(final_timestamp, timestamp + packet_duration)
    return final_timestamp


def audit_capture_session(
    session_root: Path,
    manifest: Mapping[str, object],
    probe_media: Callable[[Path], MediaProbe] = ffprobe_media,
) -> SessionAudit:
    """Validate storage truth for one capture session.

    Segment ordering comes only from the manifest's integer indexes. Filenames
    are never used to infer grouping or order.
    """

    findings: list[Finding] = []
    valid_finalized: list[tuple[int, Path]] = []
    valid_active_count = 0

    if manifest.get("schema_version") != SCHEMA_VERSION:
        findings.append(_finding("schema_version_invalid", "Unsupported capture schema"))
    if manifest.get("source_kind") != SOURCE_KIND:
        findings.append(_finding("source_kind_invalid", "Unsupported capture source kind"))

    capture_id = manifest.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id.strip():
        findings.append(_finding("capture_id_invalid", "Capture ID is missing"))

    status = manifest.get("status")
    if status not in VALID_STATUSES:
        findings.append(_finding("status_invalid", "Capture status is invalid"))

    started_at = _parse_timestamp(manifest.get("started_at"))
    ended_at = _parse_timestamp(manifest.get("ended_at"))
    if started_at is None:
        findings.append(_finding("started_at_invalid", "Start timestamp is invalid"))
    if status == "complete" and ended_at is None:
        findings.append(_finding("ended_at_missing", "Complete capture lacks an end timestamp"))
    if started_at and ended_at and ended_at < started_at:
        findings.append(_finding("timestamp_order_invalid", "End timestamp precedes start"))

    raw_source_events = manifest.get("source_events", [])
    if not isinstance(raw_source_events, list):
        findings.append(_finding("source_events_invalid", "Source events must be a list"))
        raw_source_events = []
    source_is_lost = False
    previous_event_at: datetime | None = None
    source_interruption_recovered = False
    for position, event in enumerate(raw_source_events):
        prefix = f"source_event[{position}]"
        if not isinstance(event, Mapping):
            findings.append(_finding("source_event_invalid", f"{prefix} is not an object"))
            continue
        event_type = event.get("type")
        event_at = _parse_timestamp(event.get("at"))
        if event_type not in VALID_SOURCE_EVENT_TYPES:
            findings.append(_finding("source_event_type_invalid", f"{prefix} type is invalid"))
            continue
        if event_at is None:
            findings.append(_finding("source_event_time_invalid", f"{prefix} time is invalid"))
            continue
        if previous_event_at and event_at < previous_event_at:
            findings.append(_finding("source_event_order_invalid", "Source events are out of order"))
        previous_event_at = event_at
        if started_at and event_at < started_at:
            findings.append(_finding("source_event_outside_capture", f"{prefix} predates capture"))
        if ended_at and event_at > ended_at:
            findings.append(_finding("source_event_outside_capture", f"{prefix} follows capture"))

        if event_type == "source_lost":
            source_is_lost = True
        elif event_type == "source_recovered":
            if not source_is_lost:
                findings.append(
                    _finding(
                        "source_recovery_without_loss",
                        f"{prefix} claims recovery without a preceding loss",
                    )
                )
            else:
                source_is_lost = False
                source_interruption_recovered = True

    if source_is_lost:
        findings.append(
            _finding(
                "source_not_recovered",
                "Capture source stopped and did not recover before recording ended",
            )
        )
    elif source_interruption_recovered:
        findings.append(
            _finding(
                "source_interruption_recovered",
                "Capture source stopped and was explicitly recovered",
                "warning",
            )
        )

    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list):
        findings.append(_finding("segments_invalid", "Segments must be a list"))
        raw_segments = []
    if not raw_segments:
        findings.append(_finding("no_segments", "Session has no recorded media"))

    seen_indexes: set[int] = set()
    seen_paths: set[str] = set()
    finalized_indexes: list[int] = []

    for position, item in enumerate(raw_segments):
        prefix = f"segment[{position}]"
        if not isinstance(item, Mapping):
            findings.append(_finding("segment_entry_invalid", f"{prefix} is not an object"))
            continue
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            findings.append(_finding("segment_index_invalid", f"{prefix} index is invalid"))
            continue
        if index in seen_indexes:
            findings.append(_finding("segment_index_duplicate", f"Duplicate segment index {index}"))
        seen_indexes.add(index)

        relative_path = item.get("path")
        if isinstance(relative_path, str):
            if relative_path in seen_paths:
                findings.append(_finding("segment_path_duplicate", f"Duplicate segment path {relative_path}"))
            seen_paths.add(relative_path)

        state = item.get("state")
        if state not in VALID_SEGMENT_STATES:
            findings.append(_finding("segment_state_invalid", f"{prefix} state is invalid"))
            continue

        path, path_error = _contained_regular_file(session_root, relative_path)
        if path_error:
            severity = "warning" if status == "interrupted" and state == "active" else "error"
            findings.append(_finding(path_error, f"{prefix} media is unavailable", severity))
            continue
        assert path is not None

        actual_size = path.stat().st_size
        expected_size = item.get("size_bytes")
        if actual_size <= 0:
            findings.append(_finding("segment_empty", f"{prefix} is empty"))
            continue
        if not isinstance(expected_size, int) or expected_size != actual_size:
            findings.append(_finding("segment_size_mismatch", f"{prefix} size changed"))

        probe = probe_media(path)
        if not probe.decodable:
            severity = "warning" if status == "interrupted" and state == "active" else "error"
            findings.append(_finding("segment_not_decodable", f"{prefix} cannot be decoded", severity))
            continue
        if not probe.has_audio:
            findings.append(_finding("segment_audio_missing", f"{prefix} has no audio stream"))
            continue
        if probe.duration_seconds <= 0:
            findings.append(_finding("segment_duration_invalid", f"{prefix} has no positive duration"))
            continue

        declared_duration = item.get("duration_seconds")
        if not isinstance(declared_duration, (int, float)) or declared_duration <= 0:
            findings.append(_finding("segment_duration_missing", f"{prefix} duration is invalid"))
        elif abs(float(declared_duration) - probe.duration_seconds) > 1.0:
            findings.append(_finding("segment_duration_mismatch", f"{prefix} duration changed"))

        if state == "finalized":
            expected_hash = item.get("sha256")
            if not isinstance(expected_hash, str) or expected_hash != _sha256(path):
                findings.append(_finding("segment_hash_mismatch", f"{prefix} hash changed"))
                continue
            finalized_indexes.append(index)
            valid_finalized.append((index, path))
        else:
            valid_active_count += 1
            if item.get("sha256") not in (None, ""):
                findings.append(
                    _finding(
                        "active_segment_hash_present",
                        f"{prefix} must not claim a final hash",
                        "warning",
                    )
                )

    if finalized_indexes:
        expected_indexes = list(range(min(finalized_indexes), max(finalized_indexes) + 1))
        if sorted(finalized_indexes) != expected_indexes or min(finalized_indexes) != 0:
            findings.append(_finding("segment_index_gap", "Finalized segment order has a gap"))

    lock_exists = (session_root / ".recording.lock").exists()
    if status == "recording":
        if not lock_exists:
            findings.append(_finding("recording_lock_missing", "Recording status has no live lock"))
        if valid_active_count != 1:
            findings.append(_finding("active_segment_count_invalid", "Recording requires one active segment"))
    elif status == "complete":
        if lock_exists:
            findings.append(_finding("stale_recording_lock", "Complete session retains a recording lock"))
        if valid_active_count:
            findings.append(_finding("active_segment_in_complete", "Complete session contains active media"))
    elif status == "interrupted" and lock_exists:
        findings.append(
            _finding(
                "stale_recording_lock",
                "Interrupted session retains a stale recording lock",
                "warning",
            )
        )

    error_codes = {item.code for item in findings if item.severity == "error"}
    ordered_finalized = tuple(path for _, path in sorted(valid_finalized))
    segment_errors = {
        code
        for code in error_codes
        if code.startswith("segment_") or code in {"no_segments", "active_segment_in_complete"}
    }
    merge_ready = (
        status == "complete"
        and bool(ordered_finalized)
        and not segment_errors
        and not error_codes
    )
    recovery_ready = bool(ordered_finalized) and not segment_errors.intersection(
        {
            "segment_index_duplicate",
            "segment_path_duplicate",
            "segment_index_gap",
            "segment_hash_mismatch",
            "segment_audio_missing",
            "segment_not_decodable",
            "segment_empty",
        }
    ) and "source_not_recovered" not in error_codes
    return SessionAudit(
        findings=tuple(findings),
        finalized_segments=ordered_finalized,
        merge_ready=merge_ready,
        recovery_ready=recovery_ready,
    )


def evaluate_preflight(snapshot: PreflightSnapshot) -> tuple[ControlAction, ...]:
    actions: list[ControlAction] = []
    if not snapshot.websocket_reachable:
        actions.append(ControlAction("control_unreachable", "block_start", "Recorder control is unavailable"))
    elif not snapshot.websocket_authenticated:
        actions.append(ControlAction("control_auth_failed", "block_start", "Recorder control authentication failed"))
    if not snapshot.capture_permission:
        actions.append(ControlAction("capture_permission_missing", "block_start", "System-audio permission is missing"))
    if not snapshot.microphone_permission:
        actions.append(ControlAction("microphone_permission_missing", "block_start", "Microphone permission is missing"))
    if not snapshot.output_writable:
        actions.append(ControlAction("output_not_writable", "block_start", "Capture folder is not writable"))
    if not snapshot.output_path_matches:
        actions.append(ControlAction("output_path_mismatch", "block_start", "Recorder output path does not match session"))

    missing_sources = sorted(set(snapshot.expected_sources) - set(snapshot.sources_reporting_frames))
    for source in missing_sources:
        actions.append(
            ControlAction(
                f"source_not_reporting:{source}",
                "block_start",
                f"Expected source is not delivering frames: {source}",
            )
        )
    if snapshot.disk_free_bytes <= CRITICAL_DISK_BYTES:
        actions.append(ControlAction("disk_critical", "block_start", "Disk space is critically low"))
    elif snapshot.disk_free_bytes <= LOW_DISK_WARNING_BYTES:
        actions.append(ControlAction("disk_low", "warn", "Disk space is low"))
    return tuple(actions)


def evaluate_watchdog(snapshot: WatchdogSnapshot) -> tuple[ControlAction, ...]:
    actions: list[ControlAction] = []
    if snapshot.sleep_detected:
        actions.append(
            ControlAction(
                "sleep_detected",
                "mark_interrupted",
                "macOS sleep interrupted the capture source",
            )
        )
    if snapshot.recording_reported and not snapshot.recorder_process_alive:
        actions.append(
            ControlAction(
                "recorder_process_died",
                "mark_interrupted",
                "Recorder process exited while recording was reported",
            )
        )
    if (
        snapshot.recording_reported
        and snapshot.elapsed_seconds >= FIRST_OUTPUT_DEADLINE_SECONDS
        and not snapshot.output_exists
    ):
        actions.append(
            ControlAction(
                "no_output_file",
                "mark_interrupted",
                "Recording indicator is active but no media file exists",
            )
        )
    if (
        snapshot.recording_reported
        and snapshot.output_exists
        and snapshot.seconds_since_output_growth >= OUTPUT_STALL_SECONDS
    ):
        actions.append(
            ControlAction(
                "output_stalled",
                "stop_and_finalize",
                "Media file stopped growing",
            )
        )
    if (
        snapshot.recording_reported
        and snapshot.seconds_since_audio_callback >= AUDIO_CALLBACK_STALL_SECONDS
    ):
        actions.append(
            ControlAction(
                "audio_callbacks_stalled",
                "stop_and_finalize",
                "Audio callbacks stopped arriving",
            )
        )
    missing_sources = sorted(set(snapshot.expected_sources) - set(snapshot.sources_reporting_frames))
    for source in missing_sources:
        actions.append(
            ControlAction(
                f"source_lost:{source}",
                "stop_and_finalize",
                f"Expected audio source disappeared: {source}",
            )
        )
    if not snapshot.on_ac_power:
        if snapshot.battery_percent <= CRITICAL_BATTERY_PERCENT:
            actions.append(
                ControlAction(
                    "battery_critical",
                    "stop_and_finalize",
                    "Battery reached the critical capture threshold",
                )
            )
        elif snapshot.battery_percent <= LOW_BATTERY_WARNING_PERCENT:
            actions.append(ControlAction("battery_low", "warn", "Connect the Mac to power"))
    if snapshot.disk_free_bytes <= CRITICAL_DISK_BYTES:
        actions.append(
            ControlAction(
                "disk_critical",
                "stop_and_finalize",
                "Disk space reached the critical capture threshold",
            )
        )
    elif snapshot.disk_free_bytes <= LOW_DISK_WARNING_BYTES:
        actions.append(ControlAction("disk_low", "warn", "Disk space is low"))
    return tuple(actions)


def raw_segment_cleanup_allowed(gate: CleanupGate) -> bool:
    """Raw capture is removable only after every durability gate and consent."""

    return (
        gate.session_status == "complete"
        and gate.canonical_exists
        and gate.canonical_decodable
        and gate.canonical_hash_matches
        and gate.operator_confirmed
    )


def action_codes(actions: Iterable[ControlAction]) -> set[str]:
    return {item.code for item in actions}


def recovery_order(audit: SessionAudit) -> tuple[Path, ...]:
    if not audit.recovery_ready:
        return ()
    return audit.finalized_segments


def write_manifest_atomically(path: Path, manifest: Mapping[str, object]) -> None:
    """Write a manifest with replacement semantics inside its session folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("capture manifest must be an object")
    return payload
