"""Read-only validation and discovery for normalized audio-first meeting sources."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from phone_recording_ingest import (
    CANONICAL_AUDIO_FILENAME,
    GROUPING_POLICY,
    METADATA_FILENAME,
    SOURCE_KIND,
    SOURCE_SCHEMA_VERSION,
    deterministic_folder_name,
    deterministic_group_folder_name,
    group_fingerprint_from_values,
    sha256_file,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IST_OFFSET = timedelta(hours=5, minutes=30)
ACCEPTED_STATUS = "normalized"
ACCEPTED_TRANSCRIPTION_STATUS = "not_started"
ACCEPTED_TIMEZONE = "Asia/Kolkata"


class SourceValidationError(ValueError):
    """A normalized meeting-source folder violates the audio-first contract."""


@dataclass(frozen=True)
class AudioFirstMeetingSource:
    source_folder: Path
    source_kind: str
    source_filename: str
    original_source_path: Path
    source_fingerprint: str
    created_at: datetime
    duration_seconds: float
    canonical_audio_path: Path
    normalization_status: str
    transcription_status: str
    audio_handling: str
    ingested_at: datetime
    group_fingerprint: str | None = None
    grouping_policy: str | None = None

    @property
    def identity_inputs(self) -> tuple[str, str, str, str]:
        return (
            self.source_kind,
            str(self.original_source_path),
            self.created_at.isoformat(),
            self.source_fingerprint,
        )

    @property
    def source_identity(self) -> str:
        if self.group_fingerprint is not None:
            return self.group_fingerprint
        raw = "|".join(self.identity_inputs).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceDiscoveryResult:
    source_folder: Path
    source: AudioFirstMeetingSource | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.source is not None


def _require_exact_type(payload: dict, field: str, expected_type: type):
    if field not in payload:
        raise SourceValidationError(f"missing required field: {field}")
    value = payload[field]
    if type(value) is not expected_type:
        raise SourceValidationError(f"{field} must be {expected_type.__name__}")
    return value


def _parse_offset_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SourceValidationError(f"{field} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceValidationError(f"{field} must be offset-aware")
    if parsed.utcoffset() != IST_OFFSET:
        raise SourceValidationError(f"{field} must use Asia/Kolkata offset +05:30")
    return parsed


def load_audio_first_source(source_folder: Path) -> AudioFirstMeetingSource:
    folder = source_folder.expanduser().resolve()
    if not folder.is_dir():
        raise SourceValidationError(f"source folder is missing or not a directory: {folder}")
    metadata_path = folder / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SourceValidationError(f"missing {METADATA_FILENAME}: {folder}")

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceValidationError(f"cannot read valid JSON metadata: {metadata_path}") from exc
    if type(payload) is not dict:
        raise SourceValidationError("meeting source metadata must be a JSON object")

    schema_version = _require_exact_type(payload, "schema_version", int)
    if schema_version != SOURCE_SCHEMA_VERSION:
        raise SourceValidationError(f"unsupported schema_version: {schema_version}")
    source_kind = _require_exact_type(payload, "source_kind", str)
    if source_kind != SOURCE_KIND:
        raise SourceValidationError(f"unsupported source_kind: {source_kind}")
    source_filename = _require_exact_type(payload, "source_filename", str)
    if not source_filename or Path(source_filename).name != source_filename or Path(source_filename).suffix.lower() != ".m4a":
        raise SourceValidationError("source_filename must be a plain .m4a filename")

    source_path_text = _require_exact_type(payload, "source_path", str)
    original_source_path = Path(source_path_text)
    if not source_path_text or not original_source_path.is_absolute():
        raise SourceValidationError("source_path must be a non-empty absolute path")
    if original_source_path.name != source_filename:
        raise SourceValidationError("source_filename does not match source_path")

    fingerprint = _require_exact_type(payload, "source_fingerprint_sha256", str)
    if SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise SourceValidationError("source_fingerprint_sha256 must be lowercase SHA-256")

    timezone_name = _require_exact_type(payload, "timezone", str)
    if timezone_name != ACCEPTED_TIMEZONE:
        raise SourceValidationError(f"unsupported timezone: {timezone_name}")
    created_at = _parse_offset_datetime(_require_exact_type(payload, "created_at", str), "created_at")
    ingested_at = _parse_offset_datetime(_require_exact_type(payload, "ingested_at", str), "ingested_at")

    duration = payload.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise SourceValidationError("duration_seconds must be a finite positive number")

    status = _require_exact_type(payload, "status", str)
    if status != ACCEPTED_STATUS:
        raise SourceValidationError(f"normalization status is not complete: {status}")
    transcription_status = _require_exact_type(payload, "transcription_status", str)
    if transcription_status != ACCEPTED_TRANSCRIPTION_STATUS:
        raise SourceValidationError(
            f"transcription_status contradicts audio-first contract: {transcription_status}"
        )
    audio_handling = _require_exact_type(payload, "audio_handling", str)
    if audio_handling not in {"copy", "concat_copy"}:
        raise SourceValidationError(f"unsupported audio_handling: {audio_handling}")

    group_fingerprint: str | None = None
    grouping_policy: str | None = None
    source_group = payload.get("source_group")
    source_segments = payload.get("source_segments")
    if audio_handling == "copy":
        if source_group is not None or source_segments is not None:
            raise SourceValidationError("singleton copy source cannot contain grouped provenance")
    else:
        if type(source_group) is not dict or type(source_segments) is not list:
            raise SourceValidationError("concat_copy requires source_group and source_segments")
        grouping_policy = _require_exact_type(source_group, "policy", str)
        if grouping_policy != GROUPING_POLICY:
            raise SourceValidationError(f"unsupported source grouping policy: {grouping_policy}")
        group_fingerprint = _require_exact_type(
            source_group, "group_fingerprint_sha256", str
        )
        if SHA256_PATTERN.fullmatch(group_fingerprint) is None:
            raise SourceValidationError("group_fingerprint_sha256 must be lowercase SHA-256")
        segment_count = _require_exact_type(source_group, "segment_count", int)
        if segment_count < 2 or segment_count != len(source_segments):
            raise SourceValidationError("source group segment_count mismatch")
        identity_values: list[tuple[str, str]] = []
        timestamps: list[datetime] = []
        segment_fingerprints: list[str] = []
        for index, segment in enumerate(source_segments):
            if type(segment) is not dict:
                raise SourceValidationError("source_segments entries must be objects")
            segment_path_text = _require_exact_type(segment, "source_path", str)
            segment_path = Path(segment_path_text)
            segment_filename = _require_exact_type(segment, "source_filename", str)
            if not segment_path.is_absolute() or segment_path.name != segment_filename:
                raise SourceValidationError("source segment path/filename mismatch")
            segment_fingerprint = _require_exact_type(
                segment, "fingerprint_sha256", str
            )
            if SHA256_PATTERN.fullmatch(segment_fingerprint) is None:
                raise SourceValidationError("source segment fingerprint must be lowercase SHA-256")
            segment_created_text = _require_exact_type(segment, "created_at", str)
            segment_created = _parse_offset_datetime(segment_created_text, "source segment created_at")
            if segment_created.isoformat() != segment_created_text:
                raise SourceValidationError("source segment created_at must use canonical ISO format")
            segment_duration = segment.get("duration_seconds")
            if (
                type(segment_duration) not in (int, float)
                or not math.isfinite(segment_duration)
                or segment_duration <= 0
            ):
                raise SourceValidationError("source segment duration must be positive and finite")
            if index == 0 and (
                segment_path != original_source_path
                or segment_filename != source_filename
                or segment_created != created_at
            ):
                raise SourceValidationError("first source segment contradicts primary provenance")
            timestamps.append(segment_created)
            segment_fingerprints.append(segment_fingerprint)
            identity_values.append((segment_created_text, segment_fingerprint))
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise SourceValidationError("source segment timestamps must be strictly ordered and unique")
        if len(segment_fingerprints) != len(set(segment_fingerprints)):
            raise SourceValidationError("source segment fingerprints must be unique")
        if group_fingerprint_from_values(identity_values) != group_fingerprint:
            raise SourceValidationError("source group fingerprint does not match segment provenance")

    canonical_text = _require_exact_type(payload, "canonical_audio_path", str)
    canonical_audio = Path(canonical_text)
    expected_audio = folder / CANONICAL_AUDIO_FILENAME
    if not canonical_text or not canonical_audio.is_absolute():
        raise SourceValidationError("canonical_audio_path must be absolute")
    if canonical_audio != expected_audio:
        raise SourceValidationError(
            f"canonical_audio_path must name {CANONICAL_AUDIO_FILENAME} inside the source folder"
        )
    if canonical_audio.parent.resolve() != folder or canonical_audio.is_symlink():
        raise SourceValidationError("canonical audio escapes the source folder")
    if not canonical_audio.is_file():
        raise SourceValidationError(f"canonical audio is missing: {canonical_audio}")
    if sha256_file(canonical_audio) != fingerprint:
        raise SourceValidationError("canonical audio fingerprint does not match metadata")

    expected_folder_name = (
        deterministic_group_folder_name(created_at, group_fingerprint)
        if group_fingerprint is not None
        else deterministic_folder_name(created_at, fingerprint)
    )
    if folder.name != expected_folder_name:
        raise SourceValidationError(
            f"source folder identity mismatch: expected {expected_folder_name}, got {folder.name}"
        )

    return AudioFirstMeetingSource(
        source_folder=folder,
        source_kind=source_kind,
        source_filename=source_filename,
        original_source_path=original_source_path,
        source_fingerprint=fingerprint,
        created_at=created_at,
        duration_seconds=float(duration),
        canonical_audio_path=canonical_audio,
        normalization_status=status,
        transcription_status=transcription_status,
        audio_handling=audio_handling,
        ingested_at=ingested_at,
        group_fingerprint=group_fingerprint,
        grouping_policy=grouping_policy,
    )


def discover_audio_first_sources(ingest_root: Path) -> tuple[SourceDiscoveryResult, ...]:
    root = ingest_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ingest root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"ingest root is not a directory: {root}")

    results: list[SourceDiscoveryResult] = []
    seen_folders: set[Path] = set()
    metadata_paths = sorted(root.rglob(METADATA_FILENAME), key=lambda path: str(path))
    for metadata_path in metadata_paths:
        folder = metadata_path.parent.resolve()
        if folder in seen_folders:
            continue
        seen_folders.add(folder)
        try:
            source = load_audio_first_source(folder)
        except (OSError, SourceValidationError) as exc:
            results.append(SourceDiscoveryResult(folder, None, str(exc)))
        else:
            results.append(SourceDiscoveryResult(folder, source, None))
    return tuple(results)
