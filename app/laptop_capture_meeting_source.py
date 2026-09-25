"""Strict read-only loader for explicitly selected laptop capture sources."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from laptop_capture_contract import audit_capture_session, load_manifest
from laptop_capture_ingest import (
    CANONICAL_AUDIO_FILENAME,
    CAPTURE_MANIFEST_FILENAME,
    METADATA_FILENAME,
    NORMALIZATION_POLICY,
    PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
    TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
    SOURCE_KIND,
    SOURCE_SCHEMA_VERSION,
    DEFAULT_CAPTURE_ROOT,
    build_source_identity,
    sha256_file,
    terminal_source_loss_recovery_eligible,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IST_OFFSET = timedelta(hours=5, minutes=30)


class LaptopSourceValidationError(ValueError):
    """A normalized laptop source is not safe for explicit admission."""


@dataclass(frozen=True)
class LaptopMeetingSource:
    source_folder: Path
    source_kind: str
    source_identity: str
    display_name: str
    capture_id: str
    capture_manifest_path: Path
    capture_manifest_sha256: str
    canonical_audio_path: Path
    canonical_audio_sha256: str
    created_at: datetime
    duration_seconds: float
    ingested_at: datetime
    audio_handling: str
    source_segment_count: int


@dataclass(frozen=True)
class LaptopSourceDiscoveryResult:
    source_folder: Path
    source: LaptopMeetingSource | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.source is not None


def _exact(payload: dict, field: str, expected: type):
    if field not in payload or type(payload[field]) is not expected:
        raise LaptopSourceValidationError(
            f"{field} must be {expected.__name__}"
        )
    return payload[field]


def _timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LaptopSourceValidationError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != IST_OFFSET:
        raise LaptopSourceValidationError(f"{field} must use Asia/Kolkata +05:30")
    return parsed


def _sha(value: str, field: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise LaptopSourceValidationError(f"{field} must be lowercase SHA-256")
    return value


def load_laptop_meeting_source(
    source_folder: Path,
    *,
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
) -> LaptopMeetingSource:
    lexical_folder = source_folder.expanduser().absolute()
    if lexical_folder.is_symlink():
        raise LaptopSourceValidationError("source folder must not be a symlink")
    try:
        folder = lexical_folder.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise LaptopSourceValidationError("source folder is missing") from exc
    if not folder.is_dir():
        raise LaptopSourceValidationError("source folder is not a directory")

    metadata_path = folder / METADATA_FILENAME
    audio_path = folder / CANONICAL_AUDIO_FILENAME
    if (
        metadata_path.is_symlink()
        or audio_path.is_symlink()
        or not metadata_path.is_file()
        or not audio_path.is_file()
    ):
        raise LaptopSourceValidationError("canonical source is incomplete or unsafe")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaptopSourceValidationError("canonical metadata is unreadable") from exc
    if type(payload) is not dict:
        raise LaptopSourceValidationError("canonical metadata must be an object")

    if _exact(payload, "schema_version", int) != SOURCE_SCHEMA_VERSION:
        raise LaptopSourceValidationError("unsupported schema_version")
    if _exact(payload, "source_kind", str) != SOURCE_KIND:
        raise LaptopSourceValidationError("unsupported source_kind")
    if _exact(payload, "status", str) != "normalized":
        raise LaptopSourceValidationError("source is not normalized")
    if _exact(payload, "transcription_status", str) != "not_started":
        raise LaptopSourceValidationError("source has contradictory transcription state")
    normalization_policy = _exact(payload, "normalization_policy", str)
    if normalization_policy not in {
        NORMALIZATION_POLICY,
        TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
        PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
    }:
        raise LaptopSourceValidationError("unsupported normalization policy")
    if _exact(payload, "timezone", str) != "Asia/Kolkata":
        raise LaptopSourceValidationError("unsupported timezone")

    source_identity = _sha(
        _exact(payload, "source_identity", str), "source_identity"
    )
    display_name = _exact(payload, "display_name", str)
    if (
        not display_name.strip()
        or Path(display_name).name != display_name
        or len(display_name) > 160
    ):
        raise LaptopSourceValidationError("display_name must be a plain safe label")
    capture_id = _exact(payload, "capture_id", str)
    if not capture_id.strip():
        raise LaptopSourceValidationError("capture_id is empty")

    canonical_text = _exact(payload, "canonical_audio_path", str)
    canonical_claim = Path(canonical_text)
    if (
        not canonical_claim.is_absolute()
        or canonical_claim.is_symlink()
        or canonical_claim.resolve(strict=True) != audio_path
    ):
        raise LaptopSourceValidationError("canonical_audio_path is not exact")
    canonical_sha = _sha(
        _exact(payload, "canonical_audio_sha256", str),
        "canonical_audio_sha256",
    )
    if sha256_file(audio_path) != canonical_sha:
        raise LaptopSourceValidationError("canonical audio checksum mismatch")

    manifest_text = _exact(payload, "capture_manifest_path", str)
    manifest_path = Path(manifest_text)
    if (
        not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or manifest_path.name != CAPTURE_MANIFEST_FILENAME
        or not manifest_path.is_file()
    ):
        raise LaptopSourceValidationError("capture manifest path is unsafe")
    lexical_capture_root = capture_root.expanduser().absolute()
    if lexical_capture_root.is_symlink() or not lexical_capture_root.is_dir():
        raise LaptopSourceValidationError("approved capture root is unsafe")
    try:
        resolved_capture_root = lexical_capture_root.resolve(strict=True)
        resolved_manifest = manifest_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise LaptopSourceValidationError("capture manifest path is unavailable") from exc
    if (
        resolved_manifest.parent.parent != resolved_capture_root
        or resolved_manifest.parent.is_symlink()
    ):
        raise LaptopSourceValidationError(
            "capture manifest is outside the approved recording root"
        )
    manifest_path = resolved_manifest
    manifest_sha = _sha(
        _exact(payload, "capture_manifest_sha256", str),
        "capture_manifest_sha256",
    )
    if sha256_file(manifest_path) != manifest_sha:
        raise LaptopSourceValidationError("capture manifest checksum mismatch")
    try:
        manifest = load_manifest(manifest_path)
        audit = audit_capture_session(manifest_path.parent, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LaptopSourceValidationError("capture evidence is unreadable") from exc
    normal_admission = (
        audit.accepted
        and audit.merge_ready
        and manifest.get("status") == "complete"
        and normalization_policy == NORMALIZATION_POLICY
    )
    recovery = payload.get("operator_recovery")
    recovery_admission = (
        normalization_policy in {
            TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
            PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY,
        }
        and terminal_source_loss_recovery_eligible(manifest, audit)
        and type(recovery) is dict
        and recovery.get("attestation")
        == "meeting_ended_before_terminal_source_loss"
        and recovery.get("source_loss_preserved") is True
        and isinstance(recovery.get("attested_at"), str)
    )
    if not normal_admission and not recovery_admission:
        raise LaptopSourceValidationError(
            "capture evidence is no longer eligible for its recorded policy"
        )
    if manifest.get("capture_id") != capture_id:
        raise LaptopSourceValidationError("capture_id contradicts manifest")

    raw_segments = payload.get("source_segments")
    if type(raw_segments) is not list or not raw_segments:
        raise LaptopSourceValidationError("source segment provenance is incomplete")
    prefix_trimmed = (
        normalization_policy
        == PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY
    )
    if prefix_trimmed:
        if not (
            len(raw_segments) < len(audit.finalized_segments)
            and type(recovery) is dict
            and recovery.get("retained_prefix_segment_count") == len(raw_segments)
            and recovery.get("excluded_tail_segment_count")
            == len(audit.finalized_segments) - len(raw_segments)
        ):
            raise LaptopSourceValidationError("trimmed recovery provenance is invalid")
    elif len(raw_segments) != len(audit.finalized_segments):
        raise LaptopSourceValidationError("source segment provenance is incomplete")
    if build_source_identity(
        manifest,
        retained_segment_count=len(raw_segments) if prefix_trimmed else None,
    ) != source_identity:
        raise LaptopSourceValidationError("capture identity mismatch")
    for expected_index, (item, source_path) in enumerate(
        zip(raw_segments, audit.finalized_segments[: len(raw_segments)], strict=True)
    ):
        if type(item) is not dict:
            raise LaptopSourceValidationError("source segment entry is invalid")
        if _exact(item, "index", int) != expected_index:
            raise LaptopSourceValidationError("source segment indexes are not contiguous")
        source_claim = Path(_exact(item, "source_path", str))
        if (
            not source_claim.is_absolute()
            or source_claim.is_symlink()
            or source_claim.resolve(strict=True) != source_path
        ):
            raise LaptopSourceValidationError("source segment path contradicts manifest")
        if _sha(
            _exact(item, "fingerprint_sha256", str),
            "source segment fingerprint",
        ) != sha256_file(source_path):
            raise LaptopSourceValidationError("source segment checksum mismatch")
        item_duration = item.get("duration_seconds")
        if (
            type(item_duration) not in (int, float)
            or not math.isfinite(item_duration)
            or item_duration <= 0
        ):
            raise LaptopSourceValidationError("source segment duration is invalid")

    created_at = _timestamp(_exact(payload, "created_at", str), "created_at")
    ingested_at = _timestamp(_exact(payload, "ingested_at", str), "ingested_at")
    duration = payload.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise LaptopSourceValidationError("duration_seconds is invalid")
    audio_handling = _exact(payload, "audio_handling", str)
    expected_handling = (
        "extract_copy" if len(raw_segments) == 1 else "concat_extract_copy"
    )
    if audio_handling != expected_handling:
        raise LaptopSourceValidationError("audio_handling contradicts segment count")

    expected_name = (
        f"laptop_{created_at:%Y-%m-%d_%H-%M-%S}_{source_identity[:16]}"
    )
    if folder.name != expected_name:
        raise LaptopSourceValidationError("source folder identity mismatch")

    return LaptopMeetingSource(
        source_folder=folder,
        source_kind=SOURCE_KIND,
        source_identity=source_identity,
        display_name=display_name,
        capture_id=capture_id,
        capture_manifest_path=manifest_path,
        capture_manifest_sha256=manifest_sha,
        canonical_audio_path=audio_path,
        canonical_audio_sha256=canonical_sha,
        created_at=created_at,
        duration_seconds=float(duration),
        ingested_at=ingested_at,
        audio_handling=audio_handling,
        source_segment_count=len(raw_segments),
    )


def discover_laptop_meeting_sources(
    ingest_root: Path,
    *,
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
) -> tuple[LaptopSourceDiscoveryResult, ...]:
    root = ingest_root.expanduser()
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise LaptopSourceValidationError("laptop ingest root is unsafe")
    results: list[LaptopSourceDiscoveryResult] = []
    for folder in sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    ):
        try:
            source = load_laptop_meeting_source(
                folder, capture_root=capture_root
            )
        except (LaptopSourceValidationError, OSError, json.JSONDecodeError) as exc:
            results.append(
                LaptopSourceDiscoveryResult(folder, None, str(exc))
            )
        else:
            results.append(LaptopSourceDiscoveryResult(folder, source, None))
    return tuple(results)
