"""Default-off Gate 2 synthetic voiceprint candidate lifecycle.

This module is intentionally independent of MeetingIntel processing. It accepts
only caller-supplied numeric vectors and requires a caller-supplied disposable
state root. It contains no audio, model, network, roster, or production-path
integration.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent
STATE_FILENAME = "voiceprint_candidates.json"
PENDING_TTL = timedelta(days=30)
DEFAULT_THRESHOLD = 0.80
STATE_SCHEMA_VERSION = 1


class SyntheticVoiceprintError(ValueError):
    """Base class for fail-closed synthetic lifecycle errors."""


class InvalidSyntheticVector(SyntheticVoiceprintError):
    """A vector is not a finite, non-empty, one-dimensional numeric sequence."""


class InvalidStateRoot(SyntheticVoiceprintError):
    """The caller did not provide an acceptable disposable state root."""


class InvalidLifecycleState(SyntheticVoiceprintError):
    """State on disk is malformed or an operation is not valid for that state."""


class CandidateNotFound(SyntheticVoiceprintError):
    """An enrollment or candidate no longer exists."""


@dataclass(frozen=True)
class EnrollmentRecord:
    identity_id: str
    vector: tuple[float, ...]
    created_at: datetime
    state: str
    match_confirmed: bool = False


@dataclass(frozen=True)
class MatchCandidate:
    observed_speaker: str
    identity_id: str | None
    score: float
    status: str
    confirmed: bool = False


@dataclass(frozen=True)
class TrustedMapEntry:
    """A shape-compatible candidate result after explicit confirmation."""

    speaker_label: str
    identity: str
    status: str = "confirmed"
    source: str = "operator_confirmed_synthetic_candidate"

    def as_dict(self) -> dict[str, str]:
        return {
            "speaker_label": self.speaker_label,
            "identity": self.identity,
            "status": self.status,
            "source": self.source,
        }


def _normalize_vector(value: Sequence[Real], *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise InvalidSyntheticVector(f"{label} must be a one-dimensional numeric sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise InvalidSyntheticVector(f"{label} must be a one-dimensional numeric sequence") from exc
    if not values:
        raise InvalidSyntheticVector(f"{label} must not be empty")
    normalized: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise InvalidSyntheticVector(f"{label} contains a non-numeric value")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise InvalidSyntheticVector(f"{label} contains a non-finite value")
        normalized.append(numeric)
    if not any(item != 0.0 for item in normalized):
        raise InvalidSyntheticVector(f"{label} must not be the zero vector")
    return tuple(normalized)


def _validate_identity_id(identity_id: str) -> str:
    if not isinstance(identity_id, str) or not identity_id.strip():
        raise SyntheticVoiceprintError("identity_id must be a non-empty string")
    return identity_id.strip()


def _validate_speaker_label(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise SyntheticVoiceprintError("observed speaker label must be a non-empty string")
    return label.strip()


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise SyntheticVoiceprintError("threshold must be numeric")
    value = float(threshold)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise SyntheticVoiceprintError("threshold must be finite and between -1 and 1")
    return value


def cosine_similarity(left: Sequence[Real], right: Sequence[Real]) -> float:
    """Return cosine similarity for two validated, equal-dimension vectors."""
    left_vector = _normalize_vector(left, label="left vector")
    right_vector = _normalize_vector(right, label="right vector")
    if len(left_vector) != len(right_vector):
        raise InvalidSyntheticVector("vectors must have equal dimensions")
    numerator = sum(a * b for a, b in zip(left_vector, right_vector))
    left_norm = math.sqrt(sum(a * a for a in left_vector))
    right_norm = math.sqrt(sum(b * b for b in right_vector))
    return numerator / (left_norm * right_norm)


def _utc_datetime(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SyntheticVoiceprintError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


class SyntheticVoiceprintStore:
    """Explicit-root store and matcher for synthetic voiceprint candidates."""

    def __init__(
        self,
        state_root: Path,
        *,
        clock: Callable[[], datetime],
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.state_root = self._validate_state_root(state_root)
        self.state_path = self.state_root / STATE_FILENAME
        self.clock = clock
        self.default_threshold = _validate_threshold(default_threshold)
        self._now()

    @staticmethod
    def _validate_state_root(raw_root: Path) -> Path:
        if not isinstance(raw_root, Path):
            raise InvalidStateRoot("state_root must be an explicitly supplied Path")
        candidate = raw_root.expanduser()
        # Refuse symlinks in the complete existing path, not just at the leaf.
        # A private-looking leaf beneath a redirected parent is not a private
        # state root.
        absolute = candidate.absolute()
        cursor = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            cursor /= part
            # macOS exposes /tmp as the system-owned /private/tmp symlink;
            # this fixed platform alias does not redirect to user data.
            if cursor not in {Path("/tmp"), Path("/var"), Path("/etc")} and cursor.exists() and cursor.is_symlink():
                raise InvalidStateRoot("state_root path must not contain a symlink")
        if candidate.exists() and candidate.is_symlink():
            raise InvalidStateRoot("state_root must not be a symlink")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(REPOSITORY_ROOT)
        except ValueError:
            pass
        else:
            raise InvalidStateRoot("state_root must be outside the repository")
        if resolved.exists() and not resolved.is_dir():
            raise InvalidStateRoot("state_root must be a directory")
        resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            mode = stat.S_IMODE(resolved.stat().st_mode)
            owner = resolved.stat().st_uid
        except OSError as exc:
            raise InvalidStateRoot("state_root permissions cannot be checked") from exc
        if owner != os.getuid() or mode & 0o077:
            raise InvalidStateRoot("state_root must be user-owned and mode 0700")
        os.chmod(resolved, 0o700)
        return resolved

    def _now(self) -> datetime:
        return _utc_datetime(self.clock(), label="clock value")

    def _read_records(self) -> dict[str, EnrollmentRecord]:
        if not self.state_path.exists():
            return {}
        if self.state_path.is_symlink() or not self.state_path.is_file():
            raise InvalidLifecycleState("voiceprint state file is unsafe")
        try:
            mode = stat.S_IMODE(self.state_path.stat().st_mode)
            if self.state_path.stat().st_uid != os.getuid() or mode & 0o077:
                raise InvalidLifecycleState("voiceprint state file is not private")
        except OSError as exc:
            raise InvalidLifecycleState("voiceprint state file permissions cannot be checked") from exc
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidLifecycleState("voiceprint state is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise InvalidLifecycleState("voiceprint state schema is unsupported")
        raw_records = payload.get("records")
        if not isinstance(raw_records, dict):
            raise InvalidLifecycleState("voiceprint state records are malformed")
        records: dict[str, EnrollmentRecord] = {}
        for key, raw in raw_records.items():
            if not isinstance(key, str) or not isinstance(raw, dict):
                raise InvalidLifecycleState("voiceprint state record is malformed")
            identity_id = _validate_identity_id(raw.get("identity_id"))
            if identity_id != key:
                raise InvalidLifecycleState("voiceprint identity key mismatch")
            vector = _normalize_vector(raw.get("vector"), label="stored vector")
            created_at_raw = raw.get("created_at")
            if not isinstance(created_at_raw, str):
                raise InvalidLifecycleState("stored creation time is malformed")
            try:
                created_at = _utc_datetime(datetime.fromisoformat(created_at_raw), label="stored creation time")
            except (TypeError, ValueError) as exc:
                raise InvalidLifecycleState("stored creation time is malformed") from exc
            state = raw.get("state")
            if state not in {"pending", "keep"}:
                raise InvalidLifecycleState("stored lifecycle state is invalid")
            match_confirmed = raw.get("match_confirmed", False)
            if not isinstance(match_confirmed, bool):
                raise InvalidLifecycleState("stored confirmation state is invalid")
            records[identity_id] = EnrollmentRecord(
                identity_id=identity_id,
                vector=vector,
                created_at=created_at,
                state=state,
                match_confirmed=match_confirmed,
            )
        return records

    def _write_records(self, records: Mapping[str, EnrollmentRecord]) -> None:
        if self.state_path.is_symlink():
            raise InvalidLifecycleState("voiceprint state file is unsafe")
        if not records:
            self.state_path.unlink(missing_ok=True)
            return
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "records": {
                identity_id: {
                    "identity_id": record.identity_id,
                    "vector": list(record.vector),
                    "created_at": record.created_at.isoformat(),
                    "state": record.state,
                    "match_confirmed": record.match_confirmed,
                }
                for identity_id, record in sorted(records.items())
            },
        }
        temporary = self.state_path.with_suffix(".tmp")
        if temporary.is_symlink():
            raise InvalidLifecycleState("voiceprint temporary state file is unsafe")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.state_path)
        os.chmod(self.state_path, 0o600)

    def _load_active_records(self) -> dict[str, EnrollmentRecord]:
        records = self._read_records()
        now = self._now()
        active = {
            identity_id: record
            for identity_id, record in records.items()
            if record.state != "pending" or now < record.created_at + PENDING_TTL
        }
        if len(active) != len(records):
            self._write_records(active)
        return active

    def enroll(self, identity_id: str, vector: Sequence[Real]) -> EnrollmentRecord:
        identity_id = _validate_identity_id(identity_id)
        normalized = _normalize_vector(vector, label="synthetic vector")
        records = self._load_active_records()
        if identity_id in records:
            raise SyntheticVoiceprintError("identity is already enrolled")
        if records and len(normalized) != len(next(iter(records.values())).vector):
            raise InvalidSyntheticVector("all enrolled vectors must have equal dimensions")
        record = EnrollmentRecord(
            identity_id=identity_id,
            vector=normalized,
            created_at=self._now(),
            state="pending",
        )
        records[identity_id] = record
        self._write_records(records)
        return record

    def get(self, identity_id: str) -> EnrollmentRecord | None:
        return self._load_active_records().get(_validate_identity_id(identity_id))

    def all_records(self) -> tuple[EnrollmentRecord, ...]:
        records = self._load_active_records()
        return tuple(records[key] for key in sorted(records))

    def match(
        self,
        observed: Mapping[str, Sequence[Real]],
        *,
        threshold: float | None = None,
    ) -> tuple[MatchCandidate, ...]:
        if not isinstance(observed, Mapping):
            raise SyntheticVoiceprintError("observed must be a mapping of speaker labels to vectors")
        threshold_value = self.default_threshold if threshold is None else _validate_threshold(threshold)
        records = self._load_active_records()
        observed_vectors = {
            _validate_speaker_label(label): _normalize_vector(vector, label="observed vector")
            for label, vector in observed.items()
        }
        if records:
            dimension = len(next(iter(records.values())).vector)
            if any(len(vector) != dimension for vector in observed_vectors.values()):
                raise InvalidSyntheticVector("observed vectors must match enrolled dimensions")

        scored: list[tuple[float, str, str]] = []
        best_by_observed: dict[str, tuple[float, str]] = {}
        for speaker, vector in observed_vectors.items():
            for identity_id, record in records.items():
                score = cosine_similarity(vector, record.vector)
                scored.append((score, speaker, identity_id))
                best = best_by_observed.get(speaker)
                if best is None or (score, identity_id) > best:
                    best_by_observed[speaker] = (score, identity_id)

        assigned_speakers: set[str] = set()
        assigned_identities: set[str] = set()
        assignments: dict[str, tuple[str, float]] = {}
        for score, speaker, identity_id in sorted(scored, key=lambda item: (-item[0], item[1], item[2])):
            if score < threshold_value:
                continue
            if speaker in assigned_speakers or identity_id in assigned_identities:
                continue
            assigned_speakers.add(speaker)
            assigned_identities.add(identity_id)
            assignments[speaker] = (identity_id, score)

        results: list[MatchCandidate] = []
        for speaker in sorted(observed_vectors):
            assignment = assignments.get(speaker)
            if assignment is not None:
                identity_id, score = assignment
                results.append(MatchCandidate(speaker, identity_id, score, "candidate"))
                continue
            best = best_by_observed.get(speaker)
            score = best[0] if best is not None else 0.0
            results.append(MatchCandidate(speaker, None, score, "unknown"))
        return tuple(results)

    def confirm_match(
        self,
        candidate: MatchCandidate,
        *,
        operator_confirmed: bool,
    ) -> TrustedMapEntry | None:
        if operator_confirmed is not True:
            return None
        if candidate.status != "candidate" or candidate.identity_id is None:
            raise SyntheticVoiceprintError("only a non-unknown candidate can be confirmed")
        records = self._load_active_records()
        record = records.get(candidate.identity_id)
        if record is None:
            raise CandidateNotFound("candidate enrollment no longer exists")
        records[candidate.identity_id] = EnrollmentRecord(
            identity_id=record.identity_id,
            vector=record.vector,
            created_at=record.created_at,
            state=record.state,
            match_confirmed=True,
        )
        self._write_records(records)
        return TrustedMapEntry(candidate.observed_speaker, candidate.identity_id)

    def grant_retention(
        self, identity_id: str, *, consent: bool | None
    ) -> EnrollmentRecord | None:
        if consent is False:
            self.drop(identity_id)
            return None
        if consent is not True:
            return self.get(identity_id)
        identity_id = _validate_identity_id(identity_id)
        records = self._load_active_records()
        record = records.get(identity_id)
        if record is None:
            raise CandidateNotFound("identity enrollment was not found")
        retained = EnrollmentRecord(
            identity_id=record.identity_id,
            vector=record.vector,
            created_at=record.created_at,
            state="keep",
            match_confirmed=record.match_confirmed,
        )
        records[identity_id] = retained
        self._write_records(records)
        return retained

    def drop(self, identity_id: str) -> bool:
        identity_id = _validate_identity_id(identity_id)
        records = self._load_active_records()
        if identity_id not in records:
            return False
        del records[identity_id]
        self._write_records(records)
        return True

    def revoke_retention(self, identity_id: str) -> bool:
        """Delete a retained vector and metadata; this is not a tombstone."""
        return self.drop(identity_id)
