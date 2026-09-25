"""Default-off Gate 4 readiness harness for a future consented quality test.

Preparation validates explicit manifests and isolated paths only. It does not
read audio, load a model, generate embeddings, or touch MeetingIntel paths.
Execution is available only behind an explicit flag and is never part of
normal MeetingIntel processing.
"""

from __future__ import annotations
from mi_paths import public_path

import argparse
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from voiceprint_synthetic_candidate import cosine_similarity


REPOSITORY_ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MANIFEST_TYPES = {
    "consent": "gate4_consent",
    "test_set": "gate4_test_set",
    "criteria": "gate4_criteria",
}
AGGREGATE_REPORT_FILENAME = "gate4_aggregate_metrics.json"
MEASUREMENT_DIRNAME = "measurements"
PENDING_PROVENANCE_FILENAME = "gate4_pending_provenance.json"
CANDIDATE_MATERIAL_DIRNAME = "candidate-material"
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Gate4Error(ValueError):
    """Base class for fail-closed Gate 4 readiness errors."""


class Gate4PathError(Gate4Error):
    """A path is missing, unsafe, linked, or in a normal MeetingIntel area."""


class Gate4ManifestError(Gate4Error):
    """A supplied Gate 4 manifest is malformed or inconsistent."""


class Gate4CriteriaIncomplete(Gate4ManifestError):
    """Owner-required pass/fail decisions have not been supplied."""


class Gate4ExecutionDisabled(Gate4Error):
    """The future execution gate was not explicitly supplied."""


class Gate4Expired(Gate4Error):
    """The declared evaluation expiry has passed."""


class Gate4RetentionError(Gate4Error):
    """A Gate 4 retention or cleanup operation is unsafe."""


@dataclass(frozen=True)
class ConsentSample:
    person_id: str
    sample_id: str
    audio_path: Path
    session_id: str
    capture_group: str
    candidate_id: str
    source_recording_id: str
    source_start_seconds: float
    source_end_seconds: float


@dataclass(frozen=True)
class ConsentPerson:
    person_id: str
    retention_instruction: str
    retention_expires_at: datetime | None
    samples: tuple[ConsentSample, ...]


@dataclass(frozen=True)
class ConsentManifest:
    evaluation_id: str
    created_at: datetime
    evaluation_expires_at: datetime
    persons: tuple[ConsentPerson, ...]

    @property
    def samples(self) -> dict[str, ConsentSample]:
        return {
            sample.sample_id: sample
            for person in self.persons
            for sample in person.samples
        }


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    kind: str
    left_sample_id: str
    right_sample_id: str


@dataclass(frozen=True)
class UnknownCase:
    case_id: str
    sample_id: str
    against_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class TestSetManifest:
    evaluation_id: str
    positive_comparisons: tuple[ComparisonCase, ...]
    negative_comparisons: tuple[ComparisonCase, ...]
    unknown_cases: tuple[UnknownCase, ...]


@dataclass(frozen=True)
class CriteriaManifest:
    evaluation_id: str
    evaluation_expires_at: datetime
    threshold: float
    threshold_selection_rule: str
    positive_minimum_pass_rate: float
    positive_minimum_count: int
    negative_maximum_false_match_rate: float
    negative_minimum_count: int
    unknown_minimum_abstention_rate: float
    unknown_maximum_false_candidate_rate: float
    unknown_minimum_count: int
    false_candidate_maximum_rate: float
    false_candidate_minimum_count: int
    cleanup_required_result: str


@dataclass(frozen=True)
class CleanupPlan:
    measurement_root: Path
    candidate_material_root: Path
    pending_provenance_path: Path
    retention_by_person: tuple[tuple[str, str], ...]
    delete_unretained_material: bool
    delete_on_expiry: bool
    no_tombstone: bool


@dataclass(frozen=True)
class PendingCandidate:
    candidate_id: str
    person_id: str
    sample_id: str
    retention_expires_at: datetime
    source_recording_id: str
    source_start_seconds: float
    source_end_seconds: float


@dataclass(frozen=True)
class Gate4Plan:
    isolated_root: Path
    consent_manifest_path: Path
    test_set_manifest_path: Path
    criteria_manifest_path: Path
    evaluation_id: str
    evaluation_expires_at: datetime
    consent: ConsentManifest
    test_set: TestSetManifest
    criteria: CriteriaManifest
    cleanup: CleanupPlan
    pending_candidates: tuple[PendingCandidate, ...]

    @property
    def aggregate_report_path(self) -> Path:
        return self.isolated_root / AGGREGATE_REPORT_FILENAME


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Gate4ManifestError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _now(value: datetime | None) -> datetime:
    return _utc(value if value is not None else datetime.now(timezone.utc), "current time")


def _opaque(value: Any, label: str) -> str:
    if not isinstance(value, str) or OPAQUE_ID_RE.fullmatch(value) is None:
        raise Gate4ManifestError(f"{label} must be an opaque identifier")
    return value


def _required_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise Gate4ManifestError(f"{label} has missing or unexpected fields")


def _normal_roots() -> tuple[Path, ...]:
    return (
        REPOSITORY_ROOT,
        Path.home() / "Movies" / "meetily-recordings",
        Path(str(public_path('home/Movies/meetily-recordings'))),
        Path(str(public_path('work'))),
    )


def _has_symlink_component(path: Path) -> bool:
    """Reject symlinks anywhere in an explicit path, not just at its leaf."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_normal_path(path: Path, label: str) -> None:
    for root in _normal_roots():
        resolved_root = root.resolve()
        if _under(path, resolved_root):
            raise Gate4PathError(f"{label} is in a normal or production path")
    unsafe_components = {
        "state",
        "output",
        "ledger",
        "meetily-recordings",
        "phone-recordings",
        "google drive",
        "my drive",
        "googledrive",
        "drive",
        "com.nll.asr",
    }
    unsafe_tokens = ("staging", "ingest", "drive-sync")
    if any(
        part.casefold() in unsafe_components
        or any(token in part.casefold() for token in unsafe_tokens)
        for part in path.parts
    ):
        raise Gate4PathError(f"{label} is in a normal, staging, or Drive-synced path")


def _validate_directory(raw: Path, label: str) -> Path:
    if not isinstance(raw, Path):
        raise Gate4PathError(f"{label} must be an explicit path")
    candidate = raw.expanduser()
    if not candidate.is_absolute():
        raise Gate4PathError(f"{label} must be an absolute path")
    if _has_symlink_component(candidate):
        raise Gate4PathError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Gate4PathError(f"{label} is missing") from exc
    if not resolved.is_dir():
        raise Gate4PathError(f"{label} must be a directory")
    _reject_normal_path(resolved, label)
    return resolved


def _validate_file(raw: Path, label: str) -> Path:
    if not isinstance(raw, Path):
        raise Gate4PathError(f"{label} must be an explicit path")
    candidate = raw.expanduser()
    if not candidate.is_absolute():
        raise Gate4PathError(f"{label} must be an absolute path")
    if _has_symlink_component(candidate):
        raise Gate4PathError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Gate4PathError(f"{label} is missing") from exc
    if not resolved.is_file():
        raise Gate4PathError(f"{label} must be a regular file")
    _reject_normal_path(resolved, label)
    return resolved


def _load_manifest(path: Path, manifest_type: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Gate4ManifestError(f"{manifest_type} manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise Gate4ManifestError(f"{manifest_type} manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Gate4ManifestError(f"{manifest_type} manifest schema is unsupported")
    if payload.get("manifest_type") != MANIFEST_TYPES[manifest_type]:
        raise Gate4ManifestError(f"{manifest_type} manifest type is invalid")
    return payload


def _parse_expiry(value: Any, now: datetime, label: str) -> datetime:
    if not isinstance(value, str):
        raise Gate4ManifestError(f"{label} is required")
    try:
        expiry = _utc(datetime.fromisoformat(value), label)
    except ValueError as exc:
        raise Gate4ManifestError(f"{label} is invalid") from exc
    if expiry <= now:
        raise Gate4ManifestError(f"{label} must be in the future")
    return expiry


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise Gate4ManifestError(f"{label} is required")
    try:
        return _utc(datetime.fromisoformat(value), label)
    except ValueError as exc:
        raise Gate4ManifestError(f"{label} is invalid") from exc


def _seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate4ManifestError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Gate4ManifestError(f"{label} must be a finite non-negative number")
    return result


def validate_consent_manifest(path: Path, *, now: datetime | None = None) -> ConsentManifest:
    current = _now(now)
    validated_path = _validate_file(path, "consent manifest")
    payload = _load_manifest(validated_path, "consent")
    _required_keys(
        payload,
        {
            "schema_version",
            "manifest_type",
            "evaluation_id",
            "created_at",
            "evaluation_expires_at",
            "persons",
        },
        "consent manifest",
    )
    evaluation_id = _opaque(payload["evaluation_id"], "evaluation_id")
    created_at = _parse_timestamp(payload["created_at"], "created_at")
    if created_at > current:
        raise Gate4ManifestError("created_at must not be in the future")
    expiry = _parse_expiry(payload["evaluation_expires_at"], current, "evaluation_expires_at")
    if expiry <= created_at:
        raise Gate4ManifestError("evaluation_expires_at must follow created_at")
    persons_raw = payload["persons"]
    if not isinstance(persons_raw, list) or not persons_raw:
        raise Gate4ManifestError("consent manifest must contain persons")
    persons: list[ConsentPerson] = []
    person_ids: set[str] = set()
    sample_ids: set[str] = set()
    for person_raw in persons_raw:
        if not isinstance(person_raw, dict):
            raise Gate4ManifestError("consent person entry is malformed")
        _required_keys(
            person_raw,
            {
                "person_id",
                "permission_to_evaluate",
                "retention_instruction",
                "retention_expires_at",
                "samples",
            },
            "consent person",
        )
        person_id = _opaque(person_raw["person_id"], "person_id")
        if person_id in person_ids:
            raise Gate4ManifestError("consent person IDs must be unique")
        person_ids.add(person_id)
        if person_raw["permission_to_evaluate"] != "yes":
            raise Gate4ManifestError("each evaluated person requires explicit permission yes")
        retention = person_raw["retention_instruction"]
        if retention not in {"pending", "keep", "drop"}:
            raise Gate4ManifestError("retention_instruction must be pending, keep, or drop")
        retention_expiry_raw = person_raw["retention_expires_at"]
        retention_expiry: datetime | None = None
        if retention == "pending":
            retention_expiry = _parse_expiry(
                retention_expiry_raw, current, f"{person_id}.retention_expires_at"
            )
            if retention_expiry > created_at + timedelta(days=10):
                raise Gate4ManifestError("Gate 4 pending retention cannot exceed 10 days")
            if retention_expiry > expiry:
                raise Gate4ManifestError("pending retention must expire by evaluation expiry")
        elif retention_expiry_raw is not None:
            raise Gate4ManifestError("only pending retention may have retention_expires_at")
        samples_raw = person_raw["samples"]
        if not isinstance(samples_raw, list) or not samples_raw:
            raise Gate4ManifestError("each consented person requires samples")
        samples: list[ConsentSample] = []
        for sample_raw in samples_raw:
            if not isinstance(sample_raw, dict):
                raise Gate4ManifestError("consent sample entry is malformed")
            _required_keys(
                sample_raw,
                {
                    "sample_id",
                    "audio_path",
                    "session_id",
                    "capture_group",
                    "supported_1to1_confirmed",
                    "single_speaker_attestation",
                    "candidate_id",
                    "source_locator",
                },
                "consent sample",
            )
            sample_id = _opaque(sample_raw["sample_id"], "sample_id")
            if sample_id in sample_ids:
                raise Gate4ManifestError("sample IDs must be unique")
            sample_ids.add(sample_id)
            if sample_raw["supported_1to1_confirmed"] is not True:
                raise Gate4ManifestError("each sample requires supported_1to1_confirmed true")
            attestation = sample_raw["single_speaker_attestation"]
            if not isinstance(attestation, dict):
                raise Gate4ManifestError("single_speaker_attestation is required")
            _required_keys(
                attestation,
                {"only_named_speaker", "no_meaningful_overlap"},
                "single_speaker_attestation",
            )
            if attestation["only_named_speaker"] is not True:
                raise Gate4ManifestError("sample must be attested as single-speaker only")
            if attestation["no_meaningful_overlap"] is not True:
                raise Gate4ManifestError("sample must be attested as having no meaningful overlap")
            audio_raw = sample_raw["audio_path"]
            if not isinstance(audio_raw, str):
                raise Gate4ManifestError("consent audio path must be a string")
            audio_path = _validate_file(Path(audio_raw), "consent audio path")
            session_id = _opaque(sample_raw["session_id"], "session_id")
            capture_group = _opaque(sample_raw["capture_group"], "capture_group")
            candidate_id = _opaque(sample_raw["candidate_id"], "candidate_id")
            locator = sample_raw["source_locator"]
            if not isinstance(locator, dict):
                raise Gate4ManifestError("source_locator is required")
            _required_keys(
                locator,
                {"recording_id", "start_seconds", "end_seconds"},
                "source_locator",
            )
            source_recording_id = _opaque(locator["recording_id"], "source_locator.recording_id")
            source_start = _seconds(locator["start_seconds"], "source_locator.start_seconds")
            source_end = _seconds(locator["end_seconds"], "source_locator.end_seconds")
            if source_end <= source_start:
                raise Gate4ManifestError("source_locator.end_seconds must follow start_seconds")
            samples.append(
                ConsentSample(
                    person_id,
                    sample_id,
                    audio_path,
                    session_id,
                    capture_group,
                    candidate_id,
                    source_recording_id,
                    source_start,
                    source_end,
                )
            )
        persons.append(ConsentPerson(person_id, retention, retention_expiry, tuple(samples)))
    candidate_ids = [sample.candidate_id for person in persons for sample in person.samples]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise Gate4ManifestError("candidate IDs must be unique")
    return ConsentManifest(evaluation_id, created_at, expiry, tuple(persons))


def _parse_ref(value: Any, label: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise Gate4ManifestError(f"{label} reference is malformed")
    _required_keys(value, {"person_id", "sample_id"}, label)
    return _opaque(value["person_id"], f"{label}.person_id"), _opaque(
        value["sample_id"], f"{label}.sample_id"
    )


def validate_test_set_manifest(
    path: Path, consent: ConsentManifest
) -> TestSetManifest:
    validated_path = _validate_file(path, "test-set manifest")
    payload = _load_manifest(validated_path, "test_set")
    _required_keys(
        payload,
        {
            "schema_version",
            "manifest_type",
            "evaluation_id",
            "positive_comparisons",
            "negative_comparisons",
            "unknown_cases",
        },
        "test-set manifest",
    )
    evaluation_id = _opaque(payload["evaluation_id"], "evaluation_id")
    if evaluation_id != consent.evaluation_id:
        raise Gate4ManifestError("test-set evaluation_id does not match consent")
    samples = consent.samples
    by_person = {sample.sample_id: sample.person_id for sample in samples.values()}
    by_session = {sample.sample_id: sample.session_id for sample in samples.values()}
    by_capture = {sample.sample_id: sample.capture_group for sample in samples.values()}

    def parse_comparisons(raw: Any, kind: str) -> tuple[ComparisonCase, ...]:
        if not isinstance(raw, list) or not raw:
            raise Gate4ManifestError(f"test-set requires {kind} comparisons")
        cases: list[ComparisonCase] = []
        for item in raw:
            if not isinstance(item, dict):
                raise Gate4ManifestError(f"{kind} comparison is malformed")
            _required_keys(item, {"case_id", "kind", "left", "right"}, f"{kind} comparison")
            case_id = _opaque(item["case_id"], "comparison case_id")
            if item["kind"] != kind:
                raise Gate4ManifestError(f"comparison kind must be {kind}")
            left_person, left_sample = _parse_ref(item["left"], "left")
            right_person, right_sample = _parse_ref(item["right"], "right")
            if left_sample not in samples or right_sample not in samples:
                raise Gate4ManifestError("comparison references an unknown sample")
            if by_person[left_sample] != left_person or by_person[right_sample] != right_person:
                raise Gate4ManifestError("comparison person/sample identity mismatch")
            if kind == "positive":
                if left_person != right_person:
                    raise Gate4ManifestError("positive comparison must use one person")
                if (
                    by_session[left_sample] == by_session[right_sample]
                    or by_capture[left_sample] == by_capture[right_sample]
                ):
                    raise Gate4ManifestError("positive comparison requires separate session and capture groups")
            elif left_person == right_person:
                raise Gate4ManifestError("negative comparison must use different people")
            cases.append(ComparisonCase(case_id, kind, left_sample, right_sample))
        return tuple(cases)

    positives = parse_comparisons(payload["positive_comparisons"], "positive")
    negatives = parse_comparisons(payload["negative_comparisons"], "negative")
    unknown_raw = payload["unknown_cases"]
    if not isinstance(unknown_raw, list) or not unknown_raw:
        raise Gate4ManifestError("test-set requires unknown_cases")
    unknowns: list[UnknownCase] = []
    case_ids: set[str] = set()
    for case in (*positives, *negatives):
        if case.case_id in case_ids:
            raise Gate4ManifestError("test-set case IDs must be unique")
        case_ids.add(case.case_id)
    for item in unknown_raw:
        if not isinstance(item, dict):
            raise Gate4ManifestError("unknown case is malformed")
        _required_keys(item, {"case_id", "sample", "against", "expected_behavior"}, "unknown case")
        case_id = _opaque(item["case_id"], "unknown case_id")
        if case_id in case_ids:
            raise Gate4ManifestError("test-set case IDs must be unique")
        case_ids.add(case_id)
        _person_id, sample_id = _parse_ref(item["sample"], "unknown sample")
        if sample_id not in samples:
            raise Gate4ManifestError("unknown case references an unknown sample")
        if _person_id != samples[sample_id].person_id:
            raise Gate4ManifestError("unknown sample person/sample identity mismatch")
        if item["expected_behavior"] != "abstain":
            raise Gate4ManifestError("unknown case must require abstain behavior")
        against = item["against"]
        if not isinstance(against, list) or not against:
            raise Gate4ManifestError("unknown case requires comparison references")
        against_ids: list[str] = []
        for index, reference in enumerate(against):
            _reference_person, reference_id = _parse_ref(reference, f"unknown against[{index}]")
            if reference_id not in samples:
                raise Gate4ManifestError("unknown case references an unknown comparison sample")
            if _reference_person != samples[reference_id].person_id:
                raise Gate4ManifestError("unknown comparison person/sample identity mismatch")
            against_ids.append(reference_id)
        if len(set(against_ids)) != len(against_ids):
            raise Gate4ManifestError("unknown case comparison references must be unique")
        unknowns.append(UnknownCase(case_id, sample_id, tuple(against_ids)))
    return TestSetManifest(evaluation_id, positives, negatives, tuple(unknowns))


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate4CriteriaIncomplete(f"owner must supply numeric {label}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Gate4CriteriaIncomplete(f"owner-supplied {label} must be between 0 and 1")
    return result


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise Gate4CriteriaIncomplete(f"owner must supply positive {label}")
    return value


def validate_criteria_manifest(
    path: Path, consent: ConsentManifest, *, now: datetime | None = None
) -> CriteriaManifest:
    current = _now(now)
    validated_path = _validate_file(path, "criteria manifest")
    payload = _load_manifest(validated_path, "criteria")
    _required_keys(
        payload,
        {
            "schema_version",
            "manifest_type",
            "evaluation_id",
            "evaluation_expires_at",
            "owner_approved",
            "threshold_selection",
            "positive",
            "negative",
            "unknown",
            "false_candidate",
            "cleanup",
        },
        "criteria manifest",
    )
    evaluation_id = _opaque(payload["evaluation_id"], "evaluation_id")
    if evaluation_id != consent.evaluation_id:
        raise Gate4CriteriaIncomplete("criteria evaluation_id does not match consent")
    expiry = _parse_expiry(payload["evaluation_expires_at"], current, "criteria evaluation_expires_at")
    if expiry != consent.evaluation_expires_at:
        raise Gate4CriteriaIncomplete("criteria expiry must match consent expiry")
    if payload["owner_approved"] is not True:
        raise Gate4CriteriaIncomplete("owner_approved must be explicitly true")

    threshold_raw = payload["threshold_selection"]
    if not isinstance(threshold_raw, dict):
        raise Gate4CriteriaIncomplete("threshold_selection requires owner decisions")
    _required_keys(threshold_raw, {"method", "threshold", "rule"}, "threshold_selection")
    if not isinstance(threshold_raw["method"], str) or not threshold_raw["method"].strip():
        raise Gate4CriteriaIncomplete("owner must supply threshold selection method")
    if not isinstance(threshold_raw["rule"], str) or not threshold_raw["rule"].strip():
        raise Gate4CriteriaIncomplete("owner must supply threshold selection rule")
    threshold = _rate(threshold_raw["threshold"], "threshold")

    def section(name: str, expected: set[str]) -> dict[str, Any]:
        value = payload[name]
        if not isinstance(value, dict):
            raise Gate4CriteriaIncomplete(f"{name} criteria are required")
        if set(value) != expected:
            raise Gate4CriteriaIncomplete(f"{name} criteria are incomplete")
        return value

    positive = section("positive", {"minimum_pass_rate", "minimum_count"})
    negative = section("negative", {"maximum_false_match_rate", "minimum_count"})
    unknown = section("unknown", {"minimum_abstention_rate", "maximum_false_candidate_rate", "minimum_count"})
    false_candidate = section("false_candidate", {"maximum_rate", "minimum_count"})
    cleanup = section(
        "cleanup",
        {"required_result", "delete_unretained_material", "delete_on_expiry", "no_tombstone"},
    )
    if cleanup["required_result"] != "retain_only_explicit_keep":
        raise Gate4CriteriaIncomplete("cleanup required_result must be retain_only_explicit_keep")
    for key in ("delete_unretained_material", "delete_on_expiry", "no_tombstone"):
        if cleanup[key] is not True:
            raise Gate4CriteriaIncomplete(f"cleanup.{key} must be explicitly true")
    return CriteriaManifest(
        evaluation_id=evaluation_id,
        evaluation_expires_at=expiry,
        threshold=threshold,
        threshold_selection_rule=threshold_raw["rule"],
        positive_minimum_pass_rate=_rate(positive["minimum_pass_rate"], "positive.minimum_pass_rate"),
        positive_minimum_count=_count(positive["minimum_count"], "positive.minimum_count"),
        negative_maximum_false_match_rate=_rate(negative["maximum_false_match_rate"], "negative.maximum_false_match_rate"),
        negative_minimum_count=_count(negative["minimum_count"], "negative.minimum_count"),
        unknown_minimum_abstention_rate=_rate(unknown["minimum_abstention_rate"], "unknown.minimum_abstention_rate"),
        unknown_maximum_false_candidate_rate=_rate(unknown["maximum_false_candidate_rate"], "unknown.maximum_false_candidate_rate"),
        unknown_minimum_count=_count(unknown["minimum_count"], "unknown.minimum_count"),
        false_candidate_maximum_rate=_rate(false_candidate["maximum_rate"], "false_candidate.maximum_rate"),
        false_candidate_minimum_count=_count(false_candidate["minimum_count"], "false_candidate.minimum_count"),
        cleanup_required_result=cleanup["required_result"],
    )


def _pending_candidates(consent: ConsentManifest) -> tuple[PendingCandidate, ...]:
    candidates: list[PendingCandidate] = []
    for person in consent.persons:
        if person.retention_instruction != "pending":
            continue
        if person.retention_expires_at is None:
            raise Gate4ManifestError("pending person is missing retention expiry")
        for sample in person.samples:
            candidates.append(
                PendingCandidate(
                    candidate_id=sample.candidate_id,
                    person_id=sample.person_id,
                    sample_id=sample.sample_id,
                    retention_expires_at=person.retention_expires_at,
                    source_recording_id=sample.source_recording_id,
                    source_start_seconds=sample.source_start_seconds,
                    source_end_seconds=sample.source_end_seconds,
                )
            )
    return tuple(candidates)


def _pending_provenance_payload(plan: Gate4Plan) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance_type": "gate4_pending_candidate_provenance",
        "evaluation_id": plan.evaluation_id,
        "evaluation_expires_at": plan.evaluation_expires_at.isoformat(),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "person_id": candidate.person_id,
                "sample_id": candidate.sample_id,
                "pending_expires_at": candidate.retention_expires_at.isoformat(),
                "source_locator": {
                    "recording_id": candidate.source_recording_id,
                    "start_seconds": candidate.source_start_seconds,
                    "end_seconds": candidate.source_end_seconds,
                },
            }
            for candidate in plan.pending_candidates
        ],
    }


def _write_pending_provenance(plan: Gate4Plan) -> None:
    path = plan.cleanup.pending_provenance_path
    if not plan.pending_candidates:
        return
    if path.exists() or path.is_symlink():
        raise Gate4PathError("pending provenance path must be unused before preparation")
    path.write_text(
        json.dumps(_pending_provenance_payload(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cleanup_gate4_material(plan: Gate4Plan, *, now: datetime | None = None) -> tuple[str, ...]:
    """Delete only explicit drop or expired-pending material in the isolated root."""
    current = _now(now)
    pending_expired = {
        candidate.candidate_id
        for candidate in plan.pending_candidates
        if current >= candidate.retention_expires_at or current >= plan.evaluation_expires_at
    }
    drop_ids = {
        sample.candidate_id
        for person in plan.consent.persons
        if person.retention_instruction == "drop"
        for sample in person.samples
    }
    delete_ids = pending_expired | drop_ids
    material_root = plan.cleanup.candidate_material_root
    for candidate_id in sorted(delete_ids):
        material_path = material_root / candidate_id
        if material_path.is_symlink():
            raise Gate4PathError("candidate material must not be a symlink")
        if material_path.is_dir():
            shutil.rmtree(material_path)
        elif material_path.exists():
            if not material_path.is_file():
                raise Gate4RetentionError("candidate material is not a regular file or directory")
            material_path.unlink()

    provenance_path = plan.cleanup.pending_provenance_path
    if provenance_path.is_symlink():
        raise Gate4PathError("pending provenance must not be a symlink")
    if provenance_path.exists():
        if not provenance_path.is_file():
            raise Gate4RetentionError("pending provenance is not a regular file")
        remaining = [
            candidate
            for candidate in plan.pending_candidates
            if candidate.candidate_id not in pending_expired
        ]
        if remaining:
            replacement = Gate4Plan(
                isolated_root=plan.isolated_root,
                consent_manifest_path=plan.consent_manifest_path,
                test_set_manifest_path=plan.test_set_manifest_path,
                criteria_manifest_path=plan.criteria_manifest_path,
                evaluation_id=plan.evaluation_id,
                evaluation_expires_at=plan.evaluation_expires_at,
                consent=plan.consent,
                test_set=plan.test_set,
                criteria=plan.criteria,
                cleanup=plan.cleanup,
                pending_candidates=tuple(remaining),
            )
            provenance_path.write_text(
                json.dumps(_pending_provenance_payload(replacement), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            provenance_path.unlink()
    return tuple(sorted(delete_ids))


def prepare_gate4(
    isolated_root: Path,
    consent_manifest: Path,
    test_set_manifest: Path,
    criteria_manifest: Path,
    *,
    now: datetime | None = None,
) -> Gate4Plan:
    current = _now(now)
    root = _validate_directory(isolated_root, "isolated evaluation root")
    consent_path = _validate_file(consent_manifest, "consent manifest")
    test_set_path = _validate_file(test_set_manifest, "test-set manifest")
    criteria_path = _validate_file(criteria_manifest, "criteria manifest")
    consent = validate_consent_manifest(consent_path, now=current)
    test_set = validate_test_set_manifest(test_set_path, consent)
    criteria = validate_criteria_manifest(criteria_path, consent, now=current)
    if test_set.evaluation_id != consent.evaluation_id:
        raise Gate4ManifestError("test-set evaluation_id does not match consent")
    measurement_root = root / MEASUREMENT_DIRNAME
    if measurement_root.exists() and measurement_root.is_symlink():
        raise Gate4PathError("measurement root must not be a symlink")
    if measurement_root.exists() and not measurement_root.is_dir():
        raise Gate4PathError("measurement root must be a directory")
    if measurement_root.is_dir() and any(measurement_root.iterdir()):
        raise Gate4PathError("measurement root must be empty before evaluation")
    if (root / AGGREGATE_REPORT_FILENAME).is_symlink():
        raise Gate4PathError("aggregate report path must not be a symlink")
    candidate_material_root = root / CANDIDATE_MATERIAL_DIRNAME
    if candidate_material_root.is_symlink():
        raise Gate4PathError("candidate material root must not be a symlink")
    if candidate_material_root.exists() and not candidate_material_root.is_dir():
        raise Gate4PathError("candidate material root must be a directory")
    if candidate_material_root.is_dir() and any(candidate_material_root.iterdir()):
        raise Gate4PathError("candidate material root must be empty before evaluation")
    pending_provenance_path = root / PENDING_PROVENANCE_FILENAME
    if pending_provenance_path.is_symlink():
        raise Gate4PathError("pending provenance path must not be a symlink")
    pending_candidates = _pending_candidates(consent)
    cleanup = CleanupPlan(
        measurement_root=measurement_root,
        candidate_material_root=candidate_material_root,
        pending_provenance_path=pending_provenance_path,
        retention_by_person=tuple(
            (person.person_id, person.retention_instruction) for person in consent.persons
        ),
        delete_unretained_material=True,
        delete_on_expiry=True,
        no_tombstone=True,
    )
    plan = Gate4Plan(
        isolated_root=root,
        consent_manifest_path=consent_path,
        test_set_manifest_path=test_set_path,
        criteria_manifest_path=criteria_path,
        evaluation_id=consent.evaluation_id,
        evaluation_expires_at=consent.evaluation_expires_at,
        consent=consent,
        test_set=test_set,
        criteria=criteria,
        cleanup=cleanup,
        pending_candidates=pending_candidates,
    )
    _write_pending_provenance(plan)
    return plan


def _vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise Gate4Error("embedding provider returned a malformed vector")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise Gate4Error("embedding provider returned a malformed vector") from exc
    if not values or not all(math.isfinite(item) for item in values):
        raise Gate4Error("embedding provider returned a malformed vector")
    if not any(item != 0.0 for item in values):
        raise Gate4Error("embedding provider returned a zero vector")
    return values


def _default_embedding_provider() -> Callable[[Path], Sequence[float]]:
    """Lazy future-only provider; never imported during readiness validation."""
    from pyannote.audio.pipelines import SpeakerEmbedding

    embedding_pipeline = SpeakerEmbedding(
        embedding="pyannote/embedding",
        token=os.environ.get("HF_TOKEN") or None,
    )

    def provide(audio_path: Path) -> Sequence[float]:
        result = embedding_pipeline(str(audio_path))
        array = getattr(result, "data", result)
        shape = getattr(array, "shape", None)
        if shape is not None and len(shape) == 2 and shape[0] == 1:
            array = array[0]
        return _vector(array)

    return provide


def execute_gate4(
    plan: Gate4Plan,
    *,
    execute: bool = False,
    embedding_provider: Callable[[Path], Sequence[float]] | None = None,
    now: datetime | None = None,
) -> Path:
    """Run only when explicitly enabled; write aggregate metrics only in isolation."""
    if not execute:
        raise Gate4ExecutionDisabled("Gate 4 execution requires the explicit --execute flag")
    current = _now(now)
    deleted = set(cleanup_gate4_material(plan, now=current))
    if current >= plan.evaluation_expires_at:
        raise Gate4Expired("evaluation expiry has passed")
    expired_pending = {
        candidate.candidate_id for candidate in plan.pending_candidates
        if current >= candidate.retention_expires_at
    }
    if deleted & expired_pending:
        raise Gate4Expired("a pending candidate retention expiry has passed")
    provider = embedding_provider or _default_embedding_provider()
    vectors: dict[str, tuple[float, ...]] = {}
    for sample_id, sample in plan.consent.samples.items():
        _validate_file(sample.audio_path, "consent audio path")
        vectors[sample_id] = _vector(provider(sample.audio_path))

    def score(left: str, right: str) -> float:
        try:
            return cosine_similarity(vectors[left], vectors[right])
        except KeyError as exc:
            raise Gate4Error("test-set sample was not measured") from exc

    positive_scores = [
        score(case.left_sample_id, case.right_sample_id)
        for case in plan.test_set.positive_comparisons
    ]
    negative_scores = [
        score(case.left_sample_id, case.right_sample_id)
        for case in plan.test_set.negative_comparisons
    ]
    unknown_scores = [
        max(score(case.sample_id, reference) for reference in case.against_sample_ids)
        for case in plan.test_set.unknown_cases
    ]
    threshold = plan.criteria.threshold
    positive_pass_rate = sum(value >= threshold for value in positive_scores) / len(positive_scores)
    negative_false_rate = sum(value >= threshold for value in negative_scores) / len(negative_scores)
    unknown_abstention_rate = sum(value < threshold for value in unknown_scores) / len(unknown_scores)
    false_candidate_rate = sum(value >= threshold for value in unknown_scores) / len(unknown_scores)
    criteria = plan.criteria
    meets = (
        len(positive_scores) >= criteria.positive_minimum_count
        and positive_pass_rate >= criteria.positive_minimum_pass_rate
        and len(negative_scores) >= criteria.negative_minimum_count
        and negative_false_rate <= criteria.negative_maximum_false_match_rate
        and len(unknown_scores) >= criteria.unknown_minimum_count
        and unknown_abstention_rate >= criteria.unknown_minimum_abstention_rate
        and false_candidate_rate <= criteria.unknown_maximum_false_candidate_rate
        and false_candidate_rate <= criteria.false_candidate_maximum_rate
        and len(unknown_scores) >= criteria.false_candidate_minimum_count
    )
    report = {
        "schema_version": 1,
        "report_type": "gate4_aggregate_metrics",
        "status": "pass" if meets else "fail",
        "threshold_applied": threshold,
        "positive": {"count": len(positive_scores), "pass_rate": positive_pass_rate},
        "negative": {"count": len(negative_scores), "false_match_rate": negative_false_rate},
        "unknown": {"count": len(unknown_scores), "abstention_rate": unknown_abstention_rate},
        "false_candidate": {"count": len(unknown_scores), "rate": false_candidate_rate},
        "cleanup": {
            "vectors_persisted": False,
            "unretained_material_deleted": True,
            "no_tombstone": plan.cleanup.no_tombstone,
        },
    }
    plan.aggregate_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if plan.cleanup.measurement_root.exists():
        shutil.rmtree(plan.cleanup.measurement_root)
    return plan.aggregate_report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an isolated Gate 4 voiceprint evaluation.")
    parser.add_argument("--isolated-root", type=Path, required=True)
    parser.add_argument("--consent-manifest", type=Path, required=True)
    parser.add_argument("--test-set-manifest", type=Path, required=True)
    parser.add_argument("--criteria-manifest", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicit future execution gate; never use during readiness preparation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = prepare_gate4(
            args.isolated_root,
            args.consent_manifest,
            args.test_set_manifest,
            args.criteria_manifest,
        )
        if not args.execute:
            print("GATE4_READY=1")
            print("GATE4_EXECUTION=disabled")
            return 0
        report_path = execute_gate4(plan, execute=True)
        print(f"GATE4_REPORT={report_path.name}")
        return 0
    except Gate4Error as exc:
        print(f"GATE4_REJECTED={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
