"""Private, default-off Gate 5 candidate decision boundary.

This standalone API does not discover meetings, inspect transcripts, call
``mi``, or wire identity into production. Durable biometric material remains
below the caller's explicit private root. Review results are issued,
short-lived capabilities, so a caller cannot forge a candidate by constructing
a plausible dataclass with an arbitrary score or speaker label.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Callable, Mapping, Sequence

from voiceprint_synthetic_candidate import (
    EnrollmentRecord,
    MatchCandidate,
    PENDING_TTL,
    SyntheticVoiceprintError,
    SyntheticVoiceprintStore,
    TrustedMapEntry,
)


# Synthetic fixture constants only. Real thresholds come from a locally evaluated report.
# Scoring is max cosine against each independent enrollment vector.
GATE4_PROVIDER_ID = "pyannote-community1-wespeaker-resnet34"
GATE4_FROZEN_THRESHOLD = 0.75
GATE4_AGGREGATION_RECIPE = "max_cosine_against_each_enrollment_v1"
GATE4_MODEL_ASSET_SHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
GATE5_SCHEMA_VERSION = 2
REVIEW_TTL = timedelta(hours=24)
METADATA_FILENAME = "gate5_lifecycle.json"
MAP_FILENAME = "gate5_confirmed_map.json"


class Gate5Error(SyntheticVoiceprintError):
    """A fail-closed Gate 5 contract or lifecycle error."""


@dataclass(frozen=True)
class Gate4Provenance:
    provider_id: str
    threshold: float
    aggregation_recipe: str
    report_sha256: str | None = None
    model_asset_sha256: str | None = None


@dataclass(frozen=True)
class InboxCandidate:
    """A review candidate bound to one meeting and one issued review token."""

    observed_speaker: str
    identity_id: str | None
    score: float
    status: str
    confirmed: bool = False
    meeting_id: str = ""
    candidate_id: str = ""
    issued_at: datetime | None = None
    threshold: float = GATE4_FROZEN_THRESHOLD
    audio_sha256: str | None = None


def _opaque(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", value):
        raise Gate5Error(f"{label} must be an opaque lowercase identifier")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Gate5Error(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise Gate5Error(f"{label} is malformed")
    try:
        return _utc(datetime.fromisoformat(value), label)
    except ValueError as exc:
        raise Gate5Error(f"{label} is malformed") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_file(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise Gate5Error(f"{label} must be an explicitly supplied Path")
    lexical = path.expanduser().absolute()
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor not in {Path("/tmp"), Path("/var"), Path("/etc")} and cursor.exists() and cursor.is_symlink():
            raise Gate5Error(f"{label} path contains a symlink")
    if lexical.is_symlink() or not lexical.is_file():
        raise Gate5Error(f"{label} is missing or unsafe")
    resolved = lexical.resolve(strict=True)
    repo = Path(__file__).resolve().parent
    if resolved == repo or repo in resolved.parents:
        raise Gate5Error(f"{label} must be outside the repository")
    lowered = str(resolved).casefold()
    if any(token in lowered for token in ("cloudstorage", "google drive", "staging/phone-drive")):
        raise Gate5Error(f"{label} must not be cloud-synced or staging storage")
    info = resolved.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise Gate5Error(f"{label} must be user-owned and private")
    return resolved


def _private_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise Gate5Error("state_root must be an explicitly supplied Path")
    lexical = path.expanduser().absolute()
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor not in {Path("/tmp"), Path("/var"), Path("/etc")} and cursor.exists() and cursor.is_symlink():
            raise Gate5Error("state_root path must not contain a symlink")
    resolved = lexical.resolve()
    repo = Path(__file__).resolve().parent
    if resolved == repo or repo in resolved.parents:
        raise Gate5Error("state_root must be outside the repository")
    lowered = str(resolved).casefold()
    if any(token in lowered for token in ("cloudstorage", "google drive", "staging/phone-drive")):
        raise Gate5Error("state_root must not be cloud-synced or staging storage")
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = resolved.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise Gate5Error("state_root must be user-owned and mode 0700")
    os.chmod(resolved, 0o700)
    return resolved


class Gate5Inbox:
    """Explicit-root wrapper for Gate 5 operator decisions."""

    def __init__(
        self,
        state_root: Path,
        *,
        clock: Callable[[], datetime],
        threshold: float | None = None,
        gate4_report: Path | None = None,
        provider_id: str = GATE4_PROVIDER_ID,
        model_asset: Path | None = None,
        model_asset_sha256: str | None = None,
    ) -> None:
        self.root = _private_root(state_root)
        self.clock = clock
        self.provenance = self._load_provenance(
            gate4_report, provider_id=provider_id, threshold=threshold, model_asset=model_asset, model_asset_sha256=model_asset_sha256
        )
        self.store = SyntheticVoiceprintStore(
            self.root, clock=clock, default_threshold=self.provenance.threshold
        )
        self.map_path = self.root / MAP_FILENAME
        self.metadata_path = self.root / METADATA_FILENAME
        self._cleanup_stale()

    @property
    def state_path(self) -> Path:
        return self.store.state_path

    def _load_provenance(
        self, report: Path | None, *, provider_id: str, threshold: float | None, model_asset: Path | None, model_asset_sha256: str | None
    ) -> Gate4Provenance:
        if provider_id != GATE4_PROVIDER_ID:
            raise Gate5Error("Gate 5 accepts only the verified Gate 4 provider")
        if report is None:
            raise Gate5Error("an explicit verified Gate 4 report is required")
        report_hash = None
        report_threshold = GATE4_FROZEN_THRESHOLD
        report_model_hash = None
        if report is not None:
            report_path = _private_file(report, "Gate 4 report")
            report_hash = _hash_file(report_path)
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                providers = payload["providers"]
                matching = [item for item in providers if item.get("provider_id") == provider_id and item.get("passed") is True]
                if payload.get("schema_version") != 1 or payload.get("report_type") != "voiceprint_owner_bakeoff" or len(matching) != 1:
                    raise ValueError
                report_threshold = float(matching[0]["threshold"])
                report_model_hash = matching[0].get("model_asset_sha256")
                if not isinstance(report_model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", report_model_hash):
                    raise ValueError
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise Gate5Error("Gate 4 report is malformed or provider was not passed") from exc
            if not math.isfinite(report_threshold) or not -1.0 <= report_threshold <= 1.0:
                raise Gate5Error("Gate 4 threshold must be a finite cosine threshold")
        if threshold is not None:
            try:
                supplied = float(threshold)
            except (TypeError, ValueError) as exc:
                raise Gate5Error("Gate 5 threshold must be numeric") from exc
            if supplied != report_threshold:
                raise Gate5Error("Gate 5 threshold must equal the verified Gate 4 threshold")
        model_hash = None
        if model_asset is not None:
            model_hash = _hash_file(_private_file(model_asset, "model asset"))
        if report_model_hash is not None and model_hash is not None and model_hash != report_model_hash:
            raise Gate5Error("model asset digest does not match the Gate 4 report")
        if model_asset_sha256 is not None:
            if model_asset_sha256 != report_model_hash:
                raise Gate5Error("model asset digest is not the Gate 4 attested asset")
            if model_hash is not None and model_hash != model_asset_sha256:
                raise Gate5Error("model asset digest does not match the supplied asset")
            model_hash = model_asset_sha256
        if model_hash is None or model_hash != report_model_hash:
            raise Gate5Error("the Gate 4 attested model asset is required")
        return Gate4Provenance(provider_id, report_threshold, GATE4_AGGREGATION_RECIPE, report_hash, model_hash)

    def _now(self) -> datetime:
        return _utc(self.clock(), "clock value")

    def pin_model_digest(self, digest: str) -> None:
        """Bind this inbox to the first admitted local model asset digest."""
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise Gate5Error("model asset digest is malformed")
        metadata = self._read_metadata()
        existing = {
            value.get("model_asset_sha256")
            for value in metadata["enrollments"].values()
            if isinstance(value, dict) and value.get("model_asset_sha256")
        }
        if existing and existing != {digest}:
            raise Gate5Error("model asset does not match the pinned Gate 4 model")
        if self.provenance.model_asset_sha256 not in (None, digest):
            raise Gate5Error("model asset does not match the pinned Gate 4 model")
        if self.provenance.model_asset_sha256 != digest:
            self.provenance = replace(self.provenance, model_asset_sha256=digest)

    def _read_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {"schema_version": GATE5_SCHEMA_VERSION, "enrollments": {}, "reviews": {}, "mappings": []}
        if self.metadata_path.is_symlink() or not self.metadata_path.is_file():
            raise Gate5Error("Gate 5 metadata is unsafe")
        info = self.metadata_path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise Gate5Error("Gate 5 metadata is not private")
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Gate5Error("Gate 5 metadata is malformed") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != GATE5_SCHEMA_VERSION:
            raise Gate5Error("Gate 5 metadata schema is unsupported")
        if not isinstance(payload.get("enrollments"), dict) or not isinstance(payload.get("reviews"), dict) or not isinstance(payload.get("mappings"), list):
            raise Gate5Error("Gate 5 metadata is malformed")
        return payload

    def _write_metadata(self, payload: Mapping[str, Any]) -> None:
        if not payload.get("enrollments") and not payload.get("reviews") and not payload.get("mappings"):
            if self.metadata_path.is_symlink():
                raise Gate5Error("Gate 5 metadata is unsafe")
            self.metadata_path.unlink(missing_ok=True)
            return
        temporary = self.metadata_path.with_suffix(".tmp")
        if temporary.is_symlink():
            raise Gate5Error("Gate 5 temporary metadata is unsafe")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.metadata_path)
        os.chmod(self.metadata_path, 0o600)

    def _write_map(self, mappings: list[dict[str, str]]) -> None:
        if not mappings:
            if self.map_path.is_symlink():
                raise Gate5Error("Gate 5 map is unsafe")
            self.map_path.unlink(missing_ok=True)
            return
        temporary = self.map_path.with_suffix(".tmp")
        if temporary.is_symlink() or self.map_path.is_symlink():
            raise Gate5Error("Gate 5 map is unsafe")
        temporary.write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.map_path)
        os.chmod(self.map_path, 0o600)

    def _cleanup_stale(self) -> None:
        records = {item.identity_id: item for item in self.store.all_records()}
        metadata = self._read_metadata()
        missing = set(metadata["enrollments"]) - set(records)
        for identity in missing:
            value = metadata["enrollments"].get(identity)
            try:
                expired = _parse_timestamp(value.get("created_at"), "enrollment created_at") + PENDING_TTL <= self._now()
            except (AttributeError, Gate5Error):
                expired = False
            if not expired:
                raise Gate5Error("Gate 5 vector and provenance metadata disagree")
        for value in metadata["enrollments"].values():
            if not isinstance(value, dict) or value.get("provider_id") != self.provenance.provider_id or value.get("threshold") != self.provenance.threshold or value.get("aggregation_recipe") != self.provenance.aggregation_recipe:
                raise Gate5Error("Gate 5 enrollment provenance does not match Gate 4")
            # The aggregate report contains runtime duration, so its full
            # file hash changes on a repeat of the same approved evaluation.
            # Provider, threshold, aggregation recipe, and model digest above
            # are the stable attestation; refresh the report hash below.
            if self.provenance.model_asset_sha256 is not None and value.get("model_asset_sha256") != self.provenance.model_asset_sha256:
                raise Gate5Error("model asset does not match enrolled provenance")
        metadata["enrollments"] = {key: value for key, value in metadata["enrollments"].items() if key in records}
        for value in metadata["enrollments"].values():
            value["gate4_report_sha256"] = self.provenance.report_sha256
        cutoff = self._now() - REVIEW_TTL
        fresh_reviews: dict[str, Any] = {}
        for key, value in metadata["reviews"].items():
            if not isinstance(value, dict) or value.get("consumed") is True:
                continue
            try:
                if _parse_timestamp(value.get("issued_at"), "review issued_at") >= cutoff:
                    fresh_reviews[key] = value
            except Gate5Error:
                continue
        metadata["reviews"] = fresh_reviews
        metadata["mappings"] = [
            value for value in metadata["mappings"]
            if isinstance(value, dict) and value.get("identity") in records and value.get("meeting_id") and value.get("observed_speaker")
        ]
        self._write_metadata(metadata)
        self._write_map(metadata["mappings"])

    def enroll_pending(self, identity_id: str, vector: Sequence[float], *, audio_sha256: str | None = None) -> EnrollmentRecord:
        identity = _opaque(identity_id, "identity_id")
        record = self.store.enroll(identity, vector)
        metadata = self._read_metadata()
        metadata["enrollments"][identity] = {
            "created_at": record.created_at.isoformat(),
            "provider_id": self.provenance.provider_id,
            "threshold": self.provenance.threshold,
            "aggregation_recipe": self.provenance.aggregation_recipe,
            "gate4_report_sha256": self.provenance.report_sha256,
            "model_asset_sha256": self.provenance.model_asset_sha256,
            "audio_sha256": audio_sha256,
        }
        self._write_metadata(metadata)
        return record

    def review(self, observed: Mapping[str, Sequence[float]], *, meeting_id: str, threshold: float | None = None, audio_sha256: str | None = None) -> tuple[InboxCandidate, ...]:
        meeting = _opaque(meeting_id, "meeting_id")
        if threshold is not None and float(threshold) != self.provenance.threshold:
            raise Gate5Error("review threshold must equal the verified Gate 4 threshold")
        results = self.store.match(observed, threshold=self.provenance.threshold)
        metadata = self._read_metadata()
        issued = self._now()
        candidates: list[InboxCandidate] = []
        for result in results:
            token = secrets.token_urlsafe(24)
            candidates.append(InboxCandidate(result.observed_speaker, result.identity_id, result.score, result.status, result.confirmed, meeting, token, issued, self.provenance.threshold, audio_sha256))
            metadata["reviews"][token] = {
                "meeting_id": meeting, "observed_speaker": result.observed_speaker,
                "identity_id": result.identity_id, "score": result.score,
                "status": result.status, "threshold": self.provenance.threshold,
                "issued_at": issued.isoformat(), "consumed": False, "audio_sha256": audio_sha256,
            }
        self._write_metadata(metadata)
        return tuple(candidates)

    def confirm(self, candidate: InboxCandidate, *, operator_confirmed: bool, display_name: str | None = None) -> TrustedMapEntry | None:
        if operator_confirmed is not True:
            return None
        if not isinstance(candidate, InboxCandidate) or not candidate.candidate_id:
            raise Gate5Error("only an issued Gate 5 candidate can be confirmed")
        metadata = self._read_metadata()
        issued = metadata["reviews"].get(candidate.candidate_id)
        if not isinstance(issued, dict) or issued.get("consumed") is True:
            raise Gate5Error("candidate is stale, unknown, or already consumed")
        if _parse_timestamp(issued.get("issued_at"), "review issued_at") + REVIEW_TTL <= self._now():
            raise Gate5Error("candidate has expired")
        expected = (issued.get("meeting_id"), issued.get("observed_speaker"), issued.get("identity_id"), issued.get("status"))
        actual = (candidate.meeting_id, candidate.observed_speaker, candidate.identity_id, candidate.status)
        if expected != actual or issued.get("threshold") != self.provenance.threshold or candidate.threshold != self.provenance.threshold:
            raise Gate5Error("candidate binding does not match the issued review")
        if isinstance(candidate.score, bool) or not isinstance(candidate.score, (int, float)) or not math.isfinite(float(candidate.score)):
            raise Gate5Error("candidate score is malformed")
        try:
            issued_score = float(issued["score"])
        except (KeyError, TypeError, ValueError):
            raise Gate5Error("issued candidate score is malformed")
        if not math.isfinite(issued_score) or float(candidate.score) != issued_score:
            raise Gate5Error("candidate score does not match the issued review")
        if candidate.status != "candidate" or candidate.identity_id is None:
            raise Gate5Error("only a non-unknown candidate can be confirmed")
        if display_name is not None:
            if not isinstance(display_name, str) or not re.fullmatch(r"[^\x00-\x1f\x7f]{1,80}", display_name.strip()):
                raise Gate5Error("display_name must be a short private human label")
            display_name = display_name.strip()
            enrollment = metadata["enrollments"].get(candidate.identity_id)
            if not isinstance(enrollment, dict):
                raise Gate5Error("candidate enrollment metadata is missing")
            existing_name = enrollment.get("display_name")
            if existing_name is not None and existing_name != display_name:
                raise Gate5Error("display_name does not match the retained identity metadata")
            enrollment["display_name"] = display_name
        trusted = self.store.confirm_match(MatchCandidate(candidate.observed_speaker, candidate.identity_id, candidate.score, candidate.status, False), operator_confirmed=True)
        issued["consumed"] = True
        mapping = [item for item in metadata["mappings"] if not (item.get("meeting_id") == candidate.meeting_id and item.get("observed_speaker") == candidate.observed_speaker)]
        mapping.append({
            "meeting_id": candidate.meeting_id, "observed_speaker": candidate.observed_speaker,
            "identity": trusted.identity, "confirmed_at": self._now().isoformat(),
            "status": "confirmed", "source": "operator_confirmed_gate5",
        })
        metadata["mappings"] = mapping
        self._write_metadata(metadata)
        self._write_map(mapping)
        return trusted

    def confirm_candidate(self, candidate_id: str, *, operator_confirmed: bool, display_name: str | None = None) -> TrustedMapEntry | None:
        """Confirm an issued candidate by opaque token (for CLI handoff)."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Gate5Error("candidate_id is required")
        metadata = self._read_metadata()
        value = metadata["reviews"].get(candidate_id)
        if not isinstance(value, dict):
            raise Gate5Error("candidate is stale or unknown")
        candidate = InboxCandidate(
            value.get("observed_speaker"), value.get("identity_id"), float(value.get("score")),
            value.get("status"), False, value.get("meeting_id"), candidate_id,
            _parse_timestamp(value.get("issued_at"), "review issued_at"), self.provenance.threshold,
        )
        return self.confirm(candidate, operator_confirmed=operator_confirmed, display_name=display_name)

    def set_retention(self, identity_id: str, *, consent: bool | None) -> EnrollmentRecord | None:
        identity = _opaque(identity_id, "identity_id")
        result = self.store.grant_retention(identity, consent=consent)
        if result is None:
            self._remove_identity(identity)
        else:
            self._write_metadata(self._read_metadata())
        return result

    def revoke(self, identity_id: str) -> bool:
        identity = _opaque(identity_id, "identity_id")
        existed = self.store.revoke_retention(identity)
        if existed:
            self._remove_identity(identity)
        return existed

    def _remove_identity(self, identity: str) -> None:
        metadata = self._read_metadata()
        metadata["enrollments"].pop(identity, None)
        metadata["reviews"] = {key: value for key, value in metadata["reviews"].items() if value.get("identity_id") != identity}
        metadata["mappings"] = [value for value in metadata["mappings"] if value.get("identity") != identity]
        self._write_metadata(metadata)
        self._write_map(metadata["mappings"])

    def trusted_map(self) -> tuple[dict[str, str], ...]:
        self._cleanup_stale()
        records = {item.identity_id: item for item in self.store.all_records()}
        metadata = self._read_metadata()
        entries: list[dict[str, str]] = []
        for item in metadata["mappings"]:
            record = records.get(item.get("identity"))
            if record is not None and record.state == "keep" and record.match_confirmed:
                entries.append({
                    "meeting_id": item["meeting_id"], "observed_speaker": item["observed_speaker"],
                    "identity": item["identity"], "status": "confirmed", "source": "operator_confirmed_gate5",
                })
        return tuple(sorted(entries, key=lambda item: (item["meeting_id"], item["observed_speaker"])))

    def private_display_name(self, identity_id: str) -> str | None:
        """Return the retained display name from private metadata, if present."""
        identity = _opaque(identity_id, "identity_id")
        metadata = self._read_metadata()
        value = metadata["enrollments"].get(identity)
        if not isinstance(value, dict):
            raise Gate5Error("identity is not enrolled")
        name = value.get("display_name")
        return name if isinstance(name, str) else None


__all__ = [
    "GATE4_AGGREGATION_RECIPE", "GATE4_FROZEN_THRESHOLD", "GATE4_MODEL_ASSET_SHA256", "GATE4_PROVIDER_ID",
    "Gate4Provenance", "Gate5Error", "Gate5Inbox", "InboxCandidate",
]


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Private, default-off Gate 5 candidate inbox")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--gate4-report", type=Path)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--provider-id", default=GATE4_PROVIDER_ID)
    parser.add_argument("command", choices=("status", "candidates", "confirm", "retention", "revoke", "map"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--display-name")
    args = parser.parse_args()
    try:
        inbox = Gate5Inbox(args.state_root, clock=lambda: datetime.now(timezone.utc), gate4_report=args.gate4_report, provider_id=args.provider_id, model_asset=args.model_asset)
        if args.command == "status":
            records = inbox.store.all_records()
            print(json.dumps({"pending": sum(item.state == "pending" for item in records), "kept": sum(item.state == "keep" for item in records), "confirmed_mappings": len(inbox.trusted_map())}, sort_keys=True))
        elif args.command == "map":
            print(json.dumps(list(inbox.trusted_map()), sort_keys=True))
        elif args.command == "candidates":
            metadata = inbox._read_metadata()
            rows = []
            for candidate_id, value in metadata["reviews"].items():
                if value.get("consumed") is not True:
                    rows.append({"candidate_id": candidate_id, "meeting_id": value.get("meeting_id"), "speaker": value.get("observed_speaker"), "identity": value.get("identity_id"), "status": value.get("status"), "score": value.get("score")})
            print(json.dumps(sorted(rows, key=lambda item: item["meeting_id"]), sort_keys=True))
        elif args.command == "confirm":
            if not args.value or not args.yes:
                raise Gate5Error("confirm requires candidate_id and --yes")
            result = inbox.confirm_candidate(args.value, operator_confirmed=True, display_name=args.display_name)
            print(json.dumps({"status": result.status, "identity": result.identity}, sort_keys=True))
        elif args.command in {"retention", "revoke"}:
            if not args.value:
                raise Gate5Error("identity_id is required")
            if args.command == "revoke":
                removed = inbox.revoke(args.value)
                print(json.dumps({"revoked": removed}, sort_keys=True))
            else:
                if args.keep == args.drop:
                    raise Gate5Error("retention requires exactly one of --keep or --drop")
                record = inbox.set_retention(args.value, consent=args.keep)
                print(json.dumps({"state": record.state if record else "dropped"}, sort_keys=True))
        return 0
    except (Gate5Error, OSError, ValueError) as exc:
        print(f"GATE5_REJECTED={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
