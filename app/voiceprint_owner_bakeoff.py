#!/usr/bin/env python3
"""Isolated, provider-neutral Gate 4 speaker-verification bake-off.

This module is not imported by production.  It reads only explicit manifests,
keeps embeddings in memory, writes aggregate measurements without vectors or
paths, and requires an explicit ``--execute`` flag before provider execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_NAME = "voiceprint_owner_bakeoff_report.json"
SAMPLE_CLEANUP_MARKER = "voiceprint_owner_bakeoff_cleanup.json"
ALLOWED_LANES = {"phone", "laptop"}
ALLOWED_CONDITIONS = {"clean", "noisy", "far_field"}
ALLOWED_SPLITS = {"enrollment", "calibration", "test"}
ALLOWED_ROLES = {"target", "negative", "unknown"}
ALLOWED_RETENTION = {"pending", "keep", "drop"}


class BakeoffError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    sample_id: str
    person_id: str
    role: str
    lane: str
    condition: str
    split: str
    session_id: str
    capture_group: str
    audio_path: Path
    sha256: str
    size_bytes: int
    retention: str


@dataclass(frozen=True)
class Criteria:
    minimum_enrollment: int
    minimum_calibration_target: int
    minimum_calibration_non_target: int
    minimum_test_target: int
    minimum_test_negative: int
    minimum_test_unknown: int
    minimum_overall_target_accept_rate: float
    minimum_cell_target_accept_rate: float
    maximum_false_accepts: int
    maximum_unknown_false_candidates: int
    threshold_margin: float
    required_test_cells: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    command: tuple[str, ...]
    model_asset: Path


@dataclass(frozen=True)
class Plan:
    root: Path
    sample_root: Path
    evaluation_id: str
    target_id: str
    expires_at: datetime
    samples: tuple[Sample, ...]
    criteria: Criteria
    providers: tuple[ProviderSpec, ...]


def _load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise BakeoffError("manifest is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BakeoffError("manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise BakeoffError("manifest root must be an object")
    return value


def _private_root(path: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise BakeoffError(f"{label} must be an existing ordinary directory")
    resolved = lexical.resolve(strict=True)
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise BakeoffError(f"{label} must be outside the repository")
    lowered = str(resolved).casefold()
    if any(token in lowered for token in ("cloudstorage", "google drive", "staging/phone-drive")):
        raise BakeoffError(f"{label} must not be cloud-synced or staging storage")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise BakeoffError(f"{label} must be private to the current user")
    if resolved.stat().st_uid != os.getuid():
        raise BakeoffError(f"{label} must be owned by the current user")
    return resolved


def _inside(path: Path, root: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise BakeoffError(f"{label} is missing or unsafe")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BakeoffError(f"{label} is outside the approved root") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise BakeoffError(f"{label} is not an ordinary file")
    return resolved


def _model_asset(path: Path, root: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.exists():
        raise BakeoffError("provider model asset is unsafe")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BakeoffError("provider model asset must be inside the isolated root") from exc
    if resolved.is_dir() and any(item.is_symlink() for item in resolved.rglob("*")):
        raise BakeoffError("provider model asset contains a symlink")
    if not resolved.is_dir() and not resolved.is_file():
        raise BakeoffError("provider model asset is not an ordinary file or directory")
    return resolved


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BakeoffError(f"{label} is required")
    return value.strip()


def _opaque_id(value: object, label: str) -> str:
    result = _text(value, label)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", result):
        raise BakeoffError(f"{label} must be an opaque lowercase identifier")
    return result


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BakeoffError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BakeoffError(f"{label} must be a nonnegative integer")
    return value


def _rate(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BakeoffError(f"{label} must be a rate")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise BakeoffError(f"{label} must be between zero and one")
    return result


def _parse_time(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BakeoffError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise BakeoffError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_sha256(path: Path) -> str:
    """Hash a model file or deterministic private model directory contents."""
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _criteria(payload: dict) -> Criteria:
    required = payload.get("required_test_cells")
    if not isinstance(required, list) or not required:
        raise BakeoffError("required_test_cells are required")
    cells: list[tuple[str, str]] = []
    for item in required:
        if not isinstance(item, dict):
            raise BakeoffError("required_test_cells are malformed")
        lane, condition = item.get("lane"), item.get("condition")
        if lane not in ALLOWED_LANES or condition not in ALLOWED_CONDITIONS:
            raise BakeoffError("required_test_cells contain an unsupported value")
        cells.append((lane, condition))
    if len(cells) != len(set(cells)):
        raise BakeoffError("required_test_cells contain duplicates")
    maximum_false_accepts = _nonnegative_int(
        payload.get("maximum_false_accepts"), "maximum_false_accepts"
    )
    maximum_unknown_false_candidates = _nonnegative_int(
        payload.get("maximum_unknown_false_candidates"),
        "maximum_unknown_false_candidates",
    )
    if maximum_false_accepts != 0 or maximum_unknown_false_candidates != 0:
        raise BakeoffError("false-match acceptance limits must be zero")
    return Criteria(
        _positive_int(payload.get("minimum_enrollment"), "minimum_enrollment"),
        _positive_int(payload.get("minimum_calibration_target"), "minimum_calibration_target"),
        _positive_int(payload.get("minimum_calibration_non_target"), "minimum_calibration_non_target"),
        _positive_int(payload.get("minimum_test_target"), "minimum_test_target"),
        _positive_int(payload.get("minimum_test_negative"), "minimum_test_negative"),
        _positive_int(payload.get("minimum_test_unknown"), "minimum_test_unknown"),
        _rate(payload.get("minimum_overall_target_accept_rate"), "minimum_overall_target_accept_rate"),
        _rate(payload.get("minimum_cell_target_accept_rate"), "minimum_cell_target_accept_rate"),
        maximum_false_accepts,
        maximum_unknown_false_candidates,
        _rate(payload.get("threshold_margin"), "threshold_margin"),
        tuple(cells),
    )


def load_plan(root: Path, samples_path: Path, criteria_path: Path, providers_path: Path) -> Plan:
    isolated = _private_root(root, label="isolated root")
    sample_payload = _load_json(samples_path)
    criteria_payload = _load_json(criteria_path)
    provider_payload = _load_json(providers_path)
    for payload, kind in ((sample_payload, "samples"), (criteria_payload, "criteria"), (provider_payload, "providers")):
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("manifest_type") != f"voiceprint_owner_{kind}":
            raise BakeoffError(f"{kind} manifest contract is unsupported")
    evaluation_id = _opaque_id(sample_payload.get("evaluation_id"), "evaluation_id")
    expires_at = _parse_time(sample_payload.get("evaluation_expires_at"), "evaluation_expires_at")
    for payload, label in ((criteria_payload, "criteria"), (provider_payload, "providers")):
        if _opaque_id(payload.get("evaluation_id"), f"{label} evaluation_id") != evaluation_id:
            raise BakeoffError("manifest evaluation identities disagree")
        if _parse_time(payload.get("evaluation_expires_at"), f"{label} evaluation_expires_at") != expires_at:
            raise BakeoffError("manifest evaluation expiries disagree")
    if criteria_payload.get("owner_approved") is not True:
        raise BakeoffError("explicit owner approval of criteria is required")
    sample_root = _private_root(Path(_text(sample_payload.get("sample_root"), "sample_root")), label="sample root")
    if sample_root.parent != isolated:
        raise BakeoffError("sample root must be a direct child of the isolated root")
    target_id = _opaque_id(sample_payload.get("target_id"), "target_id")
    if expires_at <= datetime.now(timezone.utc):
        raise BakeoffError("evaluation has expired")
    raw_samples = sample_payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise BakeoffError("samples are required")
    samples: list[Sample] = []
    ids: set[str] = set()
    split_sessions: dict[str, set[str]] = {split: set() for split in ALLOWED_SPLITS}
    split_groups: dict[str, set[str]] = {split: set() for split in ALLOWED_SPLITS}
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise BakeoffError("sample entry is malformed")
        sample_id = _opaque_id(raw.get("sample_id"), "sample_id")
        if sample_id in ids:
            raise BakeoffError("sample identities must be unique")
        ids.add(sample_id)
        role, lane, condition, split = raw.get("role"), raw.get("lane"), raw.get("condition"), raw.get("split")
        if role not in ALLOWED_ROLES or lane not in ALLOWED_LANES or condition not in ALLOWED_CONDITIONS or split not in ALLOWED_SPLITS:
            raise BakeoffError("sample classification is unsupported")
        person_id = _opaque_id(raw.get("person_id"), "person_id")
        if (role == "target") != (person_id == target_id):
            raise BakeoffError("target role and target identity disagree")
        if raw.get("permission_to_evaluate") != "yes":
            raise BakeoffError("explicit evaluation permission is required")
        attestation = raw.get("single_speaker_attestation")
        if not isinstance(attestation, dict) or attestation.get("one_speaker_only") is not True or attestation.get("no_meaningful_overlap") is not True:
            raise BakeoffError("single-speaker and no-overlap attestation are required")
        retention = raw.get("retention_instruction")
        if retention not in ALLOWED_RETENTION:
            raise BakeoffError("retention instruction is required")
        session_id = _opaque_id(raw.get("session_id"), "session_id")
        capture_group = _opaque_id(raw.get("capture_group"), "capture_group")
        audio = _inside(Path(_text(raw.get("audio_path"), "audio_path")), sample_root, label="sample audio")
        expected_hash = _text(raw.get("sha256"), "sha256")
        if audio.stat().st_size != raw.get("size_bytes") or _sha256(audio) != expected_hash:
            raise BakeoffError("sample audio changed after manifest creation")
        split_sessions[split].add(session_id)
        split_groups[split].add(capture_group)
        samples.append(Sample(sample_id, person_id, role, lane, condition, split, session_id, capture_group, audio, expected_hash, audio.stat().st_size, retention))
    for left, right in (("enrollment", "calibration"), ("enrollment", "test"), ("calibration", "test")):
        if split_sessions[left] & split_sessions[right] or split_groups[left] & split_groups[right]:
            raise BakeoffError("session or capture-group leakage across evaluation splits")
    criteria = _criteria(criteria_payload)
    raw_providers = provider_payload.get("providers")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise BakeoffError("providers are required")
    providers: list[ProviderSpec] = []
    provider_ids: set[str] = set()
    for raw in raw_providers:
        if not isinstance(raw, dict):
            raise BakeoffError("provider entry is malformed")
        provider_id = _opaque_id(raw.get("provider_id"), "provider_id")
        command = raw.get("command")
        if provider_id in provider_ids or not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
            raise BakeoffError("provider identity or command is invalid")
        provider_ids.add(provider_id)
        model_asset = _model_asset(
            Path(_text(raw.get("model_asset"), "model_asset")), isolated
        )
        providers.append(ProviderSpec(provider_id, tuple(command), model_asset))
    return Plan(isolated, sample_root, evaluation_id, target_id, expires_at, tuple(samples), criteria, tuple(providers))


def readiness(plan: Plan) -> tuple[str, ...]:
    gaps: list[str] = []
    counts = {(split, role): sum(sample.split == split and sample.role == role for sample in plan.samples) for split in ALLOWED_SPLITS for role in ALLOWED_ROLES}
    requirements = {
        ("enrollment", "target"): plan.criteria.minimum_enrollment,
        ("calibration", "target"): plan.criteria.minimum_calibration_target,
        ("test", "target"): plan.criteria.minimum_test_target,
        ("test", "negative"): plan.criteria.minimum_test_negative,
        ("test", "unknown"): plan.criteria.minimum_test_unknown,
    }
    calibration_non_target = counts[("calibration", "negative")] + counts[("calibration", "unknown")]
    if calibration_non_target < plan.criteria.minimum_calibration_non_target:
        gaps.append(
            "calibration_non_target_samples:missing="
            f"{plan.criteria.minimum_calibration_non_target - calibration_non_target}"
        )
    for key, minimum in requirements.items():
        if counts[key] < minimum:
            gaps.append(
                f"{key[0]}_{key[1]}_samples:missing={minimum - counts[key]}"
            )
    test_target_cells = {(sample.lane, sample.condition) for sample in plan.samples if sample.split == "test" and sample.role == "target"}
    for lane, condition in plan.criteria.required_test_cells:
        if (lane, condition) not in test_target_cells:
            gaps.append(f"test_cell_{lane}_{condition}:missing=1")
    for provider in plan.providers:
        executable = Path(provider.command[0]).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            gaps.append(f"provider_{provider.provider_id}_executable")
        if provider.model_asset.is_symlink() or not provider.model_asset.exists():
            gaps.append(f"provider_{provider.provider_id}_model_asset")
    return tuple(sorted(set(gaps)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise BakeoffError("provider vectors have incompatible dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(dot) or left_norm == 0 or right_norm == 0:
        raise BakeoffError("provider returned an invalid vector")
    score = dot / (left_norm * right_norm)
    if not math.isfinite(score):
        raise BakeoffError("provider returned a nonfinite score")
    return score


def _select_threshold(target_scores: Sequence[float], non_target_scores: Sequence[float], margin: float) -> float:
    if not target_scores or not non_target_scores:
        raise BakeoffError("calibration scores are incomplete")
    threshold = max(non_target_scores) + margin
    if threshold > 1.0:
        raise BakeoffError("no threshold can satisfy the false-accept boundary")
    return threshold


def _command_provider(spec: ProviderSpec) -> Callable[[Path], Sequence[float]]:
    def embed(audio: Path) -> Sequence[float]:
        environment = {
            key: os.environ[key]
            for key in ("LANG", "LC_ALL", "PATH", "TMPDIR")
            if key in os.environ
        }
        cache_root = spec.model_asset.parent / ".runtime-cache"
        cache_root.mkdir(mode=0o700, exist_ok=True)
        # Existing private roots may carry a macOS provenance flag that makes
        # a redundant chmod return EPERM under the managed runtime. Only try
        # the repair when the directory is actually too broad; never loosen it.
        if stat.S_IMODE(cache_root.stat().st_mode) & 0o077:
            try:
                os.chmod(cache_root, 0o700)
            except OSError as exc:
                raise BakeoffError("provider runtime cache is not private") from exc
        environment.update({
            "HOME": str(cache_root),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "PYTHONNOUSERSITE": "1",
            "TORCH_HOME": str(cache_root / "torch"),
        })
        try:
            completed = subprocess.run(
                [*spec.command, "--model-asset", str(spec.model_asset), "--audio", str(audio)],
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BakeoffError(f"provider {spec.provider_id} could not complete") from exc
        if completed.returncode != 0:
            raise BakeoffError(f"provider {spec.provider_id} failed")
        try:
            payload = json.loads(completed.stdout)
            vector = payload["embedding"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise BakeoffError(f"provider {spec.provider_id} output is malformed") from exc
        if not isinstance(vector, list) or not vector or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector):
            raise BakeoffError(f"provider {spec.provider_id} vector is malformed")
        return tuple(float(value) for value in vector)
    return embed


def evaluate(plan: Plan, providers: Mapping[str, Callable[[Path], Sequence[float]]] | None = None) -> dict:
    gaps = readiness(plan)
    if gaps:
        raise BakeoffError("readiness gaps remain: " + ",".join(gaps))
    supplied = dict(providers or {})
    reports: list[dict] = []
    for spec in plan.providers:
        embed = supplied.get(spec.provider_id) or _command_provider(spec)
        started = time.monotonic()
        vectors = {sample.sample_id: tuple(embed(sample.audio_path)) for sample in plan.samples}
        enrollment = [vectors[sample.sample_id] for sample in plan.samples if sample.split == "enrollment" and sample.role == "target"]
        def score(sample: Sample) -> float:
            return max(_cosine(vectors[sample.sample_id], enrolled) for enrolled in enrollment)
        calibration_target = [score(sample) for sample in plan.samples if sample.split == "calibration" and sample.role == "target"]
        calibration_non_target = [score(sample) for sample in plan.samples if sample.split == "calibration" and sample.role != "target"]
        threshold = _select_threshold(calibration_target, calibration_non_target, plan.criteria.threshold_margin)
        tested = [(sample, score(sample)) for sample in plan.samples if sample.split == "test"]
        target = [(sample, value) for sample, value in tested if sample.role == "target"]
        negatives = [(sample, value) for sample, value in tested if sample.role == "negative"]
        unknowns = [(sample, value) for sample, value in tested if sample.role == "unknown"]
        accepted_targets = sum(value >= threshold for _sample, value in target)
        false_accepts = sum(value >= threshold for _sample, value in negatives)
        unknown_false = sum(value >= threshold for _sample, value in unknowns)
        cells = []
        cell_pass = True
        for lane, condition in plan.criteria.required_test_cells:
            values = [value for sample, value in target if sample.lane == lane and sample.condition == condition]
            negative_values = [value for sample, value in negatives if sample.lane == lane and sample.condition == condition]
            unknown_values = [value for sample, value in unknowns if sample.lane == lane and sample.condition == condition]
            true_accepts = sum(value >= threshold for value in values)
            rate = true_accepts / len(values)
            cell_pass = cell_pass and rate >= plan.criteria.minimum_cell_target_accept_rate
            cells.append({
                "lane": lane,
                "condition": condition,
                "target_count": len(values),
                "true_accepts": true_accepts,
                "false_rejects": len(values) - true_accepts,
                "target_accept_rate": rate,
                "negative_count": len(negative_values),
                "false_accepts": sum(value >= threshold for value in negative_values),
                "unknown_count": len(unknown_values),
                "unknown_false_candidates": sum(value >= threshold for value in unknown_values),
            })
        overall = accepted_targets / len(target)
        passed = (
            false_accepts <= plan.criteria.maximum_false_accepts
            and unknown_false <= plan.criteria.maximum_unknown_false_candidates
            and overall >= plan.criteria.minimum_overall_target_accept_rate
            and cell_pass
        )
        reports.append({
            "provider_id": spec.provider_id,
            "model_asset_sha256": _asset_sha256(spec.model_asset),
            "threshold": threshold,
            "calibration_target_count": len(calibration_target),
            "calibration_non_target_count": len(calibration_non_target),
            "test_target_count": len(target),
            "target_accept_rate": overall,
            "false_accepts": false_accepts,
            "unknown_false_candidates": unknown_false,
            "cells": cells,
            "runtime_seconds": round(time.monotonic() - started, 6),
            "passed": passed,
        })
        vectors.clear()
    return {"schema_version": SCHEMA_VERSION, "report_type": "voiceprint_owner_bakeoff", "providers": reports, "passed_providers": [item["provider_id"] for item in reports if item["passed"]]}


def write_report(plan: Plan, report: dict) -> Path:
    if set(report) != {"schema_version", "report_type", "providers", "passed_providers"}:
        raise BakeoffError("report schema is not allowlisted")
    serialized = json.dumps(report, sort_keys=True)
    if any(token in serialized.casefold() for token in ("embedding", "audio_path", plan.target_id.casefold(), str(plan.root).casefold())):
        raise BakeoffError("report contains forbidden material")
    path = plan.root / REPORT_NAME
    if path.is_symlink():
        raise BakeoffError("report path is unsafe")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return path


def cleanup_expired_pending_samples(
    samples_manifest: Path,
    *,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Delete only pending owner-bakeoff sample files after evaluation expiry.

    The quality report contains no vectors or audio paths, so the manifest may
    remain as privacy-safe audit metadata after the derived audio is removed.
    This function is deliberately explicit and never touches target samples
    marked ``keep`` or any source outside the manifest's private sample root.
    """
    payload = _load_json(samples_manifest)
    expiry = _parse_time(payload.get("evaluation_expires_at"), "evaluation_expires_at")
    current = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    if current < expiry:
        return ()
    sample_root = _private_root(Path(_text(payload.get("sample_root"), "sample_root")), label="sample root")
    deleted: list[str] = []
    for raw in payload.get("samples", []):
        if not isinstance(raw, dict) or raw.get("retention_instruction") != "pending":
            continue
        sample_id = _opaque_id(raw.get("sample_id"), "sample_id")
        audio = _inside(Path(_text(raw.get("audio_path"), "audio_path")), sample_root, label="sample audio")
        if audio.is_symlink():
            raise BakeoffError("pending sample audio must not be a symlink")
        if audio.exists():
            if not audio.is_file():
                raise BakeoffError("pending sample audio must be a regular file")
            audio.unlink()
            deleted.append(sample_id)
    marker = sample_root.parent / SAMPLE_CLEANUP_MARKER
    marker.write_text(
        json.dumps({"schema_version": 1, "deleted_sample_ids": sorted(deleted)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(marker, 0o600)
    return tuple(sorted(deleted))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or execute isolated owner voiceprint Gate 4 bake-off")
    parser.add_argument("--isolated-root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--providers", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cleanup-expired", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.cleanup_expired:
            deleted = cleanup_expired_pending_samples(args.samples)
            print(f"VOICEPRINT_BAKEOFF_CLEANUP_DELETED={len(deleted)}")
            return 0
        plan = load_plan(args.isolated_root, args.samples, args.criteria, args.providers)
        gaps = readiness(plan)
        if gaps:
            print("VOICEPRINT_BAKEOFF_READY=false")
            for gap in gaps:
                print(f"MISSING={gap}")
            return 2
        print("VOICEPRINT_BAKEOFF_READY=true")
        if not args.execute:
            print("VOICEPRINT_BAKEOFF_EXECUTION=disabled")
            return 0
        path = write_report(plan, evaluate(plan))
        print(f"VOICEPRINT_BAKEOFF_REPORT={path.name}")
        return 0
    except BakeoffError as exc:
        print(f"VOICEPRINT_BAKEOFF_REJECTED={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
