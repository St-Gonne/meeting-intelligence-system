#!/usr/bin/env python3
"""Normalize one guarded laptop capture into a canonical local audio source.

This module is intentionally not wired into MeetingIntel discovery or
production processing. It is the isolated normalization boundary between a
complete guarded capture manifest and a future explicitly admitted source.
"""

from __future__ import annotations
from mi_paths import public_path

import meetingintel_observe as observe

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from laptop_capture_contract import audit_capture_session, load_manifest


CAPTURE_MANIFEST_FILENAME = "capture_manifest.json"
CANONICAL_AUDIO_FILENAME = "audio.m4a"
METADATA_FILENAME = "meeting_source.json"
LEDGER_FILENAME = "ingest_ledger.json"
SOURCE_KIND = "laptop_capture"
SOURCE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
NORMALIZATION_POLICY = "guarded_obs_manifest_v1"
TERMINAL_SOURCE_LOSS_RECOVERY_POLICY = (
    "guarded_obs_terminal_source_loss_operator_recovery_v1"
)
PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY = (
    "guarded_obs_terminal_source_loss_prefix_trim_recovery_v1"
)
IDENTITY_NAMESPACE = "meetingintel_laptop_capture_v1"
DEFAULT_CAPTURE_ROOT = public_path("project/ingest/captures")
DEFAULT_INGEST_ROOT = Path(__file__).resolve().parent / "ingest" / "laptop"
IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class NormalizeConfig:
    session_root: Path
    capture_root: Path = DEFAULT_CAPTURE_ROOT
    ingest_root: Path = DEFAULT_INGEST_ROOT
    dry_run: bool = False
    allow_terminal_source_loss_recovery: bool = False
    operator_attested_meeting_ended_before_source_loss: bool = False
    keep_finalized_segment_count: int | None = None


@dataclass(frozen=True)
class NormalizeResult:
    status: str
    source_identity: str
    target_root: Path
    segment_count: int
    duration_seconds: float | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_executable(name: str) -> str:
    preferred = Path.home() / "homebrew" / "bin" / name
    if preferred.is_file():
        return str(preferred)
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is not available")
    return executable


def _reject_symlink_chain(root: Path, candidate: Path) -> None:
    lexical_root = root.absolute()
    lexical_candidate = candidate.absolute()
    try:
        lexical_relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("capture path is outside the configured capture root") from exc

    current = lexical_root
    if current.is_symlink():
        raise ValueError("capture root must not be a symlink")
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("capture path must not traverse symlinks")

    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("capture path is outside the configured capture root") from exc


def _validated_roots(config: NormalizeConfig) -> tuple[Path, Path, Path]:
    capture_root = config.capture_root.expanduser()
    session_root = config.session_root.expanduser()
    if not capture_root.is_dir():
        raise FileNotFoundError("configured capture root does not exist")
    if not session_root.is_dir():
        raise FileNotFoundError("capture session does not exist")
    _reject_symlink_chain(capture_root, session_root)

    resolved_capture = capture_root.resolve(strict=True)
    resolved_session = session_root.resolve(strict=True)
    if resolved_session.parent != resolved_capture:
        raise ValueError("capture session must be a direct child of the capture root")

    manifest_path = resolved_session / CAPTURE_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("capture manifest is missing or unsafe")
    return resolved_capture, resolved_session, manifest_path


def build_source_identity(
    manifest: Mapping[str, object],
    *,
    retained_segment_count: int | None = None,
) -> str:
    segments = manifest.get("segments")
    if not isinstance(segments, list):
        raise ValueError("capture manifest segments are invalid")
    identities: list[dict[str, object]] = []
    for item in sorted(
        (item for item in segments if isinstance(item, Mapping)),
        key=lambda item: item.get("index", -1),
    ):
        if item.get("state") != "finalized":
            continue
        index = item.get("index")
        fingerprint = item.get("sha256")
        if not isinstance(index, int) or not isinstance(fingerprint, str):
            raise ValueError("capture segment identity is invalid")
        identities.append({"index": index, "sha256": fingerprint})
    if retained_segment_count is not None:
        if type(retained_segment_count) is not int or not (
            1 <= retained_segment_count <= len(identities)
        ):
            raise ValueError("retained segment count is invalid")
        identities = identities[:retained_segment_count]
    payload = {
        "namespace": IDENTITY_NAMESPACE,
        "capture_id": manifest.get("capture_id"),
        "started_at": manifest.get("started_at"),
        "segments": identities,
    }
    if retained_segment_count is not None:
        payload["retained_prefix_segment_count"] = retained_segment_count
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _started_at_ist(manifest: Mapping[str, object]) -> datetime:
    value = manifest.get("started_at")
    if not isinstance(value, str):
        raise ValueError("capture start timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("capture start timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("capture start timestamp must include a timezone")
    return parsed.astimezone(IST)


def _parse_aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def terminal_source_loss_recovery_eligible(
    manifest: Mapping[str, object],
    audit: object,
) -> bool:
    """Allow only a terminal, operator-attested source-loss salvage shape."""

    if manifest.get("status") != "interrupted" or manifest.get("stop_reason") != "source_lost":
        return False
    findings = getattr(audit, "findings", ())
    error_codes = {
        finding.code
        for finding in findings
        if getattr(finding, "severity", None) == "error"
    }
    if error_codes != {"source_not_recovered"}:
        return False
    finalized = getattr(audit, "finalized_segments", ())
    raw_segments = manifest.get("segments")
    if (
        not isinstance(raw_segments, list)
        or not raw_segments
        or len(finalized) != len(raw_segments)
        or any(
            not isinstance(item, Mapping) or item.get("state") != "finalized"
            for item in raw_segments
        )
    ):
        return False
    events = manifest.get("source_events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or not isinstance(events[0], Mapping)
        or events[0].get("type") != "source_lost"
    ):
        return False
    lost_at = _parse_aware_timestamp(events[0].get("at"))
    ended_at = _parse_aware_timestamp(manifest.get("ended_at"))
    if lost_at is None or ended_at is None:
        return False
    terminal_gap = (ended_at - lost_at).total_seconds()
    return 0 <= terminal_gap <= 5.0


def _target_name(started_at: datetime, source_identity: str) -> str:
    return f"laptop_{started_at:%Y-%m-%d_%H-%M-%S}_{source_identity[:16]}"


def _probe_audio(path: Path) -> tuple[dict[str, Any], float]:
    result = subprocess.run(
        [
            _find_executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,profile,sample_fmt,sample_rate,channels,channel_layout",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe could not inspect canonical audio")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        stream = streams[0]
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe returned invalid audio metadata") from exc
    if not isinstance(stream, dict) or stream.get("codec_name") != "aac":
        raise RuntimeError("capture audio must be AAC for lossless remux")
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("canonical audio duration is invalid")
    signature = {
        key: stream.get(key)
        for key in (
            "codec_name",
            "profile",
            "sample_fmt",
            "sample_rate",
            "channels",
            "channel_layout",
        )
    }
    return signature, duration


def _remux_audio(segments: Sequence[Path], output_path: Path) -> None:
    signatures = [_probe_audio(path)[0] for path in segments]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("capture segment audio streams are incompatible")

    ffmpeg = _find_executable("ffmpeg")
    if len(segments) == 1:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(segments[0]),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(output_path),
        ]
        concat_list = None
    else:
        concat_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=".concat-",
            suffix=".txt",
            delete=False,
        )
        concat_list = Path(concat_file.name)
        try:
            for segment in segments:
                escaped = str(segment).replace("'", "'\\''")
                concat_file.write(f"file '{escaped}'\n")
        finally:
            concat_file.close()
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(output_path),
        ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg could not create canonical audio")
    finally:
        if concat_list is not None:
            concat_list.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temporary, path)


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": LEDGER_SCHEMA_VERSION, "records": {}}
    if path.is_symlink() or not path.is_file():
        raise ValueError("laptop ingest ledger is unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != LEDGER_SCHEMA_VERSION
        or not isinstance(payload.get("records"), dict)
    ):
        raise ValueError("laptop ingest ledger is invalid")
    return payload


def _validate_existing_target(
    target_root: Path,
    source_identity: str,
    manifest_sha256: str,
) -> float:
    if target_root.is_symlink() or not target_root.is_dir():
        raise ValueError("existing canonical target is unsafe")
    metadata_path = target_root / METADATA_FILENAME
    audio_path = target_root / CANONICAL_AUDIO_FILENAME
    if (
        metadata_path.is_symlink()
        or audio_path.is_symlink()
        or not metadata_path.is_file()
        or not audio_path.is_file()
    ):
        raise ValueError("existing canonical target is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("source_identity") != source_identity
        or metadata.get("capture_manifest_sha256") != manifest_sha256
        or metadata.get("canonical_audio_sha256") != sha256_file(audio_path)
    ):
        raise ValueError("existing canonical target does not match capture evidence")
    _, duration = _probe_audio(audio_path)
    return duration


def normalize_capture(config: NormalizeConfig) -> NormalizeResult:
    _, session_root, manifest_path = _validated_roots(config)
    manifest = load_manifest(manifest_path)
    audit = audit_capture_session(session_root, manifest)
    normal_admission = (
        audit.accepted
        and audit.merge_ready
        and manifest.get("status") == "complete"
    )
    recovery_admission = (
        config.allow_terminal_source_loss_recovery
        and config.operator_attested_meeting_ended_before_source_loss
        and terminal_source_loss_recovery_eligible(manifest, audit)
    )
    if not normal_admission and not recovery_admission:
        codes = ",".join(sorted(audit.codes())) or "capture_not_merge_ready"
        raise ValueError(f"capture is not eligible for normalization: {codes}")
    raw_segments = sorted(
        (
            item
            for item in manifest["segments"]  # type: ignore[index]
            if isinstance(item, Mapping) and item.get("state") == "finalized"
        ),
        key=lambda item: item["index"],
    )
    keep_count = config.keep_finalized_segment_count
    if keep_count is not None:
        if not recovery_admission:
            raise ValueError("prefix trimming is allowed only for approved recovery")
        if type(keep_count) is not int or not (1 <= keep_count <= len(raw_segments)):
            raise ValueError("keep segment count is outside the finalized capture")
    retained_count = keep_count if keep_count is not None else len(raw_segments)
    prefix_trimmed = retained_count < len(raw_segments)
    normalization_policy = (
        PREFIX_TRIMMED_TERMINAL_SOURCE_LOSS_RECOVERY_POLICY
        if prefix_trimmed
        else TERMINAL_SOURCE_LOSS_RECOVERY_POLICY
        if recovery_admission
        else NORMALIZATION_POLICY
    )

    for segment in audit.finalized_segments:
        _reject_symlink_chain(session_root, segment)

    selected_segments = audit.finalized_segments[:retained_count]
    selected_raw_segments = raw_segments[:retained_count]
    identity_retained_count = retained_count if prefix_trimmed else None
    source_identity = build_source_identity(
        manifest,
        retained_segment_count=identity_retained_count,
    )
    started_at = _started_at_ist(manifest)
    ingest_root = config.ingest_root.expanduser()
    if ingest_root.exists() and (ingest_root.is_symlink() or not ingest_root.is_dir()):
        raise ValueError("laptop ingest root is unsafe")
    target_root = ingest_root / _target_name(started_at, source_identity)
    manifest_sha256 = sha256_file(manifest_path)

    if target_root.exists() or target_root.is_symlink():
        duration = _validate_existing_target(
            target_root,
            source_identity,
            manifest_sha256,
        )
        return NormalizeResult(
            status="skipped",
            source_identity=source_identity,
            target_root=target_root,
            segment_count=len(selected_segments),
            duration_seconds=duration,
        )

    if config.dry_run:
        return NormalizeResult(
            status="planned",
            source_identity=source_identity,
            target_root=target_root,
            segment_count=len(selected_segments),
        )

    ingest_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(ingest_root, stat.S_IRWXU)

    ledger_path = ingest_root / LEDGER_FILENAME
    ledger = _load_ledger(ledger_path)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target_root.name}.", dir=ingest_root)
    )
    os.chmod(temporary_root, stat.S_IRWXU)
    try:
        audio_path = temporary_root / CANONICAL_AUDIO_FILENAME
        _remux_audio(selected_segments, audio_path)
        os.chmod(audio_path, stat.S_IRUSR | stat.S_IWUSR)
        _, duration = _probe_audio(audio_path)

        declared_total = sum(
            float(item["duration_seconds"])
            for item in selected_raw_segments
        )
        tolerance = max(2.0, len(selected_segments) * 1.0)
        if abs(duration - declared_total) > tolerance:
            raise RuntimeError("canonical duration does not match capture evidence")

        canonical_sha256 = sha256_file(audio_path)
        ingested_at = datetime.now(IST).isoformat()
        metadata = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_kind": SOURCE_KIND,
            "source_identity": source_identity,
            "display_name": session_root.name,
            "capture_id": manifest["capture_id"],
            "capture_manifest_path": str(manifest_path),
            "capture_manifest_sha256": manifest_sha256,
            "created_at": started_at.isoformat(),
            "timezone": "Asia/Kolkata",
            "duration_seconds": duration,
            "ingested_at": ingested_at,
            "status": "normalized",
            "canonical_audio_path": str(target_root / CANONICAL_AUDIO_FILENAME),
            "canonical_audio_sha256": canonical_sha256,
            "audio_handling": (
                "extract_copy"
                if len(selected_segments) == 1
                else "concat_extract_copy"
            ),
            "normalization_policy": normalization_policy,
            "transcription_status": "not_started",
            "stop_reason": manifest.get("stop_reason"),
            "capture_status": manifest.get("status"),
            "operator_recovery": (
                {
                    "attestation": "meeting_ended_before_terminal_source_loss",
                    "attested_at": ingested_at,
                    "source_loss_preserved": True,
                    "retained_prefix_segment_count": len(selected_segments),
                    "excluded_tail_segment_count": (
                        len(audit.finalized_segments) - len(selected_segments)
                    ),
                }
                if recovery_admission
                else None
            ),
            "source_segments": [
                {
                    "index": item["index"],
                    "source_path": str(audit.finalized_segments[position]),
                    "fingerprint_sha256": item["sha256"],
                    "duration_seconds": item["duration_seconds"],
                }
                for position, item in enumerate(selected_raw_segments)
            ],
        }
        _atomic_write_json(temporary_root / METADATA_FILENAME, metadata)
        os.replace(temporary_root, target_root)

        records = ledger["records"]
        records[source_identity] = {
            "status": "normalized",
            "target_root": str(target_root),
            "canonical_audio_sha256": canonical_sha256,
            "capture_manifest_sha256": manifest_sha256,
            "ingested_at": ingested_at,
        }
        _atomic_write_json(ledger_path, ledger)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    observe.emit("stage.succeeded", source_id=source_identity, source_kind="laptop_capture",
                 stage="normalization", outcome="success", details={"segment_count": len(selected_segments)})
    return NormalizeResult(
        status="normalized",
        source_identity=source_identity,
        target_root=target_root,
        segment_count=len(selected_segments),
        duration_seconds=duration,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize one complete guarded laptop capture"
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--ingest-root", type=Path, default=DEFAULT_INGEST_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = normalize_capture(
            NormalizeConfig(
                session_root=args.session_root,
                capture_root=args.capture_root,
                ingest_root=args.ingest_root,
                dry_run=args.dry_run,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"LAPTOP_CAPTURE_NORMALIZE failed: {exc}")
        return 1
    print(
        "LAPTOP_CAPTURE_NORMALIZE "
        f"status={result.status} segments={result.segment_count} "
        f"source={result.source_identity[:16]} target={result.target_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
