#!/usr/bin/env python3
"""Normalize ASR Voice Recorder .m4a files into durable audio-first sources."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo


DEFAULT_SOURCE_ROOT = Path(
    str(public_path("phone-source"))
)
DEFAULT_INGEST_ROOT = Path(str(public_path('project/ingest/phone')))
LEDGER_FILENAME = "ingest_ledger.json"
METADATA_FILENAME = "meeting_source.json"
CANONICAL_AUDIO_FILENAME = "audio.m4a"
SOURCE_KIND = "phone_recording"
SOURCE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
OPERATOR_RESOLUTION_SCHEMA_VERSION = 2
OPERATOR_PLAN_SCHEMA_VERSION = 3
GROUPING_POLICY = "asr_900s_segment_group_v1"
GROUP_IDENTITY_NAMESPACE = "phone_recording_segment_group_v1"
FULL_CHUNK_MIN_SECONDS = 899.0
FULL_CHUNK_MAX_SECONDS = 901.0
MIN_CONTINUITY_GAP_SECONDS = -1.0
MAX_CONTINUITY_GAP_SECONDS = 5.0
CONCAT_DURATION_TOLERANCE_SECONDS = 1.0
IST = ZoneInfo("Asia/Kolkata")
TIMESTAMP_PATTERN = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$")


@dataclass(frozen=True)
class IngestConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    ingest_root: Path = DEFAULT_INGEST_ROOT
    dry_run: bool = False
    operator_resolution_path: Path | None = None
    selected_source_paths: tuple[Path, ...] | None = None


@dataclass
class IngestSummary:
    discovered: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    singleton: int = 0
    grouped: int = 0
    ambiguous: int = 0
    discarded: int = 0

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


@dataclass(frozen=True)
class SourceSegment:
    source_path: Path
    created_at: datetime
    duration_seconds: float
    fingerprint_sha256: str

    @property
    def is_full_chunk(self) -> bool:
        return FULL_CHUNK_MIN_SECONDS <= self.duration_seconds <= FULL_CHUNK_MAX_SECONDS


@dataclass(frozen=True)
class OperatorResolution:
    schema_version: int
    resolution: Literal["separate", "join", "discard", "plan"]
    source_fingerprints_sha256: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()
    discard_source_fingerprints_sha256: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceCandidate:
    kind: Literal["SINGLETON", "GROUP", "AMBIGUOUS", "FAILED", "DISCARDED"]
    segments: tuple[SourceSegment, ...]
    reason: str = ""
    operator_resolution: OperatorResolution | None = None

    @property
    def created_at(self) -> datetime:
        return self.segments[0].created_at

    @property
    def identity(self) -> str:
        if self.kind in {"GROUP", "DISCARDED"} and len(self.segments) > 1:
            return group_fingerprint_sha256(self.segments)
        return self.segments[0].fingerprint_sha256


def discover_sources(source_root: Path) -> list[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source root is not a directory: {source_root}")
    return sorted(
        (path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() == ".m4a"),
        key=lambda path: str(path),
    )


def parse_source_timestamp(filename: str) -> datetime:
    match = TIMESTAMP_PATTERN.fullmatch(Path(filename).stem)
    if match is None:
        raise ValueError(f"filename does not match YYYY_MM_DD_HH_MM_SS.m4a: {filename}")
    try:
        return datetime(*(int(value) for value in match.groups()), tzinfo=IST)
    except ValueError as exc:
        raise ValueError(f"filename contains an invalid date/time: {filename}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ffprobe() -> str:
    preferred = Path(str(public_path('home/homebrew/bin/ffprobe')))
    if preferred.is_file():
        return str(preferred)
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is not available")
    return executable


def find_ffmpeg() -> str:
    preferred = Path(str(public_path('home/homebrew/bin/ffmpeg')))
    if preferred.is_file():
        return str(preferred)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is not available")
    return executable


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [find_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned an invalid duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned an invalid duration for {path}: {duration}")
    return duration


def probe_stream_signature(path: Path) -> tuple[Any, ...]:
    result = subprocess.run(
        [
            find_ffprobe(), "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,profile,sample_fmt,sample_rate,channels,channel_layout,time_base,extradata_size",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise RuntimeError(f"ffprobe did not report exactly one primary audio stream: {path}")
    stream = streams[0]
    fields = ("codec_name", "profile", "sample_fmt", "sample_rate", "channels", "channel_layout", "time_base", "extradata_size")
    signature = tuple(stream.get(field) for field in fields)
    if any(value is None for value in signature):
        raise RuntimeError(f"ffprobe returned an incomplete stream signature: {path}")
    return signature


def deterministic_folder_name(created_at: datetime, fingerprint: str) -> str:
    return f"phone_{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_{fingerprint[:16]}"


def deterministic_group_folder_name(created_at: datetime, group_fingerprint: str) -> str:
    return f"phone_group_{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_{group_fingerprint[:16]}"


def group_fingerprint_from_values(values: list[tuple[str, str]]) -> str:
    payload = {
        "namespace": GROUP_IDENTITY_NAMESPACE,
        "segments": [
            {"created_at": created_at, "fingerprint_sha256": fingerprint}
            for created_at, fingerprint in values
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def group_fingerprint_sha256(segments: tuple[SourceSegment, ...]) -> str:
    return group_fingerprint_from_values(
        [(segment.created_at.isoformat(), segment.fingerprint_sha256) for segment in segments]
    )


def continuity_gap_seconds(current: SourceSegment, following: SourceSegment) -> float:
    expected_end = current.created_at + timedelta(seconds=current.duration_seconds)
    return (following.created_at - expected_end).total_seconds()


def segments_are_continuous(current: SourceSegment, following: SourceSegment) -> bool:
    gap = continuity_gap_seconds(current, following)
    return MIN_CONTINUITY_GAP_SECONDS <= gap <= MAX_CONTINUITY_GAP_SECONDS


def _candidate_has_duplicate_evidence(segments: list[SourceSegment]) -> str | None:
    timestamps = [segment.created_at for segment in segments]
    if len(timestamps) != len(set(timestamps)):
        return "duplicate_timestamp"
    fingerprints = [segment.fingerprint_sha256 for segment in segments]
    if len(fingerprints) != len(set(fingerprints)):
        return "duplicate_segment_fingerprint"
    return None


def group_source_segments(segments: list[SourceSegment]) -> list[SourceCandidate]:
    ordered = sorted(segments, key=lambda item: (str(item.source_path.parent), item.created_at, str(item.source_path)))
    buckets: dict[tuple[str, object], list[SourceSegment]] = {}
    for segment in ordered:
        buckets.setdefault((str(segment.source_path.parent), segment.created_at.date()), []).append(segment)

    candidates: list[SourceCandidate] = []
    for bucket_key in sorted(buckets, key=lambda item: (item[0], str(item[1]))):
        bucket = sorted(buckets[bucket_key], key=lambda item: (item.created_at, str(item.source_path)))
        duplicate_timestamps = {
            item.created_at
            for item in bucket
            if sum(other.created_at == item.created_at for other in bucket) > 1
        }
        if duplicate_timestamps:
            duplicates = tuple(
                item for item in bucket if item.created_at in duplicate_timestamps
            )
            candidates.append(SourceCandidate("FAILED", duplicates, "duplicate_timestamp"))
            bucket = [item for item in bucket if item.created_at not in duplicate_timestamps]
        index = 0
        while index < len(bucket):
            current = bucket[index]
            if index + 1 < len(bucket) and current.is_full_chunk and segments_are_continuous(current, bucket[index + 1]):
                following = bucket[index + 1]
                if not following.is_full_chunk:
                    pair = [current, following]
                    reason = _candidate_has_duplicate_evidence(pair) or "one_full_chunk_plus_tail"
                    candidates.append(SourceCandidate("AMBIGUOUS", tuple(pair), reason))
                    index += 2
                    continue

                grouped = [current, following]
                index += 2
                while index < len(bucket) and grouped[-1].is_full_chunk and segments_are_continuous(grouped[-1], bucket[index]):
                    grouped.append(bucket[index])
                    index += 1
                    if not grouped[-1].is_full_chunk:
                        break
                duplicate_reason = _candidate_has_duplicate_evidence(grouped)
                if duplicate_reason:
                    candidates.append(SourceCandidate("FAILED", tuple(grouped), duplicate_reason))
                else:
                    candidates.append(SourceCandidate("GROUP", tuple(grouped)))
                continue
            candidates.append(SourceCandidate("SINGLETON", (current,)))
            index += 1
    return sorted(candidates, key=lambda item: (item.created_at, str(item.segments[0].source_path)))


def empty_ledger() -> dict[str, Any]:
    return {"schema_version": LEDGER_SCHEMA_VERSION, "records": {}}


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_ledger()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION or not isinstance(payload.get("records"), dict):
        raise ValueError(f"invalid phone ingest ledger: {path}")
    return payload


def load_operator_resolution(path: Path) -> OperatorResolution:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid operator resolution: {path}") from exc
    if type(payload) is not dict:
        raise ValueError("operator resolution must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version == OPERATOR_PLAN_SCHEMA_VERSION:
        if set(payload) != {
            "schema_version",
            "source_fingerprints_sha256",
            "groups",
            "discard_source_fingerprints_sha256",
        }:
            raise ValueError("operator plan fields are invalid")
        source_identities = payload.get("source_fingerprints_sha256")
        groups_payload = payload.get("groups")
        discarded = payload.get("discard_source_fingerprints_sha256")
        if type(source_identities) is not list or type(groups_payload) is not list or type(discarded) is not list:
            raise ValueError("operator plan collections are invalid")
        _validate_fingerprint_list(source_identities, "operator plan source fingerprints", minimum=1)
        _validate_fingerprint_list(discarded, "operator plan discarded fingerprints", minimum=0)
        groups: list[tuple[str, ...]] = []
        grouped: list[str] = []
        for group in groups_payload:
            if type(group) is not dict or set(group) != {"action", "source_fingerprints_sha256"}:
                raise ValueError("operator plan group fields are invalid")
            if group.get("action") != "normalize":
                raise ValueError("operator plan group action must be normalize")
            identities = group.get("source_fingerprints_sha256")
            if type(identities) is not list:
                raise ValueError("operator plan group fingerprints must be a list")
            _validate_fingerprint_list(identities, "operator plan group fingerprints", minimum=1)
            groups.append(tuple(identities))
            grouped.extend(identities)
        all_selected = grouped + list(discarded)
        if len(set(all_selected)) != len(all_selected) or set(all_selected) != set(source_identities):
            raise ValueError("operator plan must cover every reviewed fingerprint exactly once")
        return OperatorResolution(
            OPERATOR_PLAN_SCHEMA_VERSION,
            "plan",
            tuple(source_identities),
            tuple(groups),
            tuple(discarded),
        )
    if schema_version not in {1, OPERATOR_RESOLUTION_SCHEMA_VERSION}:
        raise ValueError("unsupported operator resolution schema")
    if set(payload) != {
        "schema_version",
        "resolution",
        "source_fingerprints_sha256",
    }:
        raise ValueError("operator resolution fields are invalid")
    resolution = payload.get("resolution")
    if resolution not in {"separate", "join", "discard"}:
        raise ValueError("unsupported operator resolution")
    if schema_version == 1 and resolution != "separate":
        raise ValueError("operator resolution schema v1 supports only resolution=separate")
    identities = payload.get("source_fingerprints_sha256")
    if type(identities) is not list:
        raise ValueError("operator resolution source fingerprints must be a list")
    expected_lengths = {"separate": 2, "join": None, "discard": None}
    expected_length = expected_lengths[resolution]
    if expected_length is not None and len(identities) != expected_length:
        raise ValueError("separate operator resolution must identify exactly two source fingerprints")
    minimum_length = 2 if resolution == "join" else 1
    if expected_length is None and len(identities) < minimum_length:
        raise ValueError(
            f"{resolution} operator resolution must identify at least {minimum_length} source fingerprint"
            f"{'s' if minimum_length != 1 else ''}"
        )
    _validate_fingerprint_list(identities, "operator resolution source fingerprints", minimum=minimum_length)
    return OperatorResolution(schema_version, resolution, tuple(identities))


def _validate_fingerprint_list(values: list[Any], label: str, *, minimum: int) -> None:
    if len(values) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} fingerprint")
    if any(type(identity) is not str or re.fullmatch(r"[0-9a-f]{64}", identity) is None for identity in values):
        raise ValueError(f"{label} must be lowercase SHA-256")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _operator_resolution_payload(resolution: OperatorResolution) -> dict[str, Any]:
    if resolution.schema_version == OPERATOR_PLAN_SCHEMA_VERSION:
        return {
            "schema_version": OPERATOR_PLAN_SCHEMA_VERSION,
            "source_fingerprints_sha256": list(resolution.source_fingerprints_sha256),
            "groups": [
                {"action": "normalize", "source_fingerprints_sha256": list(group)}
                for group in resolution.groups
            ],
            "discard_source_fingerprints_sha256": list(
                resolution.discard_source_fingerprints_sha256
            ),
        }
    return {
        "schema_version": resolution.schema_version,
        "resolution": resolution.resolution,
        "source_fingerprints_sha256": list(resolution.source_fingerprints_sha256),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def now_ist() -> datetime:
    return datetime.now(IST)


def _record_is_complete(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "normalized"
        and Path(record.get("metadata_path", "")).is_file()
        and Path(record.get("canonical_audio_path", "")).is_file()
    )


def _segment_metadata(segment: SourceSegment) -> dict[str, Any]:
    return {
        "source_path": str(segment.source_path),
        "source_filename": segment.source_path.name,
        "fingerprint_sha256": segment.fingerprint_sha256,
        "created_at": segment.created_at.isoformat(),
        "duration_seconds": segment.duration_seconds,
    }


def _build_metadata(
    source: Path, fingerprint: str, created_at: datetime, duration: float,
    ingested_at: datetime, canonical_audio: Path,
    candidate: SourceCandidate | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "source_path": str(source),
        "source_filename": source.name,
        "source_fingerprint_sha256": fingerprint,
        "created_at": created_at.isoformat(),
        "timezone": "Asia/Kolkata",
        "duration_seconds": duration,
        "ingested_at": ingested_at.isoformat(),
        "status": "normalized",
        "canonical_audio_path": str(canonical_audio),
        "audio_handling": "concat_copy" if candidate and candidate.kind == "GROUP" else "copy",
        "transcription_status": "not_started",
    }
    if candidate and candidate.kind == "GROUP":
        group_fingerprint = candidate.identity
        payload["source_group"] = {
            "policy": GROUPING_POLICY,
            "group_fingerprint_sha256": group_fingerprint,
            "segment_count": len(candidate.segments),
        }
        payload["source_segments"] = [_segment_metadata(segment) for segment in candidate.segments]
    if candidate and candidate.operator_resolution is not None:
        payload["operator_resolution"] = _operator_resolution_payload(
            candidate.operator_resolution
        )
    return payload


def _normalize_source(source: Path, target: Path, metadata: dict[str, Any]) -> None:
    if target.exists():
        raise FileExistsError(f"target already exists without a complete ledger record: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        copied_audio = temporary / CANONICAL_AUDIO_FILENAME
        shutil.copy2(source, copied_audio)
        if sha256_file(copied_audio) != metadata["source_fingerprint_sha256"]:
            raise RuntimeError(f"source changed while being copied: {source}")
        temporary_metadata = dict(metadata)
        temporary_metadata["canonical_audio_path"] = str(target / CANONICAL_AUDIO_FILENAME)
        atomic_write_json(temporary / METADATA_FILENAME, temporary_metadata)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _concat_list_line(path: Path) -> str:
    raw_path = str(path)
    if any(character in raw_path for character in ("\x00", "\n", "\r")):
        raise ValueError("source path contains a concat-list control character")
    escaped = raw_path.replace("'", "'\\''")
    return f"file '{escaped}'\n"


def run_concat_copy(concat_list: Path, output_audio: Path) -> None:
    subprocess.run(
        [find_ffmpeg(), "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-map", "0:a:0", "-c", "copy", "-y", str(output_audio)],
        check=True, capture_output=True, text=True,
    )


def _normalize_group(
    candidate: SourceCandidate, target: Path, ingested_at: datetime,
    *, duration_probe: Callable[[Path], float], stream_probe: Callable[[Path], tuple[Any, ...]],
    concat_runner: Callable[[Path, Path], None],
) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"target already exists without verified grouped recovery: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        signatures: list[tuple[Any, ...]] = []
        for segment in candidate.segments:
            source = segment.source_path
            if source.is_symlink() or not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
                raise RuntimeError(f"source segment is not an ordinary regular file: {source}")
            if sha256_file(source) != segment.fingerprint_sha256:
                raise RuntimeError(f"source segment changed before concat: {source}")
            signatures.append(stream_probe(source))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError("source segment streams are incompatible for concat-copy")

        concat_list = temporary / "segments.concat.txt"
        concat_list.write_text("".join(_concat_list_line(segment.source_path) for segment in candidate.segments), encoding="utf-8")
        canonical_temporary = temporary / CANONICAL_AUDIO_FILENAME
        concat_runner(concat_list, canonical_temporary)
        concat_list.unlink(missing_ok=True)
        if canonical_temporary.is_symlink() or not canonical_temporary.is_file() or not stat.S_ISREG(canonical_temporary.stat().st_mode):
            raise RuntimeError("concat did not produce a regular canonical audio file")
        if canonical_temporary.stat().st_size <= 0:
            raise RuntimeError("concat produced an empty canonical audio file")
        canonical_duration = duration_probe(canonical_temporary)
        if not math.isfinite(canonical_duration) or canonical_duration <= 0:
            raise RuntimeError(f"concat produced an invalid canonical duration: {canonical_duration}")
        expected_duration = sum(segment.duration_seconds for segment in candidate.segments)
        if abs(canonical_duration - expected_duration) > CONCAT_DURATION_TOLERANCE_SECONDS:
            raise RuntimeError(
                f"canonical duration mismatch: expected {expected_duration}, got {canonical_duration}"
            )
        canonical_fingerprint = sha256_file(canonical_temporary)
        final_audio = target / CANONICAL_AUDIO_FILENAME
        metadata = _build_metadata(
            candidate.segments[0].source_path, canonical_fingerprint, candidate.created_at,
            canonical_duration, ingested_at, final_audio, candidate,
        )
        atomic_write_json(temporary / METADATA_FILENAME, metadata)
        os.replace(temporary, target)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _metadata_matches_group(metadata: dict[str, Any], candidate: SourceCandidate, target: Path) -> bool:
    expected_segments = [_segment_metadata(segment) for segment in candidate.segments]
    group = metadata.get("source_group")
    canonical_audio = target / CANONICAL_AUDIO_FILENAME
    basic_match = (
        metadata.get("status") == "normalized"
        and metadata.get("audio_handling") == "concat_copy"
        and metadata.get("canonical_audio_path") == str(canonical_audio)
        and canonical_audio.is_file()
        and isinstance(group, dict)
        and group.get("policy") == GROUPING_POLICY
        and group.get("group_fingerprint_sha256") == candidate.identity
        and group.get("segment_count") == len(candidate.segments)
        and metadata.get("source_segments") == expected_segments
        and sha256_file(canonical_audio) == metadata.get("source_fingerprint_sha256")
    )
    if not basic_match:
        return False
    # Local import avoids making the read-only adapter part of discovery while
    # requiring interrupted-write recovery to satisfy its complete contract.
    from audio_first_meeting_source import load_audio_first_source

    validated = load_audio_first_source(target)
    return (
        validated.group_fingerprint == candidate.identity
        and validated.grouping_policy == GROUPING_POLICY
    )


def _group_ledger_record_matches(
    record: dict[str, Any], candidate: SourceCandidate, target: Path
) -> bool:
    metadata_path = target / METADATA_FILENAME
    canonical_audio = target / CANONICAL_AUDIO_FILENAME
    if (
        record.get("source_group_policy") != GROUPING_POLICY
        or record.get("group_fingerprint_sha256") != candidate.identity
        or record.get("segment_count") != len(candidate.segments)
        or record.get("source_segment_paths")
        != [str(segment.source_path) for segment in candidate.segments]
        or record.get("target_folder") != str(target)
        or record.get("metadata_path") != str(metadata_path)
        or record.get("canonical_audio_path") != str(canonical_audio)
        or not metadata_path.is_file()
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    try:
        return type(metadata) is dict and _metadata_matches_group(metadata, candidate, target)
    except (OSError, ValueError):
        return False


def _print_candidate(candidate: SourceCandidate, target: Path | None, *, dry_run: bool) -> None:
    prefix = candidate.kind
    details = [
        f"policy={GROUPING_POLICY if candidate.kind == 'GROUP' else 'none'}",
        f"segments={','.join(segment.source_path.name for segment in candidate.segments)}",
        f"durations={','.join(str(segment.duration_seconds) for segment in candidate.segments)}",
        f"classes={','.join('full' if segment.is_full_chunk else 'tail' for segment in candidate.segments)}",
    ]
    if len(candidate.segments) > 1:
        details.append(
            "gaps=" + ",".join(
                f"{continuity_gap_seconds(left, right):.6f}"
                for left, right in zip(candidate.segments, candidate.segments[1:])
            )
        )
    if candidate.kind == "GROUP":
        details.extend([f"segment_count={len(candidate.segments)}", f"group_fingerprint={candidate.identity}"])
    if target is not None:
        details.append(f"target={target}")
    if candidate.reason:
        details.append(f"reason={candidate.reason}")
    details.append(f"dry_run={'true' if dry_run else 'false'}")
    print(f"{prefix} " + " ".join(details))


def _apply_operator_resolution(
    candidates: list[SourceCandidate],
    valid_segments: list[SourceSegment],
    resolution: OperatorResolution,
) -> list[SourceCandidate]:
    if resolution.schema_version == OPERATOR_PLAN_SCHEMA_VERSION:
        ordered_segments = sorted(
            valid_segments,
            key=lambda segment: (segment.created_at, str(segment.source_path)),
        )
        ordered_identities = tuple(
            segment.fingerprint_sha256 for segment in ordered_segments
        )
        if resolution.source_fingerprints_sha256 != ordered_identities:
            raise ValueError(
                "operator plan source fingerprints must match the complete reviewed set in chronological order"
            )
        if any(candidate.kind == "FAILED" for candidate in candidates):
            raise ValueError("operator plan cannot override a failed source candidate")
        by_identity = {segment.fingerprint_sha256: segment for segment in ordered_segments}
        used: set[str] = set()
        planned: list[SourceCandidate] = []
        for group in resolution.groups:
            if any(identity not in by_identity for identity in group):
                raise ValueError("operator plan references an unlisted source fingerprint")
            if any(identity in used for identity in group):
                raise ValueError("operator plan repeats a source fingerprint")
            segments = tuple(by_identity[identity] for identity in group)
            if tuple(segment.fingerprint_sha256 for segment in segments) != tuple(
                segment.fingerprint_sha256
                for segment in sorted(segments, key=lambda item: (item.created_at, str(item.source_path)))
            ):
                raise ValueError("operator plan group fingerprints must be chronological")
            if len({segment.source_path.parent for segment in segments}) != 1 or len(
                {segment.created_at.date() for segment in segments}
            ) != 1:
                raise ValueError("operator plan group cannot cross a source-root or date boundary")
            used.update(group)
            planned.append(
                SourceCandidate(
                    "GROUP" if len(segments) > 1 else "SINGLETON",
                    segments,
                    operator_resolution=resolution,
                )
            )
        for identity in resolution.discard_source_fingerprints_sha256:
            if identity not in by_identity or identity in used:
                raise ValueError("operator plan discard references an unlisted or repeated source fingerprint")
            used.add(identity)
            planned.append(
                SourceCandidate(
                    "DISCARDED", (by_identity[identity],), operator_resolution=resolution
                )
            )
        if used != set(ordered_identities):
            raise ValueError("operator plan must cover every reviewed source exactly once")
        return sorted(
            planned,
            key=lambda candidate: (candidate.created_at, str(candidate.segments[0].source_path)),
        )

    identities = set(resolution.source_fingerprints_sha256)
    discovered_identities = {segment.fingerprint_sha256 for segment in valid_segments}
    if not identities.issubset(discovered_identities):
        raise ValueError("operator resolution references an unlisted source fingerprint")
    if resolution.resolution == "separate":
        matches = [
            candidate
            for candidate in candidates
            if candidate.kind == "AMBIGUOUS"
            and {segment.fingerprint_sha256 for segment in candidate.segments} == identities
        ]
        if len(matches) != 1:
            raise ValueError("operator resolution is stale or does not match exactly one ambiguous pair")
        match = matches[0]
        resolved: list[SourceCandidate] = []
        for candidate in candidates:
            if candidate is not match:
                resolved.append(candidate)
                continue
            for segment in candidate.segments:
                resolved.append(
                    SourceCandidate(
                        "SINGLETON", (segment,), operator_resolution=resolution
                    )
                )
        return resolved

    selected: list[SourceCandidate] = []
    for candidate in candidates:
        candidate_identities = {
            segment.fingerprint_sha256 for segment in candidate.segments
        }
        overlap = candidate_identities & identities
        if not overlap:
            continue
        if overlap != candidate_identities:
            raise ValueError(
                "operator resolution must select every segment of each current candidate"
            )
        if candidate.kind == "FAILED":
            raise ValueError("operator resolution cannot select a failed candidate")
        selected.append(candidate)
    if {segment.fingerprint_sha256 for candidate in selected for segment in candidate.segments} != identities:
        raise ValueError("operator resolution is stale or does not match whole current candidates")

    if resolution.resolution == "discard":
        selected_ids = {id(candidate) for candidate in selected}
        return [
            SourceCandidate("DISCARDED", candidate.segments, operator_resolution=resolution)
            if id(candidate) in selected_ids
            else candidate
            for candidate in candidates
        ]

    if len(selected) < 2:
        raise ValueError("join operator resolution must select at least two current candidates")
    segments_by_identity = {
        segment.fingerprint_sha256: segment
        for candidate in selected
        for segment in candidate.segments
    }
    joined_segments = tuple(
        segments_by_identity[fingerprint]
        for fingerprint in resolution.source_fingerprints_sha256
    )
    expected_order = tuple(
        segment.fingerprint_sha256
        for segment in sorted(joined_segments, key=lambda segment: (segment.created_at, str(segment.source_path)))
    )
    if resolution.source_fingerprints_sha256 != expected_order:
        raise ValueError("join operator resolution source fingerprints must be in chronological order")
    selected_ids = {id(candidate) for candidate in selected}
    combined = SourceCandidate("GROUP", joined_segments, operator_resolution=resolution)
    return sorted(
        [candidate for candidate in candidates if id(candidate) not in selected_ids] + [combined],
        key=lambda candidate: (candidate.created_at, str(candidate.segments[0].source_path)),
    )


def _discarded_ledger_matches_candidate(record: dict[str, Any], candidate: SourceCandidate) -> bool:
    return (
        record.get("status") == "discarded"
        and record.get("candidate_identity") == candidate.identity
        and record.get("source_segment_paths")
        == [str(segment.source_path) for segment in candidate.segments]
        and record.get("source_segment_fingerprints_sha256")
        == [segment.fingerprint_sha256 for segment in candidate.segments]
    )


def _discarded_ledger_record_matches(record: dict[str, Any], candidate: SourceCandidate) -> bool:
    return (
        candidate.operator_resolution is not None
        and _discarded_ledger_matches_candidate(record, candidate)
        and record.get("operator_resolution")
        == _operator_resolution_payload(candidate.operator_resolution)
    )


def run_ingest(
    config: IngestConfig,
    *,
    duration_probe: Callable[[Path], float] = probe_duration,
    stream_probe: Callable[[Path], tuple[Any, ...]] = probe_stream_signature,
    concat_runner: Callable[[Path, Path], None] = run_concat_copy,
    clock: Callable[[], datetime] = now_ist,
) -> IngestSummary:
    source_root = config.source_root.expanduser().resolve()
    ingest_root = config.ingest_root.expanduser().resolve()
    ledger_path = ingest_root / LEDGER_FILENAME
    ledger = load_ledger(ledger_path)
    operator_resolution = (
        load_operator_resolution(config.operator_resolution_path)
        if config.operator_resolution_path is not None
        else None
    )
    sources = discover_sources(source_root)
    if config.selected_source_paths is not None:
        selected_paths = {path.expanduser().resolve() for path in config.selected_source_paths}
        sources = [path for path in sources if path.resolve() in selected_paths]
    summary = IngestSummary(discovered=len(sources))
    valid_segments: list[SourceSegment] = []

    for source in sources:
        source = source.resolve()
        print(f"DISCOVERED {source}")
        try:
            created_at = parse_source_timestamp(source.name)
            duration = duration_probe(source)
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"invalid duration: {duration}")
            valid_segments.append(SourceSegment(source, created_at, duration, sha256_file(source)))
        except Exception as exc:
            summary.failed += 1
            print(f"FAILED {source} error={type(exc).__name__}: {exc}")

    candidates = group_source_segments(valid_segments)
    if operator_resolution is not None:
        candidates = _apply_operator_resolution(
            candidates, valid_segments, operator_resolution
        )

    for candidate in candidates:
        if candidate.kind == "AMBIGUOUS":
            summary.ambiguous += 1
            summary.failed += 1
            _print_candidate(candidate, None, dry_run=config.dry_run)
            continue
        if candidate.kind == "FAILED":
            summary.failed += 1
            _print_candidate(candidate, None, dry_run=config.dry_run)
            continue

        identity = candidate.identity
        if candidate.kind == "DISCARDED":
            summary.discarded += 1
            existing = ledger["records"].get(identity)
            _print_candidate(candidate, None, dry_run=config.dry_run)
            if existing is not None:
                if _discarded_ledger_record_matches(existing, candidate):
                    summary.skipped += 1
                    print(f"SKIP_ALREADY_DISCARDED identity={identity}")
                else:
                    summary.failed += 1
                    print(f"FAILED identity={identity} error=discarded_ledger_evidence_mismatch")
                continue
            if config.dry_run:
                continue
            try:
                ledger["records"][identity] = {
                    "status": "discarded",
                    "candidate_identity": identity,
                    "source_segment_paths": [
                        str(segment.source_path) for segment in candidate.segments
                    ],
                    "source_segment_fingerprints_sha256": [
                        segment.fingerprint_sha256 for segment in candidate.segments
                    ],
                    "created_at": candidate.created_at.isoformat(),
                    "discarded_at": clock().isoformat(),
                    "operator_resolution": _operator_resolution_payload(
                        candidate.operator_resolution
                    ),
                }
                atomic_write_json(ledger_path, ledger)
                print(f"DISCARDED identity={identity} source_count={len(candidate.segments)}")
            except Exception as exc:
                summary.failed += 1
                print(f"FAILED identity={identity} error={type(exc).__name__}: {exc}")
            continue
        existing = ledger["records"].get(identity)
        if existing is not None and existing.get("status") == "discarded":
            if _discarded_ledger_matches_candidate(existing, candidate):
                summary.skipped += 1
                print(f"SKIP_ALREADY_DISCARDED identity={identity}")
            else:
                summary.failed += 1
                print(f"FAILED identity={identity} error=discarded_ledger_evidence_mismatch")
            continue
        if candidate.kind == "GROUP":
            summary.grouped += 1
            target = ingest_root / deterministic_group_folder_name(candidate.created_at, identity)
        else:
            summary.singleton += 1
            target = ingest_root / deterministic_folder_name(candidate.created_at, identity)
        _print_candidate(candidate, target, dry_run=config.dry_run)

        if config.dry_run:
            if existing is not None and _record_is_complete(existing):
                if candidate.kind == "GROUP" and not _group_ledger_record_matches(
                    existing, candidate, target
                ):
                    summary.failed += 1
                    print(f"FAILED identity={identity} error=grouped_ledger_evidence_mismatch")
                    continue
                summary.skipped += 1
                print(f"SKIP_ALREADY_INGESTED identity={identity} target={target}")
            continue

        try:
            if existing is not None and _record_is_complete(existing):
                if candidate.kind == "GROUP" and not _group_ledger_record_matches(
                    existing, candidate, target
                ):
                    raise ValueError("grouped ledger evidence does not match candidate")
                summary.skipped += 1
                print(f"SKIP_ALREADY_INGESTED identity={identity} target={target}")
                continue
            ingested_at = clock()
            if candidate.kind == "GROUP":
                metadata_path = target / METADATA_FILENAME
                canonical_audio = target / CANONICAL_AUDIO_FILENAME
                if target.exists():
                    if not metadata_path.is_file():
                        raise FileExistsError(f"incomplete pre-existing grouped target: {target}")
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not _metadata_matches_group(metadata, candidate, target):
                        raise ValueError(f"pre-existing grouped target does not match candidate: {target}")
                    ingested_at = datetime.fromisoformat(metadata["ingested_at"])
                else:
                    metadata = _normalize_group(
                        candidate, target, ingested_at, duration_probe=duration_probe,
                        stream_probe=stream_probe, concat_runner=concat_runner,
                    )
                ledger["records"][identity] = {
                    "source_path": str(candidate.segments[0].source_path),
                    "source_filename": candidate.segments[0].source_path.name,
                    "source_group_policy": GROUPING_POLICY,
                    "group_fingerprint_sha256": identity,
                    "segment_count": len(candidate.segments),
                    "source_segment_paths": [str(segment.source_path) for segment in candidate.segments],
                    "source_fingerprint_sha256": metadata["source_fingerprint_sha256"],
                    "created_at": candidate.created_at.isoformat(),
                    "duration_seconds": metadata["duration_seconds"],
                    "ingested_at": ingested_at.isoformat(),
                    "status": "normalized",
                    "target_folder": str(target),
                    "metadata_path": str(metadata_path),
                    "canonical_audio_path": str(canonical_audio),
                    "operator_resolution": metadata.get("operator_resolution"),
                }
            else:
                segment = candidate.segments[0]
                canonical_audio = target / CANONICAL_AUDIO_FILENAME
                metadata_path = target / METADATA_FILENAME
                metadata = _build_metadata(
                    segment.source_path, segment.fingerprint_sha256, segment.created_at,
                    segment.duration_seconds, ingested_at, canonical_audio, candidate,
                )
                if target.exists():
                    if not metadata_path.is_file() or not canonical_audio.is_file():
                        raise FileExistsError(f"incomplete pre-existing target: {target}")
                    existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if (
                        existing_metadata.get("source_fingerprint_sha256") != segment.fingerprint_sha256
                        or existing_metadata.get("canonical_audio_path") != str(canonical_audio)
                        or sha256_file(canonical_audio) != segment.fingerprint_sha256
                    ):
                        raise ValueError(f"pre-existing target does not match source fingerprint: {target}")
                    metadata = existing_metadata
                    ingested_at = datetime.fromisoformat(metadata["ingested_at"])
                else:
                    _normalize_source(segment.source_path, target, metadata)
                ledger["records"][identity] = {
                    "source_path": str(segment.source_path),
                    "source_filename": segment.source_path.name,
                    "source_fingerprint_sha256": segment.fingerprint_sha256,
                    "created_at": segment.created_at.isoformat(),
                    "duration_seconds": metadata["duration_seconds"],
                    "ingested_at": ingested_at.isoformat(),
                    "status": "normalized",
                    "target_folder": str(target),
                    "metadata_path": str(metadata_path),
                    "canonical_audio_path": str(canonical_audio),
                    "operator_resolution": metadata.get("operator_resolution"),
                }
            atomic_write_json(ledger_path, ledger)
            summary.ingested += 1
            print(
                f"INGESTED {candidate.segments[0].source_path} "
                f"identity={identity} target={target}"
            )
        except Exception as exc:
            summary.failed += 1
            print(f"FAILED identity={identity} error={type(exc).__name__}: {exc}")

    print(
        "SUMMARY "
        f"discovered={summary.discovered} ingested={summary.ingested} skipped={summary.skipped} "
        f"failed={summary.failed} singleton={summary.singleton} grouped={summary.grouped} "
        f"ambiguous={summary.ambiguous} discarded={summary.discarded}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ingest-root", type=Path, default=DEFAULT_INGEST_ROOT)
    parser.add_argument(
        "--operator-resolution",
        type=Path,
        help="exact SHA-256 operator resolution (separate, join, or discard)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_ingest(
            IngestConfig(
                args.source_root,
                args.ingest_root,
                args.dry_run,
                args.operator_resolution,
            )
        )
    except Exception as exc:
        print(f"FAILED ingest_setup error={type(exc).__name__}: {exc}")
        return 1
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
