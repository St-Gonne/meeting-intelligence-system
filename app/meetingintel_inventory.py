#!/usr/bin/env python3
"""Read-only inventory of Meetily and phone recording readiness."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import meetingintel_pipeline as pipeline
import phone_drive_fetch as phone_fetch
import phone_recording_ingest as phone_ingest


TIMESTAMP_PATTERN = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$")
IST = phone_ingest.IST


@dataclass(frozen=True)
class InventoryConfig:
    meetings_root: Path = pipeline.DEFAULT_MEETINGS_ROOT
    phone_staging_root: Path = phone_fetch.DEFAULT_DOWNLOAD_ROOT
    phone_fetch_ledger: Path = phone_fetch.DEFAULT_LEDGER_PATH
    phone_ingest_root: Path = phone_ingest.DEFAULT_INGEST_ROOT
    processing_ledger: Path = pipeline.DEFAULT_LEDGER_PATH


@dataclass(frozen=True)
class InventoryItem:
    created_at: datetime
    source_kind: str
    display_name: str
    status: str
    processing_mode: str = ""
    identity: str = ""
    group_key: str = ""


@dataclass(frozen=True)
class InventoryReport:
    scope: str
    audio_first: bool
    items: tuple[InventoryItem, ...]
    counts: dict[str, int]
    warnings: tuple[str, ...] = ()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    return Path(value).name or fallback


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(IST)


def _filename_datetime(value: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.fullmatch(Path(value).stem)
    if match is None:
        return None
    try:
        return datetime(*(int(part) for part in match.groups()), tzinfo=IST)
    except ValueError:
        return None


def _safe_datetime(value: Any, fallback: datetime) -> datetime:
    return _parse_datetime(value) or fallback


def _load_records(path: Path, default: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return default
    try:
        payload = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        warnings.append("local inventory ledger unreadable")
        return default
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
        warnings.append("local inventory ledger malformed")
        return default
    return payload


def _path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().absolute())


def _processed_maps(processing_ledger: dict[str, Any]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    records = processing_ledger.get("records", {})
    if not isinstance(records, dict):
        return set(), {}
    return set(records), {key: value for key, value in records.items() if isinstance(value, dict)}


def _meetily_items(
    config: InventoryConfig,
    processed_fingerprints: set[str],
    processed_records: dict[str, dict[str, Any]],
    fallback_now: datetime,
    warnings: list[str],
) -> list[InventoryItem]:
    items: list[InventoryItem] = []
    if not config.meetings_root.is_dir():
        return items
    processed_paths = {
        _path_key(record.get("folder_path"))
        for record in processed_records.values()
        if isinstance(record.get("folder_path"), str)
    }
    for folder in sorted(config.meetings_root.iterdir(), key=lambda path: path.name):
        metadata_path = folder / "metadata.json"
        transcripts_path = folder / "transcripts.json"
        if not folder.is_dir() or not metadata_path.is_file() or not transcripts_path.is_file():
            continue
        try:
            metadata = _read_json(metadata_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            warnings.append("Meetily metadata unreadable")
            continue
        if not isinstance(metadata, dict) or metadata.get("status") != "completed":
            continue
        created_at = _safe_datetime(metadata.get("created_at"), fallback_now)
        path_key = _path_key(folder)
        matching_record = next(
            (record for record in processed_records.values() if _path_key(record.get("folder_path", "")) == path_key),
            None,
        )
        processed = path_key in processed_paths
        if not processed:
            # Legacy ledgers may not retain folder_path reliably; fingerprint
            # keys are intentionally not recomputed from transcript contents.
            processed = False
        items.append(
            InventoryItem(
                created_at=created_at,
                source_kind="meetily",
                display_name=_safe_name(metadata.get("meeting_name"), folder.name),
                status="meetily already processed" if processed else "meetily ready",
                processing_mode=(matching_record or {}).get("processing_mode", "") if processed else "",
                identity=path_key,
            )
        )
    return items


def _source_identity_from_metadata(metadata: dict[str, Any]) -> str | None:
    group = metadata.get("source_group")
    if isinstance(group, dict) and isinstance(group.get("group_fingerprint_sha256"), str):
        return group["group_fingerprint_sha256"]
    source_kind = metadata.get("source_kind")
    source_path = metadata.get("source_path")
    created_at = metadata.get("created_at")
    fingerprint = metadata.get("source_fingerprint_sha256")
    if not all(isinstance(value, str) and value for value in (source_kind, source_path, created_at, fingerprint)):
        return None
    raw = "|".join((source_kind, source_path, created_at, fingerprint)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _phone_normalized_items(
    config: InventoryConfig,
    processed_fingerprints: set[str],
    processed_records: dict[str, dict[str, Any]],
    fallback_now: datetime,
    warnings: list[str],
) -> tuple[list[InventoryItem], set[str], set[str], set[str]]:
    items: list[InventoryItem] = []
    normalized_source_paths: set[str] = set()
    normalized_source_fingerprints: set[str] = set()
    discarded_source_paths: set[str] = set()
    ingest_ledger = _load_records(
        config.phone_ingest_root / phone_ingest.LEDGER_FILENAME,
        phone_ingest.empty_ledger(),
        warnings,
    )
    records = ingest_ledger.get("records", {})
    if not isinstance(records, dict):
        records = {}
    for identity, record in sorted(records.items(), key=lambda pair: str(pair[0])):
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        source_paths = record.get("source_segment_paths", [])
        if not isinstance(source_paths, list):
            source_paths = []
        normalized_paths = {_path_key(path) for path in source_paths if isinstance(path, str)}
        if status == "discarded":
            discarded_source_paths.update(normalized_paths)
            created_at = _safe_datetime(record.get("created_at"), fallback_now)
            items.append(
                InventoryItem(
                    created_at=created_at,
                    source_kind="phone_recording",
                    display_name=_safe_name(record.get("source_filename"), "discarded phone recording"),
                    status="discarded",
                    identity=str(identity),
                    group_key="discarded:" + str(identity),
                )
            )
            continue
        if status != "normalized":
            continue
        metadata: dict[str, Any] = {}
        metadata_path = record.get("metadata_path")
        if isinstance(metadata_path, str) and Path(metadata_path).is_file():
            try:
                loaded = _read_json(Path(metadata_path))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, UnicodeError, json.JSONDecodeError):
                warnings.append("normalized phone metadata unreadable")
        normalized_source_paths.update(normalized_paths)
        if isinstance(metadata.get("source_path"), str):
            normalized_source_paths.add(_path_key(metadata["source_path"]))
        source_fingerprint = metadata.get("source_fingerprint_sha256")
        if isinstance(source_fingerprint, str) and source_fingerprint:
            normalized_source_fingerprints.add(source_fingerprint)
        source_segments = metadata.get("source_segments")
        if isinstance(source_segments, list):
            for segment in source_segments:
                if not isinstance(segment, dict):
                    continue
                segment_fingerprint = segment.get("fingerprint_sha256")
                if isinstance(segment_fingerprint, str) and segment_fingerprint:
                    normalized_source_fingerprints.add(segment_fingerprint)
        source_identity = _source_identity_from_metadata(metadata)
        fingerprint = pipeline.build_audio_first_fingerprint(source_identity) if source_identity else str(identity)
        target_folder = record.get("target_folder")
        processed = fingerprint in processed_fingerprints or str(identity) in processed_fingerprints
        if isinstance(target_folder, str):
            target_key = _path_key(target_folder)
            processed = processed or any(
                _path_key(processed_record.get("folder_path")) == target_key
                for processed_record in processed_records.values()
                if isinstance(processed_record.get("folder_path"), str)
            )
        processing_record = processed_records.get(fingerprint) or processed_records.get(str(identity))
        status_label = "phone already processed" if processed else "phone normalized and ready"
        if not processed and source_identity:
            failure_category = pipeline.qwen_latest_failure_category(source_identity)
            if failure_category == "unsupported_detected_language":
                status_label = "phone processing stopped: ambiguous language tag"
            elif failure_category:
                status_label = "phone processing stopped; recording is safe"
        items.append(
            InventoryItem(
                created_at=_safe_datetime(metadata.get("created_at", record.get("created_at")), fallback_now),
                source_kind="phone_recording",
                display_name=_safe_name(metadata.get("source_filename", record.get("source_filename")), "phone recording"),
                status=status_label,
                processing_mode=(processing_record or {}).get("processing_mode", "") if processed else "",
                identity=fingerprint,
            )
        )
    return (
        items,
        normalized_source_paths,
        normalized_source_fingerprints,
        discarded_source_paths,
    )


def _raw_phone_items(
    config: InventoryConfig,
    normalized_source_paths: set[str],
    normalized_source_fingerprints: set[str],
    discarded_source_paths: set[str],
    fallback_now: datetime,
    warnings: list[str],
) -> list[InventoryItem]:
    fetch_ledger = _load_records(config.phone_fetch_ledger, {"schema_version": phone_fetch.LEDGER_SCHEMA_VERSION, "records": {}}, warnings)
    records = fetch_ledger.get("records", {})
    if not isinstance(records, dict):
        return []
    raw: list[tuple[datetime, str, str, dict[str, Any]]] = []
    for remote_id, record in sorted(records.items(), key=lambda pair: str(pair[0])):
        if not isinstance(record, dict) or not isinstance(record.get("local_relative_path"), str):
            continue
        relative = Path(record["local_relative_path"])
        absolute = _path_key(config.phone_staging_root / relative)
        if absolute in normalized_source_paths or absolute in discarded_source_paths:
            continue
        stored_fingerprint = record.get("sha256")
        if (
            isinstance(stored_fingerprint, str)
            and stored_fingerprint in normalized_source_fingerprints
        ):
            continue
        if record.get("status") == "discarded":
            continue
        created_at = _filename_datetime(relative.name) or _parse_datetime(record.get("modified_time")) or fallback_now
        group_key = relative.parent.as_posix()
        raw.append((created_at, relative.name, group_key, record))

    groups: dict[str, list[tuple[datetime, str, str, dict[str, Any]]]] = {}
    for item in raw:
        groups.setdefault(item[2], []).append(item)
    items: list[InventoryItem] = []

    def raw_group_key(relative_parent: str, created_at: datetime) -> str:
        return f"raw:{relative_parent}:{created_at.astimezone(IST).date().isoformat()}"

    for group_key, group in sorted(groups.items()):
        group.sort(key=lambda item: (item[0], item[1]))
        statuses = {item[3].get("status") for item in group}
        explicit_grouping = all(item[3].get("grouping_status") in {"deterministic_group", "grouped"} for item in group)
        if len(group) > 1:
            if statuses.intersection({"downloaded", "pending_settlement"}):
                # A multi-file batch is not reviewable until every member has
                # passed the two-observation settle rule.
                status = "phone pending settlement"
            elif "requires_operator_decision" in statuses or not explicit_grouping:
                status = "operator decision required"
            else:
                status = "phone pending normalization"
            display_name = f"{group_key} ({len(group)} staged segments)"
            group_identity = raw_group_key(group_key, group[0][0])
            items.append(
                InventoryItem(
                    created_at=group[0][0],
                    source_kind="phone_recording",
                    display_name=display_name,
                    status=status,
                    group_key=group_identity,
                )
            )
            continue
        created_at, display_name, _group_key, record = group[0]
        status_value = record.get("status")
        status = {
            "settled_ready": "phone downloaded and settled (not normalized)",
            "downloaded": "phone pending settlement",
            "pending_settlement": "phone pending settlement",
            "requires_operator_decision": "operator decision required",
            "failed": "phone fetch failed",
        }.get(status_value, "phone pending normalization")
        items.append(
            InventoryItem(
                created_at=created_at,
                source_kind="phone_recording",
                display_name=display_name,
                status=status,
                identity=str(record.get("drive_file_id", "")),
                group_key=raw_group_key(group_key, created_at),
            )
        )
    return items


def _state_key(item: InventoryItem) -> str:
    return item.status


def _scope_filter(items: Sequence[InventoryItem], scope: str, audio_first: bool, now: datetime) -> list[InventoryItem]:
    normalized_phone_statuses = {"phone normalized and ready", "phone already processed"}
    filtered = [
        item for item in items
        if not audio_first or (
            item.source_kind == "phone_recording"
            and (
                item.status in normalized_phone_statuses
                or item.status.startswith("phone processing stopped")
            )
        )
    ]
    if scope == "today":
        return [item for item in filtered if item.created_at.astimezone(IST).date() == now.astimezone(IST).date()]
    if scope == "last":
        return [max(filtered, key=lambda item: (item.created_at, item.source_kind, item.display_name))] if filtered else []
    if scope == "new":
        filtered = [item for item in filtered if "already processed" not in item.status and item.status != "discarded"]
        return filtered
    if scope == "all":
        return filtered
    raise ValueError(f"unsupported scope: {scope}")


def inventory(
    scope: str,
    *,
    config: InventoryConfig = InventoryConfig(),
    audio_first: bool = False,
    clock: Callable[[], datetime] = pipeline.now_ist,
) -> InventoryReport:
    now = clock().astimezone(IST)
    warnings: list[str] = []
    processing_ledger = _load_records(config.processing_ledger, {"records": {}}, warnings)
    processed_fingerprints, processed_records = _processed_maps(processing_ledger)
    normalized, normalized_paths, normalized_fingerprints, discarded_paths = _phone_normalized_items(
        config, processed_fingerprints, processed_records, now, warnings,
    )
    items = normalized + _raw_phone_items(
        config, normalized_paths, normalized_fingerprints, discarded_paths, now, warnings,
    )
    if not audio_first:
        items = _meetily_items(config, processed_fingerprints, processed_records, now, warnings) + items
    selected = sorted(
        _scope_filter(items, scope, audio_first, now),
        key=lambda item: (item.created_at, item.source_kind, item.display_name),
    )
    counts: dict[str, int] = {}
    for item in selected:
        counts[_state_key(item)] = counts.get(_state_key(item), 0) + 1
    return InventoryReport(scope, audio_first, tuple(selected), counts, tuple(warnings))


def format_inventory(report: InventoryReport) -> str:
    lines = [
        "MEETINGINTEL_INVENTORY "
        f"scope={report.scope} audio_first={str(report.audio_first).lower()} total={len(report.items)} "
        + " ".join(f"{key.replace(' ', '_')}={value}" for key, value in sorted(report.counts.items()))
    ]
    for warning in report.warnings:
        lines.append(f"WARNING {warning}")
    if not report.items:
        lines.append("No matching meetings or phone recordings.")
        return "\n".join(lines)
    for item in report.items:
        created = item.created_at.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
        mode = f" | mode={item.processing_mode}" if item.processing_mode else ""
        lines.append(f"  {created} | {item.source_kind} | {item.display_name} | {item.status}{mode}")
    return "\n".join(lines)
