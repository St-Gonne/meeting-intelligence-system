#!/usr/bin/env python3
"""Read-only-by-default, source-aware operator inbox.

This module is deliberately an adapter over the existing local inventories and
laptop capture evidence.  It owns no ledger and has no write or processing
path.  Actions remain in the CLI and must select one exact item before calling
an existing operator workflow.
"""

from __future__ import annotations
from mi_paths import public_path

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from laptop_capture_contract import audit_capture_session, load_manifest
import laptop_capture_ingest as capture_ingest
from laptop_capture_meeting_source import (
    LaptopMeetingSource,
    LaptopSourceValidationError,
    discover_laptop_meeting_sources,
)
import meetingintel_inventory as local_inventory
import meetingintel_pipeline as pipeline
from meetingintel_runtime import source_annotation


IST = local_inventory.IST
ACTIVE_CAPTURE_GRACE = timedelta(hours=12)


@dataclass(frozen=True)
class InboxConfig:
    inventory: local_inventory.InventoryConfig = local_inventory.InventoryConfig()
    laptop_ingest_root: Path = public_path("project/ingest/laptop")
    laptop_capture_root: Path = public_path("project/ingest/captures")


@dataclass(frozen=True)
class InboxItem:
    created_at: datetime
    source_kind: str
    display_name: str
    status: str
    next_action: str
    category: str
    identity: str = ""
    segment_count: int = 0
    processed_mode: str = ""

    @property
    def state(self) -> str:
        """Stable alias for callers that use state rather than status."""
        return self.status


@dataclass(frozen=True)
class InboxReport:
    items: tuple[InboxItem, ...]
    counts: dict[str, int]
    warnings: tuple[str, ...] = ()
    today_counts: dict[str, int] = field(default_factory=dict)
    today_date: date | None = None

    @property
    def current_action_item(self) -> InboxItem | None:
        candidates = [item for item in self.items if item.category == "needs_decision"]
        return max(candidates, key=lambda item: (item.created_at, item.source_kind, item.identity), default=None)

    @property
    def numbered_items(self) -> tuple[InboxItem, ...]:
        actionable = [item for item in self.items if item.category in {"needs_decision", "ready"}]
        if self.today_date is None:
            return tuple(actionable)
        today = [item for item in actionable if item.created_at.astimezone(IST).date() == self.today_date]
        older = [item for item in actionable if item.created_at.astimezone(IST).date() != self.today_date]
        return tuple(today + older)


def _read_records(path: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        warnings.append("local processing ledger unreadable")
        return {}
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        warnings.append("local processing ledger malformed")
        return {}
    return {str(key): value for key, value in records.items() if isinstance(value, dict)}


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(IST)


def _category_for_inventory(status: str) -> tuple[str, str]:
    if status == "phone pending settlement":
        return (
            "waiting",
            "Waiting for the second unchanged Drive check; minimum 10 minutes. "
            "The scheduler checks every 2 hours; run mi sync for an immediate check after the minimum.",
        )
    if status in {"phone normalized and ready", "meetily ready", "laptop normalized and ready"}:
        return "ready", "Ready for an explicit processing action"
    if status.startswith("phone processing stopped"):
        return "ready", "Previous processing failed; recording is safe and ready for a corrected retry"
    if status in {"phone already processed", "meetily already processed", "laptop already processed", "discarded"}:
        return "completed", "No action needed"
    if status == "phone fetch failed":
        return "needs_decision", "Check the local fetch status"
    return "needs_decision", "Review this recording before normalization"


def _phone_or_meetily_items(
    config: InboxConfig,
    *,
    clock: Callable[[], datetime],
    warnings: list[str],
) -> list[InboxItem]:
    report = local_inventory.inventory("all", config=config.inventory, clock=clock)
    warnings.extend(report.warnings)
    items: list[InboxItem] = []
    for source in report.items:
        category, action = _category_for_inventory(source.status)
        if source.source_kind == "phone_recording":
            label = "Phone recording"
            kind = "phone"
        else:
            label = "Meetily recording"
            kind = "meetily"
        segment_count = 0
        if source.display_name.endswith(" staged segments)"):
            try:
                segment_count = int(source.display_name.rsplit("(", 1)[1].split(" ", 1)[0])
            except (IndexError, ValueError):
                segment_count = 0
        items.append(
            InboxItem(
                source.created_at.astimezone(IST),
                kind,
                label,
                source.status,
                action,
                category,
                identity=source.group_key or source.identity,
                segment_count=segment_count,
                processed_mode=source.processing_mode,
            )
        )
    return items


def _processed_laptop(
    source: LaptopMeetingSource,
    records: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    fingerprint = pipeline.build_laptop_capture_fingerprint(source.source_identity)
    matching = records.get(fingerprint)
    if matching is None:
        matching = next(
            (
                record
                for record in records.values()
                if record.get("source_identity") == source.source_identity
                or record.get("folder_path") == str(source.source_folder)
            ),
            None,
        )
    return matching is not None, str((matching or {}).get("processing_mode", ""))


def _laptop_items(
    config: InboxConfig,
    *,
    now: datetime,
    warnings: list[str],
    discover: Callable[..., Sequence[Any]] = discover_laptop_meeting_sources,
    audit: Callable[..., Any] = audit_capture_session,
) -> list[InboxItem]:
    records = _read_records(config.inventory.processing_ledger, warnings)
    items: list[InboxItem] = []
    normalized_capture_ids: set[str] = set()
    try:
        discoveries = discover(config.laptop_ingest_root, capture_root=config.laptop_capture_root)
    except (OSError, LaptopSourceValidationError, ValueError) as exc:
        warnings.append(f"laptop source discovery unavailable: {type(exc).__name__}")
        discoveries = ()
    for result in discoveries:
        source = getattr(result, "source", None)
        if source is None:
            continue
        normalized_capture_ids.add(source.capture_id)
        processed, mode = _processed_laptop(source, records)
        items.append(
            InboxItem(
                source.created_at.astimezone(IST),
                "laptop",
                "Laptop recording",
                "laptop already processed" if processed else "laptop normalized and ready",
                "No action needed" if processed else "Ready for an explicit processing action",
                "completed" if processed else "ready",
                identity=source.source_identity,
                segment_count=source.source_segment_count,
                processed_mode=mode,
            )
        )

    root = config.laptop_capture_root.expanduser()
    if not root.is_dir() or root.is_symlink():
        return items
    for session in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        manifest_path = session / capture_ingest.CAPTURE_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest(manifest_path)
            capture_id = manifest.get("capture_id")
            if isinstance(capture_id, str) and capture_id in normalized_capture_ids:
                continue
            started = _timestamp(manifest.get("started_at"))
            if started is None:
                continue
            session_audit = audit(session, manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            warnings.append("laptop capture evidence unreadable")
            continue
        status = manifest.get("status")
        if status == "complete" and session_audit.accepted and session_audit.merge_ready:
            label = "laptop validated; waiting for normalization"
            action = "Normalize this exact capture after confirmation"
        elif status == "interrupted" and session_audit.recovery_ready:
            label = "laptop interrupted; recovery eligible"
            action = "Review recovery evidence before normalization"
        elif status == "recording" and now - started <= ACTIVE_CAPTURE_GRACE:
            label = "laptop recording in progress"
            action = "Leave this active capture alone"
        elif status == "recording":
            label = "laptop capture appears abandoned"
            action = "Review capture evidence; this is not an active recording"
        else:
            label = "laptop capture needs attention"
            action = "Review capture evidence"
        items.append(
            InboxItem(
                started,
                "laptop",
                "Laptop recording",
                label,
                action,
                (
                    "waiting"
                    if status == "recording" and now - started <= ACTIVE_CAPTURE_GRACE
                    else "needs_decision"
                ),
                identity=str(capture_id) if isinstance(capture_id, str) and capture_id else "",
                segment_count=len(getattr(session_audit, "finalized_segments", ())),
            )
        )
    return items


def build_inbox(
    *,
    config: InboxConfig = InboxConfig(),
    clock: Callable[[], datetime] = pipeline.now_ist,
    laptop_discover: Callable[..., Sequence[Any]] = discover_laptop_meeting_sources,
    laptop_audit: Callable[..., Any] = audit_capture_session,
) -> InboxReport:
    now = clock().astimezone(IST)
    warnings: list[str] = []
    items = _phone_or_meetily_items(config, clock=lambda: now, warnings=warnings)
    items.extend(
        _laptop_items(
            config,
            now=now,
            warnings=warnings,
            discover=laptop_discover,
            audit=laptop_audit,
        )
    )
    corrected = []
    for item in items:
        annotation = source_annotation(item.identity) if item.source_kind == "laptop" and item.identity else None
        if annotation and annotation.get("completeness") == "incomplete":
            item = replace(item, status="laptop incomplete; meeting continued after capture stopped",
                           next_action="Retained audio only. Full-meeting processing blocked; review missing capture.", category="needs_decision")
        corrected.append(item)
    ordered = tuple(sorted(corrected, key=lambda item: (item.created_at, item.source_kind, item.display_name), reverse=True))
    counts = {category: sum(item.category == category for item in ordered) for category in ("needs_decision", "ready", "waiting", "completed")}
    today_counts = {
        category: sum(
            item.category == category and item.created_at.astimezone(IST).date() == now.date()
            for item in ordered
        )
        for category in ("needs_decision", "ready", "waiting", "completed")
    }
    return InboxReport(ordered, counts, tuple(warnings), today_counts, now.date())


def _row(item: InboxItem, number: int | None = None) -> str:
    created = item.created_at.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    segments = f" — {item.segment_count} chunks" if item.segment_count else ""
    mode = f" — mode={item.processed_mode}" if item.processed_mode else ""
    prefix = f"[{number}] " if number is not None else "• "
    return f"{prefix}{item.display_name} — {created}{segments} — {item.status}{mode}\n  Next: {item.next_action}"


def format_inbox(report: InboxReport, *, details: bool = False) -> str:
    today_needs = report.today_counts.get("needs_decision", 0)
    today_ready = report.today_counts.get("ready", 0)
    today_waiting = report.today_counts.get("waiting", 0)
    lines = [
        "MEETING INBOX",
        "",
        "Summary: "
        f"today actionable={today_needs + today_ready}; "
        f"today waiting={today_waiting}; "
        f"completed/closed={report.counts.get('completed', 0)}",
    ]
    for warning in report.warnings:
        lines.append(f"Warning: {warning}")
    if not report.items:
        lines.append("No matching meetings or phone recordings.")
        return "\n".join(lines)

    numbered = report.numbered_items
    number_by_identity = {id(item): index for index, item in enumerate(numbered, 1)}
    today_action = [
        item for item in numbered
        if report.today_date is not None and item.created_at.astimezone(IST).date() == report.today_date
    ]
    today_ids = {id(item) for item in today_action}
    older_action = [item for item in numbered if id(item) not in today_ids]
    if today_action:
        lines.extend(["", "Today"])
        lines.extend(_row(item, number_by_identity[id(item)]) for item in today_action)
    if older_action:
        lines.extend(["", f"Older unresolved / ready ({len(older_action)})"])
        if details:
            lines.extend(_row(item, number_by_identity[id(item)]) for item in older_action)
        else:
            lines.append("  Hidden from the daily view. Press d to review older items.")

    waiting = [item for item in report.items if item.category == "waiting"]
    if waiting:
        lines.extend(["", f"Waiting safely ({len(waiting)})"])
        shown = waiting if details else waiting[:1]
        lines.extend(_row(item) for item in shown)
        if not details and len(waiting) > 1:
            lines.append(f"  Additional waiting items: {len(waiting) - 1}")

    completed = [item for item in report.items if item.category == "completed"]
    if completed:
        lines.extend(["", f"Completed / closed ({len(completed)})"])
        if details:
            lines.extend(_row(item) for item in completed)
    return "\n".join(lines)
