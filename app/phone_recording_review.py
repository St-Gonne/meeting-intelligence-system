#!/usr/bin/env python3
"""Guided, local-only review of settled phone recordings.

This module measures staged container evidence and creates an explicit
schema-v3 operator plan for the existing phone normalizer.  It never contacts
Drive, loads model credentials, or enters MeetingIntel production processing.
"""

from __future__ import annotations

import meetingintel_observe as observe

import contextlib
import io
import json
import math
import stat
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import phone_drive_fetch as fetch
import phone_recording_ingest as ingest
from audio_first_meeting_source import (
    AudioFirstMeetingSource,
    CANONICAL_AUDIO_FILENAME,
    METADATA_FILENAME,
    load_audio_first_source,
)


MAX_INPUT_ATTEMPTS = 3


class ReviewError(RuntimeError):
    """A review cannot proceed safely with the available local evidence."""


@dataclass(frozen=True)
class ReviewSegment:
    source_path: Path
    created_at: datetime
    duration_seconds: float
    fingerprint_sha256: str
    size_bytes: int

    @property
    def end_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.duration_seconds)


@dataclass(frozen=True)
class ReviewSet:
    source_parent: Path
    recording_date: date
    segments: tuple[ReviewSegment, ...]
    candidates: tuple[ingest.SourceCandidate, ...]
    suggestion: ingest.OperatorResolution | None
    suggestion_reason: str
    group_key: str = ""


@dataclass(frozen=True)
class ReviewCommitResult:
    summary: ingest.IngestSummary
    normalized_sources: tuple[AudioFirstMeetingSource, ...]


@dataclass(frozen=True)
class ReviewConfig:
    staging_root: Path = fetch.DEFAULT_DOWNLOAD_ROOT
    fetch_ledger_path: Path = fetch.DEFAULT_LEDGER_PATH
    ingest_root: Path = ingest.DEFAULT_INGEST_ROOT


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str):
        raise ReviewError("settled fetch record has no local relative path")
    try:
        return fetch._safe_relative_path(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError("settled fetch record has an unsafe local path") from exc


def _operator_plan_payload(plan: ingest.OperatorResolution) -> dict[str, Any]:
    return ingest._operator_resolution_payload(plan)


def _normalized_evidence(ingest_root: Path) -> tuple[set[str], set[str]]:
    ledger = ingest.load_ledger(ingest_root / ingest.LEDGER_FILENAME)
    fingerprints: set[str] = set()
    paths: set[str] = set()
    for record in ledger.get("records", {}).values():
        if not isinstance(record, dict) or record.get("status") not in {"normalized", "discarded"}:
            continue
        for key in ("source_fingerprint_sha256", "candidate_identity"):
            value = record.get(key)
            if isinstance(value, str):
                fingerprints.add(value)
        values = record.get("source_segment_fingerprints_sha256", [])
        if isinstance(values, list):
            fingerprints.update(value for value in values if isinstance(value, str))
        for key in ("source_path", "source_segment_paths"):
            value = record.get(key)
            if isinstance(value, str):
                paths.add(str(Path(value).expanduser().absolute()))
            elif isinstance(value, list):
                paths.update(
                    str(Path(item).expanduser().absolute())
                    for item in value
                    if isinstance(item, str)
                )
    return fingerprints, paths


def _check_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ReviewError(f"source is not an ordinary file: {path.name}")


def _build_segment(path: Path, record: dict[str, Any], duration_probe: Callable[[Path], float]) -> ReviewSegment:
    _check_regular(path)
    try:
        created_at = ingest.parse_source_timestamp(path.name)
    except ValueError as exc:
        raise ReviewError(f"malformed phone recording filename: {path.name}") from exc
    duration = duration_probe(path)
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise ReviewError(f"invalid duration for {path.name}")
    fingerprint = ingest.sha256_file(path)
    expected_size = record.get("size")
    if isinstance(expected_size, int) and expected_size != path.stat().st_size:
        raise ReviewError(f"settled source size changed: {path.name}")
    expected_fingerprint = record.get("sha256")
    if isinstance(expected_fingerprint, str) and expected_fingerprint != fingerprint:
        raise ReviewError(f"settled source fingerprint changed: {path.name}")
    return ReviewSegment(path.resolve(), created_at, float(duration), fingerprint, path.stat().st_size)


def _as_ingest_segment(segment: ReviewSegment) -> ingest.SourceSegment:
    return ingest.SourceSegment(
        segment.source_path,
        segment.created_at,
        segment.duration_seconds,
        segment.fingerprint_sha256,
    )


def collect_review_sets(
    config: ReviewConfig = ReviewConfig(),
    *,
    duration_probe: Callable[[Path], float] = ingest.probe_duration,
) -> tuple[ReviewSet, ...]:
    """Collect only settled, locally staged, unresolved source segments."""
    fetch_ledger = fetch.load_ledger(config.fetch_ledger_path)
    normalized_fingerprints, normalized_paths = _normalized_evidence(config.ingest_root)
    buckets: dict[tuple[Path, date], list[ReviewSegment]] = {}
    seen_paths: set[Path] = set()
    seen_fingerprints: set[str] = set()
    for record in fetch_ledger.get("records", {}).values():
        if not isinstance(record, dict) or record.get("status") != "settled_ready":
            continue
        relative = _safe_relative(record.get("local_relative_path"))
        source = config.staging_root / relative
        source_key = source.resolve() if source.exists() and not source.is_symlink() else source.absolute()
        if str(source_key) in normalized_paths:
            continue
        if source.is_symlink() or source_key in seen_paths:
            raise ReviewError("duplicate or symlinked staged source evidence")
        segment = _build_segment(source, record, duration_probe)
        if segment.fingerprint_sha256 in normalized_fingerprints:
            continue
        if segment.fingerprint_sha256 in seen_fingerprints:
            raise ReviewError("duplicate source fingerprint evidence")
        seen_paths.add(source_key)
        seen_fingerprints.add(segment.fingerprint_sha256)
        buckets.setdefault((segment.source_path.parent, segment.created_at.date()), []).append(segment)

    result: list[ReviewSet] = []
    for (source_parent, recording_date), segments in sorted(
        buckets.items(), key=lambda item: (item[0][1], str(item[0][0]))
    ):
        ordered = tuple(sorted(segments, key=lambda item: (item.created_at, item.source_path.name)));
        if len({segment.created_at for segment in ordered}) != len(ordered):
            raise ReviewError("duplicate timestamp evidence in a review set")
        candidates = tuple(
            ingest.group_source_segments([_as_ingest_segment(segment) for segment in ordered])
        )
        if any(candidate.kind == "FAILED" for candidate in candidates):
            raise ReviewError("review set contains failed duplicate evidence")
        suggestion: ingest.OperatorResolution | None = None
        reason = ""
        if all(candidate.kind in {"GROUP", "SINGLETON"} for candidate in candidates):
            groups = tuple(
                tuple(segment.fingerprint_sha256 for segment in candidate.segments)
                for candidate in candidates
            )
            suggestion = _make_plan(ordered, groups=groups)
            reason = f"existing {ingest.GROUPING_POLICY} continuity policy"
        else:
            reasons = sorted({candidate.reason for candidate in candidates if candidate.reason})
            reason = reasons[0] if reasons else "existing grouping policy found no safe complete plan"
        try:
            relative_parent = source_parent.resolve().relative_to(
                config.staging_root.expanduser().resolve()
            )
        except ValueError as exc:
            raise ReviewError("review source parent is outside the configured staging root") from exc
        group_key = f"raw:{relative_parent.as_posix()}:{recording_date.isoformat()}"
        result.append(
            ReviewSet(
                source_parent,
                recording_date,
                ordered,
                candidates,
                suggestion,
                reason,
                group_key,
            )
        )
    return tuple(result)


def _make_plan(
    segments: Sequence[ReviewSegment],
    *,
    groups: Sequence[Sequence[str]],
    discarded: Sequence[str] = (),
) -> ingest.OperatorResolution:
    identities = tuple(segment.fingerprint_sha256 for segment in segments)
    return ingest.OperatorResolution(
        ingest.OPERATOR_PLAN_SCHEMA_VERSION,
        "plan",
        identities,
        tuple(tuple(group) for group in groups),
        tuple(discarded),
    )


def _display_duration(seconds: float) -> str:
    whole = int(seconds)
    fraction = seconds - whole
    minutes, remainder = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    remainder_text = f"{remainder + fraction:06.3f}"
    if hours:
        return f"{hours}h {minutes:02d}m {remainder_text}s"
    if minutes:
        return f"{minutes}m {remainder_text}s"
    return f"{remainder_text}s"


def _display_time(value: datetime) -> str:
    fraction = value.microsecond / 1_000_000
    suffix = f".{int(round(fraction * 1000)):03d}" if fraction else ""
    return value.strftime("%H:%M:%S") + suffix


def format_review(review: ReviewSet, *, plain_language: bool = False) -> str:
    header = (
        " #  Date IST    Filename                         Start          End*           Duration       Gap"
        if plain_language
        else " #  Filename                         Start          End*           Duration       Gap"
    )
    lines = [
        f"PHONE RECORDING REVIEW — {review.recording_date.strftime('%d %b %Y')}",
        f"{len(review.segments)} settled segments · not normalized",
        "",
        header,
    ]
    previous: ReviewSegment | None = None
    for index, segment in enumerate(review.segments, 1):
        gap = "—" if previous is None else f"{(segment.created_at - previous.end_at).total_seconds():+.3f}s"
        date_text = segment.created_at.strftime("%Y-%m-%d ") if plain_language else ""
        lines.append(
            f"{index:>2}  {date_text}{segment.source_path.name:<31} "
            f"{_display_time(segment.created_at):<14} "
            f"{_display_time(segment.end_at):<14} "
            f"{_display_duration(segment.duration_seconds):<14} {gap}"
        )
        previous = segment
    lines.extend([
        "",
        "* End is calculated from the filename start time plus measured duration; times are Asia/Kolkata.",
    ])
    if review.suggestion is None:
        lines.extend([
            "",
            "No safe automatic suggestion.",
            (
                "Reason: no unambiguous grouping was proven from the local evidence."
                if plain_language
                else f"Reason: {review.suggestion_reason}."
            ),
        ])
    else:
        lines.extend(["", "Suggested plan:"])
        for index, group in enumerate(review.suggestion.groups, 1):
            segment_numbers = [
                str(review.segments.index(next(segment for segment in review.segments if segment.fingerprint_sha256 == identity)) + 1)
                for identity in group
            ]
            lines.append(f"  Meeting {index}: segments {', '.join(segment_numbers)}")
        lines.append(
            "  Reason: the local continuity check produced this safe grouping."
            if plain_language
            else f"  Reason: {review.suggestion_reason}."
        )
    return "\n".join(lines)


def _parse_partition(value: str, segment_count: int) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    try:
        for raw_group in value.split(";"):
            if not raw_group.strip():
                raise ValueError
            indexes = tuple(int(item.strip()) for item in raw_group.split(","))
            if not indexes or any(index < 1 or index > segment_count for index in indexes):
                raise ValueError
            if tuple(sorted(indexes)) != indexes or len(set(indexes)) != len(indexes):
                raise ValueError
            groups.append(indexes)
    except (TypeError, ValueError) as exc:
        raise ReviewError("invalid grouping; use every segment exactly once, e.g. 1,2;3") from exc
    flattened = [index for group in groups for index in group]
    if flattened != list(range(1, segment_count + 1)):
        raise ReviewError("grouping must include every segment exactly once in chronological order")
    return tuple(groups)


def _parse_indexes(value: str, segment_count: int) -> tuple[int, ...]:
    try:
        indexes = tuple(int(item.strip()) for item in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ReviewError("invalid segment selection") from exc
    if not indexes or any(index < 1 or index > segment_count for index in indexes):
        raise ReviewError("segment selection is out of range")
    if len(set(indexes)) != len(indexes) or tuple(sorted(indexes)) != indexes:
        raise ReviewError("segment selection must be unique and chronological")
    return indexes


def _plan_from_indexes(review: ReviewSet, groups: Sequence[Sequence[int]], discarded: Sequence[int] = ()) -> ingest.OperatorResolution:
    identities = [segment.fingerprint_sha256 for segment in review.segments]
    return _make_plan(
        review.segments,
        groups=[[identities[index - 1] for index in group] for group in groups],
        discarded=[identities[index - 1] for index in discarded],
    )


def _preview_plan(review: ReviewSet, plan: ingest.OperatorResolution) -> str:
    by_identity = {segment.fingerprint_sha256: segment for segment in review.segments}
    lines = ["Proposed normalization:"]
    meeting_index = 0
    for group in plan.groups:
        meeting_index += 1
        segments = [by_identity[identity] for identity in group]
        lines.extend([
            "",
            f"  Meeting {meeting_index}",
            f"    Segments: {', '.join(str(review.segments.index(segment) + 1) for segment in segments)}",
            f"    Start: {_display_time(segments[0].created_at)}",
            f"    End: {_display_time(segments[-1].end_at)}",
            f"    Duration: {_display_duration(sum(segment.duration_seconds for segment in segments))}",
        ])
    if plan.discard_source_fingerprints_sha256:
        lines.extend(["", "  Discard:"])
        for identity in plan.discard_source_fingerprints_sha256:
            segment = by_identity[identity]
            lines.append(f"    Segment {review.segments.index(segment) + 1}: {segment.source_path.name} at {_display_time(segment.created_at)}")
    return "\n".join(lines)


def _revalidate(review: ReviewSet, config: ReviewConfig) -> None:
    fetch_ledger = fetch.load_ledger(config.fetch_ledger_path)
    normalized_fingerprints, normalized_paths = _normalized_evidence(config.ingest_root)
    current: list[ReviewSegment] = []
    by_relative = {
        str(segment.source_path.relative_to(config.staging_root.expanduser().resolve())): segment
        for segment in review.segments
    }
    for record in fetch_ledger.get("records", {}).values():
        if not isinstance(record, dict) or record.get("status") != "settled_ready":
            continue
        relative = _safe_relative(record.get("local_relative_path"))
        path = config.staging_root / relative
        if path.parent.resolve() != review.source_parent or ingest.parse_source_timestamp(path.name).date() != review.recording_date:
            continue
        key = str(path.resolve().relative_to(config.staging_root.expanduser().resolve()))
        if key not in by_relative:
            record_fingerprint = record.get("sha256")
            if (
                str(path.resolve()) in normalized_paths
                or isinstance(record_fingerprint, str)
                and record_fingerprint in normalized_fingerprints
            ):
                # A completed or discarded recording in the same date folder is
                # unrelated historical evidence, not a change to this review.
                continue
            raise ReviewError("review source set changed; restart the review")
        current.append(_build_segment(path, record, lambda _path: by_relative[key].duration_seconds))
    current_ids = tuple(segment.fingerprint_sha256 for segment in sorted(current, key=lambda item: item.created_at))
    expected_ids = tuple(segment.fingerprint_sha256 for segment in review.segments)
    if current_ids != expected_ids:
        raise ReviewError("review source set changed; restart the review")
    for segment in review.segments:
        if segment.fingerprint_sha256 in normalized_fingerprints or str(segment.source_path) in normalized_paths:
            raise ReviewError("a reviewed source was normalized or discarded before confirmation")
        _check_regular(segment.source_path)
        if segment.source_path.stat().st_size != segment.size_bytes or ingest.sha256_file(segment.source_path) != segment.fingerprint_sha256:
            raise ReviewError("a reviewed source changed before confirmation")


def _prompt(input_func: Callable[[str], str], text: str) -> str | None:
    try:
        return input_func(text)
    except (EOFError, KeyboardInterrupt, StopIteration):
        return None


def _review_context(review: ReviewSet) -> str:
    start = review.segments[0].created_at.astimezone(ingest.IST).strftime("%Y-%m-%d %H:%M IST")
    return f"Phone recording — {start} — {len(review.segments)} segments"


def _interactive_plan(
    review: ReviewSet,
    input_func: Callable[[str], str],
    *,
    plain_language: bool = False,
) -> ingest.OperatorResolution | None:
    if plain_language:
        return _interactive_plain_plan(review, input_func)

    context = _review_context(review)
    for _ in range(MAX_INPUT_ATTEMPTS):
        choice = _prompt(
            input_func,
            f"\n[a] Accept  [e] Edit  [s] Separate  [d] Discard  [q] Leave unresolved: "
            f"({context})",
        )
        if choice is None or choice == "q":
            return None
        if choice == "a" and review.suggestion is not None:
            print(_preview_plan(review, review.suggestion))
            if _prompt(input_func, "\n[c] Confirm  [e] Edit  [q] Cancel: ") == "c":
                return review.suggestion
            continue
        if choice == "e":
            print("Examples: 1,2,3   1,2;3   1;2;3")
            partition = _prompt(input_func, f"{context}\nGrouping: ")
            if partition is None or partition == "q":
                return None
            try:
                groups = _parse_partition(partition, len(review.segments))
                plan = _plan_from_indexes(review, groups)
            except ReviewError as exc:
                print(f"Invalid grouping: {exc}")
                continue
            print(_preview_plan(review, plan))
            confirmation = _prompt(input_func, f"\n{context}\n[c] Confirm  [e] Edit  [q] Cancel: ")
            if confirmation == "c":
                return plan
            if confirmation == "q" or confirmation is None:
                return None
            continue
        if choice == "s":
            plan = _plan_from_indexes(review, [(index,) for index in range(1, len(review.segments) + 1)])
            print(_preview_plan(review, plan))
            if _prompt(input_func, f"\n{context}\n[c] Confirm  [e] Edit  [q] Cancel: ") == "c":
                return plan
            continue
        if choice == "d":
            selected = _prompt(input_func, f"{context}\nSegments to discard: ")
            if selected is None:
                return None
            try:
                discard_indexes = _parse_indexes(selected, len(review.segments))
                remaining = [index for index in range(1, len(review.segments) + 1) if index not in discard_indexes]
                plan = _plan_from_indexes(review, [(index,) for index in remaining], discard_indexes)
            except ReviewError as exc:
                print(f"Invalid discard selection: {exc}")
                continue
            print(_preview_plan(review, plan))
            if _prompt(input_func, f"{context}\nType DISCARD to confirm: ") == "DISCARD":
                return plan
            return None
        print("Invalid choice; no default action will be selected.")
    return None


def _plain_plan_from_indexes(
    review: ReviewSet,
    groups: Sequence[Sequence[int]],
    discarded: Sequence[int] = (),
) -> ingest.OperatorResolution:
    return _plan_from_indexes(review, groups, discarded)


def _interactive_plain_plan(
    review: ReviewSet,
    input_func: Callable[[str], str],
    *,
    show_evidence: bool = True,
) -> ingest.OperatorResolution | None:
    """Use the operator-contract grammar for the normal inbox route."""
    if show_evidence:
        print(format_review(review, plain_language=True))
    context = _review_context(review)
    for _ in range(MAX_INPUT_ATTEMPTS):
        choice = _prompt(
            input_func,
            f"\n{context}\n"
            "Choose a grouping action:\n"
            "  [1] Use suggested plan\n"
            "  [2] Join all chunks as one meeting\n"
            "  [3] Keep every chunk separate\n"
            "  [4] Enter a custom grouping\n"
            "  [5] Discard selected chunks\n"
            "  [d] Show details again\n"
            "  [q] Leave unchanged\n"
            "Choice: ",
        )
        if choice is None or choice in {"", "q"}:
            return None
        if choice == "d":
            print(format_review(review, plain_language=True))
            continue
        # Undocumented compatibility aliases are accepted only at this first
        # menu; they are never accepted at plan confirmation.
        choice = {"a": "1", "s": "3", "e": "4"}.get(choice, choice)
        plan: ingest.OperatorResolution | None = None
        if choice == "1":
            if review.suggestion is None:
                print("There is no safe suggested plan; choose a different action.")
                continue
            plan = review.suggestion
        elif choice == "2":
            plan = _plain_plan_from_indexes(
                review, [tuple(range(1, len(review.segments) + 1))]
            )
        elif choice == "3":
            plan = _plain_plan_from_indexes(
                review,
                [(index,) for index in range(1, len(review.segments) + 1)],
            )
        elif choice == "4":
            print("Examples: 1,2,3   1,2;3   1;2;3")
            partition = _prompt(input_func, f"{context}\nCustom grouping: ")
            if partition is None or partition in {"", "q"}:
                return None
            if partition == "b":
                continue
            try:
                groups = _parse_partition(partition, len(review.segments))
                plan = _plain_plan_from_indexes(review, groups)
            except ReviewError as exc:
                print(f"Invalid grouping; the review is unchanged: {exc}")
                continue
        elif choice == "5":
            selected = _prompt(input_func, f"{context}\nChunks to discard: ")
            if selected is None or selected in {"", "q"}:
                return None
            try:
                discard_indexes = _parse_indexes(selected, len(review.segments))
                remaining = [
                    index
                    for index in range(1, len(review.segments) + 1)
                    if index not in discard_indexes
                ]
                plan = _plain_plan_from_indexes(
                    review,
                    [(index,) for index in remaining],
                    discard_indexes,
                )
            except ReviewError as exc:
                print(f"Invalid discard selection; the review is unchanged: {exc}")
                continue
        else:
            print("Invalid choice; the review is unchanged.")
            continue

        if plan is None:
            continue
        if plan.discard_source_fingerprints_sha256:
            while True:
                print(_preview_plan(review, plan))
                confirmation = _prompt(
                    input_func,
                    f"\n{context}\nType DISCARD to confirm; [b] Back; [q] Leave unchanged: ",
                )
                if confirmation == "DISCARD":
                    return plan
                if confirmation in {None, "", "q"}:
                    return None
                if confirmation == "b":
                    break
                print("The plan is unchanged; type DISCARD, b, or q.")
            continue

        while True:
            print(_preview_plan(review, plan))
            confirmation = _prompt(
                input_func,
                f"\n{context}\n[y] Confirm this plan  [b] Back and edit  [q] Leave unchanged: ",
            )
            if confirmation == "y":
                return plan
            if confirmation in {None, "", "q"}:
                return None
            if confirmation == "b":
                break
            print("The plan is unchanged; choose y, b, or q.")
        # Back returns to the single grouping menu level.  It never restores
        # the earlier automatic suggestion.
    return None


def _commit_plan(
    review: ReviewSet,
    plan: ingest.OperatorResolution,
    config: ReviewConfig,
    *,
    duration_probe: Callable[[Path], float],
    ingest_runner: Callable[..., ingest.IngestSummary],
    clock: Callable[[], datetime],
) -> ReviewCommitResult:
    _revalidate(review, config)
    before_ledger = ingest.load_ledger(config.ingest_root / ingest.LEDGER_FILENAME)
    before_records = before_ledger.get("records", {})
    started_at = clock()
    with tempfile.TemporaryDirectory(prefix="meetingintel-phone-review-") as temporary:
        plan_path = Path(temporary) / "operator-plan.json"
        plan_path.write_text(json.dumps(_operator_plan_payload(plan), sort_keys=True), encoding="utf-8")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            summary = ingest_runner(
                ingest.IngestConfig(
                    review.source_parent,
                    config.ingest_root,
                    False,
                    plan_path,
                    tuple(segment.source_path for segment in review.segments),
                ),
                duration_probe=duration_probe,
            )
    if summary.failed:
        raise ReviewError("normalization failed; no production processing started")
    after_ledger = ingest.load_ledger(config.ingest_root / ingest.LEDGER_FILENAME)
    after_records = after_ledger.get("records", {})
    if not isinstance(before_records, dict) or not isinstance(after_records, dict):
        raise ReviewError("normalizer ledger is malformed after normalization")

    expected_plan = _operator_plan_payload(plan)
    expected_groups = {
        tuple(group)
        for group in plan.groups
    }
    normalized_sources: list[AudioFirstMeetingSource] = []
    for identity, record in after_records.items():
        if identity in before_records or not isinstance(record, dict):
            continue
        if record.get("status") != "normalized":
            continue
        if record.get("operator_resolution") != expected_plan:
            raise ReviewError("normalized output provenance does not match the confirmed plan")
        try:
            completed_at = datetime.fromisoformat(str(record["ingested_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError("normalized output has invalid completion evidence") from exc
        if completed_at < started_at:
            raise ReviewError("normalized output was not completed by this review invocation")
        try:
            folder = Path(str(record["target_folder"])).expanduser()
            folder_resolved = folder.resolve()
            ingest_root = config.ingest_root.expanduser().resolve()
            folder_resolved.relative_to(ingest_root)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError("normalized output is outside the configured ingest root") from exc
        if folder.is_symlink() or not folder.is_dir():
            raise ReviewError("normalized output folder is not a regular local folder")
        metadata_path = folder_resolved / METADATA_FILENAME
        canonical_audio = folder_resolved / CANONICAL_AUDIO_FILENAME
        if record.get("metadata_path") != str(metadata_path) or record.get("canonical_audio_path") != str(canonical_audio):
            raise ReviewError("normalized output ledger paths do not match its canonical folder")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewError("normalized output metadata is unreadable") from exc
        if metadata.get("operator_resolution") != expected_plan:
            raise ReviewError("normalized metadata provenance does not match the confirmed plan")
        source_ids: tuple[str, ...]
        if metadata.get("audio_handling") == "concat_copy":
            segments = metadata.get("source_segments")
            if not isinstance(segments, list):
                raise ReviewError("grouped normalized output has incomplete source provenance")
            source_ids = tuple(
                str(segment.get("fingerprint_sha256"))
                for segment in segments
                if isinstance(segment, dict)
            )
        else:
            source_ids = (str(metadata.get("source_fingerprint_sha256")),)
        if source_ids not in expected_groups:
            raise ReviewError("normalized output source identities do not match the confirmed plan")
        expected_groups.remove(source_ids)
        try:
            normalized_sources.append(load_audio_first_source(folder_resolved))
        except (OSError, ValueError) as exc:
            raise ReviewError("normalized output failed audio-first source validation") from exc

    if len(normalized_sources) != summary.ingested or expected_groups:
        raise ReviewError("normalizer completion evidence does not match normalized outputs")
    for source in normalized_sources:
        from meetingintel_pipeline import build_audio_first_fingerprint
        observe.emit("phone.normalized", source_id=build_audio_first_fingerprint(source.source_identity),
                     source_kind="phone_recording", stage="normalization", outcome="success",
                     details={"source_refs": [source.source_identity, source.source_fingerprint]})
    return ReviewCommitResult(summary, tuple(sorted(normalized_sources, key=lambda source: source.created_at)))


def _process_now_prompt(
    sources: Sequence[AudioFirstMeetingSource],
    input_func: Callable[[str], str],
) -> str | None:
    count = len(sources)
    noun = "meeting" if count == 1 else "meetings"
    print(f"\nProcess exactly this {noun} now?" if count == 1 else f"\nProcess exactly these {count} meetings now?")
    print("  [y] Yes")
    print("  [l] List exact meeting(s)")
    print("  [n] Not now")
    invalid = 0
    while invalid < MAX_INPUT_ATTEMPTS:
        answer = _prompt(input_func, "Choice: ")
        if answer is None or answer in {"", "n"}:
            return answer
        if answer == "y":
            return answer
        if answer == "l":
            for source in sources:
                created = source.created_at.strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {created} IST | phone | {source.source_filename}")
            continue
        invalid += 1
        if invalid < MAX_INPUT_ATTEMPTS:
            print("Invalid choice; no processing will be started by default.")
    return None


def _run_review_sets(
    reviews: Sequence[ReviewSet],
    *,
    config: ReviewConfig,
    input_func: Callable[[str], str],
    terminal_isatty: Callable[[], bool],
    dry_run: bool,
    duration_probe: Callable[[Path], float],
    ingest_runner: Callable[..., ingest.IngestSummary],
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None,
    clock: Callable[[], datetime],
    plain_language: bool = False,
) -> int:
    if not reviews:
        print("No settled phone recordings require review.")
        return 0
    for review in reviews:
        # Evidence is always visible before the first grouping decision,
        # including the single-item inbox route.
        print(format_review(review, plain_language=plain_language))
    if dry_run:
        return 0
    if not terminal_isatty():
        print("Non-interactive terminal; no changes made.")
        return 0
    normalized = 0
    discarded = 0
    normalized_sources: list[AudioFirstMeetingSource] = []
    for review in reviews:
        plan = (
            _interactive_plain_plan(review, input_func, show_evidence=False)
            if plain_language
            else _interactive_plan(review, input_func)
        )
        if plan is None:
            continue
        try:
            result = _commit_plan(
                review,
                plan,
                config,
                duration_probe=duration_probe,
                ingest_runner=ingest_runner,
                clock=clock,
            )
        except (ReviewError, OSError, ValueError) as exc:
            print(f"Phone review stopped safely: {exc}")
            return 1
        normalized += result.summary.ingested
        discarded += result.summary.discarded
        normalized_sources.extend(result.normalized_sources)
    if normalized or discarded:
        print(f"Normalized: {normalized} phone meetings")
        if discarded:
            print(f"Discarded: {discarded} phone recordings")
        print("Original staged files: retained")
        print("Production processing: not started")
        if normalized_sources and process_runner is not None:
            choice = _process_now_prompt(normalized_sources, input_func)
            if choice == "y":
                try:
                    processing_code = process_runner(tuple(normalized_sources))
                except Exception:
                    print("Normalization succeeded.")
                    print("Processing failed: exact-scoped production execution error")
                    print("Normalized source remains available.")
                    print("\nRecovery:\n  mi --audio-first new list\n  mi --audio-first new")
                    return 1
                if processing_code:
                    print("Normalization succeeded.")
                    print("Processing failed: see the safe categorized error above")
                    print("Normalized source remains available.")
                    print("\nRecovery:\n  mi --audio-first new list\n  mi --audio-first new")
                    return 1
                print("Processing completed for the exact normalized set.")
            else:
                print("Processing not started.")
        print("\nNext:\n  mi --audio-first new list\n  mi --audio-first new")
    return 0


def run_review(
    scope: str = "today",
    *,
    config: ReviewConfig = ReviewConfig(),
    clock: Callable[[], datetime] = ingest.now_ist,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: False,
    dry_run: bool = False,
    duration_probe: Callable[[Path], float] = ingest.probe_duration,
    ingest_runner: Callable[..., ingest.IngestSummary] = ingest.run_ingest,
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None = None,
) -> int:
    if scope not in {"today", "last", "new", "all"}:
        raise ValueError(f"unsupported review scope: {scope}")
    reviews = list(collect_review_sets(config, duration_probe=duration_probe))
    if scope == "today":
        today = clock().astimezone(ingest.IST).date()
        reviews = [review for review in reviews if review.recording_date == today]
    elif scope == "last" and reviews:
        latest = max(review.recording_date for review in reviews)
        reviews = [review for review in reviews if review.recording_date == latest]
    return _run_review_sets(
        reviews,
        config=config,
        input_func=input_func,
        terminal_isatty=terminal_isatty,
        dry_run=dry_run,
        duration_probe=duration_probe,
        ingest_runner=ingest_runner,
        process_runner=process_runner,
        clock=clock,
    )


def run_selected_review(
    review: ReviewSet,
    *,
    config: ReviewConfig = ReviewConfig(),
    clock: Callable[[], datetime] = ingest.now_ist,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: False,
    dry_run: bool = False,
    duration_probe: Callable[[Path], float] = ingest.probe_duration,
    ingest_runner: Callable[..., ingest.IngestSummary] = ingest.run_ingest,
    process_runner: Callable[[Sequence[AudioFirstMeetingSource]], int] | None = None,
) -> int:
    """Review exactly one already-selected set through the calm inbox path."""
    return _run_review_sets(
        (review,),
        config=config,
        input_func=input_func,
        terminal_isatty=terminal_isatty,
        dry_run=dry_run,
        duration_probe=duration_probe,
        ingest_runner=ingest_runner,
        process_runner=process_runner,
        clock=clock,
        plain_language=True,
    )


def parse_review_args(argv: Sequence[str]) -> tuple[str, bool]:
    allowed = {"today", "last", "new", "all"}
    values = [value for value in argv if value != "--dry-run"]
    if len(values) > 1 or (values and values[0] not in allowed):
        raise ValueError("use mi phone review [today|last|new|all] [--dry-run]")
    return (values[0] if values else "today", "--dry-run" in argv)
