#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "normalized_meeting_artifact_v1"
TRUTH_LAYER_ENGINE = "whispermlx"
SPEAKER_ATTRIBUTION_WARNING = (
    "Speaker labels are evidence, not absolute truth. Use them to understand turn-taking "
    "and likely ownership, but do not confidently assign actions, commitments, or sensitive "
    "claims if the wording is ambiguous. Prefer “likely SPEAKER_00”, “likely SPEAKER_01”, "
    "or “unclear speaker” when attribution is uncertain."
)
SUPPORTED_OUTPUT_SUFFIXES = {".json", ".srt", ".txt", ".tsv", ".vtt"}


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or "meeting"


def new_artifact_id() -> str:
    return uuid.uuid4().hex[:10]


def meeting_id_from_fields(meeting_date_ist: str, meeting_time_ist: str, meeting_name: str) -> str:
    hhmm = meeting_time_ist.replace(":", "")[:4]
    return f"{meeting_date_ist}__{hhmm}__{slugify(meeting_name)}"


def artifact_dir_name(meeting_date_ist: str, meeting_time_ist: str, meeting_name: str, artifact_id: str) -> str:
    hhmm = meeting_time_ist.replace(":", "")[:4]
    return f"{meeting_date_ist}__{hhmm}__{slugify(meeting_name)}__{artifact_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_single_file(output_dir: Path, suffix: str) -> Path | None:
    matches = sorted(path for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == suffix)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one {suffix} file in {output_dir}, found {len(matches)}")
    return matches[0]


def collect_source_files(output_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
            continue
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "suffix": suffix,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def normalize_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(payload.get("segments") or [], start=1):
        text = (segment.get("text") or "").strip()
        normalized.append(
            {
                "segment_id": segment.get("id", index),
                "start_seconds": segment.get("start"),
                "end_seconds": segment.get("end"),
                "speaker_label": segment.get("speaker") or "UNCLEAR_SPEAKER",
                "speaker_name": None,
                "text": text,
            }
        )
    return normalized


def final_speaker_labels(segments: list[dict[str, Any]]) -> list[str]:
    labels = sorted(
        {
            segment["speaker_label"]
            for segment in segments
            if segment.get("speaker_label", "").startswith("SPEAKER_")
        }
    )
    return labels


def unclear_segment_count(segments: list[dict[str, Any]]) -> int:
    return sum(1 for segment in segments if segment.get("speaker_label") == "UNCLEAR_SPEAKER")


def seconds_to_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_quality_warnings(
    speaker_count_mode: str,
    detected_speakers_before_forcing: int | None,
    final_speaker_count: int,
    unclear_segments: int,
) -> list[str]:
    warnings = [SPEAKER_ATTRIBUTION_WARNING]
    if speaker_count_mode.startswith("forced"):
        warnings.append(
            f"Speaker count mode was {speaker_count_mode}; diarization labels were constrained rather than left fully open."
        )
    if detected_speakers_before_forcing is not None:
        warnings.append(
            "Detected speakers before forcing was "
            f"{detected_speakers_before_forcing}; final labels in this artifact total {final_speaker_count}."
        )
    if unclear_segments:
        warnings.append(
            f"{unclear_segments} segment(s) had no diarization label and were preserved as UNCLEAR_SPEAKER."
        )
    if final_speaker_count == 0:
        warnings.append("No speaker labels were found in the WhisperMLX segments.")
    return warnings


def infer_confidence_label(final_speaker_count: int, has_segments: bool) -> str:
    if not has_segments:
        return "low"
    if final_speaker_count >= 2:
        return "medium"
    if final_speaker_count == 1:
        return "low"
    return "low"


def render_transcript_markdown(
    meeting_name: str,
    meeting_date_ist: str,
    meeting_time_ist: str,
    source_type: str,
    quality_note: str,
    segments: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {meeting_name}",
        "",
        f"- Date (IST): {meeting_date_ist}",
        f"- Time (IST): {meeting_time_ist}",
        f"- Source type: {source_type}",
        f"- Diarization quality note: {quality_note}",
        f"- Speaker attribution warning: {SPEAKER_ATTRIBUTION_WARNING}",
        "",
        "## Speaker-Labeled Transcript",
        "",
    ]
    for segment in segments:
        label = segment.get("speaker_label") or "UNKNOWN_SPEAKER"
        start = seconds_to_timestamp(segment.get("start_seconds"))
        end = seconds_to_timestamp(segment.get("end_seconds"))
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{start} - {end}] {label}: {text}")
    return "\n".join(lines).strip() + "\n"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
