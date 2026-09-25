#!/usr/bin/env python3
"""Explicit owner-only Voice ID operator surface."""

from __future__ import annotations
from mi_paths import public_path

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from voiceprint_gate5_inbox import Gate5Error, Gate5Inbox
from voiceprint_gate5_preview import PreviewError, run_preview
from voiceprint_provider_pyannote import ProviderError


PROJECT_ROOT = Path(str(public_path('project')))
PHONE_ROOT = public_path("project/ingest/phone")
DIARIZATION_ROOT = Path(str(public_path('work/diarization')))
STATE_ROOT = Path(str(public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate5')))
GATE4_REPORT = Path(str(public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate4/voiceprint_owner_bakeoff_report.json')))
MODEL_ASSET = Path(str(public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate4/models/community1-embedding.bin')))


class VoiceCommandError(ValueError):
    """An explicit Voice ID command could not proceed safely."""


def _inbox() -> Gate5Inbox:
    return Gate5Inbox(
        STATE_ROOT,
        clock=lambda: datetime.now(timezone.utc),
        gate4_report=GATE4_REPORT,
        model_asset=MODEL_ASSET,
    )


def _single_retained_identity(inbox: Gate5Inbox) -> tuple[str, str]:
    records = [record for record in inbox.store.all_records() if record.state == "keep"]
    if len(records) != 1:
        raise VoiceCommandError("Voice ID requires exactly one retained owner enrollment")
    identity = records[0].identity_id
    name = inbox.private_display_name(identity)
    if not name:
        raise VoiceCommandError("the retained owner enrollment has no private display name")
    return identity, name


def _latest_successful_phone_diarization() -> tuple[Path, Path, str]:
    candidates: list[tuple[datetime, Path, Path, str]] = []
    if not DIARIZATION_ROOT.is_dir():
        raise VoiceCommandError("no completed diarization root is available")
    phone_root = PHONE_ROOT.resolve(strict=True)
    for manifest_path in DIARIZATION_ROOT.glob("*/run_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") != "success" or payload.get("source_mode") != "audio_first":
                continue
            source_identity = payload.get("source_identity")
            source_folder_text = payload.get("source_folder")
            completed_text = payload.get("completed_at")
            if not all(isinstance(value, str) and value for value in (source_identity, source_folder_text, completed_text)):
                continue
            source_folder = Path(source_folder_text).expanduser().resolve(strict=True)
            if source_folder.parent != phone_root:
                continue
            completed = datetime.fromisoformat(completed_text)
            if completed.tzinfo is None:
                continue
            candidates.append((completed, source_folder, manifest_path.parent, source_identity))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise VoiceCommandError("no successful supported phone diarization is available")
    _completed, source, diarization, identity = max(candidates, key=lambda row: row[0])
    return source, diarization, identity


def _pending_candidate(inbox: Gate5Inbox, meeting_id: str, speaker: str) -> str:
    metadata = inbox._read_metadata()
    matches = [
        candidate_id
        for candidate_id, value in metadata["reviews"].items()
        if value.get("meeting_id") == meeting_id
        and value.get("observed_speaker") == speaker
        and value.get("status") == "candidate"
        and value.get("consumed") is not True
    ]
    if len(matches) != 1:
        raise VoiceCommandError("Voice ID did not issue exactly one confirmable owner candidate")
    return matches[0]


def run_status() -> int:
    inbox = _inbox()
    records = inbox.store.all_records()
    print("MeetingIntel Voice ID")
    print(f"Retained owner enrollments: {sum(record.state == 'keep' for record in records)}")
    print(f"Pending enrollments: {sum(record.state == 'pending' for record in records)}")
    print(f"Confirmed meeting mappings: {len(inbox.trusted_map())}")
    print("Mode: local, owner-only, explicit review")
    return 0


def run_review_last(
    *,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
) -> int:
    if not terminal_isatty():
        raise VoiceCommandError("Voice ID review requires an interactive terminal")
    inbox = _inbox()
    identity_id, display_name = _single_retained_identity(inbox)
    source, diarization, source_identity = _latest_successful_phone_diarization()
    print("Selected: latest successful supported phone meeting")
    if input_func("Use this meeting's local audio for Voice ID? Type VOICE: ") != "VOICE":
        print("Left unchanged.")
        return 0
    meeting_id = f"meeting-{source_identity[:12]}"
    result = run_preview(
        state_root=STATE_ROOT,
        gate4_report=GATE4_REPORT,
        model_asset=MODEL_ASSET,
        source_folder=source,
        diarization_folder=diarization,
        identity_id=identity_id,
        meeting_id=meeting_id,
        consent="yes",
    )
    named_label = result.get("named_label")
    for label, row in sorted(result["scores"].items()):
        print(f"{label}: {row['status']} (score {row['score']:.4f})")
    if not isinstance(named_label, str):
        print("No unique owner match. Speakers remain anonymous.")
        return 0
    if input_func(f"Confirm {named_label} is {display_name}? Type YES: ") != "YES":
        print("Candidate not confirmed. Speakers remain anonymous.")
        return 0
    candidate_id = _pending_candidate(inbox, meeting_id, named_label)
    inbox.confirm_candidate(candidate_id, operator_confirmed=True, display_name=display_name)
    print(f"Confirmed for this meeting: {named_label} is {display_name}")
    print("Normal transcript, prompts, ledger, and brief remain unchanged.")
    return 0


def run_voice(
    argv: Sequence[str],
    *,
    input_func: Callable[[str], str] = input,
    terminal_isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
) -> int:
    parser = argparse.ArgumentParser(prog="mi voice", description="Explicit local owner Voice ID")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status")
    review = subparsers.add_parser("review")
    review.add_argument("scope", choices=("last",))
    args = parser.parse_args(list(argv))
    try:
        if args.action == "status":
            return run_status()
        return run_review_last(input_func=input_func, terminal_isatty=terminal_isatty)
    except (Gate5Error, PreviewError, ProviderError, VoiceCommandError, OSError, ValueError) as exc:
        print(f"Voice ID stopped safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_voice(sys.argv[1:]))
