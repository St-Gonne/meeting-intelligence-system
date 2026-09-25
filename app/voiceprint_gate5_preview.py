"""Explicit private Gate 5 presentation preview for one existing diarization.

This command is intentionally outside the normal ``mi`` processing lane.  It
reuses a completed isolated diarization, builds short private speaker tracks,
and records only a measured voice-match suggestion.  It never edits the
source, production ledger, normal output, or Gate 5 confirmation map.
"""

from __future__ import annotations
from mi_paths import public_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

from audio_first_meeting_source import load_audio_first_source
from voiceprint_gate5_audio import review_audio
from voiceprint_gate5_inbox import Gate5Error, Gate5Inbox
from voiceprint_provider_pyannote import ProviderError


LABEL_RE = re.compile(r"SPEAKER_\d+")
MAX_TRACK_SECONDS = 180.0
MIN_TRACK_SECONDS = 30.0


class PreviewError(Gate5Error):
    """The explicit preview inputs or output contract are unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_only_file(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise PreviewError(f"{label} must be an explicit path")
    lexical = path.expanduser().absolute()
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor not in {Path("/tmp"), Path("/var"), Path("/etc")} and cursor.exists() and cursor.is_symlink():
            raise PreviewError(f"{label} path contains a symlink")
    if lexical.is_symlink() or not lexical.is_file():
        raise PreviewError(f"{label} is missing or unsafe")
    resolved = lexical.resolve(strict=True)
    if resolved.stat().st_uid != os.getuid():
        raise PreviewError(f"{label} is not user-owned")
    lowered = str(resolved).casefold()
    if any(token in lowered for token in ("cloudstorage", "google drive", "staging/phone-drive")):
        raise PreviewError(f"{label} must not be cloud-synced or staging storage")
    return resolved


def _validated_source_audio(path: Path) -> Path:
    """Permit only the canonical path obtained from validated source metadata.

    Gate5's general audio admission rejects repository paths.  This one narrow
    exception is for the already-normalized source selected by
    ``load_audio_first_source``; it is copied into a 0600 disposable file
    before inference and is never passed to the production pipeline.
    """
    resolved = _read_only_file(path, "selected canonical audio")
    controlled_root = public_path("project/ingest/phone")
    try:
        resolved.relative_to(controlled_root.resolve())
    except ValueError as exc:
        raise PreviewError("selected audio is outside the controlled ingest tree") from exc
    return resolved


def _load_diarization(folder: Path, source_identity: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    if folder.expanduser().absolute().is_symlink() or not folder.is_dir():
        raise PreviewError("diarization folder is missing or unsafe")
    manifest_path = _read_only_file(folder / "run_manifest.json", "diarization manifest")
    audio_json = _read_only_file(folder / "audio.json", "diarization JSON")
    srt_path = _read_only_file(folder / "audio.srt", "diarized SRT")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(audio_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreviewError("diarization artifacts are malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "success":
        raise PreviewError("diarization manifest is not successful")
    if manifest.get("source_identity") != source_identity or manifest.get("source_mode") != "audio_first":
        raise PreviewError("diarization source lineage does not match the selected source")
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        raise PreviewError("diarization JSON has no segments")
    valid: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("speaker"), str):
            continue
        label = segment["speaker"]
        if LABEL_RE.fullmatch(label) is None:
            continue
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        valid.append({"speaker": label, "start": start, "end": end})
    if not valid:
        raise PreviewError("diarization JSON has no valid speaker segments")
    return manifest, valid, audio_json, srt_path


def _make_track(source_audio: Path, segments: list[dict[str, Any]], output: Path) -> float:
    """Extract one bounded label track without modifying the source audio."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreviewError("ffmpeg is unavailable")
    selected: list[dict[str, Any]] = []
    total = 0.0
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if duration <= 0:
            continue
        selected.append(segment)
        total += duration
        if total >= MAX_TRACK_SECONDS:
            break
    if total < MIN_TRACK_SECONDS:
        raise PreviewError("speaker track is too short for a measured voice match")
    with tempfile.TemporaryDirectory(prefix="mi-g5-track-") as work:
        work_path = Path(work)
        files: list[Path] = []
        for index, segment in enumerate(selected):
            part = work_path / f"part-{index:03d}.wav"
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-ss", str(segment["start"]),
                 "-to", str(segment["end"]), "-i", str(source_audio), "-ac", "1",
                 "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(part)],
                check=False, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 or not part.is_file() or part.stat().st_size == 0:
                raise PreviewError("speaker track extraction failed")
            os.chmod(part, 0o600)
            files.append(part)
        concat = work_path / "concat.txt"
        concat.write_text("".join(f"file '{part}'\n" for part in files), encoding="utf-8")
        os.chmod(concat, 0o600)
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(concat), "-c:a", "pcm_s16le", "-y", str(output)],
            check=False, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise PreviewError("speaker track concatenation failed")
    os.chmod(output, 0o600)
    return min(total, MAX_TRACK_SECONDS)


def _render_preview(srt_path: Path, *, named_label: str | None, display_name: str, scores: dict[str, dict[str, Any]], source_date: str, source_duration: float) -> str:
    text = srt_path.read_text(encoding="utf-8")
    if named_label is not None:
        replacement = f"[{named_label} — {display_name} (voice-match suggestion; THIS CALL not human-confirmed)]"
        text = re.sub(rf"\[{re.escape(named_label)}\]", replacement, text)
    score_lines = []
    for label in sorted(scores):
        row = scores[label]
        score_lines.append(f"- {label}: {row['score']:.6f} — {row['status']} (Gate 5 voice-match measurement)")
    if named_label is None:
        decision = "No label was named: the measured result was not uniquely candidate-level."
    else:
        decision = f"Measured suggestion: {named_label} may be {display_name}. This call has NOT been human-confirmed."
    header = (
        "# Private Gate 5 named transcript preview\n\n"
        f"Source date: {source_date}\n"
        f"Source duration: {source_duration:.3f} seconds\n"
        "Diarization bound: 2 speakers (processing bound only; not a claim about\n"
        "the call's actual speaker count).\n"
        "Identity status: retained identity used for a measured suggestion only; "
        "this call remains unconfirmed.\n\n"
        "## Voice-match measurements\n" + "\n".join(score_lines) + "\n\n"
        f"{decision}\n\n"
        "## Diarized transcript (presentation-only decoration)\n\n"
    )
    return header + text + ("\n" if not text.endswith("\n") else "")


def run_preview(*, state_root: Path, gate4_report: Path, model_asset: Path, source_folder: Path, diarization_folder: Path, identity_id: str, meeting_id: str, consent: str, output_root: Path | None = None) -> dict[str, Any]:
    if consent != "yes":
        raise PreviewError("explicit audio evaluation consent is required")
    source = load_audio_first_source(source_folder.expanduser().absolute())
    manifest, segments, audio_json, srt_path = _load_diarization(diarization_folder.expanduser().absolute(), source.source_identity)
    inbox = Gate5Inbox(state_root, clock=lambda: datetime.now(timezone.utc), gate4_report=gate4_report, model_asset=model_asset)
    display_name = inbox.private_display_name(identity_id)
    if not display_name:
        raise PreviewError("retained identity has no private display name")
    labels = sorted({segment["speaker"] for segment in segments})
    out_root = output_root or (inbox.root / "previews")
    out_root = out_root.expanduser().absolute()
    try:
        out_root.relative_to(inbox.root)
    except ValueError as exc:
        raise PreviewError("preview output must remain below the private Gate 5 root") from exc
    out_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_root, 0o700)
    with tempfile.TemporaryDirectory(prefix="mi-g5-preview-") as work:
        work_path = Path(work)
        private_audio = work_path / "source-audio.m4a"
        shutil.copyfile(_validated_source_audio(source.canonical_audio_path), private_audio)
        os.chmod(private_audio, 0o600)
        scores: dict[str, dict[str, Any]] = {}
        for label in labels:
            track = work_path / f"{label}.wav"
            track_seconds = _make_track(private_audio, [segment for segment in segments if segment["speaker"] == label], track)
            candidate = review_audio(inbox, label, meeting_id=meeting_id, model_asset=model_asset, audio=track, consent="yes")
            scores[label] = {"score": candidate.score, "status": candidate.status, "candidate_id": candidate.candidate_id, "track_seconds": track_seconds}
    candidate_labels = [label for label, row in scores.items() if row["status"] == "candidate"]
    named_label = candidate_labels[0] if len(candidate_labels) == 1 else None
    preview_path = out_root / f"named_preview_{source.created_at.date().isoformat()}_{source.source_identity[:12]}.md"
    receipt_path = out_root / f"named_preview_{source.created_at.date().isoformat()}_{source.source_identity[:12]}.json"
    preview_path.write_text(_render_preview(srt_path, named_label=named_label, display_name=display_name, scores=scores, source_date=source.created_at.isoformat(), source_duration=source.duration_seconds), encoding="utf-8")
    os.chmod(preview_path, 0o600)
    receipt = {
        "schema_version": 1,
        "source_identity": source.source_identity,
        "source_date": source.created_at.isoformat(),
        "duration_seconds": source.duration_seconds,
        "diarization_manifest_sha256": _sha256(_read_only_file(diarization_folder / "run_manifest.json", "diarization manifest")),
        "diarization_audio_json_sha256": _sha256(audio_json),
        "diarization_model": manifest.get("model"),
        "diarization_min_speakers": manifest.get("min_speakers"),
        "diarization_max_speakers": manifest.get("max_speakers"),
        "observed_speaker_labels": labels,
        "gate4_provider": inbox.provenance.provider_id,
        "gate4_threshold": inbox.provenance.threshold,
        "gate4_aggregation_recipe": inbox.provenance.aggregation_recipe,
        "identity_id": identity_id,
        "suggested_display_name": display_name,
        "meeting_id": meeting_id,
        "human_confirmed_this_call": False,
        "named_label": named_label,
        "scores": {label: {"score": row["score"], "status": row["status"], "track_seconds": row["track_seconds"]} for label, row in scores.items()},
        "preview_file": preview_path.name,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(receipt_path, 0o600)
    return {"preview": str(preview_path), "receipt": str(receipt_path), "named_label": named_label, "scores": receipt["scores"], "human_confirmed_this_call": False}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Explicit private Gate 5 named transcript preview")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--gate4-report", type=Path, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--diarization-folder", type=Path, required=True)
    parser.add_argument("--identity-id", required=True)
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--consent", choices=("yes", "no"), required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run_preview(state_root=args.state_root, gate4_report=args.gate4_report, model_asset=args.model_asset, source_folder=args.source_folder, diarization_folder=args.diarization_folder, identity_id=args.identity_id, meeting_id=args.meeting_id, consent=args.consent, output_root=args.output_root), sort_keys=True))
        return 0
    except (Gate5Error, ProviderError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"GATE5_PREVIEW_REJECTED={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
