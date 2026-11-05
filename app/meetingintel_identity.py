"""Owner-only automatic voice matching and post-model transcript presentation.

The lightweight parent/UI functions never load the embedding runtime. The
worker uses the existing pinned local provider and retained owner enrollment.
Machine matches are stored separately from the human-confirmed Gate 5 map.
"""
from __future__ import annotations
from mi_paths import public_path

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import uuid
from meetingintel_audio_tools import audio_tool_environment, AudioToolUnavailable

ROOT = Path(__file__).resolve().parent


@contextmanager
def _embedding_audio_environment():
    """Make the existing local audio tool visible to the frozen decoder only.

    Track extraction already has this fallback, but the embedding provider does
    its own PATH lookup. Scope the fallback to this worker's embedding call;
    preserve the provider bytes, service environment and current-tool precedence.
    """
    try:
        with audio_tool_environment(("ffmpeg",)):
            yield
    except AudioToolUnavailable as error:
        from voiceprint_gate5_preview import PreviewError
        raise PreviewError(str(error)) from error


STATE_ROOT = public_path("home/Library/Application Support/MeetingIntel/voiceprint-gate5")
GATE4_ROOT = STATE_ROOT.parent / "voiceprint-gate4"
PYTHON = public_path("home/whispermlx-env/bin/python")
MIN_SPEECH = 40.0
MAX_SPEECH = 180.0
POLICY_FILE = "automatic_owner.json"
FINGERPRINT = re.compile(r"[a-f0-9]{64}")
REASONS = {"below_threshold", "multiple_candidates", "insufficient_speech",
           "provider_mismatch", "no_enrollment", "disabled", "revoked",
           "unavailable", "matched"}
REASONS.update({"audio_extraction_failed", "invalid_stage"})
WORKER_FAILURE_REASONS = {"worker_launch_failed", "worker_timeout", "worker_terminated",
                          "worker_exit_failed", "worker_output_invalid", "worker_exception"}
REASONS.update(WORKER_FAILURE_REASONS)
# Exact, fixed messages only. Never persist exception text, paths or stderr.
# These codes travel through the existing projection and run-health reason.
PREVIEW_FAILURE_REASONS = {
    "ffmpeg is unavailable": "ffmpeg_unavailable",
    "ffprobe is unavailable": "ffprobe_unavailable",
    "speaker track is too short for a measured voice match": "speaker_track_too_short",
    "speaker track extraction failed": "audio_extraction_failed",
    "speaker track concatenation failed": "audio_concatenation_failed",
    "diarization manifest is not successful": "stage_not_successful",
    "diarization source lineage does not match the selected source": "stage_lineage_mismatch",
    "diarization artifacts are malformed": "stage_artifacts_malformed",
    "diarization JSON has no segments": "stage_segments_missing",
    "diarization JSON has no valid speaker segments": "stage_segments_invalid",
    "diarization folder is missing or unsafe": "stage_folder_unsafe",
}
for _label in ("diarization manifest", "diarization JSON", "diarized SRT"):
    for _suffix, _reason in (
        ("must be an explicit path", "stage_artifact_missing_or_unsafe"),
        ("is missing or unsafe", "stage_artifact_missing_or_unsafe"),
        ("path contains a symlink", "stage_artifact_symlink"),
        ("is not user-owned", "stage_artifact_not_owned"),
        ("must not be cloud-synced or staging storage", "stage_artifact_disallowed_location"),
    ):
        PREVIEW_FAILURE_REASONS[f"{_label} {_suffix}"] = _reason
REASONS.update(PREVIEW_FAILURE_REASONS.values())


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path):
    lexical = Path(path).expanduser().absolute()
    for part in (lexical, *lexical.parents):
        if part not in {Path("/tmp"), Path("/var"), Path("/etc")} and part.is_symlink():
            raise ValueError("unsafe identity path")
    return lexical


def _json(path):
    path = _safe_path(path)
    if not path.is_file() or path.stat().st_size > 65536:
        return {}
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("identity metadata must be private")
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _atomic(path, value):
    path = _safe_path(path)
    _safe_path(path.parent).mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".identity-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def enabled(state_root=None):
    try:
        policy = _json(Path(state_root or STATE_ROOT) / POLICY_FILE)
        return policy.get("schema_version") == 1 and policy.get("enabled") is True
    except (OSError, ValueError, TypeError):
        return False


def set_enabled(value, *, state_root=None):
    root = Path(state_root or STATE_ROOT)
    if value:
        inbox, records, _model = _owner(root)
        if len(records) != 1 or not inbox.private_display_name(records[0].identity_id):
            raise ValueError("automatic matching requires one retained owner enrollment")
    _atomic(root / POLICY_FILE, {"schema_version": 1, "enabled": bool(value),
            "method": "auto_match_v1", "updated_at": datetime.now(timezone.utc).isoformat()})


def clean_ranges(segments, label):
    """Union a label's intervals, then remove every other label's overlap."""
    own = sorted((float(s["start"]), float(s["end"])) for s in segments if s["speaker"] == label)
    merged = []
    for start, end in own:
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError("invalid speaker interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    others = [(float(s["start"]), float(s["end"])) for s in segments if s["speaker"] != label]
    result = []
    for start, end in merged:
        pieces = [(start, end)]
        for left, right in others:
            remaining = []
            for a, b in pieces:
                if right <= a or left >= b:
                    remaining.append((a, b))
                else:
                    if left > a:
                        remaining.append((a, left))
                    if right < b:
                        remaining.append((right, b))
            pieces = remaining
        result.extend((a, b) for a, b in pieces if b - a >= 0.5)
    return result


def decide(rows, threshold):
    """Use raw threshold scores; exclusive assignment cannot hide ambiguity."""
    eligible = [r for r in rows if r["eligible_seconds"] >= MIN_SPEECH]
    candidates = [r for r in rows if r.get("score") is not None and
                  math.isfinite(r["score"]) and r["score"] >= threshold]
    if len(candidates) == 1:
        if candidates[0]["eligible_seconds"] >= MIN_SPEECH:
            return candidates[0]["speaker_label"], "matched"
        return None, "insufficient_speech"
    if len(candidates) > 1:
        return None, "multiple_candidates"
    return None, "below_threshold" if eligible else "insufficient_speech"


def bounded_track_segments(ranges):
    """Respect the existing half-second fragment floor at the crop boundary."""
    remaining, selected = MAX_SPEECH, []
    for start, end in ranges:
        if remaining < 0.5:
            break
        stop = min(end, start + remaining)
        duration = stop - start
        if duration >= 0.5:
            selected.append({"start": start, "end": stop})
            remaining -= duration
    return selected


def render(srt, name=None, label=None, reason="below_threshold"):
    if reason not in REASONS:
        raise ValueError("invalid identity reason")
    if label is not None:
        if not re.fullmatch(r"SPEAKER_\d+", label) or not isinstance(name, str) or not re.fullmatch(r"[^\x00-\x1f\x7f]{1,80}", name):
            raise ValueError("invalid identity presentation")
        # Replace only anchored SRT speaker prefixes, never spoken text.
        srt = re.sub(rf"(?m)^\[{re.escape(label)}\](?=:|\s|$)",
                     lambda _m: f"[{label} — {name}]", srt)
        header = "Owner voice recognized automatically; this meeting was not human-confirmed."
    else:
        header = "Speakers remain anonymous: " + reason.replace("_", " ") + "."
    return "# Meeting transcript\n\n" + header + "\n\n" + srt.rstrip() + "\n"


def _owner(state_root):
    from voiceprint_gate5_inbox import Gate5Inbox
    gate4 = GATE4_ROOT if Path(state_root) == STATE_ROOT else Path(state_root).parent / "voiceprint-gate4"
    model = gate4 / "models/community1-embedding.bin"
    inbox = Gate5Inbox(Path(state_root), clock=lambda: datetime.now(timezone.utc),
                      gate4_report=gate4 / "voiceprint_owner_bakeoff_report.json", model_asset=model,
                      maintain_state=False)
    records = [r for r in inbox.store._read_records().values() if r.state == "keep"]
    return inbox, records, model


def _source(folder):
    from audio_first_meeting_source import load_audio_first_source
    from laptop_capture_meeting_source import load_laptop_meeting_source
    folder = _safe_path(folder).resolve(strict=True)
    if folder.parent == ROOT / "ingest/phone":
        return load_audio_first_source(folder)
    if folder.parent == ROOT / "ingest/laptop":
        return load_laptop_meeting_source(folder)
    raise ValueError("identity source is outside the admitted lanes")


def anonymous_presentation(stage, output_root, fingerprint, reason="unavailable", *, revoked=False):
    """Publish an anonymous version after failure or revocation; retain history."""
    if FINGERPRINT.fullmatch(fingerprint) is None or reason not in REASONS:
        raise ValueError("invalid anonymous presentation")
    srt = _safe_path(Path(stage) / "audio.srt")
    if not srt.is_file() or srt.stat().st_size > 10_000_000:
        return {"status": "unavailable", "reason": reason}
    directory = _safe_path(Path(output_root) / "transcripts/identity")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"{fingerprint}-{uuid.uuid4().hex}-anonymous.md"
    path.write_text(render(srt.read_text(encoding="utf-8"), reason=reason), encoding="utf-8")
    path.chmod(0o600)
    digest = _hash(path)
    status = "revoked" if revoked else "anonymous"
    _atomic(directory / (fingerprint + ".json"), {
        "schema_version": 1, "fingerprint": fingerprint, "method": "auto_match_v1",
        "status": status, "reason": reason, "presentation_path": str(path),
        "presentation_sha256": digest, "anonymous_path": str(path), "anonymous_sha256": digest,
        "enrollment_state_sha256": None, "enrollment_metadata_sha256": None})
    return {"status": status, "reason": reason, "presentation_path": str(path)}


def match_and_present(source_folder, diarization_folder, output_root, fingerprint, *, state_root=None):
    """Executed in the pinned local runtime; no names returned to the model path."""
    from voiceprint_gate5_preview import _load_diarization, _make_track
    from voiceprint_gate5_audio import _offline_environment
    from voiceprint_provider_pyannote import embed
    from voiceprint_synthetic_candidate import cosine_similarity
    if FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("invalid source fingerprint")
    state = _safe_path(state_root or STATE_ROOT)
    if not enabled(state):
        return {"status": "disabled", "reason": "disabled"}
    source = _source(source_folder)
    namespace = "laptop_capture" if source.source_folder.parent == ROOT / "ingest/laptop" else "audio_first"
    expected = hashlib.sha256(f"meetingintel|{namespace}|v1|{source.source_identity}".encode()).hexdigest()
    if fingerprint != expected:
        raise ValueError("identity fingerprint does not match source")
    manifest, segments, _audio_json, srt_path = _load_diarization(Path(diarization_folder), source.source_identity, include_unassigned=True)
    if (Path(manifest.get("source_folder", "")).resolve() != source.source_folder or
            Path(manifest.get("source_audio", "")).resolve() != source.canonical_audio_path):
        raise ValueError("identity source lineage mismatch")
    revocation = _json(state / "automatic" / (fingerprint + "-revocation.json"))
    if revocation.get("fingerprint") == fingerprint and revocation.get("method") == "auto_match_v1":
        return anonymous_presentation(diarization_folder, output_root, fingerprint, "revoked", revoked=True)
    from voiceprint_gate5_inbox import Gate5Error, METADATA_FILENAME
    try:
        inbox, records, model = _owner(state)
    except Gate5Error:
        return anonymous_presentation(diarization_folder, output_root, fingerprint, "provider_mismatch")
    rows = []
    named = None
    reason = "no_enrollment"
    name = None
    if len(records) == 1:
        owner = records[0]
        name = inbox.private_display_name(owner.identity_id)
        if name:
            with tempfile.TemporaryDirectory(prefix="mi-auto-owner-") as temporary:
                for label in sorted({s["speaker"] for s in segments if isinstance(s["speaker"], str) and re.fullmatch(r"SPEAKER_\d+", s["speaker"])}):
                    ranges = clean_ranges(segments, label)
                    total = sum(b - a for a, b in ranges)
                    row = {"speaker_label": label, "eligible_seconds": total, "score": None}
                    if total >= MIN_SPEECH:
                        selected = bounded_track_segments(ranges)
                        track = Path(temporary) / (label + ".wav")
                        _make_track(source.canonical_audio_path, selected, track)
                        with _offline_environment(), _embedding_audio_environment():
                            vector = embed(model, track)
                        row["score"] = cosine_similarity(vector, owner.vector)
                    rows.append(row)
            named, reason = decide(rows, inbox.provenance.threshold)
    output = _safe_path(output_root)
    directory = output / "transcripts/identity"
    _safe_path(directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    srt = srt_path.read_text(encoding="utf-8")
    anonymous = directory / f"{fingerprint}-{version}-anonymous.md"
    presentation = directory / f"{fingerprint}-{version}.md"
    for path, body in ((anonymous, render(srt)), (presentation, render(srt, name, named, reason))):
        path.write_text(body, encoding="utf-8")
        path.chmod(0o600)
    receipt = {"schema_version": 1, "method": "auto_match_v1", "human_confirmed": False,
               "fingerprint": fingerprint, "source_identity": source.source_identity,
               "source_audio_sha256": _hash(source.canonical_audio_path),
               "stage_manifest_sha256": _hash(Path(diarization_folder) / "run_manifest.json"),
               "provider": inbox.provenance.provider_id, "model_sha256": inbox.provenance.model_asset_sha256,
               "threshold": inbox.provenance.threshold, "aggregation": inbox.provenance.aggregation_recipe,
               "enrollment_id": records[0].identity_id if len(records) == 1 else None,
               "enrollment_state_sha256": _hash(inbox.state_path) if inbox.state_path.is_file() else None,
               "enrollment_metadata_sha256": _hash(state / METADATA_FILENAME) if (state / METADATA_FILENAME).is_file() else None,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "named_label": named, "reason": reason, "clusters": rows}
    receipt_root = os.environ.get('MI_IDENTITY_RECEIPT_ROOT') if os.environ.get('MI_CANARY') == '1' else None
    private_receipt = _safe_path(Path(receipt_root) if receipt_root else state / "automatic") / f"{fingerprint}-{version}.json"
    _atomic(private_receipt, receipt)
    projection = {"schema_version": 1, "fingerprint": fingerprint,
                  "status": "matched" if named else "anonymous", "reason": reason,
                  "method": "auto_match_v1", "presentation_path": str(presentation),
                  "presentation_sha256": _hash(presentation), "anonymous_path": str(anonymous),
                  "anonymous_sha256": _hash(anonymous), "receipt_path": str(private_receipt),
                  "enrollment_state_sha256": receipt["enrollment_state_sha256"],
                  "enrollment_metadata_sha256": receipt["enrollment_metadata_sha256"]}
    _atomic(directory / (fingerprint + ".json"), projection)
    return {"status": projection["status"], "reason": reason, "presentation_path": str(presentation)}


def presentation(fingerprint, output_root, *, state_root=None):
    """Read only verified local presentation metadata for a UI/CLI result."""
    if not isinstance(fingerprint, str) or FINGERPRINT.fullmatch(fingerprint) is None:
        return {}
    state = Path(state_root or STATE_ROOT)
    directory = _safe_path(Path(output_root) / "transcripts/identity")
    try:
        record = _json(directory / (fingerprint + ".json"))
        if record.get("schema_version") != 1 or record.get("fingerprint") != fingerprint:
            return {}
        status = record.get("status")
        if status not in {"matched", "anonymous", "revoked"} or record.get("reason") not in REASONS:
            return {}
        from voiceprint_synthetic_candidate import STATE_FILENAME
        from voiceprint_gate5_inbox import METADATA_FILENAME
        enrollment = _safe_path(state / STATE_FILENAME)
        metadata = _safe_path(state / METADATA_FILENAME)
        valid_enrollment = (enabled(state) and enrollment.is_file() and
                            metadata.is_file() and record.get("enrollment_state_sha256") == _hash(enrollment) and
                            record.get("enrollment_metadata_sha256") == _hash(metadata))
        use_named = status == "matched" and valid_enrollment
        path = _safe_path(record["presentation_path"] if use_named else record["anonymous_path"])
        expected = record["presentation_sha256"] if use_named else record["anonymous_sha256"]
        if path.parent != directory or not path.is_file() or _hash(path) != expected:
            return {}
        return {"status": "matched" if use_named else ("revoked" if status == "revoked" else "anonymous"),
                "reason": record["reason"] if use_named or status != "matched" else "disabled",
                "path": str(path), "header": "Owner voice recognized automatically" if use_named else "Speakers remain anonymous"}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def revoke(fingerprint, output_root, *, state_root=None):
    if FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("use the exact meeting fingerprint")
    directory = _safe_path(Path(output_root) / "transcripts/identity")
    path = directory / (fingerprint + ".json")
    record = _json(path)
    if record.get("fingerprint") != fingerprint:
        raise ValueError("meeting has no automatic identity result")
    # Verify the anonymous version before publishing the revocation.
    anonymous = _safe_path(record["anonymous_path"])
    if anonymous.parent != directory or _hash(anonymous) != record["anonymous_sha256"]:
        raise ValueError("anonymous presentation is unavailable")
    record.update(status="revoked", reason="revoked", revoked_at=datetime.now(timezone.utc).isoformat())
    _atomic(path, record)
    _atomic(Path(state_root or STATE_ROOT) / "automatic" / (fingerprint + "-revocation.json"),
            {"fingerprint": fingerprint, "method": "auto_match_v1", "revoked_at": record["revoked_at"]})


def create_presentation(source, stage, output_root):
    if not enabled():
        return {"status": "disabled", "reason": "disabled"}
    import meetingintel_gpu as gpu
    command = [str(PYTHON), str(Path(__file__)), "match", "--source-folder", str(source.source_folder),
               "--diarization-folder", str(stage), "--output-root", str(output_root),
               "--fingerprint", source.fingerprint]
    try:
        result = gpu.run(command, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        reason = "worker_timeout"
    except OSError:
        reason = "worker_launch_failed"
    else:
        if result.returncode in {-signal.SIGALRM, 128 + signal.SIGALRM}:
            reason = "worker_timeout"
        elif result.returncode < 0 or result.returncode in {128 + signal.SIGKILL, 128 + signal.SIGTERM}:
            reason = "worker_terminated"
        else:
            try:
                record = json.loads(result.stdout)
            except (TypeError, ValueError):
                record = None
            valid = (isinstance(record, dict) and record.get("reason") in REASONS
                     and record.get("status") in {"matched", "anonymous", "disabled", "revoked", "unavailable"})
            if valid:
                if result.returncode == 0:
                    return record
                reason = record["reason"]
            else:
                reason = "worker_exit_failed" if result.returncode != 0 else "worker_output_invalid"
    try:
        return anonymous_presentation(stage, output_root, source.fingerprint, reason)
    except (OSError, ValueError):
        return {"status": "unavailable", "reason": reason}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local automatic owner presentation")
    parser.add_argument("command", choices=("match",))
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--diarization-folder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    args = parser.parse_args()
    import signal
    signal.alarm(600)
    try:
        with audio_tool_environment():
            result = match_and_present(args.source_folder, args.diarization_folder, args.output_root, args.fingerprint)
    except Exception as error:
        from voiceprint_gate5_preview import PreviewError
        reason = "worker_exception"
        if isinstance(error, (PreviewError, AudioToolUnavailable)):
            reason = PREVIEW_FAILURE_REASONS.get(str(error), "invalid_stage")
        print(json.dumps({"status": "unavailable", "reason": reason}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    import meetingintel_gpu as gpu
    raise SystemExit(gpu.supervised_main(main))
