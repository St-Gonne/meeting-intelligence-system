"""Exact-source GUI adapter. Listing reads metadata; execution revalidates media."""
from __future__ import annotations
from mi_paths import public_path

import argparse
import hashlib
import json
import math
import os
import re
import stat
from datetime import date, datetime
from pathlib import Path

import meetingintel_cli as cli
import meetingintel_inbox as inbox
import meetingintel_inventory as inventory
import meetingintel_pipeline as pipeline
import meetingintel_learning as learning
import meetingintel_observe as observe
from meetingintel_runtime import operation_lock, source_annotation, status as runtime_status, BusyError

STAGING_ROOT = public_path("work/diarization")
CAPTURE_REASONS = {
    "source_lost": "The audio source was lost.",
    "source_not_recovered": "The audio source was lost.",
    "control_lost": "The recorder control connection was lost.",
    "recorder_process_died": "The recorder exited unexpectedly.",
    "recording_stopped_externally": "Recording stopped outside the guarded stop command.",
    "output_stalled": "The recorder stopped writing audio.",
    "disk_critical": "Available disk space became too low.",
    "battery_critical": "The battery became too low.",
    "capture_permission_failed": "Audio capture permission was unavailable.",
    "stop_unconfirmed": "The recorder could not confirm its stop.",
    "recorder_exit_unconfirmed": "Recorder shutdown could not be confirmed.",
    "signal_sigterm": "The recording controller received a termination request.",
    "signal_sighup": "The recording controller lost its Terminal session.",
    "terminal_closed": "The recording controller lost its Terminal session.",
    "termination_requested": "The recording controller received a termination request.",
    "operator_cancelled_before_start": "Recording was cancelled before its start was confirmed.",
    "cleanup_failed": "Recorder cleanup could not be completed.",
    "manifest_write_failed": "The final recording evidence could not be saved.",
}


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000:
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _safe_file(path, root):
    if not path:
        return None
    p, root = Path(path), Path(root).absolute()
    if any(x.is_symlink() for x in (root, *root.parents)):
        return None
    root = root.resolve()
    try:
        if p.is_symlink() or not p.is_file() or root not in p.resolve().parents:
            return None
        if any(x.is_symlink() for x in p.parents if x != root and root in x.parents):
            return None
        return p
    except OSError:
        return None


def _capture_runtime():
    try:
        return runtime_status()
    except (OSError, ValueError, RuntimeError):
        return {"active": None, "owner": None}


def _safe_directory(path):
    path = Path(path).absolute()
    try:
        return not any(item.is_symlink() for item in (path, *path.parents)) and path.is_dir()
    except OSError:
        return False


def _capture_owner(folder, active):
    """A session is live only when its PID matches the held shared capture lock."""
    descriptor = -1
    try:
        if not _safe_directory(folder):
            return "unknown"
        descriptor = os.open(folder / ".recording.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 32:
            return "unknown"
        raw = os.read(descriptor, 33).decode("ascii").strip()
        if not re.fullmatch(r"[1-9][0-9]{0,9}", raw):
            return "unknown"
        pid = int(raw)
    except FileNotFoundError:
        return "absent" if active.get("active") is not None else "unknown"
    except (OSError, UnicodeError, ValueError):
        return "unknown"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if active.get("active") is None:
        return "unknown"
    owner = active.get("owner") or {}
    if active.get("active") is True and (not isinstance(owner, dict) or type(owner.get("pid")) is not int or not isinstance(owner.get("kind"), str)):
        return "unknown"
    return "matching" if active.get("active") is True and owner.get("kind") in {"capture", "recording"} and owner.get("pid") == pid else "absent"


def _capture_duration(meta, ownership):
    segments = meta.get("segments")
    if not isinstance(segments, list) or not segments:
        return None, False
    durations = [item.get("duration_seconds") if isinstance(item, dict) else None for item in segments]
    if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in durations):
        return None, False
    total = sum(durations)
    if not math.isfinite(total):
        return None, False
    final = (meta.get("status") in {"complete", "interrupted"} and ownership == "absent"
             and meta.get("finalizing") is not True
             and bool(inbox._timestamp(meta.get("ended_at")))
             and total > 0
             and all(item.get("state") == "finalized" and isinstance(item.get("sha256"), str)
                     and re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in segments))
    return total, bool(final)


def _capture_row(meta, folder, active):
    state = meta.get("status", "unknown")
    ownership = _capture_owner(folder, active)
    duration, duration_final = _capture_duration(meta, ownership)
    raw_reason = meta.get("stop_reason")
    reason = CAPTURE_REASONS.get(raw_reason if isinstance(raw_reason, str) else "", "The capture did not finish with a confirmed guarded stop.")
    category = "needs_decision"
    if state == "recording" and ownership == "matching" and meta.get("capture_phase") == "starting":
        status, recording_state, category, warning = "Starting recorder", "starting; audio not yet confirmed", "waiting", ""
        action = "Keep the lid open and wait for Capture guard active in the recording Terminal. Audio capture has not been confirmed yet."
    elif state == "recording" and ownership == "matching":
        status, recording_state, category, warning = "Recording", "recording", "waiting", ""
        action = "Keep the lid open. Stop with Control-C in the recording Terminal and wait for the saved recording message."
    elif state == "interrupted":
        recording_state = "interrupted; preserving available audio" if ownership == "matching" else "interrupted"
        status = "Recording interrupted — preserving available audio" if ownership == "matching" else "Recording interrupted"
        warning = reason + " Later conversation may be missing."
        action = "Keep the lid open. Wait for the recorder to stop; if the meeting continues, start a new mi record. Review the retained audio with mi inbox."
    elif state == "recording":
        status, recording_state = "Recording interrupted — guard not verified", "interrupted; guard not verified"
        warning = ("The recording controller could not be verified." if ownership == "unknown" else "This session has no matching active recording controller.") + " Recording cannot be confirmed; later conversation may be missing."
        action = "Check the recording Terminal and keep the lid open. Wait for the recorder to stop before starting a new mi record; review with mi inbox."
    elif state == "complete":
        status = "Recording saved — ready for review" if duration_final else "Recording stopped — saved audio needs checking"
        recording_state = "retained" if duration_final else "stopped; final evidence unverified"
        warning = "" if duration_final else "The saved recording evidence is not yet complete. Review it before processing."
        action = "Review this capture with mi inbox to prepare the retained audio."
    else:
        status, recording_state = "Capture needs review", "unknown"
        warning = "The recording state could not be verified. Review it before processing."
        action = "Check the recording Terminal and review this capture with mi inbox."
    if ownership == "unknown" and state == "interrupted":
        warning += " Recorder shutdown could not be verified."
    return dict(status=status, recording_state=recording_state, category=category,
                next_action=action, warning=warning, duration_seconds=duration, duration_is_final=duration_final)


def _catalog(active=None):
    """No audio hashing, ffprobe, transcript reads or model calls in the inbox."""
    config = inventory.InventoryConfig(meetings_root=cli.MEETINGS_ROOT,
        phone_ingest_root=cli.AUDIO_FIRST_ROOT, processing_ledger=cli.LEDGER_PATH)
    report = inventory.inventory("all", config=config)
    rows, warnings = [], list(report.warnings)
    active = _capture_runtime() if active is None else active
    if active.get("active") is None:
        warnings.append("Recording ownership is unavailable. Check the recording Terminal before starting another operation.")
    for item in report.items:
        kind = "phone" if item.source_kind == "phone_recording" else "meetily"
        identity = item.group_key or item.identity
        sid = observe.meetily_id(identity) if kind == "meetily" else identity
        if not re.fullmatch(r"[a-f0-9]{64}", sid or ""):
            sid = hashlib.sha256((kind + ":" + identity).encode()).hexdigest()
        category, action = inbox._category_for_inventory(item.status)
        rows.append(dict(id=sid, identity=identity, source_kind=kind,
            created_at=item.created_at.isoformat(), title="Phone recording" if kind == "phone" else "Meetily recording",
            status=item.status, category=category, next_action=action,
            source_identity=None, recording_state="retained; completeness unverified", warning=""))
    normalized = set()
    root = cli.LAPTOP_INGEST_ROOT
    if _safe_directory(root):
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                continue
            meta = _json(folder / "meeting_source.json")
            identity = meta.get("source_identity")
            if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
                continue
            created = meta.get("created_at")
            if not inbox._timestamp(created):
                continue
            normalized.add(meta.get("capture_id"))
            annotation = source_annotation(identity)
            incomplete = bool(annotation and annotation.get("completeness") == "incomplete")
            interrupted = meta.get("capture_status") == "interrupted"
            rows.append(dict(id=pipeline.build_laptop_capture_fingerprint(identity), identity=identity,
                source_identity=identity, capture_id=meta.get("capture_id"), source_kind="laptop", created_at=created, title="Laptop recording",
                status="laptop recording incomplete" if incomplete else "laptop normalized and ready",
                category="needs_decision" if incomplete else "ready", next_action="Review retained recording" if incomplete else "Process this recording",
                recording_state="incomplete" if incomplete else ("interrupted; recovered by operator" if interrupted else "retained"),
                duration_seconds=meta.get("duration_seconds"), duration_is_final=True,
                warning="The meeting continued after recording stopped. Only the retained portion was transcribed. Full-meeting processing is blocked." if incomplete else
                        ("Capture was interrupted. Recovery does not prove that the whole meeting was recorded." if interrupted else "")))
    root = cli.LAPTOP_RECORDINGS_ROOT
    if _safe_directory(root):
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                continue
            meta = _json(folder / "capture_manifest.json")
            cid = meta.get("capture_id")
            created = meta.get("started_at")
            if not isinstance(cid, str) or cid in normalized or not inbox._timestamp(created):
                continue
            rows.append(dict(id=hashlib.sha256(("capture:"+cid).encode()).hexdigest(), identity=cid,
                source_identity=None, source_kind="capture", title="Laptop capture", created_at=created,
                **_capture_row(meta, folder, active)))
    return rows, warnings


def _record(row, records):
    if row["id"] in records:
        return records[row["id"]]
    if row["source_kind"] == "meetily":
        return next((r for r in records.values() if r.get("folder_path") == row["identity"]), None)
    return None


def _staged_transcript(row):
    identity = row.get("source_identity")
    if not identity and row["source_kind"] == "phone":
        # Resolve source identity from normalization metadata, without audio validation.
        root = cli.AUDIO_FIRST_ROOT
        if root.is_dir() and not root.is_symlink():
            for folder in root.iterdir():
                if folder.is_symlink() or not folder.is_dir():
                    continue
                meta = _json(folder / "meeting_source.json")
                candidate = inventory._source_identity_from_metadata(meta)
                if candidate and pipeline.build_audio_first_fingerprint(candidate) == row["id"]:
                    identity = candidate
                    break
    if not identity:
        return None
    paths = [STAGING_ROOT / ("audio_first_" + identity)]
    if pipeline.QWEN_STAGING_ROOT.is_dir():
        paths.extend(sorted(pipeline.QWEN_STAGING_ROOT.glob("audio_first_" + identity[:16] + "__*"), reverse=True))
    for folder in paths:
        if any(x.is_symlink() for x in (folder, *folder.parents)):
            continue
        meta = _json(folder / "run_manifest.json")
        if meta.get("status") != "success" or meta.get("source_identity") != identity:
            continue
        for name in ("audio.srt", "audio.txt"):
            path = _safe_file(folder / name, folder)
            if path:
                return path
    return None


def _project(row, records):
    r = dict(row)
    record = _record(row, records)
    transcript = _safe_file((record or {}).get("transcript_path"), cli.OUTPUT_DIR)
    staged = None if transcript else _staged_transcript(row)
    report = _safe_file((record or {}).get("layer2_report_path"), cli.OUTPUT_DIR)
    r.update(transcript_state="saved" if transcript else ("retained portion transcribed" if staged else "not verified"),
        report_state="saved" if report else "not saved", brief_state="not saved",
        can_process=row["category"] == "ready" and record is None,
        can_repair_brief=False, artifacts=[])
    if transcript or staged:
        r["artifacts"].append({"kind": "transcript"})
    if report:
        r["artifacts"].append({"kind": "report"})
    if record:
        r["can_process"] = False
        target = str(record.get("meeting_date_ist", ""))
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", target):
            brief = cli.OUTPUT_DIR / ("brief_input_" + target + ".md")
            try:
                expected = pipeline.render_brief(date.fromisoformat(target), pipeline.collect_today_records({"records": records}, date.fromisoformat(target)))
                current = _safe_file(brief, cli.OUTPUT_DIR)
                good = current is not None and current.read_text() == expected
            except (OSError, ValueError, TypeError):
                good = False
            r["brief_state"] = "saved" if good else "missing or outdated"
            r["can_repair_brief"] = not good
            if good:
                r["artifacts"].append({"kind": "brief"})
        if row["recording_state"] != "incomplete":
            r["status"] = "Report saved" if report else "Saved record; report missing"
            r["category"] = "completed" if report and transcript and r["brief_state"] == "saved" else "needs_decision"
            if not transcript:
                r["warning"] = (r["warning"] + " Saved record has no verified durable transcript.").strip()
    observed = learning.source_state(row["id"])
    if observed.get("outcome") in {"failure", "failed", "interrupted"} and not record:
        r["status"] = "Processing stopped; retained recording available" if r["can_process"] else r["status"]
        r["last_error"] = observed.get("error_code") or "operation_failed"
        r["next_action"] = "Review and retry this exact recording. Transcription may run again." if r["can_process"] else r["next_action"]
    if record and r["category"] == "completed":
        r["next_action"] = "Open the saved transcript, report or daily brief."
    elif r["can_repair_brief"]:
        r["next_action"] = "Repair the daily brief from saved records without rerunning models."
    r["handoff"] = "mi inbox" if r["category"] == "needs_decision" else ""
    # Return opaque identities only. Paths stay inside the adapter.
    r.pop("identity", None)
    r.pop("source_identity", None)
    r.pop("capture_id", None)
    return r


def snapshot():
    active = _capture_runtime()
    rows, warnings = _catalog(active)
    operational_warnings = []
    if active.get("active") and (active.get("owner") or {}).get("kind") != "review":
        kind = (active.get("owner") or {}).get("kind", "operation")
        operational_warnings.append("MeetingIntel " + kind + " is active. A competing processing request will receive Busy; no work is queued.")
    records = inbox._read_records(cli.LEDGER_PATH, warnings)
    if any("ledger" in warning for warning in warnings):
        warnings.append("Processing is disabled until the ledger can be read safely.")
    items = [_project(row, records) for row in rows]
    if any("ledger" in warning for warning in warnings):
        for item in items:
            item["can_process"] = item["can_repair_brief"] = False
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return {"items": items, "warnings": warnings + operational_warnings, "coverage_warnings": warnings}


def detail(source_id):
    matches = [row for row in snapshot()["items"] if row["id"] == source_id]
    if len(matches) != 1:
        raise ValueError("Source unavailable or ambiguous; refresh the inbox.")
    return matches[0]


def reconcile():
    data = snapshot()
    rows, _ = _catalog()
    references = {row["id"]: [value for value in (row.get("source_identity"), row.get("capture_id"))
                              if isinstance(value, str) and learning.IDENTITY.fullmatch(value)] for row in rows}
    snapshots = []
    for row in data["items"]:
        common = {"source_id": row["id"],
            "source_kind": {"phone":"phone_recording", "laptop":"laptop_capture", "capture":"laptop_capture"}.get(row["source_kind"], row["source_kind"]),
            "details": {"reconstructed": True, "source_refs": references.get(row["id"], []),
                        "known_incomplete": row["recording_state"] == "incomplete"}}
        stages = {"processing": "success" if row["category"] == "completed" else "unresolved",
                  "capture": "interrupted" if "interrupt" in row["recording_state"] or row["recording_state"] == "incomplete" else "unknown",
                  "transcription": "saved" if row["transcript_state"] in {"saved", "retained portion transcribed"} else "unknown",
                  "report": "saved" if row["report_state"] == "saved" else "missing",
                  "brief": "saved" if row["brief_state"] == "saved" else "missing"}
        snapshots.extend(dict(common, stage=stage, outcome=outcome) for stage, outcome in stages.items())
    learning.reconcile(snapshots)
    return data


def _exact_row(source_id):
    if not isinstance(source_id, str) or not re.fullmatch(r"[a-f0-9]{64}", source_id):
        raise ValueError("Invalid source selection.")
    rows, _ = _catalog()
    selected = [row for row in rows if row["id"] == source_id]
    if len(selected) != 1:
        raise ValueError("Source unavailable or ambiguous; refresh the inbox.")
    return selected[0]


def perform(action, source_id):
    if action not in {"process", "repair_brief"}:
        raise ValueError("Unsupported action.")
    with operation_lock("processing", os.environ.get("MI_OPERATION_ID")):
        row = _exact_row(source_id)
        shown = detail(source_id)
        ledger = pipeline.load_ledger(cli.LEDGER_PATH)
        if action == "repair_brief":
            record = _record(row, ledger["records"])
            if not record or not shown["can_repair_brief"]:
                return {"ok": True, "outcome": "already_done"}
            target = date.fromisoformat(record["meeting_date_ist"])
            pipeline.write_brief(cli.OUTPUT_DIR, target, pipeline.render_brief(target, pipeline.collect_today_records(ledger, target)))
            return {"ok": True, "outcome": "succeeded", "message": "Daily brief repaired from saved records."}
        if _record(row, ledger["records"]):
            return {"ok": True, "outcome": "already_done"}
        if not shown["can_process"]:
            raise ValueError("This source needs Terminal review before processing.")
        item = inbox.InboxItem(datetime.fromisoformat(row["created_at"]), row["source_kind"], row["title"],
            row["status"], row["next_action"], row["category"], row["identity"])
        if row["source_kind"] == "laptop":
            source = cli._load_exact_laptop_source(item)
            code = cli._run_exact_laptop_source(source, clock=pipeline.now_ist, ollama_manager=None, suppress_private_paths=True)
        elif row["source_kind"] == "phone":
            source = cli._load_exact_phone_source(item)
            code = cli._run_exact_phone_sources((source,), clock=pipeline.now_ist,
                pipeline_runner=pipeline.process_meetings, ollama_manager=None, suppress_private_paths=True)
        elif row["source_kind"] == "meetily":
            folder = cli._exact_direct_child(Path(item.identity), cli.MEETINGS_ROOT, "Meetily source")
            for filename in ("metadata.json", "transcripts.json"):
                path = folder / filename
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Meetily source is incomplete or unsafe.")
            metadata, _, transcript_hash, text, created = pipeline.parse_meeting_folder(folder)
            if metadata.get("status") != "completed" or not text:
                raise ValueError("Meetily source is no longer ready.")
            fp = pipeline.build_fingerprint(folder.name, metadata["created_at"], transcript_hash)
            candidate = cli.SourceCandidate(fp, created, "meetily", "Meetily recording")
            code = cli._run_production((candidate,), scope="exact GUI recording", processed=set(ledger["records"]),
                clock=pipeline.now_ist, pipeline_runner=pipeline.process_meetings, ollama_manager=None,
                audio_first_only=False, explicit_meeting_folder=folder, suppress_private_paths=True)
        else:
            raise ValueError("Review this source in Terminal.")
        # The record/artifacts, not a CLI return code, establish completion.
        final = detail(source_id)
        complete = code == 0 and final["transcript_state"] == "saved" and final["report_state"] == "saved" and final["brief_state"] == "saved"
        return {"ok": complete, "outcome": "succeeded" if complete else "failed",
                "message": "Processing completed." if complete else "Processing stopped. Saved artifacts remain available; check the source details."}


def artifact(source_id, kind):
    row = _exact_row(source_id)
    records = pipeline.load_ledger(cli.LEDGER_PATH)["records"]
    record = _record(row, records) or {}
    if kind == "transcript":
        path = _safe_file(record.get("transcript_path"), cli.OUTPUT_DIR) or _staged_transcript(row)
    elif kind == "report":
        path = _safe_file(record.get("layer2_report_path"), cli.OUTPUT_DIR)
    elif kind == "brief" and record:
        target = str(record.get("meeting_date_ist", ""))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", target):
            raise ValueError("Invalid brief date.")
        path = _safe_file(cli.OUTPUT_DIR / ("brief_input_"+target+".md"), cli.OUTPUT_DIR)
    else:
        raise ValueError("Unsupported artifact.")
    if path is None or path.stat().st_size > 10_000_000:
        raise ValueError("Artifact unavailable.")
    learning.emit("artifact.opened", source_id=source_id, stage=kind, origin="gui")
    return path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("action", choices=["process", "repair_brief"])
    parser.add_argument("source_id")
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    learning.emit("operation.started", source_id=args.source_id)
    try:
        result = perform(args.action, args.source_id)
    except pipeline.gpu.Deferred:
        result = {"ok": False, "outcome": "deferred", "message": "GPU processing deferred. Your recording is saved; retry when local AI work finishes."}
    except BusyError:
        result = {"ok": False, "outcome": "busy", "message": "Recording or processing is already active. Try again when it finishes."}
    except Exception as exc:
        result = {"ok": False, "outcome": "failed", "message": "Action stopped safely. Check source details or use mi inbox.", "error_code": observe.safe_error(exc)}
    learning.emit("operation." + ("succeeded" if result["ok"] else ("busy" if result["outcome"] in {"busy", "deferred"} else "failed")), source_id=args.source_id)
    pipeline.atomic_write_bytes(args.result_file, json.dumps(result).encode())
    return 0 if result["ok"] else (75 if result["outcome"] == "deferred" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
