#!/usr/bin/env python3
"""Supervised synthetic-tone proof of the real guarded OBS recording lifecycle.

No microphone or ambient system audio is admitted to the recording mix. Hardware
inputs are instantiated by the existing guard but muted and removed from all
tracks before StartRecord. This does not prove their signal continuity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import laptop_capture_guard as guard
import laptop_capture_contract as contract
import meetingintel_pipeline as pipeline
import obs_capture_probe as obs
from meetingintel_runtime import operation_lock, status as operation_status


class GateError(RuntimeError):
    pass


def safe_path(path, exists=True):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise GateError("unsafe_path")
    if exists and not path.exists():
        raise GateError("missing_path")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, payload):
    safe_path(path, False)
    pipeline.atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True)+"\n").encode())
    path.chmod(0o600)


def config_files():
    root = safe_path(obs.DEFAULT_OBS_CONFIG_ROOT)
    paths = [root/name for name in ("global.ini", "user.ini", "basic.ini") if (root/name).exists()]
    basic = root/"basic"
    if basic.exists():
        for path in basic.rglob("*"):
            safe_path(path)
            if path.is_file():
                paths.append(path)
    for path in paths:
        safe_path(path)
        if not path.is_file() or path.stat().st_nlink != 1:
            raise GateError("unsafe_obs_config_file")
    return sorted(paths)


def backup_config(destination):
    destination.mkdir(mode=0o700)
    records = {}
    for source in config_files():
        relative = source.relative_to(obs.DEFAULT_OBS_CONFIG_ROOT)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(source, target)
        target.chmod(0o600)
        records[str(relative)] = {"sha256": sha(source), "mode": stat.S_IMODE(source.stat().st_mode)}
    save(destination/"inventory.json", records)
    return records


def restore_config(backup, before, after):
    """Preserve the proof configuration, then restore exact preexisting files.

    New uniquely named proof profiles/collections remain on disk. No unrelated
    file or directory is deleted to make the inventories match.
    """
    if guard.obs_app_is_running() or obs.websocket_reachable():
        raise GateError("obs_exit_unconfirmed_restore_deferred")
    backup_config(after)
    for relative, facts in before.items():
        source = safe_path(backup/relative)
        target = safe_path(obs.DEFAULT_OBS_CONFIG_ROOT/relative, False)
        if sha(source) != facts["sha256"]:
            raise GateError("config_backup_changed")
        pipeline.atomic_write_bytes(target, source.read_bytes())
        target.chmod(facts["mode"])
    return all(sha(obs.DEFAULT_OBS_CONFIG_ROOT/name) == facts["sha256"]
               for name, facts in before.items())


def power_assertions():
    completed = subprocess.run(["/usr/bin/pmset", "-g", "assertions"], capture_output=True, text=True,
                               check=False, timeout=5)
    if completed.returncode:
        raise GateError("power_assertions_unavailable")
    return completed.stdout


def worker(run_root, mode):
    """Runs in a child so terminate-mode cannot kill the restore supervisor."""
    run_root = safe_path(run_root)
    os.environ["MI_LEARNING_ROOT"] = str(run_root/"learning")
    os.umask(0o077)
    tone = safe_path(run_root/"tone.wav")
    unique = "MI Synthetic Proof " + run_root.name.rsplit("-", 1)[-1]
    config = guard.GuardConfig(session_root=run_root/"capture-session", allowed_root=run_root,
                              profile_name=unique, scene_name=unique,
                              segment_seconds=60, poll_seconds=1)
    actual_configure, actual_request = guard.configure_guard_profile, guard.request_json
    actual_started = guard.wait_for_recording_started
    tone_name = "Synthetic Proof Tone"
    timer = None
    known_audio = set()
    privacy_receipt = {}

    def privacy_check():
        all_inputs = actual_request("GetInputList").get("inputs", [])
        special = actual_request("GetSpecialInputs")
        names = {value for value in special.values() if isinstance(value, str) and value}
        for item in all_inputs:
            name, kind = item.get("inputName"), item.get("inputKind")
            if name == tone_name:
                continue
            if kind in {"color_source_v3", "color_source"}:
                continue
            if not isinstance(name, str):
                raise GateError("unknown_input_shape")
            names.add(name)
        if names != known_audio:
            raise GateError("audio_source_set_changed")
        checked = []
        for name in sorted(names):
            muted = actual_request("GetInputMute", {"inputName": name}).get("inputMuted")
            tracks = actual_request("GetInputAudioTracks", {"inputName": name}).get("inputAudioTracks")
            if muted is not True or not isinstance(tracks, dict) or len(tracks) != 6 or any(value is not False for value in tracks.values()):
                raise GateError("non_tone_audio_not_excluded")
            checked.append({"muted": True, "all_tracks_disabled": True})
        tone_muted = actual_request("GetInputMute", {"inputName": tone_name}).get("inputMuted")
        tone_tracks = actual_request("GetInputAudioTracks", {"inputName": tone_name}).get("inputAudioTracks")
        monitor = actual_request("GetInputAudioMonitorType", {"inputName": tone_name}).get("monitorType")
        if tone_muted is not False or tone_tracks != {str(index): index == 1 for index in range(1, 7)} or monitor != "OBS_MONITORING_TYPE_NONE":
            raise GateError("tone_routing_not_confirmed")
        settings = actual_request("GetInputSettings", {"inputName": tone_name})
        if settings.get("inputKind") != "ffmpeg_source" or settings.get("inputSettings", {}).get("local_file") != str(tone):
            raise GateError("tone_source_not_confirmed")
        privacy_receipt.update(non_tone_inputs=checked, tone_track=1, monitoring="none",
                               tone_sha256=sha(tone), checked_before_start=True)
        save(run_root/"audio-routing.json", privacy_receipt)

    def configure(config):
        actual_configure(config)
        if actual_request("GetRecordStatus").get("outputActive"):
            raise GateError("recording_started_before_privacy_check")
        inputs = actual_request("GetInputList").get("inputs", [])
        for item in inputs:
            if item.get("inputKind") not in {"color_source_v3", "color_source"}:
                known_audio.add(item["inputName"])
        known_audio.update(value for value in actual_request("GetSpecialInputs").values()
                           if isinstance(value, str) and value)
        for name in sorted(known_audio):
            actual_request("SetInputMute", {"inputName": name, "inputMuted": True})
            actual_request("SetInputAudioTracks", {"inputName": name, "inputAudioTracks": {str(i): False for i in range(1, 7)}})
            actual_request("SetInputAudioMonitorType", {"inputName": name, "monitorType": "OBS_MONITORING_TYPE_NONE"})
        actual_request("CreateInput", {"sceneName": config.scene_name, "inputName": tone_name,
            "inputKind": "ffmpeg_source", "inputSettings": {"is_local_file": True, "local_file": str(tone),
                "looping": True, "restart_on_activate": True, "clear_on_media_end": False}, "sceneItemEnabled": True})
        actual_request("SetInputMute", {"inputName": tone_name, "inputMuted": False})
        actual_request("SetInputAudioTracks", {"inputName": tone_name, "inputAudioTracks": {str(i): i == 1 for i in range(1, 7)}})
        actual_request("SetInputAudioMonitorType", {"inputName": tone_name, "monitorType": "OBS_MONITORING_TYPE_NONE"})
        privacy_check()

    def request(name, data=None):
        if name == "StartRecord":
            privacy_check()
        return actual_request(name, data)

    def started(*args, **kwargs):
        nonlocal timer
        result = actual_started(*args, **kwargs)
        save(run_root/"active.json", {"pid": os.getpid(), "mode": mode,
             "started_at": datetime.now(timezone.utc).isoformat()})
        def stop():
            try:
                save(run_root/"stop-trigger.json", {"mode": mode,
                     "triggered_at": datetime.now(timezone.utc).isoformat(),
                     "active_seconds_target": 70 if mode == "complete" else 15})
                if mode == "external-stop":
                    actual_request("StopRecord")
                else:
                    os.kill(os.getpid(), signal.SIGINT if mode == "complete" else signal.SIGTERM)
            except Exception as error:
                save(run_root/"stop-error.json", {"error_type": type(error).__name__})
        timer = threading.Timer(70 if mode == "complete" else 15, stop)
        timer.daemon = True
        timer.start()
        return result

    guard.configure_guard_profile, guard.request_json, guard.wait_for_recording_started = configure, request, started
    try:
        result = guard.run_guard(config)
        save(run_root/"worker-result.json", {"guard_exit_code": result})
        return result
    finally:
        if timer is not None:
            timer.cancel()
        guard.configure_guard_profile, guard.request_json, guard.wait_for_recording_started = actual_configure, actual_request, actual_started


def inspect_media(run_root, mode):
    session = run_root/"capture-session"
    manifest = contract.load_manifest(session/"capture_manifest.json")
    trigger_path = safe_path(run_root/"stop-trigger.json")
    if json.loads(trigger_path.read_text()).get("mode") != mode or (run_root/"stop-error.json").exists():
        raise GateError("intended_stop_trigger_not_confirmed")
    audited = contract.audit_capture_session(session, manifest)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise GateError("ffmpeg_unavailable")
    evidence = []
    for segment in manifest.get("segments", []):
        relative = segment.get("relative_path") or segment.get("path")
        if not isinstance(relative, str):
            raise GateError("segment_path_missing")
        path = safe_path(session/relative)
        if not path.is_relative_to(session) or not path.is_file():
            raise GateError("segment_outside_session")
        decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
                                 capture_output=True, text=True, check=False, timeout=60)
        if decoded.returncode:
            raise GateError("full_decode_failed")
        tone = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", "8000", "pipe:1"],
                              capture_output=True, check=False, timeout=60)
        if tone.returncode or not tone.stdout:
            raise GateError("audio_decode_failed")
        import array, math
        samples = array.array("f")
        samples.frombytes(tone.stdout)
        rms = math.sqrt(sum(value*value for value in samples)/len(samples))
        if rms <= 0.0001:
            raise GateError("recorded_tone_is_silent")
        evidence.append({"bytes": path.stat().st_size, "sha256": sha(path),
                         "duration_seconds": segment.get("duration_seconds"), "full_decode": True,
                         "audio_rms": round(rms, 8), "manifest_hash_matches": sha(path) == segment.get("sha256")})
    if not evidence or not all(item["manifest_hash_matches"] for item in evidence):
        raise GateError("manifest_media_mismatch")
    if mode == "complete" and (manifest.get("status") != "complete" or not audited.accepted or len(evidence) < 2):
        raise GateError("complete_split_audit_failed")
    if mode != "complete" and manifest.get("status") != "interrupted":
        raise GateError("interruption_not_preserved")
    if mode == "external-stop" and manifest.get("stop_reason") != "recording_stopped_externally":
        raise GateError("unexpected_interruption_cause")
    return {"capture_status": manifest.get("status"), "stop_reason": manifest.get("stop_reason"),
            "audit_accepted": audited.accepted, "audit_merge_ready": audited.merge_ready,
            "audit_recovery_ready": audited.recovery_ready,
            "segments": evidence, "recording_lock_removed": not (session/".recording.lock").exists()}


def run(proof_root, mode):
    root = safe_path(proof_root)
    if not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise GateError("proof_root_must_be_private")
    for forbidden in (PROJECT, obs.DEFAULT_OBS_CONFIG_ROOT, Path.home()/"Movies"):
        if root == forbidden or root.is_relative_to(forbidden) or forbidden.is_relative_to(root):
            raise GateError("proof_root_overlaps_production")
    if operation_status()["active"] or guard.obs_app_is_running() or obs.websocket_reachable():
        raise GateError("existing_operation_or_obs_blocks_proof")
    run_root = root/("guard-proof-"+uuid.uuid4().hex)
    run_root.mkdir(mode=0o700)
    old_umask = os.umask(0o077)
    before = None
    summary = {"schema_version": 1, "mode": mode, "status": "failed",
               "evidence_scope": "real_guard_synthetic_tone_only",
               "real_microphone_or_system_audio_continuity_proven": False,
               "source_loss_root_cause_proven": False}
    try:
        before = backup_config(run_root/"config-before")
        assertions_before = power_assertions()
        (run_root/"assertions-before.txt").write_text(assertions_before)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise GateError("ffmpeg_unavailable")
        generated = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=997:sample_rate=48000:duration=90", "-af", "volume=0.05", "-c:a", "pcm_s16le", str(run_root/"tone.wav")],
            capture_output=True, check=False, timeout=30)
        if generated.returncode:
            raise GateError("tone_generation_failed")
        with (run_root/"worker.log").open("wb") as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(run_root), "--mode", mode],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                summary["worker_exit_code"] = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise GateError("guard_worker_timeout")
        if mode == "complete" and summary["worker_exit_code"] != 0:
            raise GateError("clean_guard_exit_not_confirmed")
        if mode != "complete" and summary["worker_exit_code"] == 0:
            raise GateError("interrupted_guard_exit_not_preserved")
        summary.update(inspect_media(run_root, mode))
        if not (run_root/"audio-routing.json").is_file():
            raise GateError("privacy_routing_receipt_missing")
        summary["audio_routing_verified_before_recording"] = True
        summary["status"] = "passed"
    except Exception as error:
        summary["error_code"] = str(error) if isinstance(error, GateError) else type(error).__name__
    finally:
        try:
            summary["obs_exited"] = not guard.obs_app_is_running()
            summary["control_listener_closed"] = not obs.websocket_reachable()
            summary["shared_operation_lock_free"] = not operation_status()["active"]
            assertions_after = power_assertions()
            (run_root/"assertions-after.txt").write_text(assertions_after)
            # Compare newly present caffeinate assertion lines; unrelated
            # applications may legitimately retain their own assertions.
            before_pids = set(re.findall(r"pid\s+(\d+)\(caffeinate\)", assertions_before)) if 'assertions_before' in locals() else set()
            lingering = set(re.findall(r"pid\s+(\d+)\(caffeinate\)", assertions_after)) - before_pids
            summary["new_caffeinate_assertions_absent"] = not lingering
            if before is not None and summary["obs_exited"] and summary["control_listener_closed"]:
                summary["original_config_bytes_restored"] = restore_config(run_root/"config-before", before, run_root/"config-after-proof")
            else:
                summary["original_config_bytes_restored"] = False
            if not all(summary.get(key) for key in ("obs_exited", "control_listener_closed", "shared_operation_lock_free", "new_caffeinate_assertions_absent", "original_config_bytes_restored", "recording_lock_removed")):
                summary.update(status="failed", cleanup="requires_review")
        except Exception as error:
            summary.update(status="failed", cleanup="requires_review", cleanup_error=type(error).__name__)
        save(run_root/"summary.json", summary)
        summary["proof_directory"] = str(run_root)
        os.umask(old_umask)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-root", type=Path)
    parser.add_argument("--mode", choices=("complete", "external-stop", "terminate"), default="complete")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.mode)
    if args.proof_root is None:
        parser.error("--proof-root is required")
    try:
        summary = run(args.proof_root, args.mode)
    except Exception as error:
        summary = {"status": "failed", "error_code": str(error) if isinstance(error, GateError) else type(error).__name__}
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
