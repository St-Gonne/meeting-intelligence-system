#!/usr/bin/env python3
"""Run one explicit synthetic laptop source through the real local pipeline.

This deliberately performs real ASR, diarization and local Gemma calls when
invoked. All supplied media must be inside the operator's isolated proof root.
Only output-root constants are redirected through small import wrappers; no
model result, subprocess, routing decision or persistence function is mocked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meetingintel_pipeline as pipeline
from laptop_capture_meeting_source import load_laptop_meeting_source
from meetingintel_runtime import operation_lock
from ollama_endpoint import OLLAMA_APPROVED_DIGEST

ENDPOINT = "http://127.0.0.1:11435/api/generate"


class ProofError(RuntimeError):
    """A content-free failure category suitable for the summary."""


def safe_path(path: Path, *, exists: bool = True) -> Path:
    path = path.expanduser().absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ProofError("unsafe_path")
    if exists and not path.exists():
        raise ProofError("missing_input")
    return path


def layout(source_folder: Path, capture_root: Path, proof_root: Path) -> dict[str, Path]:
    root = safe_path(proof_root)
    if not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise ProofError("proof_root_must_be_private")
    protected = (PROJECT_ROOT, pipeline.DEFAULT_MEETINGS_ROOT,
                 Path.home() / "Movies" / "MeetingIntel Recordings",
                 Path.home() / "Library" / "Application Support" / "MeetingIntel")
    for path in protected:
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ProofError("proof_root_overlaps_production")
    source, capture = safe_path(source_folder), safe_path(capture_root)
    if not source.is_dir() or not capture.is_dir():
        raise ProofError("input_must_be_directory")
    if source == root or capture == root or not source.is_relative_to(root) or not capture.is_relative_to(root):
        raise ProofError("synthetic_inputs_must_be_inside_proof_root")
    validation = root / "validation"
    if source == validation or capture == validation or source.is_relative_to(validation) or capture.is_relative_to(validation):
        raise ProofError("input_overlaps_validation_output")
    if validation.is_relative_to(source) or validation.is_relative_to(capture):
        raise ProofError("validation_output_overlaps_input")
    paths = {"proof": root, "source": source, "capture": capture, "validation": validation}
    for name in ("output", "state", "staging", "shadow", "wrappers", "logs", "learning"):
        paths[name] = validation / name
    for name, path in paths.items():
        if name not in {"proof", "source", "capture"}:
            safe_path(path, exists=False)
    return paths


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict:
    """Hash only the exact synthetic source/capture or isolated output tree."""
    found = {}
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProofError("symlink_in_proof_evidence")
        if path.is_file():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ProofError("unsafe_proof_evidence")
            found[str(path.relative_to(root))] = {
                "bytes": info.st_size, "sha256": digest(path), "mtime_ns": info.st_mtime_ns,
            }
    return found


def private_json(path: Path, payload: dict) -> None:
    safe_path(path, exists=False)
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ProofError("unsafe_receipt")
    pipeline.atomic_write_bytes(path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())
    path.chmod(0o600)


def make_wrappers(paths: dict[str, Path]) -> tuple[Path, Path]:
    """Redirect real helper output roots without copying or changing its logic."""
    prefix = (
        "import os, sys, json\nfrom pathlib import Path\nfrom datetime import datetime, timezone\n"
        "os.umask(0o077)\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
    )
    helper = paths["wrappers"] / "real_diarization.py"
    shadow = paths["wrappers"] / "real_shadow.py"
    for target, module, assignments, role in (
        (helper, "whispermlx_diarization_helper", {"OUTPUT_ROOT": paths["staging"]}, "helper"),
        (shadow, "diarized_layer2_shadow", {"DIARIZATION_ROOT": paths["staging"], "SHADOW_OUTPUT_ROOT": paths["shadow"]}, "shadow"),
    ):
        receipt = paths["logs"] / (role + "-invocations.jsonl")
        content = prefix + f"import {module} as actual\n"
        for key, value in assignments.items():
            content += f"actual.{key} = Path({str(value)!r})\n"
        content += (
            f"receipt = Path({str(receipt)!r})\n"
            "fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)\n"
            "with os.fdopen(fd, 'a') as stream:\n"
            f" stream.write(json.dumps({{'role': {role!r}, 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()}}) + '\\n')\n"
            " stream.flush()\n os.fsync(stream.fileno())\n"
            "raise SystemExit(actual.main())\n"
        )
        pipeline.atomic_write_bytes(target, content.encode())
        target.chmod(0o600)
    return helper, shadow


def call_counts(paths: dict[str, Path]) -> dict[str, int]:
    result = {}
    for role in ("helper", "shadow"):
        path = safe_path(paths["logs"] / (role + "-invocations.jsonl"), exists=False)
        result[role] = len(path.read_text().splitlines()) if path.exists() else 0
    return result


def arguments(paths: dict[str, Path], source) -> argparse.Namespace:
    return argparse.Namespace(
        ledger_path=paths["state"] / "processed_ledger.json", output_dir=paths["output"],
        layer2_prompt_path=pipeline.DEFAULT_LAYER2_PROMPT_PATH, prompt_path=pipeline.DEFAULT_PROMPT_PATH,
        context_pack=None, brief_date=source.created_at.astimezone(pipeline.IST).date().isoformat(),
        ollama_url=ENDPOINT, model=pipeline.DEFAULT_MODEL, dry_run=False, refresh_existing=False,
        phone_transcription_backend="legacy",
    )


@contextmanager
def isolated_configuration(paths: dict[str, Path], wrappers: tuple[Path, Path]):
    originals = pipeline.DIARIZATION_HELPER_SCRIPT, pipeline.DIARIZED_SHADOW_SCRIPT
    old_learning = os.environ.get("MI_LEARNING_ROOT")
    previous_umask = os.umask(0o077)
    pipeline.DIARIZATION_HELPER_SCRIPT, pipeline.DIARIZED_SHADOW_SCRIPT = wrappers
    os.environ["MI_LEARNING_ROOT"] = str(paths["learning"])
    try:
        yield
    finally:
        pipeline.DIARIZATION_HELPER_SCRIPT, pipeline.DIARIZED_SHADOW_SCRIPT = originals
        os.umask(previous_umask)
        if old_learning is None:
            os.environ.pop("MI_LEARNING_ROOT", None)
        else:
            os.environ["MI_LEARNING_ROOT"] = old_learning


def verified_artifacts(args: argparse.Namespace, fingerprint: str, brief: Path) -> tuple[dict, dict]:
    ledger = pipeline.load_ledger(args.ledger_path)
    if set(ledger.get("records", {})) != {fingerprint}:
        raise ProofError("isolated_ledger_must_contain_one_exact_record")
    record = ledger["records"][fingerprint]
    artifacts = {}
    for name, raw in (("transcript", record.get("transcript_path")),
                      ("report", record.get("layer2_report_path")), ("brief", brief)):
        if not raw:
            raise ProofError("required_artifact_missing")
        path = safe_path(Path(raw))
        if not path.is_relative_to(args.output_dir) or not path.is_file() or path.stat().st_size == 0:
            raise ProofError("required_artifact_missing_or_uncontained")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if artifacts["transcript"]["sha256"] != record.get("transcript_sha256"):
        raise ProofError("durable_transcript_hash_mismatch")
    expected_brief = pipeline.render_brief(
        source_date := datetime.strptime(args.brief_date, "%Y-%m-%d").date(),
        pipeline.collect_today_records(ledger, source_date),
    )
    if brief.read_text() != expected_brief:
        raise ProofError("brief_does_not_match_isolated_ledger")
    return record, artifacts


def safe_metrics(record: dict) -> dict:
    metrics = {}
    for stage, values in (record.get("ollama_metrics") or {}).items():
        if stage not in {"layer2", "layer3"} or not isinstance(values, dict):
            continue
        selected = {}
        for key in pipeline.OLLAMA_METRIC_FIELDS:
            value = values.get(key)
            if key == "done_reason" and value in {"stop", "length"}:
                selected[key] = value
            elif key != "done_reason" and type(value) in (int, float) and value >= 0:
                selected[key] = value
        metrics[stage] = selected
    return metrics


def run_proof(source_folder: Path, capture_root: Path, proof_root: Path) -> dict:
    paths = layout(source_folder, capture_root, proof_root)
    # Repeat invocations must not follow a substituted ledger/artifact link
    # into the ordinary workspace. Inspect only this isolated validation tree.
    inventory(paths["validation"])
    for name in ("validation", "output", "state", "staging", "shadow", "wrappers", "logs", "learning"):
        path = paths[name]
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ProofError("proof_subdirectories_must_be_private")
    invocation = uuid.uuid4().hex
    log = paths["logs"] / ("pipeline-" + invocation + ".log")
    summary = {"schema_version": 1, "status": "failed", "invocation_id": invocation,
               "evidence_scope": "synthetic_local_real_models", "model": pipeline.DEFAULT_MODEL,
               "ollama_url": ENDPOINT, "approved_model_digest": OLLAMA_APPROVED_DIGEST,
               "production_ledger_output_used": False, "historical_discovery_used": False,
               "capture_engine_proven_by_this_run": False, "word_accuracy_measured": False,
               "root_overrides_only": True}
    started = time.monotonic()
    source_before = capture_before = None
    capture_session = None
    try:
        source = load_laptop_meeting_source(paths["source"], capture_root=paths["capture"])
        capture_session = safe_path(source.capture_manifest_path.parent)
        if not capture_session.is_relative_to(paths["capture"]):
            raise ProofError("capture_manifest_outside_synthetic_capture_root")
        source_before, capture_before = inventory(paths["source"]), inventory(capture_session)
        args = arguments(paths, source)
        selected = pipeline.production_source_from_laptop_capture(source)
        summary["source_identity"] = source.source_identity
        summary["audio_seconds"] = source.duration_seconds
        binding = {"source_identity": source.source_identity,
                   "canonical_audio_sha256": source.canonical_audio_sha256,
                   "capture_manifest_sha256": source.capture_manifest_sha256}
        binding_path = paths["validation"] / "source-binding.json"
        if binding_path.exists():
            safe_path(binding_path)
            if json.loads(binding_path.read_text()) != binding:
                raise ProofError("proof_root_already_bound_to_different_source")
        else:
            private_json(binding_path, binding)
        wrappers = make_wrappers(paths)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle, redirect_stdout(handle), redirect_stderr(handle), \
             isolated_configuration(paths, wrappers), operation_lock("processing"):
            from meetingintel_cli import load_secret_environment
            if not load_secret_environment():
                raise ProofError("required_local_credentials_unavailable")
            pipeline.prepare_ollama(args)
            summary["preflight"] = "approved_alias_and_digest_passed"
            before_counts = call_counts(paths)
            initial = pipeline.load_ledger(args.ledger_path)
            expected = 0 if selected.fingerprint in initial.get("records", {}) else 1
            if set(initial.get("records", {})) - {selected.fingerprint}:
                raise ProofError("unexpected_source_in_isolated_ledger")
            first_count, brief = pipeline.process_explicit_laptop_sources(args, (selected,))
            after_counts = call_counts(paths)
            summary["first_processed"] = first_count
            summary["helper_calls_first"] = after_counts["helper"] - before_counts["helper"]
            if first_count != expected or summary["helper_calls_first"] != expected:
                raise ProofError("first_pass_did_not_persist_exact_expected_result")
            if expected == 0 and after_counts != before_counts:
                raise ProofError("already_persisted_source_invoked_processing_helper")
            record, artifacts = verified_artifacts(args, selected.fingerprint, brief)
            before_repeat = {name: inventory(paths[name]) for name in ("state", "staging", "shadow", "output")}
            # This is the same real entrypoint again, not an early return in
            # the harness and not a replacement/helper stub.
            repeat_count, repeat_brief = pipeline.process_explicit_laptop_sources(args, (selected,))
            repeat_counts = call_counts(paths)
            _, repeated_artifacts = verified_artifacts(args, selected.fingerprint, repeat_brief)
            after_repeat = {name: inventory(paths[name]) for name in ("state", "staging", "shadow", "output")}
            for name in ("state", "staging", "shadow"):
                if before_repeat[name] != after_repeat[name]:
                    raise ProofError("deduplicated_pass_changed_processing_evidence")
            for name, facts in before_repeat["output"].items():
                later = after_repeat["output"].get(name)
                if not later or facts["sha256"] != later["sha256"] or facts["bytes"] != later["bytes"]:
                    raise ProofError("deduplicated_pass_changed_saved_artifact")
                if not name.startswith("brief_input_") and facts["mtime_ns"] != later["mtime_ns"]:
                    raise ProofError("deduplicated_pass_rewrote_saved_artifact")
            if repeat_count != 0 or repeat_counts != after_counts or artifacts != repeated_artifacts:
                raise ProofError("deduplication_gate_failed")
            summary.update(status="passed", repeated_processed=repeat_count,
                           helper_calls_repeat=repeat_counts["helper"]-after_counts["helper"],
                           shadow_calls_repeat=repeat_counts["shadow"]-after_counts["shadow"],
                           deduplication="no_new_helper_or_shadow_calls_and_artifacts_preserved",
                           processing_mode=record.get("processing_mode"),
                           artifacts=artifacts, ollama_metrics=safe_metrics(record))
    except Exception as error:
        summary["error_code"] = str(error) if isinstance(error, ProofError) else type(error).__name__
        summary["status"] = "failed"
    finally:
        if source_before is not None and capture_before is not None and capture_session is not None:
            try:
                unchanged = inventory(paths["source"]) == source_before and inventory(capture_session) == capture_before
            except Exception:
                unchanged = False
            summary["synthetic_source_and_capture_unchanged"] = unchanged
            if not unchanged:
                summary.update(status="failed", error_code="synthetic_input_changed")
        summary["elapsed_seconds"] = round(time.monotonic()-started, 3)
        private_json(paths["validation"] / ("summary-"+invocation+".json"), summary)
        private_json(paths["validation"] / "latest-summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--proof-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_proof(args.source_folder, args.capture_root, args.proof_root)
    except Exception as error:
        summary = {"status": "failed", "error_code": str(error) if isinstance(error, ProofError) else type(error).__name__}
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
