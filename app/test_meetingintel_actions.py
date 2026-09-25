import json
import hashlib
import os
import tempfile
import unittest
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from unittest.mock import patch

import meetingintel_actions as actions
import meetingintel_cli as cli
import meetingintel_inventory as inventory
import meetingintel_pipeline as pipeline
import meetingintel_runtime as runtime
from test_meetingintel_pipeline import legacy_record


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory(dir="/private/tmp"))
        self.root = Path(root)
        self.stack.enter_context(patch.dict(os.environ, {
            "MI_RUNTIME_ROOT": str(self.root / "runtime"), "MI_LEARNING_ROOT": str(self.root / "learning")}))
        for name, child in {"MEETINGS_ROOT": "meetings", "AUDIO_FIRST_ROOT": "phone", "LAPTOP_INGEST_ROOT": "laptop",
            "LAPTOP_RECORDINGS_ROOT": "captures", "LEDGER_PATH": "ledger.json", "OUTPUT_DIR": "output"}.items():
            self.stack.enter_context(patch.object(cli, name, self.root / child))
        self.stack.enter_context(patch.object(actions, "STAGING_ROOT", self.root / "staged"))
        self.stack.enter_context(patch.object(pipeline, "QWEN_STAGING_ROOT", self.root / "qwen"))
        self.stack.enter_context(patch.object(inventory, "inventory", return_value=inventory.InventoryReport("all", False, (), {})))
        self.identity = "a" * 64
        self.sid = pipeline.build_laptop_capture_fingerprint(self.identity)
        self.folder = cli.LAPTOP_INGEST_ROOT / "source"
        self.folder.mkdir(parents=True)
        self.meta = dict(source_identity=self.identity, capture_id="capture-1", created_at="2026-09-21T14:00:53+05:30",
            capture_status="interrupted", duration_seconds=351.594)
        (self.folder / "meeting_source.json").write_text(json.dumps(self.meta))

    def record(self):
        record = legacy_record(self.sid)
        record["source_identity"] = self.identity
        record["source_kind"] = "laptop_capture"
        record["folder_path"] = str(self.folder)
        for field, name in (("transcript_path", "transcripts/test.md"), ("layer2_report_path", "layer2/test.md")):
            path = cli.OUTPUT_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic test content only.")
            record[field] = str(path)
        pipeline.save_ledger(cli.LEDGER_PATH, {"records": {self.sid: record}})
        return record

    def test_opening_metadata_does_not_validate_media_or_process(self):
        with patch.object(cli, "_load_exact_laptop_source", side_effect=AssertionError("audio validation")), \
             patch.object(pipeline, "process_meetings", side_effect=AssertionError("model work")):
            row = actions.detail(self.sid)
        self.assertTrue(row["can_process"])
        self.assertNotIn("identity", row)
        self.assertFalse(cli.LEDGER_PATH.exists())

    def test_operator_correction_preserves_metadata_and_blocks_processing(self):
        before = (self.folder / "meeting_source.json").read_bytes()
        runtime.annotate_source(self.identity)
        row = actions.detail(self.sid)
        self.assertEqual("incomplete", row["recording_state"])
        self.assertFalse(row["can_process"])
        with patch.object(cli, "_load_exact_laptop_source") as load:
            with self.assertRaises(ValueError):
                actions.perform("process", self.sid)
            load.assert_not_called()
        self.assertEqual(before, (self.folder / "meeting_source.json").read_bytes())

    def test_transcript_survives_report_failure_in_projection(self):
        staged = actions.STAGING_ROOT / ("audio_first_"+self.identity)
        staged.mkdir(parents=True)
        (staged / "run_manifest.json").write_text(json.dumps({"source_identity": self.identity, "status": "success"}))
        (staged / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSynthetic.")
        runtime.annotate_source(self.identity)
        row = actions.detail(self.sid)
        self.assertEqual("retained portion transcribed", row["transcript_state"])
        self.assertEqual("not saved", row["report_state"])
        self.assertIn("Synthetic", actions.artifact(self.sid, "transcript"))

    def test_brief_repair_uses_saved_records_no_model(self):
        record = self.record()
        before = cli.LEDGER_PATH.read_bytes()
        self.assertTrue(actions.detail(self.sid)["can_repair_brief"])
        with patch.object(pipeline, "summarize_with_ollama", side_effect=AssertionError("model work")):
            result = actions.perform("repair_brief", self.sid)
        self.assertTrue(result["ok"])
        self.assertEqual("saved", actions.detail(self.sid)["brief_state"])
        self.assertEqual(before, cli.LEDGER_PATH.read_bytes())
        self.assertEqual("already_done", actions.perform("process", self.sid)["outcome"])

    def test_missing_durable_transcript_prevents_completed_label(self):
        record = self.record()
        actions.perform("repair_brief", self.sid)
        Path(record["transcript_path"]).unlink()
        row = actions.detail(self.sid)
        self.assertEqual("needs_decision", row["category"])
        self.assertIn("no verified durable transcript", row["warning"])

    def test_symlinked_artifact_or_staging_never_exposed(self):
        record = self.record()
        path = Path(record["transcript_path"])
        path.unlink()
        secret = self.root / "outside.txt"
        secret.write_text("must not expose")
        path.symlink_to(secret)
        with self.assertRaises(ValueError):
            actions.artifact(self.sid, "transcript")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "run_manifest.json").write_text(json.dumps({"source_identity": self.identity,"status":"success"}))
        (outside / "audio.srt").write_text("must not expose")
        actions.STAGING_ROOT.mkdir()
        (actions.STAGING_ROOT / ("audio_first_"+self.identity)).symlink_to(outside, target_is_directory=True)
        self.assertEqual("not verified", actions.detail(self.sid)["transcript_state"])

    def test_malformed_ledger_disables_processing(self):
        cli.LEDGER_PATH.write_text("broken")
        self.assertFalse(actions.detail(self.sid)["can_process"])
        with self.assertRaises(ValueError):
            actions.perform("process", self.sid)

    def test_untrusted_identifiers_actions_rejected(self):
        for sid in ("../../file", "", "b"*64):
            with self.assertRaises(ValueError):
                actions.perform("process", sid)
        with self.assertRaises(ValueError):
            actions.perform("shell", self.sid)

    def test_partial_correction_checked_inside_pipeline(self):
        runtime.annotate_source(self.identity)
        from types import SimpleNamespace
        source = SimpleNamespace(source_identity=self.identity, fingerprint=self.sid)
        with patch.object(pipeline, "stage_diarized_eligibility") as stage:
            with self.assertRaisesRegex(pipeline.ProcessingFallback, "capture_incomplete"):
                pipeline.select_audio_first_result(SimpleNamespace(dry_run=False), source, "", "")
            stage.assert_not_called()

    def test_exact_ui_adapter_real_persistence_then_brief_failure_repair(self):
        """Real adapter/pipeline/ledger path, synthetic selection substitutes models."""
        folder = cli.MEETINGS_ROOT / "synthetic"
        folder.mkdir(parents=True)
        created = pipeline.now_ist()
        (folder / "metadata.json").write_text(json.dumps({"created_at": created.isoformat(),
            "status": "completed", "meeting_name": "Synthetic", "duration_seconds": 10}))
        (folder / "transcripts.json").write_text(json.dumps({"segments": [{"text": "Synthetic fixture only."}]}))
        item = inventory.InventoryItem(created, "meetily", "Synthetic", "meetily ready", identity=str(folder))
        report_path = cli.OUTPUT_DIR / "layer2" / "synthetic.md"
        report_path.parent.mkdir(parents=True)
        report_path.write_text("Synthetic report.")
        selected = pipeline.SelectedMeetingResult("Synthetic summary.", str(report_path), {}, "flat",
                                                  layer2_transcript_input="Synthetic fixture only.")
        sid = actions.observe.meetily_id(folder)
        with patch.object(inventory, "inventory", return_value=inventory.InventoryReport("all", False, (item,), {})), \
             patch.object(cli, "load_secret_environment", return_value=True), \
             patch.object(cli, "preflight_ollama"), \
             patch.object(pipeline, "select_meeting_result", return_value=selected) as model, \
             patch.object(pipeline, "write_brief", side_effect=OSError("simulated full disk")):
            result = actions.perform("process", sid)
            row = actions.detail(sid)
            self.assertFalse(result["ok"])
            self.assertEqual("saved", row["report_state"])
            self.assertEqual("saved", row["transcript_state"])
            self.assertTrue(row["can_repair_brief"])
            self.assertEqual(1, model.call_count)
        with patch.object(inventory, "inventory", return_value=inventory.InventoryReport("all", False, (item,), {})), \
             patch.object(pipeline, "select_meeting_result", side_effect=AssertionError("must not rerun")):
            self.assertTrue(actions.perform("repair_brief", sid)["ok"])
            self.assertEqual("completed", actions.detail(sid)["category"])
            self.assertEqual("already_done", actions.perform("process", sid)["outcome"])

class RawCaptureTests(unittest.TestCase):
    setUp = ActionTests.setUp

    def capture(self, *, state="recording", lock_pid=None, finalized=False):
        folder = cli.LAPTOP_RECORDINGS_ROOT / "synthetic-capture"
        folder.mkdir(parents=True)
        meta = {"capture_id": "raw-capture-1", "started_at": "2026-09-21T14:00:53+05:30",
                "ended_at": "2026-09-21T14:06:00+05:30" if state != "recording" else None,
                "status": state, "stop_reason": "source_lost" if state == "interrupted" else None,
                "segments": [{"state": "finalized" if finalized else "active", "duration_seconds": 307.2,
                              "sha256": "f" * 64 if finalized else None}]}
        (folder / "capture_manifest.json").write_text(json.dumps(meta))
        if lock_pid is not None:
            (folder / ".recording.lock").write_text(str(lock_pid) + "\n")
        self.capture_id = hashlib.sha256(b"capture:raw-capture-1").hexdigest()
        return folder, meta

    def test_matching_held_capture_owner_is_waiting_and_duration_is_observed(self):
        self.capture(lock_pid=os.getpid())
        with runtime.operation_lock("capture"):
            row = actions.detail(self.capture_id)
        self.assertEqual("recording", row["recording_state"])
        self.assertEqual("waiting", row["category"])
        self.assertEqual("", row["warning"])
        self.assertFalse(row["duration_is_final"])
        self.assertFalse(row["can_process"])

    def test_unrelated_capture_pid_cannot_make_abandoned_session_active(self):
        self.capture(lock_pid=os.getpid() + 1)
        with runtime.operation_lock("capture"):
            row = actions.detail(self.capture_id)
        self.assertIn("interrupted", row["recording_state"])
        self.assertEqual("needs_decision", row["category"])
        self.assertIn("no matching active", row["warning"])

    def test_starting_guard_does_not_claim_audio_recording_before_readiness(self):
        folder, meta = self.capture(lock_pid=os.getpid())
        meta["capture_phase"] = "starting"
        (folder / "capture_manifest.json").write_text(json.dumps(meta))
        with runtime.operation_lock("capture"):
            row = actions.detail(self.capture_id)
        self.assertEqual("Starting recorder", row["status"])
        self.assertEqual("starting; audio not yet confirmed", row["recording_state"])
        self.assertEqual("waiting", row["category"])
        self.assertEqual("", row["warning"])
        self.assertIn("wait for Capture guard active", row["next_action"])
        self.assertFalse(row["duration_is_final"])
        self.assertFalse(row["can_process"])

    def test_processing_owner_with_same_pid_is_not_a_recording_owner(self):
        self.capture(lock_pid=os.getpid())
        with runtime.operation_lock("processing"):
            self.assertIn("interrupted", actions.detail(self.capture_id)["recording_state"])

    def test_abandoned_capture_is_attention_without_waiting_twelve_hours(self):
        self.capture(lock_pid=os.getpid())
        row = actions.detail(self.capture_id)
        self.assertEqual("needs_decision", row["category"])
        self.assertIn("later conversation may be missing", row["warning"])
        self.assertEqual("mi inbox", row["handoff"])
        self.assertFalse(row["duration_is_final"])

    def test_unreadable_runtime_is_unknown_and_does_not_claim_recording(self):
        self.capture(lock_pid=os.getpid())
        with patch.object(actions, "runtime_status", side_effect=runtime.RuntimeSafetyError("unavailable")):
            result = actions.snapshot()
        row = next(item for item in result["items"] if item["id"] == self.capture_id)
        self.assertIn("could not be verified", row["warning"])
        self.assertTrue(any("ownership is unavailable" in item for item in result["warnings"]))
        self.assertFalse(row["duration_is_final"])

    def test_interrupted_capture_with_live_owner_is_preserving_not_saved(self):
        self.capture(state="interrupted", lock_pid=os.getpid(), finalized=True)
        with runtime.operation_lock("capture"):
            row = actions.detail(self.capture_id)
        self.assertIn("preserving available audio", row["status"])
        self.assertIn("audio source was lost", row["warning"])
        self.assertFalse(row["duration_is_final"])
        self.assertIn("start a new mi record", row["next_action"])

    def test_held_lock_without_owner_metadata_remains_unverified(self):
        self.capture(lock_pid=os.getpid())
        with patch.object(actions, "runtime_status", return_value={"active": True, "owner": None}):
            row = actions.detail(self.capture_id)
        self.assertIn("could not be verified", row["warning"])
        self.assertNotEqual("waiting", row["category"])

    def test_finalized_interruption_stays_visible_with_saved_duration(self):
        self.capture(state="interrupted", finalized=True)
        row = actions.detail(self.capture_id)
        self.assertEqual("interrupted", row["recording_state"])
        self.assertEqual("needs_decision", row["category"])
        self.assertTrue(row["warning"])
        self.assertTrue(row["duration_is_final"])
        self.assertEqual(307.2, row["duration_seconds"])

    def test_finalizing_checkpoint_never_claims_final_duration(self):
        folder, meta = self.capture(state="interrupted", finalized=True)
        meta["finalizing"] = True
        (folder / "capture_manifest.json").write_text(json.dumps(meta))
        self.assertFalse(actions.detail(self.capture_id)["duration_is_final"])

    def test_complete_final_metadata_is_ready_for_review_not_already_processed(self):
        self.capture(state="complete", finalized=True)
        row = actions.detail(self.capture_id)
        self.assertTrue(row["duration_is_final"])
        self.assertEqual("retained", row["recording_state"])
        self.assertFalse(row["can_process"])
        self.assertEqual("mi inbox", row["handoff"])

    def test_missing_duration_remains_unavailable_and_raw_error_is_not_displayed(self):
        folder, meta = self.capture(state="interrupted", finalized=True)
        meta["segments"][0].pop("duration_seconds")
        meta["stop_reason"] = "/private/sensitive-path secret diagnostic"
        (folder / "capture_manifest.json").write_text(json.dumps(meta))
        row = actions.detail(self.capture_id)
        self.assertIsNone(row["duration_seconds"])
        self.assertFalse(row["duration_is_final"])
        self.assertNotIn("sensitive-path", json.dumps(row))

    def test_symlinked_session_lock_cannot_establish_live_ownership(self):
        folder, _ = self.capture()
        outside = self.root / "external-lock"
        outside.write_text(str(os.getpid()))
        (folder / ".recording.lock").symlink_to(outside)
        with runtime.operation_lock("capture"):
            row = actions.detail(self.capture_id)
        self.assertIn("could not be verified", row["warning"])
        self.assertNotEqual("waiting", row["category"])

    def test_symlinked_capture_root_is_not_followed(self):
        self.capture()
        alias = self.root / "capture-link"
        alias.symlink_to(cli.LAPTOP_RECORDINGS_ROOT, target_is_directory=True)
        with patch.object(cli, "LAPTOP_RECORDINGS_ROOT", alias):
            self.assertFalse(any(row["source_kind"] == "capture" for row in actions.snapshot()["items"]))


if __name__ == "__main__":
    unittest.main()
