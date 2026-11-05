from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import meetingintel_progress as progress


class MeetingIntelProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context_token = progress._current_operation.set(None)
        self.env_patch = patch.dict(os.environ, {}, clear=False)
        self.env_patch.start()
        os.environ.pop("MI_OPERATION_ID", None)

    def tearDown(self):
        self.env_patch.stop()
        progress._current_operation.reset(self.context_token)
        self.temp.cleanup()

    def test_start_uses_environment_or_generates_a_safe_id_and_writes_private_state(self):
        with patch.dict(os.environ, {"MI_OPERATION_ID": "ui-job-7"}):
            snapshot = progress.start(
                "gpu_wait", fingerprint64hex="a" * 64, stdout_bytes=0, stderr_bytes=12, root=self.root
            )
        self.assertEqual(snapshot["operation_id"], "ui-job-7")
        self.assertEqual(progress.current_operation_id(), "ui-job-7")
        self.assertEqual(snapshot["stage_label"], "Waiting for GPU")
        self.assertTrue(snapshot["start_at"].endswith("Z"))
        self.assertEqual(snapshot["stdout_bytes"], 0)
        self.assertEqual(snapshot["stderr_bytes"], 12)
        record = self.root / "state" / "progress" / "ui-job-7.json"
        self.assertEqual(record.stat().st_mode & 0o777, 0o600)
        self.assertEqual((record.parent.stat().st_mode & 0o777), 0o700)
        self.assertEqual((record.parent.parent.stat().st_mode & 0o777), 0o700)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MI_OPERATION_ID", None)
            generated = progress.start(root=self.root)
        self.assertRegex(generated["operation_id"], r"^[0-9a-f]{32}$")

    def test_stage_changes_activity_and_elapsed_time_are_derived(self):
        base = datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc)
        ticks = [base, base + timedelta(seconds=10), base + timedelta(seconds=20),
                 base + timedelta(seconds=30), base + timedelta(seconds=40),
                 base + timedelta(seconds=50)]
        with patch.object(progress, "_utc_now", side_effect=ticks):
            started = progress.start("gpu_wait", operation_id="timed-job", root=self.root)
            first = progress.update("transcription", stdout_bytes=15, root=self.root)
            same_stage = progress.update_current("transcription", stdout_bytes=25, stderr_bytes=4, root=self.root)
            observed = progress.read("timed-job", root=self.root)
            finished = progress.finish(root=self.root)
            after_finish = progress.read("timed-job", root=self.root)

        self.assertEqual(started["stage_started_at"], started["start_at"])
        self.assertEqual(first["stage_started_at"], first["last_activity_at"])
        self.assertEqual(same_stage["stage_started_at"], first["stage_started_at"])
        self.assertEqual(same_stage["stdout_bytes"], 25)
        self.assertEqual(same_stage["stderr_bytes"], 4)
        self.assertEqual(observed["elapsed_seconds"], 30)
        self.assertEqual(observed["seconds_since_activity"], 10)
        self.assertEqual(finished["stage"], "complete")
        self.assertEqual(finished["finished_at"], finished["last_activity_at"])
        self.assertEqual(finished["elapsed_seconds"], 40)
        self.assertEqual(after_finish["elapsed_seconds"], 40)
        self.assertEqual(after_finish["seconds_since_activity"], 10)

    def test_invalid_ids_stages_fingerprints_and_counters_fail_before_writing(self):
        for operation_id in ("../escape", "a/b", "", "x" * 65):
            with self.subTest(operation_id=operation_id), self.assertRaises(progress.ProgressError):
                progress.start(operation_id=operation_id, root=self.root)
        with self.assertRaises(progress.ProgressError):
            progress.start("made_up", operation_id="valid", root=self.root)
        with self.assertRaises(progress.ProgressError):
            progress.start(fingerprint64hex="g" * 64, operation_id="valid", root=self.root)
        with self.assertRaises(progress.ProgressError):
            progress.start(stdout_bytes=True, operation_id="valid", root=self.root)
        self.assertFalse((self.root / "state" / "progress").exists())
        self.assertIsNone(progress.read("../escape", root=self.root))

    def test_reader_projects_only_safe_fields_and_fails_closed_on_malformed_or_public_state(self):
        snapshot = progress.start(operation_id="safe-job", root=self.root)
        path = self.root / "state" / "progress" / "safe-job.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["transcript"] = "private meeting words"
        payload["exception"] = "sensitive error text"
        path.write_text(json.dumps(payload), encoding="utf-8")

        safe = progress.read("safe-job", root=self.root)
        self.assertEqual(safe["operation_id"], snapshot["operation_id"])
        self.assertNotIn("transcript", safe)
        self.assertNotIn("exception", safe)
        self.assertNotIn("private meeting words", json.dumps(safe))
        self.assertNotIn("sensitive error text", json.dumps(safe))

        path.write_text("{malformed", encoding="utf-8")
        self.assertIsNone(progress.read("safe-job", root=self.root))
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o644)
        self.assertIsNone(progress.read("safe-job", root=self.root))

    def test_reader_and_writer_reject_symlink_boundaries(self):
        outside = Path(self.temp.name).parent / (Path(self.temp.name).name + "-outside")
        outside.mkdir(mode=0o700)
        try:
            state = self.root / "state"
            state.mkdir(mode=0o700)
            (state / "progress").symlink_to(outside, target_is_directory=True)
            self.assertIsNone(progress.read("escape", root=self.root))
            with self.assertRaises(progress.ProgressError):
                progress.start(operation_id="escape", root=self.root)
            self.assertFalse((outside / "escape.json").exists())
        finally:
            (self.root / "state" / "progress").unlink(missing_ok=True)
            outside.rmdir()

    def test_writer_refuses_symlink_record_instead_of_replacing_its_target(self):
        progress.start(operation_id="link-job", root=self.root)
        path = self.root / "state" / "progress" / "link-job.json"
        outside = self.root / "outside.json"
        outside.write_text('{"keep":"unchanged"}', encoding="utf-8")
        outside.chmod(0o600)
        path.unlink()
        path.symlink_to(outside)
        self.assertIsNone(progress.read("link-job", root=self.root))
        with self.assertRaises(progress.ProgressError):
            progress.update("analysis", operation_id="link-job", root=self.root)
        self.assertEqual(outside.read_text(encoding="utf-8"), '{"keep":"unchanged"}')


if __name__ == "__main__":
    unittest.main()
