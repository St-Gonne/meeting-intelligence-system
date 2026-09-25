"""Independent synthetic regression cases for exact GUI artifact boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import meetingintel_actions as actions


class ActionBoundaryReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.environment = patch.dict(os.environ, {
            "MI_RUNTIME_ROOT": str(self.root / "runtime"),
            "MI_LEARNING_ROOT": str(self.root / "learning"),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_missing_durable_transcript_is_not_completed(self):
        output = self.root / "output"
        output.mkdir()
        report = output / "report.txt"
        report.write_text("Synthetic report")
        brief = output / "brief_input_2026-09-22.md"
        brief.write_text("Synthetic brief")
        source_id = "a" * 64
        row = dict(id=source_id, identity="b" * 64, source_identity="b" * 64,
                   source_kind="laptop", created_at="2026-09-22T10:00:00+05:30",
                   category="ready", recording_state="retained", status="ready", warning="")
        records = {source_id: {"transcript_path": str(output / "missing.txt"),
                              "layer2_report_path": str(report),
                              "meeting_date_ist": "2026-09-22"}}
        with patch.object(actions.cli, "OUTPUT_DIR", output), \
             patch.object(actions, "_staged_transcript", return_value=None), \
             patch.object(actions.pipeline, "collect_today_records", return_value=[]), \
             patch.object(actions.pipeline, "render_brief", return_value="Synthetic brief"):
            result = actions._project(row, records)
        self.assertNotEqual("completed", result["category"])
        self.assertFalse(result["can_process"])

    def test_staged_transcript_does_not_follow_source_folder_symlink(self):
        staging = self.root / "staging"
        staging.mkdir()
        external = self.root / "external"
        external.mkdir()
        identity = "b" * 64
        (external / "run_manifest.json").write_text(json.dumps({
            "status": "success", "source_identity": identity,
        }))
        (external / "audio.txt").write_text("Synthetic external text")
        (staging / ("audio_first_" + identity)).symlink_to(external, target_is_directory=True)
        with patch.object(actions, "STAGING_ROOT", staging), \
             patch.object(actions.pipeline, "QWEN_STAGING_ROOT", self.root / "absent"):
            result = actions._staged_transcript({"source_identity": identity, "source_kind": "laptop"})
        self.assertIsNone(result)

    def test_artifact_root_symlink_is_rejected(self):
        external = self.root / "external"
        external.mkdir()
        (external / "report.txt").write_text("Synthetic external text")
        root_link = self.root / "output"
        root_link.symlink_to(external, target_is_directory=True)
        self.assertIsNone(actions._safe_file(root_link / "report.txt", root_link))


if __name__ == "__main__":
    unittest.main()
