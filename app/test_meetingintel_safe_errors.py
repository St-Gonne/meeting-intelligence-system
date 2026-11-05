"""Focused privacy tests for safe failure classification and storage."""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from gpu_lock import Deferred
import meetingintel_learning as learning
from meetingintel_observe import safe_error
from meetingintel_runtime import BusyError, RuntimeSafetyError


class SafeErrorTests(unittest.TestCase):
    def test_handled_cli_failure_preserves_stage_code_in_terminal_observation(self):
        from types import SimpleNamespace
        import meetingintel_observe as observe
        from meetingintel_pipeline import ProcessingFallback
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            with patch.dict(os.environ, {"MI_LEARNING_ROOT": str(Path(temporary)/"learning"), "MI_ORIGIN": "cli"}):
                source = SimpleNamespace(fingerprint="a"*64, source_kind="laptop_capture")
                @observe.observed("transcription", source_scope=True)
                def rejected(source):
                    raise ProcessingFallback("diarization_helper:exit_1:transcription_repetition_rejected")
                @observe.production_attempt
                def attempt(selected, **kwargs):
                    try:
                        rejected(selected[0])
                    except ProcessingFallback:
                        return 1
                self.assertEqual(1, attempt((source,), processed=set()))
                final = learning.source_state(source.fingerprint)
                self.assertEqual("operation.failed", final["event_type"])
                self.assertEqual("transcription_repetition_rejected", final["error_code"])

    def test_known_failure_shapes_are_coarsened_without_losing_safe_codes(self):
        self.assertEqual(
            safe_error(RuntimeError(
                "diarization_helper:exit_1:transcription_repetition_rejected"
            )),
            "transcription_repetition_rejected",
        )
        self.assertEqual(
            safe_error(RuntimeError("diarization_helper:exit_2")),
            "diarization_failed",
        )
        self.assertEqual(safe_error(RuntimeError("transcription_failed")), "transcription_failed")
        self.assertEqual(safe_error(RuntimeError("transcription_qwen:window_did_not_finish")),
                         "transcription_failed")
        self.assertEqual(safe_error(RuntimeError("invalid_owner_field")), "invalid_owner_field")
        self.assertEqual(
            safe_error(RuntimeError("flat_layer2:invalid_owner_field")),
            "invalid_owner_field",
        )
        self.assertEqual(
            safe_error(RuntimeError("capture_incomplete:operator_correction_requires_review")),
            "capture_incomplete",
        )
        self.assertEqual(safe_error(BusyError("local operation lock is occupied")), "busy")
        self.assertEqual(safe_error(Deferred("private GPU detail")), "deferred")
        self.assertEqual(
            safe_error(RuntimeSafetyError("runtime directory unavailable")),
            "runtime_unavailable",
        )

    def test_unknown_or_embedded_codes_are_not_promoted(self):
        self.assertEqual(
            safe_error(RuntimeError(
                "model output mentions transcription_failed at /private/model/secret"
            )),
            "operation_failed",
        )
        self.assertEqual(
            safe_error(RuntimeError(
                "diarization_helper:exit_1:private_model_path"
            )),
            "operation_failed",
        )
        self.assertEqual(
            safe_error(RuntimeSafetyError("runtime directory unavailable: /private/root")),
            "operation_failed",
        )

    def test_new_allowlisted_codes_survive_isolated_storage_without_raw_text(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp", prefix="mi-safe-errors-") as temporary_root:
            root = Path(temporary_root) / "learning"
            with patch.dict(os.environ, {"MI_LEARNING_ROOT": str(root)}):
                raw_private = "/private/meeting/model-secret"
                cases = (
                    (
                        RuntimeError(
                            "diarization_helper:exit_1:transcription_repetition_rejected"
                        ),
                        "transcription",
                    ),
                    (Deferred(f"GPU busy at {raw_private}"), "processing"),
                    (RuntimeSafetyError("runtime directory unavailable"), "preflight"),
                    (
                        RuntimeError(f"unrecognized model output: {raw_private}"),
                        "analysis",
                    ),
                )
                for index, (exception, stage) in enumerate(cases, start=1):
                    learning.emit(
                        "stage.failed",
                        source_id=f"{index:064x}",
                        stage=stage,
                        outcome="failure",
                        error_code=safe_error(exception),
                    )

                with sqlite3.connect(root / "events.sqlite") as connection:
                    stored = connection.execute(
                        "SELECT error_code FROM events WHERE event_type='stage.failed' ORDER BY source_id"
                    ).fetchall()

            self.assertEqual(
                [row[0] for row in stored],
                [
                    "transcription_repetition_rejected",
                    "deferred",
                    "runtime_unavailable",
                    "operation_failed",
                ],
            )
            database_bytes = (root / "events.sqlite").read_bytes()
            self.assertNotIn(b"/private/meeting/model-secret", database_bytes)
            self.assertNotIn(b"GPU busy at", database_bytes)


if __name__ == "__main__":
    unittest.main()
