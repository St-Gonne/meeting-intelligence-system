"""Synthetic harness-boundary tests; these never run ASR or local models."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).parent / "scripts" / "validate_local_reliability.py"
spec = importlib.util.spec_from_file_location("validate_local_reliability", SCRIPT)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class ReliabilityHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.capture = self.root / "capture"
        self.source.mkdir()
        self.capture.mkdir()

    def test_layout_has_only_explicit_proof_outputs(self):
        paths = proof.layout(self.source, self.capture, self.root)
        source = SimpleNamespace(created_at=datetime.now(timezone.utc))
        args = proof.arguments(paths, source)
        self.assertTrue(args.ledger_path.is_relative_to(self.root))
        self.assertTrue(args.output_dir.is_relative_to(self.root))
        self.assertEqual("http://127.0.0.1:11435/api/generate", args.ollama_url)
        self.assertFalse(args.refresh_existing)
        self.assertIsNone(args.context_pack)
        self.assertFalse(paths["validation"].exists())

    def test_source_outside_proof_root_rejected_without_reads(self):
        with self.assertRaisesRegex(proof.ProofError, "inside_proof_root"):
            proof.layout(self.root.parent, self.capture, self.root)

    def test_production_root_overlap_rejected(self):
        with patch.object(proof, "PROJECT_ROOT", self.source):
            with self.assertRaisesRegex(proof.ProofError, "overlaps_production"):
                proof.layout(self.source, self.capture, self.root)

    def test_source_output_overlap_rejected(self):
        source = self.root / "validation" / "source"
        source.mkdir(parents=True)
        with self.assertRaisesRegex(proof.ProofError, "input_overlaps"):
            proof.layout(source, self.capture, self.root)

    def test_symlink_ancestor_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(proof.ProofError, "unsafe_path"):
            proof.layout(link, self.capture, self.root)

    def test_inventory_rejects_hardlinks_and_preserves_content_hash(self):
        path = self.source / "synthetic.txt"
        path.write_text("Synthetic evidence")
        before = proof.inventory(self.source)
        self.assertEqual(18, before["synthetic.txt"]["bytes"])
        os.link(path, self.source / "link.txt")
        with self.assertRaisesRegex(proof.ProofError, "unsafe_proof_evidence"):
            proof.inventory(self.source)

    def test_wrappers_invoke_real_modules_and_redirect_only_output_roots(self):
        paths = proof.layout(self.source, self.capture, self.root)
        for name in ("wrappers", "logs", "staging", "shadow", "learning"):
            paths[name].mkdir(parents=True, mode=0o700)
        helper, shadow = proof.make_wrappers(paths)
        for path in (helper, shadow):
            text = path.read_text()
            compile(text, str(path), "exec")
            self.assertIn("actual.main()", text)
            self.assertNotIn("unittest.mock", text)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
        self.assertIn("actual.OUTPUT_ROOT =", helper.read_text())
        self.assertIn("actual.DIARIZATION_ROOT =", shadow.read_text())
        self.assertIn("actual.SHADOW_OUTPUT_ROOT =", shadow.read_text())
        original = proof.pipeline.DIARIZATION_HELPER_SCRIPT
        with self.assertRaisesRegex(ValueError, "injected"):
            with proof.isolated_configuration(paths, (helper, shadow)):
                self.assertEqual(helper, proof.pipeline.DIARIZATION_HELPER_SCRIPT)
                self.assertEqual(str(paths["learning"]), os.environ["MI_LEARNING_ROOT"])
                raise ValueError("injected")
        self.assertEqual(original, proof.pipeline.DIARIZATION_HELPER_SCRIPT)

    def test_invalid_source_fails_before_model_or_pipeline_work_and_hides_details(self):
        with patch.object(proof, "load_laptop_meeting_source", side_effect=ValueError("PRIVATE SYNTHETIC DETAIL")), \
             patch.object(proof.pipeline, "prepare_ollama") as preflight, \
             patch.object(proof.pipeline, "process_explicit_laptop_sources") as processing:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = proof.main(["--source-folder", str(self.source), "--capture-root", str(self.capture), "--proof-root", str(self.root)])
        self.assertEqual(1, code)
        preflight.assert_not_called()
        processing.assert_not_called()
        self.assertNotIn("PRIVATE SYNTHETIC DETAIL", output.getvalue())
        self.assertEqual("ValueError", json.loads(output.getvalue())["error_code"])

    def test_summary_metrics_exclude_content_and_arbitrary_fields(self):
        value = proof.safe_metrics({"ollama_metrics": {
            "layer2": {"done_reason": "stop", "eval_count": 20, "response": "text", "secret": "value"},
            "private": {"eval_count": 1},
        }})
        self.assertEqual({"layer2": {"done_reason": "stop", "eval_count": 20}}, value)

    def test_substituted_ledger_link_rejected_before_source_or_model_reads(self):
        state = self.root / "validation" / "state"
        state.mkdir(parents=True, mode=0o700)
        target = self.root / "not-the-proof-ledger.json"
        target.write_text("untouched synthetic sentinel")
        (state / "processed_ledger.json").symlink_to(target)
        with patch.object(proof, "load_laptop_meeting_source") as loader:
            with self.assertRaisesRegex(proof.ProofError, "symlink_in_proof_evidence"):
                proof.run_proof(self.source, self.capture, self.root)
        loader.assert_not_called()
        self.assertEqual("untouched synthetic sentinel", target.read_text())


if __name__ == "__main__":
    unittest.main()
