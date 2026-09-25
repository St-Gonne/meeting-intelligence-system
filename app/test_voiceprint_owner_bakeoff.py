from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import voiceprint_owner_bakeoff as bakeoff


class OwnerBakeoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "gate4"
        self.root.mkdir(mode=0o700)
        self.samples = self.root / "samples"
        self.samples.mkdir(mode=0o700)
        self.asset = self.root / "model.bin"
        self.asset.write_bytes(b"model fixture")
        self.sample_manifest = self.root / "samples.json"
        self.criteria_manifest = self.root / "criteria.json"
        self.providers_manifest = self.root / "providers.json"
        self.target_id = "owner-target-opaque"
        self.evaluation_id = "owner-gate4-evaluation"
        self.rows = []
        definitions = [
            ("enroll-phone", "target", "phone", "clean", "enrollment"),
            ("enroll-laptop", "target", "laptop", "clean", "enrollment"),
            ("cal-target-1", "target", "phone", "noisy", "calibration"),
            ("cal-target-2", "target", "laptop", "noisy", "calibration"),
            ("cal-negative", "negative", "phone", "clean", "calibration"),
            ("cal-unknown", "unknown", "laptop", "clean", "calibration"),
            ("test-phone-clean", "target", "phone", "clean", "test"),
            ("test-phone-noisy", "target", "phone", "noisy", "test"),
            ("test-laptop-clean", "target", "laptop", "clean", "test"),
            ("test-laptop-noisy", "target", "laptop", "noisy", "test"),
            ("test-negative-1", "negative", "phone", "noisy", "test"),
            ("test-negative-2", "negative", "laptop", "noisy", "test"),
            ("test-unknown-1", "unknown", "phone", "clean", "test"),
            ("test-unknown-2", "unknown", "laptop", "clean", "test"),
        ]
        for index, (sample_id, role, lane, condition, split) in enumerate(definitions):
            path = self.samples / f"sample-{index}.wav"
            path.write_text(sample_id, encoding="utf-8")
            self.rows.append({
                "sample_id": sample_id,
                "person_id": self.target_id if role == "target" else f"person-{sample_id}",
                "role": role,
                "lane": lane,
                "condition": condition,
                "split": split,
                "session_id": f"{split}-session-{index}",
                "capture_group": f"{split}-capture-{index}",
                "audio_path": str(path),
                "sha256": bakeoff._sha256(path),
                "size_bytes": path.stat().st_size,
                "permission_to_evaluate": "yes",
                "retention_instruction": "keep" if role == "target" else "drop",
                "single_speaker_attestation": {"one_speaker_only": True, "no_meaningful_overlap": True},
            })
        self.write_manifests()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifests(self):
        expiry = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        self.sample_manifest.write_text(json.dumps({
            "schema_version": 1,
            "manifest_type": "voiceprint_owner_samples",
            "evaluation_id": self.evaluation_id,
            "target_id": self.target_id,
            "sample_root": str(self.samples),
            "evaluation_expires_at": expiry,
            "samples": self.rows,
        }), encoding="utf-8")
        self.criteria_manifest.write_text(json.dumps({
            "schema_version": 1,
            "manifest_type": "voiceprint_owner_criteria",
            "evaluation_id": self.evaluation_id,
            "evaluation_expires_at": expiry,
            "owner_approved": True,
            "minimum_enrollment": 2,
            "minimum_calibration_target": 2,
            "minimum_calibration_non_target": 2,
            "minimum_test_target": 4,
            "minimum_test_negative": 2,
            "minimum_test_unknown": 2,
            "minimum_overall_target_accept_rate": 1.0,
            "minimum_cell_target_accept_rate": 1.0,
            "maximum_false_accepts": 0,
            "maximum_unknown_false_candidates": 0,
            "threshold_margin": 0.05,
            "required_test_cells": [
                {"lane": "phone", "condition": "clean"},
                {"lane": "phone", "condition": "noisy"},
                {"lane": "laptop", "condition": "clean"},
                {"lane": "laptop", "condition": "noisy"},
            ],
        }), encoding="utf-8")
        self.providers_manifest.write_text(json.dumps({
            "schema_version": 1,
            "manifest_type": "voiceprint_owner_providers",
            "evaluation_id": self.evaluation_id,
            "evaluation_expires_at": expiry,
            "providers": [{"provider_id": "mock", "command": ["/bin/echo"], "model_asset": str(self.asset)}],
        }), encoding="utf-8")

    def plan(self):
        return bakeoff.load_plan(self.root, self.sample_manifest, self.criteria_manifest, self.providers_manifest)

    @staticmethod
    def good_provider(path: Path):
        sample_id = path.read_text(encoding="utf-8")
        return (1.0, 0.05) if "target" in sample_id or sample_id.startswith(("enroll", "test-phone", "test-laptop")) else (0.05, 1.0)

    def test_valid_plan_is_ready_and_execution_is_inert_by_default(self):
        self.assertEqual((), bakeoff.readiness(self.plan()))
        output = io.StringIO()
        with redirect_stdout(output):
            result = bakeoff.main([
                "--isolated-root", str(self.root), "--samples", str(self.sample_manifest),
                "--criteria", str(self.criteria_manifest), "--providers", str(self.providers_manifest),
            ])
        self.assertEqual(0, result)
        self.assertIn("EXECUTION=disabled", output.getvalue())
        self.assertFalse((self.root / bakeoff.REPORT_NAME).exists())

    def test_passing_report_has_no_vectors_paths_or_target_identity(self):
        report = bakeoff.evaluate(self.plan(), {"mock": self.good_provider})
        self.assertEqual(["mock"], report["passed_providers"])
        serialized = json.dumps(report)
        self.assertNotIn("embedding", serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn(self.target_id, serialized)
        self.assertEqual(0, report["providers"][0]["false_accepts"])

    def test_false_accept_prevents_provider_from_passing(self):
        def unsafe(path: Path):
            sample_id = path.read_text(encoding="utf-8")
            if sample_id == "test-negative-1":
                return (1.0, 0.0)
            return self.good_provider(path)
        report = bakeoff.evaluate(self.plan(), {"mock": unsafe})
        self.assertEqual([], report["passed_providers"])
        self.assertEqual(1, report["providers"][0]["false_accepts"])

    def test_session_leakage_fails_closed(self):
        self.rows[-1]["session_id"] = self.rows[0]["session_id"]
        self.write_manifests()
        with self.assertRaisesRegex(bakeoff.BakeoffError, "leakage"):
            self.plan()

    def test_missing_required_condition_is_reported(self):
        self.rows = [row for row in self.rows if row["sample_id"] != "test-phone-noisy"]
        self.write_manifests()
        gaps = bakeoff.readiness(self.plan())
        self.assertIn("test_target_samples:missing=1", gaps)
        self.assertIn("test_cell_phone_noisy:missing=1", gaps)

    def test_changed_audio_fails_closed(self):
        Path(self.rows[0]["audio_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "changed"):
            self.plan()

    def test_overlap_or_missing_permission_fails_closed(self):
        for field in ("permission", "overlap"):
            with self.subTest(field=field):
                original = json.loads(json.dumps(self.rows))
                if field == "permission":
                    self.rows[0]["permission_to_evaluate"] = "no"
                else:
                    self.rows[0]["single_speaker_attestation"]["no_meaningful_overlap"] = False
                self.write_manifests()
                with self.assertRaises(bakeoff.BakeoffError):
                    self.plan()
                self.rows = original

    def test_nonfinite_or_wrong_dimension_provider_fails_closed(self):
        for vector in ((float("nan"), 1.0), (1.0,)):
            with self.subTest(vector=vector):
                calls = 0
                def provider(path: Path):
                    nonlocal calls
                    calls += 1
                    return vector if calls == len(self.rows) else self.good_provider(path)
                with self.assertRaises(bakeoff.BakeoffError):
                    bakeoff.evaluate(self.plan(), {"mock": provider})

    def test_report_write_is_private_and_atomic(self):
        path = bakeoff.write_report(self.plan(), bakeoff.evaluate(self.plan(), {"mock": self.good_provider}))
        self.assertEqual(0o600, path.stat().st_mode & 0o777)
        self.assertFalse(path.with_suffix(".tmp").exists())

    def test_expired_pending_cleanup_deletes_only_pending_audio(self):
        pending_ids = {"cal-negative", "test-negative-1"}
        for row in self.rows:
            if row["sample_id"] in pending_ids:
                row["retention_instruction"] = "pending"
        self.write_manifests()
        deleted = bakeoff.cleanup_expired_pending_samples(
            self.sample_manifest,
            now=datetime.now(timezone.utc) + timedelta(days=11),
        )
        self.assertEqual(tuple(sorted(pending_ids)), deleted)
        for row in self.rows:
            path = Path(row["audio_path"])
            if row["sample_id"] in pending_ids:
                self.assertFalse(path.exists())
            else:
                self.assertTrue(path.exists())
        marker = self.root / bakeoff.SAMPLE_CLEANUP_MARKER
        self.assertEqual(0o600, marker.stat().st_mode & 0o777)
        self.assertEqual(sorted(pending_ids), json.loads(marker.read_text())["deleted_sample_ids"])

    def test_cell_report_includes_accept_reject_and_unknown_accounting(self):
        report = bakeoff.evaluate(self.plan(), {"mock": self.good_provider})
        cells = report["providers"][0]["cells"]
        self.assertEqual(4, len(cells))
        for cell in cells:
            self.assertIn("true_accepts", cell)
            self.assertIn("false_rejects", cell)
            self.assertIn("false_accepts", cell)
            self.assertIn("unknown_false_candidates", cell)

    def test_provider_timeout_is_sanitized(self):
        spec = self.plan().providers[0]
        with patch.object(
            bakeoff.subprocess,
            "run",
            side_effect=bakeoff.subprocess.TimeoutExpired(spec.command, 300),
        ):
            with self.assertRaisesRegex(bakeoff.BakeoffError, "could not complete"):
                bakeoff._command_provider(spec)(Path(self.rows[0]["audio_path"]))

    def test_invalid_acceptance_limits_fail_closed(self):
        payload = json.loads(self.criteria_manifest.read_text(encoding="utf-8"))
        for field, value in (
            ("maximum_false_accepts", -1),
            ("maximum_false_accepts", True),
            ("maximum_unknown_false_candidates", "0"),
        ):
            with self.subTest(field=field, value=value):
                changed = dict(payload)
                changed[field] = value
                self.criteria_manifest.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(bakeoff.BakeoffError, "nonnegative integer"):
                    self.plan()
        self.criteria_manifest.write_text(json.dumps(payload), encoding="utf-8")

        changed = dict(payload)
        changed["maximum_false_accepts"] = 1
        self.criteria_manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "must be zero"):
            self.plan()

    def test_manifests_are_cross_linked_owner_approved_and_ids_are_opaque(self):
        criteria = json.loads(self.criteria_manifest.read_text(encoding="utf-8"))
        criteria["evaluation_id"] = "different-evaluation"
        self.criteria_manifest.write_text(json.dumps(criteria), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "identities disagree"):
            self.plan()

        self.write_manifests()
        criteria = json.loads(self.criteria_manifest.read_text(encoding="utf-8"))
        criteria["owner_approved"] = False
        self.criteria_manifest.write_text(json.dumps(criteria), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "owner approval"):
            self.plan()

        self.write_manifests()
        self.rows[0]["person_id"] = "Real Person Name"
        self.write_manifests()
        with self.assertRaisesRegex(bakeoff.BakeoffError, "opaque lowercase"):
            self.plan()

    def test_model_asset_must_be_local_to_gate_and_not_a_symlink(self):
        payload = json.loads(self.providers_manifest.read_text(encoding="utf-8"))
        outside = self.root.parent / "outside-model.bin"
        outside.write_bytes(b"outside")
        payload["providers"][0]["model_asset"] = str(outside)
        self.providers_manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "inside the isolated root"):
            self.plan()

        linked = self.root / "linked-model.bin"
        linked.symlink_to(self.asset)
        payload["providers"][0]["model_asset"] = str(linked)
        self.providers_manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "unsafe"):
            self.plan()

        model_dir = self.root / "model-directory"
        model_dir.mkdir()
        (model_dir / "nested-link").symlink_to(self.asset)
        payload["providers"][0]["model_asset"] = str(model_dir)
        self.providers_manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(bakeoff.BakeoffError, "contains a symlink"):
            self.plan()

    def test_write_report_rejects_non_allowlisted_or_sensitive_payload(self):
        plan = self.plan()
        with self.assertRaisesRegex(bakeoff.BakeoffError, "schema"):
            bakeoff.write_report(plan, {"embedding": [1.0]})
        report = bakeoff.evaluate(plan, {"mock": self.good_provider})
        report["providers"][0]["leak"] = plan.target_id
        with self.assertRaisesRegex(bakeoff.BakeoffError, "forbidden"):
            bakeoff.write_report(plan, report)


if __name__ == "__main__":
    unittest.main()
