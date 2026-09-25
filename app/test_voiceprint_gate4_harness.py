from __future__ import annotations
from mi_paths import public_path

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import voiceprint_gate4_harness as harness


NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
CREATED_AT = "2026-07-19T12:00:00+00:00"
EXPIRY = "2026-08-01T00:00:00+00:00"
PENDING_EXPIRY = "2026-07-25T12:00:00+00:00"


class Gate4HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = (Path(self.temporary.name) / "isolated-evaluation").resolve()
        self.root.mkdir()
        self.samples_root = (Path(self.temporary.name) / "synthetic-samples").resolve()
        self.samples_root.mkdir()
        for sample_id in ("a-1", "a-2", "b-1", "b-2"):
            (self.samples_root / f"{sample_id}.m4a").write_bytes(b"synthetic fixture only")
        self.consent_path = self.root / "consent.json"
        self.test_set_path = self.root / "test-set.json"
        self.criteria_path = self.root / "criteria.json"
        self._write_manifests()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_json(self, path: Path, payload: dict):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _consent(self):
        def sample(person_id, sample_id, session, capture):
            return {
                "sample_id": sample_id,
                "audio_path": str(self.samples_root / f"{sample_id}.m4a"),
                "session_id": session,
                "capture_group": capture,
                "supported_1to1_confirmed": True,
                "single_speaker_attestation": {
                    "only_named_speaker": True,
                    "no_meaningful_overlap": True,
                },
                "candidate_id": f"candidate-{sample_id}",
                "source_locator": {
                    "recording_id": f"recording-{sample_id}",
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                },
            }

        return {
            "schema_version": 1,
            "manifest_type": "gate4_consent",
            "evaluation_id": "synthetic-evaluation-1",
            "created_at": CREATED_AT,
            "evaluation_expires_at": EXPIRY,
            "persons": [
                {
                    "person_id": "person-a",
                    "permission_to_evaluate": "yes",
                    "retention_instruction": "pending",
                    "retention_expires_at": PENDING_EXPIRY,
                    "samples": [
                        sample("person-a", "a-1", "session-a-1", "capture-a-1"),
                        sample("person-a", "a-2", "session-a-2", "capture-a-2"),
                    ],
                },
                {
                    "person_id": "person-b",
                    "permission_to_evaluate": "yes",
                    "retention_instruction": "keep",
                    "retention_expires_at": None,
                    "samples": [
                        sample("person-b", "b-1", "session-b-1", "capture-b-1"),
                        sample("person-b", "b-2", "session-b-2", "capture-b-2"),
                    ],
                },
            ],
        }

    def _test_set(self):
        ref = lambda person_id, sample_id: {
            "person_id": person_id,
            "sample_id": sample_id,
        }
        return {
            "schema_version": 1,
            "manifest_type": "gate4_test_set",
            "evaluation_id": "synthetic-evaluation-1",
            "positive_comparisons": [
                {
                    "case_id": "positive-1",
                    "kind": "positive",
                    "left": ref("person-a", "a-1"),
                    "right": ref("person-a", "a-2"),
                }
            ],
            "negative_comparisons": [
                {
                    "case_id": "negative-1",
                    "kind": "negative",
                    "left": ref("person-a", "a-1"),
                    "right": ref("person-b", "b-1"),
                }
            ],
            "unknown_cases": [
                {
                    "case_id": "unknown-1",
                    "sample": ref("person-b", "b-2"),
                    "against": [ref("person-a", "a-1")],
                    "expected_behavior": "abstain",
                }
            ],
        }

    def _criteria(self):
        return {
            "schema_version": 1,
            "manifest_type": "gate4_criteria",
            "evaluation_id": "synthetic-evaluation-1",
            "evaluation_expires_at": EXPIRY,
            "owner_approved": True,
            "threshold_selection": {
                "method": "owner-supplied synthetic fixture decision",
                "threshold": 0.8,
                "rule": "owner-supplied threshold rule",
            },
            "positive": {"minimum_pass_rate": 1.0, "minimum_count": 1},
            "negative": {"maximum_false_match_rate": 0.0, "minimum_count": 1},
            "unknown": {
                "minimum_abstention_rate": 1.0,
                "maximum_false_candidate_rate": 0.0,
                "minimum_count": 1,
            },
            "false_candidate": {"maximum_rate": 0.0, "minimum_count": 1},
            "cleanup": {
                "required_result": "retain_only_explicit_keep",
                "delete_unretained_material": True,
                "delete_on_expiry": True,
                "no_tombstone": True,
            },
        }

    def _write_manifests(self):
        self._write_json(self.consent_path, self._consent())
        self._write_json(self.test_set_path, self._test_set())
        self._write_json(self.criteria_path, self._criteria())

    def _prepare(self):
        return harness.prepare_gate4(
            self.root,
            self.consent_path,
            self.test_set_path,
            self.criteria_path,
            now=NOW,
        )

    def test_valid_manifests_prepare_in_memory_plan_and_cleanup_contract(self):
        plan = self._prepare()
        self.assertEqual("synthetic-evaluation-1", plan.evaluation_id)
        self.assertEqual(
            (("person-a", "pending"), ("person-b", "keep")),
            plan.cleanup.retention_by_person,
        )
        self.assertTrue(plan.cleanup.delete_unretained_material)
        self.assertTrue(plan.cleanup.delete_on_expiry)
        self.assertTrue(plan.cleanup.no_tombstone)
        self.assertFalse(plan.aggregate_report_path.exists())
        self.assertFalse(plan.cleanup.measurement_root.exists())
        self.assertTrue(plan.cleanup.pending_provenance_path.exists())
        provenance = json.loads(plan.cleanup.pending_provenance_path.read_text())
        self.assertEqual(["candidate-a-1", "candidate-a-2"], [item["candidate_id"] for item in provenance["candidates"]])
        self.assertNotIn("audio_path", plan.cleanup.pending_provenance_path.read_text())

    def test_readiness_is_inert_without_execute_flag(self):
        with patch.object(harness, "_now", return_value=NOW), patch.object(
            harness, "execute_gate4"
        ) as execute, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                0,
                harness.main(
                    [
                        "--isolated-root", str(self.root),
                        "--consent-manifest", str(self.consent_path),
                        "--test-set-manifest", str(self.test_set_path),
                        "--criteria-manifest", str(self.criteria_path),
                    ]
                ),
            )
        execute.assert_not_called()
        self.assertIn("GATE4_EXECUTION=disabled", output.getvalue())
        self.assertFalse(plan_path_exists(self.root, harness.AGGREGATE_REPORT_FILENAME))

    def test_execution_requires_explicit_flag_and_does_not_create_state(self):
        plan = self._prepare()
        with self.assertRaises(harness.Gate4ExecutionDisabled):
            harness.execute_gate4(plan)
        self.assertEqual([], list(self.root.glob("measurements/*")))
        self.assertFalse(plan.aggregate_report_path.exists())

    def test_missing_or_malformed_manifests_fail_closed(self):
        missing = self.root / "missing.json"
        with self.assertRaises(harness.Gate4PathError):
            harness.prepare_gate4(self.root, missing, self.test_set_path, self.criteria_path, now=NOW)
        self.criteria_path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

    def test_repository_normal_and_symlink_paths_are_rejected(self):
        with self.assertRaises(harness.Gate4PathError):
            harness.prepare_gate4(
                Path(__file__).parent,
                self.consent_path,
                self.test_set_path,
                self.criteria_path,
                now=NOW,
            )
        with self.assertRaises(harness.Gate4PathError):
            harness._validate_directory(Path(str(public_path('home/Movies/meetily-recordings'))), "normal root")

        symlink_root = Path(self.temporary.name) / "root-link"
        symlink_root.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(harness.Gate4PathError):
            harness.prepare_gate4(
                symlink_root,
                self.consent_path,
                self.test_set_path,
                self.criteria_path,
                now=NOW,
            )

        symlink_manifest = self.root / "consent-link.json"
        symlink_manifest.symlink_to(self.consent_path)
        with self.assertRaises(harness.Gate4PathError):
            harness.prepare_gate4(
                self.root,
                symlink_manifest,
                self.test_set_path,
                self.criteria_path,
                now=NOW,
            )

        for unsafe_name in ("staging-clips", "drive-sync-clips"):
            unsafe_root = (Path(self.temporary.name) / unsafe_name).resolve()
            unsafe_root.mkdir()
            with self.assertRaises(harness.Gate4PathError):
                harness._validate_directory(unsafe_root, "unsafe root")

    def test_consent_requires_explicit_permission_retention_and_supported_scope(self):
        payload = self._consent()
        payload["persons"][0]["retention_expires_at"] = None
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["permission_to_evaluate"] = "no"
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["samples"][0]["supported_1to1_confirmed"] = False
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["retention_expires_at"] = "2026-08-01T00:00:00+00:00"
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["samples"][0]["single_speaker_attestation"]["only_named_speaker"] = False
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["samples"][0]["single_speaker_attestation"]["no_meaningful_overlap"] = False
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

    def test_symlink_audio_and_malformed_test_set_fail_closed(self):
        audio_link = self.samples_root / "audio-link.m4a"
        audio_link.symlink_to(self.samples_root / "a-1.m4a")
        payload = self._consent()
        payload["persons"][0]["samples"][0]["audio_path"] = str(audio_link)
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4PathError):
            self._prepare()

        self._write_json(self.consent_path, self._consent())
        test_set = self._test_set()
        test_set["positive_comparisons"][0]["right"]["sample_id"] = "b-1"
        self._write_json(self.test_set_path, test_set)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

    def test_criteria_fails_closed_until_owner_supplies_all_decisions(self):
        criteria = self._criteria()
        criteria["threshold_selection"]["threshold"] = None
        self._write_json(self.criteria_path, criteria)
        with self.assertRaises(harness.Gate4CriteriaIncomplete):
            self._prepare()

        criteria = self._criteria()
        criteria["owner_approved"] = False
        self._write_json(self.criteria_path, criteria)
        with self.assertRaises(harness.Gate4CriteriaIncomplete):
            self._prepare()

        criteria = self._criteria()
        criteria["unknown"].pop("maximum_false_candidate_rate")
        self._write_json(self.criteria_path, criteria)
        with self.assertRaises(harness.Gate4CriteriaIncomplete):
            self._prepare()

    def test_explicit_root_contains_no_normal_state_or_output_writes(self):
        plan = self._prepare()
        self.assertTrue(plan.isolated_root.is_relative_to(Path(self.temporary.name).resolve()))
        self.assertFalse((Path(__file__).parent / harness.AGGREGATE_REPORT_FILENAME).exists())
        self.assertFalse((Path(__file__).parent / "measurements").exists())
        self.assertEqual(
            {"consent.json", "test-set.json", "criteria.json", harness.PENDING_PROVENANCE_FILENAME},
            {path.name for path in self.root.iterdir()},
        )

    def test_pending_expiry_at_boundary_deletes_material_and_provenance_without_tombstone(self):
        plan = self._prepare()
        material = plan.cleanup.candidate_material_root / "candidate-a-1"
        material.mkdir(parents=True)
        (material / "synthetic-vector.bin").write_bytes(b"synthetic only")
        deleted = harness.cleanup_gate4_material(plan, now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(("candidate-a-1", "candidate-a-2"), deleted)
        self.assertFalse(material.exists())
        self.assertFalse(plan.cleanup.pending_provenance_path.exists())
        self.assertFalse((plan.isolated_root / "disabled-candidate.json").exists())

    def test_drop_deletes_candidate_material_immediately_and_keep_remains_isolated(self):
        payload = self._consent()
        payload["persons"][0]["retention_instruction"] = "drop"
        payload["persons"][0]["retention_expires_at"] = None
        self._write_json(self.consent_path, payload)
        plan = self._prepare()
        drop_material = plan.cleanup.candidate_material_root / "candidate-a-1"
        keep_material = plan.cleanup.candidate_material_root / "candidate-b-1"
        drop_material.mkdir(parents=True)
        keep_material.mkdir(parents=True)
        harness.cleanup_gate4_material(plan, now=NOW)
        self.assertFalse(drop_material.exists())
        self.assertTrue(keep_material.exists())
        self.assertTrue(keep_material.is_relative_to(plan.isolated_root))

    def test_pending_expiry_is_timezone_aware_and_ten_day_trial_only(self):
        payload = self._consent()
        payload["persons"][0]["retention_expires_at"] = "2026-07-25T12:00:00"
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()

        payload = self._consent()
        payload["persons"][0]["retention_expires_at"] = "2026-07-30T12:00:00+00:00"
        self._write_json(self.consent_path, payload)
        with self.assertRaises(harness.Gate4ManifestError):
            self._prepare()


def plan_path_exists(root: Path, filename: str) -> bool:
    return (root / filename).exists()


if __name__ == "__main__":
    unittest.main()
