from __future__ import annotations

import ast
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from voiceprint_synthetic_candidate import (
    InvalidSyntheticVector,
    InvalidStateRoot,
    SyntheticVoiceprintStore,
    TrustedMapEntry,
    cosine_similarity,
)


class FrozenClock:
    def __init__(self):
        self.value = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


class SyntheticVoiceprintCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "disposable-voiceprint-state"
        self.clock = FrozenClock()
        self.store = SyntheticVoiceprintStore(self.root, clock=self.clock)

    def tearDown(self):
        self.temporary.cleanup()

    def test_cosine_threshold_pass_miss_and_configurable_threshold(self):
        self.store.enroll("synthetic-a", (1.0, 0.0))
        results = self.store.match(
            {"SPEAKER_00": (1.0, 0.0), "SPEAKER_01": (0.0, 1.0)},
            threshold=0.9,
        )
        self.assertEqual("candidate", results[0].status)
        self.assertEqual("synthetic-a", results[0].identity_id)
        self.assertAlmostEqual(1.0, results[0].score)
        self.assertEqual("unknown", results[1].status)
        self.assertIsNone(results[1].identity_id)
        self.assertAlmostEqual(0.0, results[1].score)
        self.assertAlmostEqual(0.0, cosine_similarity((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(
            "unknown",
            self.store.match({"SPEAKER_00": (0.8, 0.6)}, threshold=0.9)[0].status,
        )
        self.assertEqual(
            "candidate",
            self.store.match({"SPEAKER_00": (0.8, 0.6)}, threshold=0.7)[0].status,
        )

    def test_one_to_one_collision_and_ties_are_exclusive_and_deterministic(self):
        self.store.enroll("identity-b", (1.0, 0.0))
        self.store.enroll("identity-a", (1.0, 0.0))
        results = self.store.match(
            {
                "SPEAKER_B": (1.0, 0.0),
                "SPEAKER_A": (1.0, 0.0),
            },
            threshold=0.9,
        )
        by_speaker = {result.observed_speaker: result for result in results}
        self.assertEqual("identity-a", by_speaker["SPEAKER_A"].identity_id)
        self.assertEqual("identity-b", by_speaker["SPEAKER_B"].identity_id)
        self.assertEqual(
            2,
            len({result.identity_id for result in results if result.identity_id is not None}),
        )

        collision_store = SyntheticVoiceprintStore(
            self.root.parent / "collision", clock=self.clock
        )
        collision_store.enroll("identity-one", (1.0, 0.0))
        collision_store.enroll("identity-two", (0.0, 1.0))
        collision = collision_store.match(
            {"SPEAKER_A": (1.0, 0.0), "SPEAKER_B": (1.0, 0.0)}, threshold=0.9
        )
        self.assertEqual("identity-one", collision[0].identity_id)
        self.assertEqual("unknown", collision[1].status)

    def test_candidate_is_not_confirmed_without_explicit_confirmation(self):
        self.store.enroll("synthetic-a", (1.0, 0.0))
        candidate = self.store.match({"SPEAKER_00": (1.0, 0.0)})[0]
        self.assertFalse(candidate.confirmed)
        self.assertIsNone(
            self.store.confirm_match(candidate, operator_confirmed=False)
        )
        self.assertFalse(self.store.get("synthetic-a").match_confirmed)
        confirmed = self.store.confirm_match(candidate, operator_confirmed=True)
        self.assertIsInstance(confirmed, TrustedMapEntry)
        self.assertEqual(
            {
                "speaker_label": "SPEAKER_00",
                "identity": "synthetic-a",
                "status": "confirmed",
                "source": "operator_confirmed_synthetic_candidate",
            },
            confirmed.as_dict(),
        )

    def test_confirmation_and_retention_consent_are_separate(self):
        self.store.enroll("synthetic-a", (1.0, 0.0))
        candidate = self.store.match({"SPEAKER_00": (1.0, 0.0)})[0]
        self.store.confirm_match(candidate, operator_confirmed=True)
        self.assertEqual("pending", self.store.get("synthetic-a").state)
        self.store.grant_retention("synthetic-a", consent=None)
        self.assertEqual("pending", self.store.get("synthetic-a").state)
        retained = self.store.grant_retention("synthetic-a", consent=True)
        self.assertEqual("keep", retained.state)
        self.assertTrue(retained.match_confirmed)

    def test_explicit_no_after_confirmed_match_deletes_state(self):
        self.store.enroll("confirmed-a", (1.0, 0.0))
        candidate = self.store.match({"SPEAKER_00": (1.0, 0.0)})[0]
        self.store.confirm_match(candidate, operator_confirmed=True)
        self.assertEqual("pending", self.store.get("confirmed-a").state)
        self.assertIsNone(self.store.grant_retention("confirmed-a", consent=False))
        self.assertIsNone(self.store.get("confirmed-a"))
        self.assertFalse(self.store.state_path.exists())
        self.assertEqual([], list(self.root.rglob("*")))

    def test_explicit_no_for_unconfirmed_pending_deletes_without_tombstone(self):
        self.store.enroll("pending-a", (1.0, 0.0))
        self.assertIsNone(self.store.grant_retention("pending-a", consent=False))
        self.assertIsNone(self.store.get("pending-a"))
        self.assertFalse(self.store.state_path.exists())
        self.assertEqual([], list(self.root.rglob("*")))

    def test_deleted_identity_requires_fresh_enrollment_before_retention(self):
        self.store.enroll("deleted-a", (1.0, 0.0))
        self.store.grant_retention("deleted-a", consent=False)
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.grant_retention("deleted-a", consent=True)
        self.assertFalse(self.store.state_path.exists())
        self.assertEqual([], list(self.root.rglob("*")))
        self.store.enroll("deleted-a", (1.0, 0.0))
        self.assertEqual("keep", self.store.grant_retention("deleted-a", consent=True).state)

    def test_pending_expires_at_exactly_thirty_days(self):
        self.store.enroll("synthetic-a", (1.0, 0.0))
        self.clock.advance(days=30)
        self.assertIsNone(self.store.get("synthetic-a"))
        self.assertFalse(self.store.state_path.exists())
        self.assertEqual((), self.store.all_records())

    def test_drop_deletes_pending_and_revoke_keep_deletes_without_tombstone(self):
        self.store.enroll("pending-a", (1.0, 0.0))
        self.assertTrue(self.store.drop("pending-a"))
        self.assertIsNone(self.store.get("pending-a"))
        self.assertFalse(self.store.state_path.exists())

        self.store.enroll("kept-a", (1.0, 0.0))
        self.store.grant_retention("kept-a", consent=True)
        self.assertTrue(self.store.revoke_retention("kept-a"))
        self.assertIsNone(self.store.get("kept-a"))
        self.assertFalse(self.store.state_path.exists())
        self.assertEqual([], list(self.root.rglob("*")))

    def test_invalid_dimensions_and_malformed_vectors_fail_closed(self):
        invalid = ((), (0.0, 0.0), (1.0, math.nan), (1.0, math.inf), (True, 0.0), ((1.0, 2.0),))
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaises(InvalidSyntheticVector):
                    self.store.enroll("invalid", vector)
        self.store.enroll("synthetic-a", (1.0, 0.0))
        with self.assertRaises(InvalidSyntheticVector):
            self.store.enroll("synthetic-b", (1.0, 0.0, 0.0))
        with self.assertRaises(InvalidSyntheticVector):
            self.store.match({"SPEAKER_00": (1.0, 0.0, 0.0)})
        with self.assertRaises(InvalidSyntheticVector):
            cosine_similarity((1.0, 0.0), (0.0, 0.0))

    def test_all_state_is_under_explicit_disposable_root(self):
        self.store.enroll("synthetic-a", (1.0, 0.0))
        self.assertEqual(self.root.resolve(), self.store.state_path.parent)
        self.assertTrue(self.store.state_path.is_file())
        self.assertFalse((Path(__file__).parent / "voiceprint_candidates.json").exists())
        with self.assertRaises(InvalidStateRoot):
            SyntheticVoiceprintStore(Path(__file__).parent / "not-a-state-root", clock=self.clock)

    def test_normal_meetingintel_modules_do_not_import_or_invoke_component(self):
        module_names = (
            "meetingintel_pipeline.py",
            "meetingintel_cli.py",
            "diarized_layer2_shadow.py",
        )
        component_name = "voiceprint_synthetic_candidate"
        for module_name in module_names:
            tree = ast.parse((Path(__file__).parent / module_name).read_text(encoding="utf-8"))
            imported = []
            called_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_names.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called_names.append(node.func.attr)
            self.assertNotIn(component_name, imported)
            self.assertNotIn("SyntheticVoiceprintStore", called_names)


if __name__ == "__main__":
    unittest.main()
