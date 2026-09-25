from __future__ import annotations

import tempfile
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from voiceprint_gate5_inbox import Gate5Error, Gate5Inbox, InboxCandidate, GATE4_FROZEN_THRESHOLD, GATE4_MODEL_ASSET_SHA256, GATE4_PROVIDER_ID


def synthetic_gate4_report(root: Path, model_hash: str = GATE4_MODEL_ASSET_SHA256) -> Path:
    report = root / "synthetic_gate4_report.json"
    report.write_text(json.dumps({
        "schema_version": 1, "report_type": "voiceprint_owner_bakeoff",
        "providers": [{"provider_id": GATE4_PROVIDER_ID, "passed": True,
                       "threshold": GATE4_FROZEN_THRESHOLD, "model_asset_sha256": model_hash}],
    }))
    report.chmod(0o600)
    return report


class Gate5InboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = lambda: datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        self.report = synthetic_gate4_report(Path(self.temp.name))
        self.inbox = Gate5Inbox(Path(self.temp.name) / "private", clock=self.clock, gate4_report=self.report, model_asset_sha256=GATE4_MODEL_ASSET_SHA256)

    def tearDown(self):
        self.temp.cleanup()

    def test_confirmation_and_retention_are_separate(self):
        self.inbox.enroll_pending("owner", (1.0, 0.0))
        candidate = self.inbox.review({"SPEAKER_00": (1.0, 0.0)}, meeting_id="meeting-review")[0]
        self.assertEqual("candidate", candidate.status)
        trusted = self.inbox.confirm(candidate, operator_confirmed=True)
        self.assertEqual("confirmed", trusted.status)
        self.assertEqual((), self.inbox.trusted_map())
        self.inbox.set_retention("owner", consent=True)
        self.assertEqual(
            ({"meeting_id": "meeting-review", "observed_speaker": "SPEAKER_00", "identity": "owner", "status": "confirmed", "source": "operator_confirmed_gate5"},),
            self.inbox.trusted_map(),
        )

    def test_forged_and_cross_meeting_candidates_are_rejected(self):
        self.inbox.enroll_pending("owner", (1.0, 0.0))
        candidate = self.inbox.review({"SPEAKER_00": (1.0, 0.0)}, meeting_id="meeting-a")[0]
        forged = InboxCandidate("SPEAKER_00", "owner", 0.99, "candidate", meeting_id="meeting-a", candidate_id=candidate.candidate_id, threshold=GATE4_FROZEN_THRESHOLD)
        with self.assertRaises(Gate5Error):
            self.inbox.confirm(forged, operator_confirmed=True)
        cross_meeting = InboxCandidate(candidate.observed_speaker, candidate.identity_id, candidate.score, candidate.status, meeting_id="meeting-b", candidate_id=candidate.candidate_id, threshold=GATE4_FROZEN_THRESHOLD)
        with self.assertRaises(Gate5Error):
            self.inbox.confirm(cross_meeting, operator_confirmed=True)
        self.inbox.confirm(candidate, operator_confirmed=True)
        with self.assertRaises(Gate5Error):
            self.inbox.confirm(candidate, operator_confirmed=True)

    def test_expiry_removes_pending_vector_and_mapping_metadata(self):
        self.inbox.enroll_pending("owner", (1.0, 0.0))
        candidate = self.inbox.review({"SPEAKER_00": (1.0, 0.0)}, meeting_id="meeting-a")[0]
        self.inbox.confirm(candidate, operator_confirmed=True)
        self.clock = lambda: datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        refreshed = Gate5Inbox(Path(self.temp.name) / "private", clock=self.clock, gate4_report=self.report, model_asset_sha256=GATE4_MODEL_ASSET_SHA256)
        self.assertEqual((), refreshed.trusted_map())
        self.assertFalse(refreshed.state_path.exists())
        self.assertFalse(refreshed.map_path.exists())

    def test_gate4_report_and_model_attestation_are_required(self):
        with self.assertRaises(Gate5Error):
            Gate5Inbox(Path(self.temp.name) / "missing-report", clock=self.clock)
        with self.assertRaises(Gate5Error):
            Gate5Inbox(Path(self.temp.name) / "missing-model", clock=self.clock, gate4_report=self.report)

    def test_unknown_is_not_confirmable_and_drop_removes_state(self):
        self.inbox.enroll_pending("owner", (1.0, 0.0))
        unknown = self.inbox.review({"SPEAKER_00": (0.0, 1.0)}, meeting_id="meeting-review")[0]
        self.assertEqual("unknown", unknown.status)
        with self.assertRaises(ValueError):
            self.inbox.confirm(unknown, operator_confirmed=True)
        self.assertIsNone(self.inbox.set_retention("owner", consent=False))
        self.assertFalse(self.inbox.state_path.exists())


if __name__ == "__main__":
    unittest.main()
