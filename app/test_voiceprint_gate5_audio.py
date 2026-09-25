from __future__ import annotations

import tempfile
import hashlib
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from test_voiceprint_gate5_inbox import synthetic_gate4_report

from voiceprint_gate5_audio import enroll_audio, review_audio
from voiceprint_provider_pyannote import ProviderError
from voiceprint_gate5_inbox import Gate5Error, Gate5Inbox


class Gate5AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.audio = root / "clip.wav"
        self.model = root / "community1-embedding.bin"
        self.model.write_bytes(b"synthetic model fixture, never loaded")
        self.model.chmod(0o600)
        model_hash = hashlib.sha256(self.model.read_bytes()).hexdigest()
        # Bind the test-only attestation to this synthetic asset, preserving real
        # hashing, report validation and rejection of changed model contents.
        attestation = patch("voiceprint_gate5_inbox.GATE4_MODEL_ASSET_SHA256", model_hash)
        attestation.start()
        self.addCleanup(attestation.stop)
        self.audio.write_bytes(b"audio")
        self.audio.chmod(0o600)
        self.inbox = Gate5Inbox(
            root / "state",
            clock=lambda: datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
            gate4_report=synthetic_gate4_report(root, model_hash),
            model_asset=self.model,
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def fake_embedder(model: Path, audio: Path) -> tuple[float, ...]:
        assert model.name == "community1-embedding.bin"
        assert audio.name == "clip.wav"
        return (1.0, 0.0)

    def test_explicit_audio_enroll_and_review(self):
        record = enroll_audio(
            self.inbox,
            "owner",
            model_asset=self.model,
            audio=self.audio,
            consent="yes",
            embedder=self.fake_embedder,
        )
        self.assertEqual("pending", record.state)
        candidate = review_audio(
            self.inbox,
            "SPEAKER_00",
            meeting_id="meeting-review",
            model_asset=self.model,
            audio=self.audio,
            consent="yes",
            embedder=self.fake_embedder,
        )
        self.assertEqual("candidate", candidate.status)
        self.assertEqual("owner", candidate.identity_id)

    def test_changed_model_is_rejected_before_embedding(self):
        self.model.write_bytes(b"changed synthetic model")
        with self.assertRaisesRegex(Gate5Error, "pinned Gate 4 model"):
            enroll_audio(self.inbox, "owner", model_asset=self.model, audio=self.audio,
                         consent="yes", embedder=lambda *_: self.fail("must not embed"))

    def test_missing_explicit_audio_fails_closed(self):
        with self.assertRaises(ProviderError):
            review_audio(
                self.inbox,
                "SPEAKER_00",
                meeting_id="meeting-review",
                model_asset=self.model,
                audio=Path(self.temp.name) / "missing.wav",
                consent="yes",
                embedder=self.fake_embedder,
            )

    def test_audio_without_explicit_consent_is_rejected(self):
        with self.assertRaises(Gate5Error):
            enroll_audio(
                self.inbox,
                "owner",
                model_asset=self.model,
                audio=self.audio,
                embedder=self.fake_embedder,
            )


if __name__ == "__main__":
    unittest.main()
