import tempfile
import unittest
from pathlib import Path

from voiceprint_gate5_preview import _render_preview


class Gate5PreviewPresentationTests(unittest.TestCase):
    def test_only_unique_candidate_label_is_decorated_and_unconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            srt = Path(directory) / "audio.srt"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: anonymous\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_01]: retained\n",
                encoding="utf-8",
            )
            rendered = _render_preview(
                srt,
                named_label="SPEAKER_01",
                display_name="Alex",
                scores={
                    "SPEAKER_00": {"score": 0.2, "status": "unknown"},
                    "SPEAKER_01": {"score": 0.6, "status": "candidate"},
                },
                source_date="2026-09-07T12:41:41+05:30",
                source_duration=10.0,
            )
            self.assertIn("SPEAKER_01 — Alex (voice-match suggestion; THIS CALL not human-confirmed)", rendered)
            self.assertIn("[SPEAKER_00]: anonymous", rendered)
            self.assertNotIn("SPEAKER_00 — Alex", rendered)

    def test_ambiguous_candidates_keep_all_labels_anonymous(self):
        with tempfile.TemporaryDirectory() as directory:
            srt = Path(directory) / "audio.srt"
            srt.write_text("[SPEAKER_00]: a\n[SPEAKER_01]: b\n", encoding="utf-8")
            rendered = _render_preview(
                srt,
                named_label=None,
                display_name="Alex",
                scores={
                    "SPEAKER_00": {"score": 0.6, "status": "candidate"},
                    "SPEAKER_01": {"score": 0.7, "status": "candidate"},
                },
                source_date="2026-09-07T12:41:41+05:30",
                source_duration=10.0,
            )
            self.assertIn("No label was named", rendered)
            self.assertNotIn("voice-match suggestion; THIS CALL", rendered)
            self.assertIn("[SPEAKER_00]: a", rendered)
            self.assertIn("[SPEAKER_01]: b", rendered)


if __name__ == "__main__":
    unittest.main()
