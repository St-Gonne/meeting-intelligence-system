from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import meetingintel_voice_cli as voice


class MeetingIntelVoiceCliTests(unittest.TestCase):
    def test_review_requires_interactive_terminal(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(
                1,
                voice.run_voice(["review", "last"], terminal_isatty=lambda: False),
            )

    def test_declined_audio_review_writes_nothing(self):
        inbox = Mock()
        inbox.store.all_records.return_value = [Mock(state="keep", identity_id="owner-id")]
        inbox.private_display_name.return_value = "Owner"
        preview = Mock()
        with (
            patch.object(voice, "_inbox", return_value=inbox),
            patch.object(
                voice,
                "_latest_successful_phone_diarization",
                return_value=(Path("/source"), Path("/diarization"), "a" * 64),
            ),
            patch.object(voice, "run_preview", preview),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                0,
                voice.run_review_last(
                    input_func=lambda _prompt: "no", terminal_isatty=lambda: True
                ),
            )
        preview.assert_not_called()

    def test_unique_candidate_requires_then_records_exact_confirmation(self):
        inbox = Mock()
        inbox.store.all_records.return_value = [Mock(state="keep", identity_id="owner-id")]
        inbox.private_display_name.return_value = "Owner"
        inbox._read_metadata.return_value = {
            "reviews": {
                "candidate-token": {
                    "meeting_id": "meeting-aaaaaaaaaaaa",
                    "observed_speaker": "SPEAKER_01",
                    "status": "candidate",
                    "consumed": False,
                }
            }
        }
        answers = iter(("VOICE", "YES"))
        with (
            patch.object(voice, "_inbox", return_value=inbox),
            patch.object(
                voice,
                "_latest_successful_phone_diarization",
                return_value=(Path("/source"), Path("/diarization"), "a" * 64),
            ),
            patch.object(
                voice,
                "run_preview",
                return_value={
                    "named_label": "SPEAKER_01",
                    "scores": {
                        "SPEAKER_00": {"status": "unknown", "score": 0.2},
                        "SPEAKER_01": {"status": "candidate", "score": 0.8},
                    },
                },
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                0,
                voice.run_review_last(
                    input_func=lambda _prompt: next(answers),
                    terminal_isatty=lambda: True,
                ),
            )
        inbox.confirm_candidate.assert_called_once_with(
            "candidate-token", operator_confirmed=True, display_name="Owner"
        )


if __name__ == "__main__":
    unittest.main()
