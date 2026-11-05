import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import voiceprint_gate5_preview as preview
from voiceprint_gate5_preview import _render_preview


class Gate5TrackToolTests(unittest.TestCase):
    def test_scheduler_path_uses_existing_local_ffmpeg_without_executing_it(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            binary = root / "homebrew/bin/ffmpeg"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"synthetic executable; never run")
            binary.chmod(0o700)
            output = root / "track.wav"

            def synthetic_run(command, **kwargs):
                self.assertEqual(str(binary), command[0])
                Path(command[-1]).write_bytes(b"synthetic track")
                return SimpleNamespace(returncode=0)

            with patch.object(preview.shutil, "which", return_value=None), \
                 patch.object(preview.Path, "home", return_value=root), \
                 patch.object(preview.subprocess, "run", side_effect=synthetic_run) as run:
                self.assertEqual(40., preview._make_track(root / "source.m4a",
                    [{"start": 0., "end": 40.}], output))
                self.assertEqual(2, run.call_count)
                self.assertEqual(0o600, output.stat().st_mode & 0o777)

    def test_missing_or_nonexecutable_fallback_fails_before_extraction(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            binary = root / "homebrew/bin/ffmpeg"
            binary.parent.mkdir(parents=True)
            with patch.object(preview.shutil, "which", return_value=None), \
                 patch.object(preview.Path, "home", return_value=root), \
                 patch.object(preview.subprocess, "run") as run:
                for exists in (False, True):
                    if exists:
                        binary.write_bytes(b"not executable")
                        binary.chmod(0o600)
                    with self.subTest(exists=exists), self.assertRaisesRegex(
                            preview.PreviewError, "ffmpeg is unavailable"):
                        preview._make_track(root / "source.m4a", [{"start": 0., "end": 40.}],
                                            root / "track.wav")
                run.assert_not_called()


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
                display_name="Owner",
                scores={
                    "SPEAKER_00": {"score": 0.2, "status": "unknown"},
                    "SPEAKER_01": {"score": 0.6, "status": "candidate"},
                },
                source_date="2026-09-07T12:41:41+05:30",
                source_duration=10.0,
            )
            self.assertIn("SPEAKER_01 — Owner (voice-match suggestion; THIS CALL not human-confirmed)", rendered)
            self.assertIn("[SPEAKER_00]: anonymous", rendered)
            self.assertNotIn("SPEAKER_00 — Owner", rendered)

    def test_ambiguous_candidates_keep_all_labels_anonymous(self):
        with tempfile.TemporaryDirectory() as directory:
            srt = Path(directory) / "audio.srt"
            srt.write_text("[SPEAKER_00]: a\n[SPEAKER_01]: b\n", encoding="utf-8")
            rendered = _render_preview(
                srt,
                named_label=None,
                display_name="Owner",
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
