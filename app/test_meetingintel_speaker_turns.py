"""Synthetic regressions for speaker changes inside a single ASR segment."""
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
import importlib.util

HAS_UPSTREAM = all(importlib.util.find_spec(x) is not None for x in ("whispermlx", "pandas"))

from meetingintel_speaker_turns import project_speaker_turns, preserve_overlap_uncertainty


def conversation():
    return {"language": "en", "segments": [{
        "start": 1., "end": 5., "speaker": "SPEAKER_00",
        "text": "  Can we proceed?  Only after approval. ", "avg_logprob": -.2,
        "words": [
            {"word": "Can", "start": 1., "end": 1.3, "speaker": "SPEAKER_00"},
            {"word": "we", "start": 1.3, "end": 1.5, "speaker": "SPEAKER_00"},
            {"word": "proceed?", "start": 1.5, "end": 2., "speaker": "SPEAKER_00"},
            {"word": "Only", "start": 2.5, "end": 3., "speaker": "SPEAKER_01"},
            {"word": "after", "start": 3., "end": 3.5, "speaker": "SPEAKER_01"},
            {"word": "approval.", "start": 3.5, "end": 4.8, "speaker": "SPEAKER_01"},
        ]}]}


class SpeakerTurnTests(unittest.TestCase):
    @unittest.skipUnless(HAS_UPSTREAM, "optional Whisper/diarization runtime integration")
    def test_live_assignment_overlap_and_writers_keep_accepted_speaker_behavior(self):
        """Exercise real upstream assignment/export with live alignment types."""
        import numpy as np
        import pandas as pd
        from whispermlx.diarize import assign_word_speakers
        from whispermlx.utils import WriteSRT, WriteTXT
        from diarized_layer2_shadow import load_turns
        raw = conversation()
        for item in [*raw['segments'], *raw['segments'][0]['words']]:
            for key in ('start', 'end'):
                item[key] = np.float64(item[key])
        intervals = [
            {'start': 1., 'end': 2., 'speaker': 'SPEAKER_00'},
            {'start': 2.5, 'end': 5., 'speaker': 'SPEAKER_01'},
            {'start': 3., 'end': 3.5, 'speaker': 'SPEAKER_00'},
        ]
        expected = ['SPEAKER_00', 'SPEAKER_01', 'UNKNOWN', 'SPEAKER_01']
        options = {'highlight_words': False, 'max_line_count': None, 'max_line_width': None}
        for rows in (intervals, intervals[::-1]):
            assigned = assign_word_speakers(pd.DataFrame(rows), deepcopy(raw))
            projected = project_speaker_turns(preserve_overlap_uncertainty(assigned, rows))
            self.assertEqual(expected, [s['speaker'] for s in projected['segments']])
            self.assertEqual(6, projected['word_speaker_uncertainty']['timed_words'])
            self.assertEqual(1, projected['word_speaker_uncertainty']['equal_overlap_words'])
            self.assertEqual(raw['segments'][0]['text'], ''.join(s['text'] for s in projected['segments']))
            self.assertEqual([(w['start'], w['end']) for w in raw['segments'][0]['words']],
                             [(w['start'], w['end']) for s in projected['segments'] for w in s['words']])
            with tempfile.TemporaryDirectory() as directory:
                for writer in (WriteSRT, WriteTXT):
                    writer(directory)(projected, 'audio.m4a', options)
                turns, _ = load_turns(Path(directory))
                self.assertEqual(['SPEAKER_00', 'SPEAKER_01', 'UNKNOWN_SPEAKER', 'SPEAKER_01'],
                                 [t.speaker for t in turns])
                self.assertEqual(['Can we proceed?', 'Only', 'after', 'approval.'],
                                 [t.text for t in turns])
                text = (Path(directory) / 'audio.txt').read_text()
                for label, phrase in zip(expected, ('Can we proceed?', 'Only', 'after', 'approval.')):
                    self.assertIn(f'[{label}]: {phrase}', text)

    def test_live_alignment_numeric_scalars_match_saved_json_projection(self):
        import json
        import numpy as np
        for scalar in (np.float64, np.float32):
            raw = conversation()
            for item in [*raw['segments'], *raw['segments'][0]['words']]:
                for key in ('start', 'end'):
                    item[key] = scalar(item[key])
            # Alignment obtains times from pandas/NumPy. Saving its checkpoint
            # must not change which speaker turns the live exporter can retain.
            saved = json.loads(json.dumps(raw, default=float))
            with self.subTest(scalar=scalar.__name__):
                live = project_speaker_turns(raw)
                self.assertEqual(project_speaker_turns(saved), live)
                self.assertEqual(['SPEAKER_00', 'SPEAKER_01'],
                                 [s['speaker'] for s in live['segments']])

    def test_live_alignment_numeric_scalars_preserve_overlap_abstention(self):
        import numpy as np
        raw = conversation()
        for word in raw['segments'][0]['words']:
            word['start'], word['end'] = np.float64(word['start']), np.float64(word['end'])
        intervals = [dict(start=np.float64(1), end=np.float64(5), speaker=label)
                     for label in ('SPEAKER_00', 'SPEAKER_01')]
        result = preserve_overlap_uncertainty(raw, intervals)
        self.assertEqual(6, result['word_speaker_uncertainty']['timed_words'])
        self.assertEqual(6, result['word_speaker_uncertainty']['equal_overlap_words'])
        self.assertTrue(all(w['speaker'] == 'UNKNOWN' for w in result['segments'][0]['words']))

    @unittest.skipUnless(HAS_UPSTREAM, "optional Whisper/diarization runtime integration")
    def test_equal_overlap_cannot_choose_a_speaker_from_input_row_order(self):
        from whispermlx.diarize import assign_word_speakers
        import pandas as pd
        raw = {"language": "en", "segments": [{"text": "Agreed.", "start": 1., "end": 2., "words": [
            {"word": "Agreed.", "start": 1., "end": 2.}]}]}
        rows = [{"start": 0., "end": 3., "speaker": s} for s in ["SPEAKER_00", "SPEAKER_01"]]
        for ordered in [rows, rows[::-1]]:
            assigned = assign_word_speakers(pd.DataFrame(ordered), deepcopy(raw))
            result = project_speaker_turns(preserve_overlap_uncertainty(assigned, ordered))
            self.assertEqual("UNKNOWN", result["segments"][0]["speaker"])
            word = result["segments"][0]["words"][0]
            self.assertEqual("UNKNOWN", word["speaker"])
            self.assertEqual(["SPEAKER_00", "SPEAKER_01"], word["overlap_speakers"])
            self.assertEqual(raw["segments"][0]["text"], result["segments"][0]["text"])

    def test_overlap_policy_retains_clear_winner_and_preserves_input_evidence(self):
        raw = conversation(); before = deepcopy(raw)
        rows = [{"start": 1., "end": 5., "speaker": "SPEAKER_00"},
                {"start": 2.9, "end": 3.1, "speaker": "SPEAKER_01"}]
        result = preserve_overlap_uncertainty(raw, rows)
        self.assertEqual(raw["segments"], result["segments"])
        self.assertEqual(before, raw)
        self.assertEqual(0, result["word_speaker_uncertainty"]["equal_overlap_words"])

    @unittest.skipUnless(HAS_UPSTREAM, "optional Whisper/diarization runtime integration")
    def test_overlap_abstention_survives_export_reader_and_word_index(self):
        from whispermlx.utils import WriteSRT
        from diarized_layer2_shadow import load_turns
        raw = conversation(); raw["word_segments"] = deepcopy(raw["segments"][0]["words"])
        rows = [{"start": 2.5, "end": 3., "speaker": s} for s in ["SPEAKER_00", "SPEAKER_01"]]
        result = project_speaker_turns(preserve_overlap_uncertainty(raw, rows))
        self.assertEqual("UNKNOWN", result["word_segments"][3]["speaker"])
        with tempfile.TemporaryDirectory() as directory:
            WriteSRT(directory)(result, "audio.m4a", {"highlight_words": False, "max_line_count": None, "max_line_width": None})
            turns, _ = load_turns(Path(directory))
        self.assertEqual(["SPEAKER_00", "UNKNOWN_SPEAKER", "SPEAKER_01"], [t.speaker for t in turns])
        self.assertEqual(["Can we proceed?", "Only", "after approval."], [t.text for t in turns])

    @unittest.skipUnless(HAS_UPSTREAM, "optional Whisper/diarization runtime integration")
    def test_exported_unassigned_word_is_read_as_unknown_without_losing_its_text(self):
        from whispermlx.utils import WriteSRT
        from diarized_layer2_shadow import load_turns
        raw = conversation(); del raw["segments"][0]["words"][3]["speaker"]
        result = project_speaker_turns(raw)
        with tempfile.TemporaryDirectory() as directory:
            WriteSRT(directory)(result, "audio.m4a", {"highlight_words": False, "max_line_count": None, "max_line_width": None})
            turns, _ = load_turns(Path(directory))
        self.assertEqual(["SPEAKER_00", "UNKNOWN_SPEAKER", "SPEAKER_01"], [t.speaker for t in turns])
        self.assertEqual(["Can we proceed?", "Only", "after approval."], [t.text for t in turns])

    @unittest.skipUnless(HAS_UPSTREAM, "optional Whisper/diarization runtime integration")
    def test_same_segment_question_and_answer_keep_distinct_speakers_in_actual_writers(self):
        from whispermlx.utils import WriteSRT, WriteTXT
        result = project_speaker_turns(conversation())
        options = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
        for writer in (WriteSRT, WriteTXT):
            out = io.StringIO()
            writer("/unused").write_result(result, out, options)
            self.assertIn("[SPEAKER_00]: Can we proceed?", out.getvalue())
            self.assertIn("[SPEAKER_01]: Only after approval.", out.getvalue())
            self.assertNotIn("[SPEAKER_00]: Can we proceed? Only", out.getvalue())

    def test_exact_text_words_and_outer_timing_survive_without_mutating_input(self):
        raw = conversation(); before = deepcopy(raw)
        result = project_speaker_turns(raw)
        self.assertEqual(before, raw)
        self.assertEqual(raw["segments"][0]["text"], "".join(s["text"] for s in result["segments"]))
        self.assertEqual(raw["segments"][0]["words"], [w for s in result["segments"] for w in s["words"]])
        self.assertEqual(["SPEAKER_00", "SPEAKER_01"], [s["speaker"] for s in result["segments"]])
        self.assertEqual([(1., 2.), (2.5, 5.)], [(s["start"], s["end"]) for s in result["segments"]])
        result["segments"][0]["words"][0]["word"] = "Changed"
        self.assertEqual(before, raw)

    def test_unlabeled_word_stays_unknown_rather_than_inheriting_majority(self):
        raw = conversation(); del raw["segments"][0]["words"][3]["speaker"]
        result = project_speaker_turns(raw)
        self.assertEqual(["SPEAKER_00", "UNKNOWN", "SPEAKER_01"], [s["speaker"] for s in result["segments"]])

    def test_incomplete_alignment_keeps_all_text_and_abstains_on_mixed_segment(self):
        for damage in ("missing_time", "unrepresented_text", "nonfinite_time", "reversed_time"):
            raw = conversation()
            if damage == "missing_time": del raw["segments"][0]["words"][3]["start"]
            if damage == "unrepresented_text": raw["segments"][0]["text"] += " omitted"
            if damage == "nonfinite_time": raw["segments"][0]["words"][3]["start"] = float("nan")
            if damage == "reversed_time": raw["segments"][0]["words"][3]["end"] = 1.
            with self.subTest(damage=damage):
                result = project_speaker_turns(raw)
                self.assertEqual(1, len(result["segments"]))
                self.assertEqual("UNKNOWN", result["segments"][0]["speaker"])
                self.assertEqual(raw["segments"][0]["text"], result["segments"][0]["text"])

    def test_same_speaker_and_unaligned_legacy_segments_remain_compatible(self):
        raw = conversation()
        for word in raw["segments"][0]["words"]: word["speaker"] = "SPEAKER_00"
        self.assertEqual(raw["segments"], project_speaker_turns(raw)["segments"])
        del raw["segments"][0]["words"]
        self.assertEqual(raw["segments"], project_speaker_turns(raw)["segments"])

    def test_unicode_and_repeated_tokens_do_not_lose_text_or_punctuation(self):
        raw = {"language": "hi", "segments": [{"start": 0., "end": 2., "text": "हाँ, हाँ, ठीक।", "speaker": "SPEAKER_00", "words": [
            {"word": "हाँ,", "start": 0., "end": .5, "speaker": "SPEAKER_00"},
            {"word": "हाँ,", "start": .5, "end": 1., "speaker": "SPEAKER_01"},
            {"word": "ठीक।", "start": 1., "end": 2., "speaker": "SPEAKER_01"}]}]}
        result = project_speaker_turns(raw)
        self.assertEqual(2, len(result["segments"]))
        self.assertEqual(raw["segments"][0]["text"], "".join(s["text"] for s in result["segments"]))


if __name__ == "__main__":
    unittest.main()
