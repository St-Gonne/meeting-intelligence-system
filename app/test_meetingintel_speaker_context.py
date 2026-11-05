"""Synthetic regressions for keeping speaker evidence through general analysis."""

import argparse
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import meetingintel_pipeline as pipeline
from test_meetingintel_pipeline import valid_eligibility_summary, _SyntheticSession


REPORT = """===== LAYER 2 INTELLIGENCE REPORT =====
## TL;DR
SPEAKER_00 proposed Friday; SPEAKER_01 required approval.
## SPEAKER MAP
- SPEAKER_00
- SPEAKER_01
- SPEAKER_02
## MEETING ARC
- SPEAKER_00 proposed Friday; SPEAKER_01 objected; SPEAKER_02 offered to ask.
## SIGNALS & STAGE
- Delivery — conditional — owner/source speaker: SPEAKER_01 — evidence: SPEAKER_01: "Only if finance approves."
## NOTABLE EXCHANGES
- SPEAKER_00: "Could we deliver Friday?" — SPEAKER_01: "Only if finance approves."
## OPEN THREADS / ACTION CANDIDATES
- Ask finance — timing: [NOT STATED]
  OWNER: SPEAKER_02
## ANALYST NOTES
Friday delivery is a proposal, conditional on approval, not an agreed deadline.
===== END REPORT =====
"""
BRIEF = """MEETING_TYPE: Mixed/other
SIGNAL: Friday delivery remains conditional on finance approval.
INTEL: none
ACTION: Ask finance for approval; no agreed delivery deadline.
OWNER: SPEAKER_02
URGENCY: this-week
ROLODEX_FLAG: null
"""
SRT = """1
00:00:01,000 --> 00:00:02,000
[SPEAKER_00]: Could we deliver Friday?

2
00:00:03,000 --> 00:00:04,000
[SPEAKER_01]: Only if finance approves.

3
00:00:05,000 --> 00:00:06,000
[SPEAKER_02]: I'll ask finance.
"""


class SpeakerContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        (self.stage / 'audio.srt').write_text(SRT)
        (self.stage / 'run_manifest.json').write_text(json.dumps({
            'transcription_mode': 'qwen3-asr-1.7b-contiguous',
            'candidate_stage_provenance': {'qwen_asr': {'model': {
                'identifier': pipeline.QWEN_MODEL_IDENTIFIER,
                'revision': pipeline.QWEN_MODEL_REVISION,
            }}},
        }))
        folder = self.root / 'source'
        folder.mkdir()
        audio = folder / 'audio.m4a'
        audio.write_bytes(b'synthetic only')
        self.created = datetime.fromisoformat('2026-10-20T20:34:04+05:30')
        self.metadata = dict(meeting_name='Synthetic meeting', duration_seconds=6,
                             created_at=self.created.isoformat(), status='completed')
        self.source = pipeline.ProductionSource(
            'laptop_capture', 'a' * 64, folder, audio, 'b' * 64,
            self.created, 6, 'Synthetic meeting', self.metadata, 'c' * 64)
        self.args = argparse.Namespace(dry_run=False, output_dir=self.root / 'output',
            ollama_url='unused', model='unused', context_pack_snapshot=None,
            phone_transcription_backend='legacy')
        self.summary = valid_eligibility_summary(('SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'))
        for p in (
            patch.dict(os.environ, MI_LEARNING_ROOT=str(self.root / 'learning'),
                       MI_RUNTIME_ROOT=str(self.root / 'runtime'),
                       MI_PROGRESS_ROOT=str(self.root / 'progress'),
                       MI_RUN_HEALTH_ROOT=str(self.root / 'health'), MI_ACTOR='agent'),
            patch.object(pipeline.gpu, 'Session', _SyntheticSession),
            patch.object(pipeline.identity, 'create_presentation', return_value={'status': 'disabled'}),
            patch.object(pipeline, 'stage_diarized_eligibility',
                         return_value=(self.stage, self.root / 'eligibility', self.summary)),
        ):
            p.start()
            self.addCleanup(p.stop)

    def select(self):
        return pipeline.select_audio_first_result(self.args, self.source,
            pipeline.load_prompt(pipeline.DEFAULT_LAYER2_PROMPT_PATH),
            pipeline.load_prompt(pipeline.DEFAULT_PROMPT_PATH))

    def test_general_analysis_keeps_turns_timing_and_speaker_owner(self):
        with patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(REPORT, {}), (BRIEF, {})]) as model:
            result = self.select()
        prompt = model.call_args_list[0].args[0]
        for label, words, start in (
            ('SPEAKER_00', 'Could we deliver Friday?', '00:00:01.000'),
            ('SPEAKER_01', 'Only if finance approves.', '00:00:03.000'),
            ('SPEAKER_02', "I'll ask finance.", '00:00:05.000'),
        ):
            self.assertIn(f'**{label}:** {words}', prompt)
            self.assertIn(start, prompt)
            self.assertIn(f'**{label}:** {words}', result.layer2_transcript_input)
        self.assertEqual('diarized_general', result.processing_mode)
        self.assertFalse(result.routing_eligibility['eligible'])
        self.assertIn('OWNER: SPEAKER_02', result.summary_text)
        self.assertNotIn('FLAT-TRANSCRIPT SAFETY', model.call_args_list[1].args[0])
        self.assertIn('Friday delivery is a proposal', model.call_args_list[1].args[0])
        self.assertEqual(2, model.call_count)
        path, digest = pipeline.persist_selected_transcript(self.args.output_dir,
            self.source.fingerprint, self.source.display_name, '2026-10-20',
            result.layer2_transcript_input)
        self.assertEqual(result.layer2_transcript_input, path.read_text())
        self.assertEqual(pipeline.sha256_file(path), digest)

    def test_supported_route_failure_keeps_speakers_in_general_fallback(self):
        self.summary.clear()
        self.summary.update(valid_eligibility_summary())
        with patch.object(pipeline, 'run_diarized_production',
                          side_effect=pipeline.ProcessingFallback('synthetic failure')), \
             patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(REPORT, {}), (BRIEF, {})]) as model:
            result = self.select()
        self.assertIn('**SPEAKER_02:**', model.call_args_list[0].args[0])
        self.assertEqual('diarized_general_fallback', result.processing_mode)
        self.assertEqual('synthetic failure', result.fallback_reason)

    def test_general_analysis_rejects_named_or_unobserved_action_owner_before_saving(self):
        for owner in ('Fixture Person', 'SPEAKER_99'):
            with self.subTest(owner=owner), patch.object(pipeline, 'summarize_with_ollama',
                    return_value=(REPORT.replace('OWNER: SPEAKER_02', f'OWNER: {owner}'), {})) as model, \
                 patch.object(pipeline, 'save_layer2_report') as save:
                with self.assertRaises(pipeline.ProcessingFallback):
                    self.select()
                self.assertEqual(1, model.call_count)
                save.assert_not_called()

    def test_general_brief_rejects_unobserved_owner(self):
        with patch.object(pipeline, 'summarize_with_ollama',
                side_effect=[(REPORT, {}), (BRIEF.replace('OWNER: SPEAKER_02', 'OWNER: SPEAKER_99'), {})]):
            with self.assertRaises(pipeline.ProcessingFallback):
                self.select()

    def test_general_report_rejects_invented_speaker_in_claim_even_with_valid_owner(self):
        with patch.object(pipeline, 'summarize_with_ollama',
                          return_value=(REPORT.replace('SPEAKER_00 proposed', 'SPEAKER_99 proposed'), {})) as model:
            with self.assertRaises(pipeline.ProcessingFallback):
                self.select()
            self.assertEqual(1, model.call_count)

    def test_report_speaker_list_is_completed_from_observed_labels_without_regeneration(self):
        incomplete = REPORT.replace('- SPEAKER_01\n', '').replace('- SPEAKER_02\n', '')
        with patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(incomplete, {}), (BRIEF, {})]) as model:
            result = self.select()
        report = Path(result.layer2_report_path).read_text()
        import diarized_layer2_shadow as shadow
        speaker_map = shadow.extract_report_section(report, 'SPEAKER MAP')
        for label in ('SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'):
            self.assertIn(label, speaker_map)
        self.assertEqual(2, model.call_count)
        self.assertIn("SPEAKER_02", model.call_args_list[1].args[0])

    def test_general_meetily_uses_accepted_turns_instead_of_flat_source(self):
        with patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(REPORT, {}), (BRIEF, {})]) as model:
            result = pipeline.select_meeting_result(self.args, self.metadata, self.created,
                'old untagged export', self.source.display_name, self.source.fingerprint,
                pipeline.load_prompt(pipeline.DEFAULT_LAYER2_PROMPT_PATH),
                pipeline.load_prompt(pipeline.DEFAULT_PROMPT_PATH), self.source.source_folder)
        self.assertIn('**SPEAKER_02:**', model.call_args_list[0].args[0])
        self.assertNotIn('old untagged export', model.call_args_list[0].args[0])
        self.assertEqual('diarized_general', result.processing_mode)

    def test_unknown_turn_and_unclear_owner_are_preserved(self):
        (self.stage / 'audio.srt').write_text(SRT.replace('[SPEAKER_02]', '[UNKNOWN_SPEAKER]'))
        report = REPORT.replace('OWNER: SPEAKER_02', 'OWNER: owner unclear').replace('SPEAKER_02', 'UNKNOWN_SPEAKER')
        brief = BRIEF.replace('OWNER: SPEAKER_02', 'OWNER: owner unclear')
        with patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(report, {}), (brief, {})]) as model:
            result = self.select()
        self.assertIn("**UNKNOWN_SPEAKER:** I'll ask finance.", model.call_args_list[0].args[0])
        self.assertIn('OWNER: owner unclear', result.summary_text)

    def test_out_of_order_turns_keep_every_speaker_and_timestamp(self):
        blocks = SRT.strip().split('\n\n')
        (self.stage / 'audio.srt').write_text('\n\n'.join(reversed(blocks)) + '\n')
        with patch.object(pipeline, 'summarize_with_ollama',
                          side_effect=[(REPORT, {}), (BRIEF, {})]):
            result = self.select()
        text = result.layer2_transcript_input
        self.assertLess(text.index('**SPEAKER_00:**'), text.index('**SPEAKER_01:**'))
        self.assertLess(text.index('**SPEAKER_01:**'), text.index('**SPEAKER_02:**'))
        for stamp in ('00:00:01.000', '00:00:02.000', '00:00:03.000',
                      '00:00:04.000', '00:00:05.000', '00:00:06.000'):
            self.assertIn(stamp, text)


if __name__ == '__main__':
    unittest.main()
