"""Stability evidence, privacy and cross-process progress regressions."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import meetingintel_run_health as health
import meetingintel_progress as progress
import meetingintel_observe as observe
import meetingintel_pipeline as pipeline


class RunHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'MI_LEARNING_ROOT': str(self.root / 'learning'),
            'MI_RUN_HEALTH_ROOT': str(self.root / 'health'), 'MI_RUNTIME_ROOT': str(self.root / 'runtime'),
            'MI_PROGRESS_ROOT': str(self.root), 'MI_ORIGIN': 'cli', 'MI_ACTOR': 'human'})
        self.env.start()
        for key in ('MI_OPERATION_ID', 'MI_CANARY', 'MI_RUN_HEALTH_OFF'):
            os.environ.pop(key, None)
        self.token = health._runs.set({})
        self.active = health._active.set(None)
        self.progress = progress._current_operation.set(None)
        health.record_suite_pass(1)

    def tearDown(self):
        health._runs.reset(self.token)
        health._active.reset(self.active)
        progress._current_operation.reset(self.progress)
        self.env.stop()
        self.tmp.cleanup()

    def complete(self, index, *, canary=False):
        with patch.dict(os.environ, {'MI_CANARY': '1' if canary else '0'}):
            fingerprint = f'{index:064x}'
            health.begin(fingerprint, 'phone_recording', select=True)
            for stage in ('transcription', 'analysis', 'processing', 'transcript', 'ledger'):
                health.event('stage.started', stage=stage)
                health.event('stage.succeeded', stage=stage)
            health.identity_result({'status': 'matched', 'reason': 'matched', 'private_name': 'must not be stored'})
            health.reuse(False)
            health.gpu_timing(12, 345)
            health.delivered([fingerprint])
            return health._active.get()

    def test_streak_requires_tested_build_five_distinct_human_ordinary_runs(self):
        for index in range(1, 5):
            self.complete(index)
        self.complete(99, canary=True)
        self.assertEqual(health.summary()['stability_streak'], 4)
        self.complete(5)
        self.assertTrue(health.summary()['stable'])
        self.assertEqual(health.summary()['canary_runs'], 1)
        with patch.object(health.learning, 'build_id', return_value='e' * 64):
            self.assertFalse(health.summary()['stable'])

    def test_specific_worker_reason_survives_parent_projection_and_canary_health(self):
        import meetingintel_identity as identity
        stage = self.root / 'synthetic-stage'
        stage.mkdir()
        (stage / 'audio.srt').write_text('[SPEAKER_00]: Synthetic\n')
        for index, reason in enumerate(('ffmpeg_unavailable', 'stage_lineage_mismatch',
                                        'speaker_track_too_short', 'worker_launch_failed', 'worker_timeout',
                                        'worker_terminated', 'worker_exit_failed',
                                        'worker_output_invalid', 'worker_exception'), 1):
            with self.subTest(reason=reason), patch.dict(os.environ, {'MI_CANARY': '1',
                    'MI_CANARY_PHASE': 'cheap', 'MI_ORIGIN': 'scheduled', 'MI_ACTOR': 'scheduler'}):
                source = SimpleNamespace(source_folder=self.root / 'source', fingerprint=f'{index:064x}')
                health.begin(source.fingerprint, 'phone_recording', select=True)
                with patch.object(identity, 'enabled', return_value=True), \
                     patch('meetingintel_gpu.run', return_value=SimpleNamespace(returncode=1,
                        stdout=json.dumps({'status':'unavailable', 'reason':reason}))):
                    result = identity.create_presentation(source, stage, self.root / 'isolated-output')
                health.identity_result(result)
                health.finish(health._active.get(), 'failed', 'validation_failure')
                rows, gaps = health.records()
                self.assertEqual(0, gaps)
                self.assertEqual('anonymous', rows[-1]['identity_status'])
                self.assertEqual(reason, rows[-1]['identity_reason'])
                self.assertEqual('validation_failure', rows[-1]['failure_code'])
                self.assertIn(reason, health.summary()['header'])
                projection = identity._json(self.root / 'isolated-output/transcripts/identity' /
                                            (source.fingerprint + '.json'))
                self.assertEqual(reason, projection['reason'])
                self.assertNotIn('Synthetic', json.dumps(rows[-1]))
        self.assertEqual(0, health.summary()['stability_streak'])

    def test_failure_fallback_and_retry_do_not_become_first_try_pass(self):
        self.complete(1)
        health.begin('2' * 64, 'phone_recording', select=True)
        health.event('stage.failed', stage='analysis', error_code='invalid_owner_field')
        for stage in ('processing', 'transcript', 'ledger'):
            health.event('stage.succeeded', stage=stage)
        health.delivered(['2' * 64])
        rows, gaps = health.records()
        self.assertEqual(gaps, 0)
        self.assertFalse(rows[-1]['first_try'])
        self.assertEqual(rows[-1]['failure_codes'], ['invalid_owner_field'])
        self.assertEqual(health.summary()['stability_streak'], 0)

    def test_pre_health_staging_attempt_is_not_first_try(self):
        import whispermlx_diarization_helper as helper
        for index, (kind, evidence, prior) in enumerate((
            ('phone_recording', 'legacy_empty', True),
            ('laptop_capture', 'legacy_suffix', True),
            ('phone_recording', 'qwen_empty', True),
            ('phone_recording', 'failed_manifest', True),
            ('laptop_capture', 'failed_manifest', True),
            ('phone_recording', 'unrelated_manifest', False),
            ('phone_recording', 'prefix_collision', False),
            ('phone_recording', 'none', False),
            ('meetily', 'meetily_empty', True),
        ), 1):
            with self.subTest(kind=kind, evidence=evidence):
                staging = self.root / f'staging-{index}'
                staging.mkdir()
                identity = f'{index:064x}'
                source_folder = self.root / f'source-{index}'
                fingerprint = (pipeline.build_laptop_capture_fingerprint(identity) if kind == 'laptop_capture'
                               else pipeline.build_audio_first_fingerprint(identity))
                if evidence != 'none':
                    name = {'legacy_empty': 'audio_first_' + identity,
                            'legacy_suffix': 'audio_first_' + identity + '__old',
                            'qwen_empty': 'audio_first_' + identity[:16] + '__old',
                            'prefix_collision': 'audio_first_' + identity[:16] + '__other',
                            'meetily_empty': source_folder.name}.get(evidence, 'old-failed-run')
                    folder = staging / name
                    folder.mkdir()
                    if evidence in {'failed_manifest', 'unrelated_manifest', 'prefix_collision'}:
                        (folder / 'run_manifest.json').write_text(json.dumps({
                            'status': 'failure', 'source_identity': identity if prior else 'f' * 64}))
                source = SimpleNamespace(fingerprint=fingerprint, source_kind=kind,
                                         source_identity=identity if kind != 'meetily' else None,
                                         source_folder=source_folder)
                @observe.observed('processing', source_scope=True)
                def selected(args, source):
                    # Evidence created by this new attempt must not make it a retry.
                    (staging / ('audio_first_' + identity)).mkdir(exist_ok=True)
                    return 'unchanged processing result'
                with patch.object(pipeline, 'QWEN_STAGING_ROOT', staging), \
                        patch.object(helper, 'OUTPUT_ROOT', staging):
                    health.begin(fingerprint, kind)  # CLI preflight has no source identity yet.
                    self.assertEqual(selected(SimpleNamespace(dry_run=False), source), 'unchanged processing result')
                record = health._runs.get()[fingerprint]
                self.assertEqual(record['first_try'], not prior)
                self.assertNotIn('source_identity', record)
                self.assertNotIn('source_folder', record)

    def test_separate_process_reads_complete_after_persistence(self):
        record = self.complete(1)
        child = subprocess.run([sys.executable, '-c',
            'import json,meetingintel_progress as p; print(json.dumps(p.read()))'],
            env=dict(os.environ, MI_OPERATION_ID=record['progress_operation_id']),
            capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(child.stdout)['stage'], 'complete')
        child = subprocess.run([sys.executable, '-c',
            'import meetingintel_progress as p; print(p.read())'],
            env=dict(os.environ), capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(child.stdout.strip(), 'None')

    def test_next_source_rebinds_fingerprint_before_previous_terminal(self):
        health.begin('a' * 64, 'phone_recording', select=True)
        first = progress.current_operation_id()
        health.begin('b' * 64, 'laptop_capture', select=True)
        self.assertNotEqual(first, progress.current_operation_id())
        self.assertEqual(progress.read()['fingerprint64hex'], 'b' * 64)

    def test_corrupt_or_unsafe_record_is_gap_not_success_and_no_raw_text_saved(self):
        record = self.complete(1)
        path = self.root / 'health' / (record['run_id'] + '.json')
        text = path.read_text()
        self.assertNotIn('private_name', text)
        self.assertNotIn('must not be stored', text)
        raw = json.loads(text)
        raw['private_path'] = '/private/secret'
        path.write_text(json.dumps(raw))
        self.assertEqual(health.records()[1], 1)
        self.assertFalse(health.summary()['stable'])

    def test_persistence_exception_finishes_failed_not_complete(self):
        @health.batch
        def run():
            health.begin('a' * 64, 'laptop_capture', select=True)
            raise pipeline.ProductionPersistenceError('ledger persistence failed')
        # Exercise a fresh outer batch rather than this test's open measurement scope.
        token = health._runs.set(None)
        try:
            with self.assertRaises(pipeline.ProductionPersistenceError):
                run()
        finally:
            health._runs.reset(token)
        record = health.records()[0][0]
        self.assertEqual(record['failure_code'], 'ledger_write_failed')
        self.assertEqual(record['progress_terminal'], 'failed')

    def test_actual_pipeline_persistence_and_brief_are_in_same_receipt(self):
        source = pipeline.ProductionSource('laptop_capture', 'a' * 64, self.root,
            self.root / 'audio.m4a', 'b' * 64, datetime.now(pipeline.IST), 10,
            'Synthetic', {'created_at': datetime.now(pipeline.IST).isoformat()}, 'c' * 64)
        prompt = self.root / 'prompt.txt'
        prompt.write_text('Synthetic prompt')
        report = self.root / 'report.md'
        report.write_text('Synthetic report')
        args = argparse.Namespace(dry_run=False, refresh_existing=False, context_pack=None,
            ledger_path=self.root / 'ledger.json', output_dir=self.root / 'output',
            prompt_path=prompt, layer2_prompt_path=prompt, brief_date=datetime.now(pipeline.IST).date().isoformat())
        @observe.observed('processing', source_scope=True)
        def select(args, source, layer2_prompt_template, prompt_template):
            return pipeline.SelectedMeetingResult('Synthetic summary', str(report), None,
                'flat_from_diarized_transcript', layer2_transcript_input='Synthetic transcript')
        token = health._runs.set(None)
        try:
            with patch.object(pipeline, 'select_audio_first_result', select):
                count, brief = pipeline.process_explicit_laptop_sources(args, (source,))
            self.assertEqual(count, 1)
            self.assertTrue(brief.is_file())
            record = health.records()[0][0]
            self.assertEqual(record['outcome'], 'complete')
            self.assertEqual(record['progress_terminal'], 'complete')
            self.assertEqual(record['stages']['ledger']['result'], 'passed')
        finally:
            health._runs.reset(token)


if __name__ == '__main__':
    unittest.main()
