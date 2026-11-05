"""Canary admission is synthetic: no models, real references or scheduling."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import meetingintel_canary as canary
import meetingintel_run_health as health


class CanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'MI_LEARNING_ROOT': str(self.root / 'learning'),
            'MI_RUN_HEALTH_ROOT': str(self.root / 'health'), 'MI_RUNTIME_ROOT': str(self.root / 'runtime')})
        self.env.start()
        self.refs = [dict(lane=lane, source_folder=str(self.root / lane), stage=str(self.root / 'stage'),
            source_audio_sha256='a' * 64, stage_manifest_sha256='b' * 64,
            stage_artifact_sha256={name:'c'*64 for name in ('audio.json','audio.srt','audio.txt')},
            expected_label='SPEAKER_00') for lane in ('phone', 'laptop')]
        canary.configure(self.refs)
        health.record_suite_pass(1)
        self.time = patch.object(canary, 'datetime')
        self.clock = self.time.start()
        self.clock.now.return_value = datetime(2026, 10, 5, 2, tzinfo=canary.IST)
        self.clock.fromisoformat.side_effect = datetime.fromisoformat

    def tearDown(self):
        self.time.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_daytime_never_touches_gpu(self):
        self.clock.now.return_value = datetime(2026, 10, 5, 12, tzinfo=canary.IST)
        with patch.object(canary.gpu_lock, 'Session') as session:
            self.assertEqual(canary.due()['status'], 'not_night')
        session.assert_not_called()

    def test_worker_scopes_both_audio_tools_for_source_validation_and_identity(self):
        import shutil
        root = canary._root(); run = root / 'run-synthetic'; run.mkdir()
        binary_dir = self.root / 'homebrew/bin'; binary_dir.mkdir(parents=True)
        for name in ('ffmpeg', 'ffprobe'):
            binary = binary_dir / name
            binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o700)
        def checks(config, output, lane):
            self.assertEqual(str(binary_dir / 'ffmpeg'), shutil.which('ffmpeg'))
            self.assertEqual(str(binary_dir / 'ffprobe'), shutil.which('ffprobe'))
            return 0
        with patch.dict(os.environ, {'PATH':'/usr/bin:/bin', 'MI_CANARY':'1'}), \
             patch.object(Path, 'home', return_value=self.root), \
             patch.object(canary.gpu_lock, 'assert_child'), \
             patch.object(canary.signal, 'signal'), patch.object(canary.signal, 'alarm'), \
             patch.object(canary, '_checks', side_effect=checks):
            self.assertEqual(0, canary.worker(run, 'phone'))
            self.assertEqual('/usr/bin:/bin', os.environ['PATH'])

    def test_busy_and_uncertain_residency_skip_with_no_retry(self):
        with patch.object(canary.runtime, 'status', return_value={'active': True}), patch.object(canary.gpu_lock, 'Session') as session:
            self.assertEqual(canary.due()['status'], 'deferred_busy')
            session.assert_not_called()
        with patch.object(canary.runtime, 'status', return_value={'active': False}), patch.object(canary.gpu_lock, 'loaded', return_value=None), patch.object(canary.gpu_lock, 'Session') as session:
            self.assertEqual(canary.due()['status'], 'deferred_gpu')
            session.assert_not_called()

    def test_one_full_attempt_per_seven_days_alternates_lanes_zero_wait(self):
        with patch.object(canary.runtime, 'status', return_value={'active': False}), \
             patch.object(canary.gpu_lock, 'loaded', return_value=[]), patch.object(canary.gpu_lock, 'Session') as session:
            session.return_value.__enter__.return_value.run.return_value.returncode = 0
            self.assertEqual(canary.due()['status'], 'passed')
            session.assert_called_once_with('meetingintel', wait_seconds=0)
            self.assertEqual(canary.due()['status'], 'not_due')
            self.clock.now.return_value = datetime(2026, 10, 12, 2, tzinfo=canary.IST)
            self.assertEqual(canary.due()['status'], 'passed')
            commands = session.return_value.__enter__.return_value.run.call_args_list
            self.assertEqual([call.args[0][-1] for call in commands], ['laptop', 'phone'])

    def test_failure_consumes_week_and_is_observable_in_header(self):
        with patch.object(canary.runtime, 'status', return_value={'active': False}), \
             patch.object(canary.gpu_lock, 'loaded', return_value=[]), patch.object(canary.gpu_lock, 'Session') as session:
            session.return_value.__enter__.return_value.run.return_value.returncode = 1
            self.assertEqual(canary.due()['status'], 'failed')
            self.assertEqual(canary.due()['status'], 'not_due')
            self.assertIn('reference check failed', health.summary()['header'])

    def test_overdue_reference_check_uses_attempt_age_in_header_and_heartbeat_aggregate(self):
        root = canary._root()
        configured_at = datetime(2026, 10, 4, tzinfo=timezone.utc)
        config_path = root / 'references.json'
        os.utime(config_path, (configured_at.timestamp(), configured_at.timestamp()))
        for age, expected in ((timedelta(days=8), False), (timedelta(days=8, seconds=1), True)):
            with self.subTest(age=age), patch.object(health, 'now', return_value=configured_at + age):
                aggregate = health.summary()
                self.assertEqual(' · reference check overdue' in aggregate['header'], expected)
                self.assertEqual(aggregate['reference_check_overdue'], expected)
        current = configured_at + timedelta(days=20)
        for age, expected in ((timedelta(days=8), False), (timedelta(days=8, seconds=1), True),
                              (timedelta(days=1), False)):
            with self.subTest(attempt_age=age):
                health.runtime._write_json(root, 'schedule.json',
                    {'last_full_at': (current - age).isoformat(), 'next_lane': 'phone'})
                # Deferrals are not attempts and must not hide an overdue check.
                health.runtime._write_json(root, 'receipt.json',
                    {'at': current.isoformat(), 'status': 'deferred_busy', 'lane': 'phone'})
                with patch.object(health, 'now', return_value=current):
                    aggregate = health.summary()
                self.assertEqual(' · reference check overdue' in aggregate['header'], expected)
                self.assertEqual(aggregate['reference_check_overdue'], expected)

    def test_prestart_deferral_does_not_consume_the_week(self):
        with patch.object(canary.runtime, 'status', return_value={'active': False}), \
             patch.object(canary.gpu_lock, 'loaded', return_value=[]), patch.object(canary.gpu_lock, 'Session') as session:
            session.return_value.__enter__.return_value.run.return_value.returncode = 75
            self.assertEqual(canary.due()['status'], 'deferred_busy')
            self.clock.now.return_value = datetime(2026, 10, 6, 2, tzinfo=canary.IST)
            self.assertEqual(canary.due()['status'], 'deferred_busy')
            self.assertEqual(session.call_count, 2)

    def test_config_requires_exact_two_lanes_and_no_symlink(self):
        with self.assertRaises(ValueError):
            canary.configure(self.refs[:1])
        target = self.root / 'target'
        target.mkdir()
        link = self.root / 'link'
        link.symlink_to(target)
        self.refs[0]['stage'] = str(link)
        with self.assertRaises(ValueError):
            canary.configure(self.refs)

    def test_asr_and_analysis_destinations_are_isolated_without_changing_defaults(self):
        import whispermlx_diarization_helper as helper
        import diarized_layer2_shadow as shadow
        original_asr, original_shadow = helper.OUTPUT_ROOT, shadow.SHADOW_OUTPUT_ROOT
        with patch.dict(os.environ, {'MI_CANARY': '1', 'MI_CANARY_ASR_ROOT': str(self.root / 'asr'),
                                   'MI_CANARY_SHADOW_ROOT': str(self.root / 'analysis')}):
            output = helper.create_output_folder(self.root / 'source')
            report = shadow.create_output_folder('Synthetic meeting')
        self.assertEqual(output.parent, self.root / 'asr')
        self.assertEqual(report.parent, self.root / 'analysis')
        self.assertEqual(helper.OUTPUT_ROOT, original_asr)
        self.assertEqual(shadow.SHADOW_OUTPUT_ROOT, original_shadow)

    def test_old_human_anchor_is_identity_reference_not_weakened_asr_cache(self):
        import meetingintel_identity as identity
        import meetingintel_pipeline as pipeline
        stage = self.root / 'stage'
        stage.mkdir()
        source = self.root / 'phone'
        source.mkdir()
        audio = source / 'audio.m4a'
        audio.write_bytes(b'synthetic')
        production = SimpleNamespace(source_identity='a'*64, source_folder=source, source_audio_path=audio)
        # Historical success has no backend/runtime sidecar, as in the selected anchor.
        manifest = dict(status='success', exit_code=0, source_mode='audio_first', source_identity='a'*64,
            source_folder=str(source), source_audio=str(audio), output_folder=str(stage),
            completed_at='2026-09-12T00:00:00+05:30')
        (stage/'run_manifest.json').write_text(json.dumps(manifest))
        (stage/'audio.json').write_text(json.dumps({'segments':[{'speaker':'SPEAKER_00','start':0,'end':3,'text':'Synthetic'}]}))
        (stage/'audio.srt').write_text('1\n00:00:00,000 --> 00:00:03,000\n[SPEAKER_00]: Synthetic\n')
        (stage/'audio.txt').write_text('Synthetic')
        ref=dict(self.refs[0], source_audio_sha256=pipeline.sha256_file(audio),
                 stage_manifest_sha256=pipeline.sha256_file(stage/'run_manifest.json'),
                 stage_artifact_sha256={name:pipeline.sha256_file(stage/name) for name in ('audio.json','audio.srt','audio.txt')})
        with patch.object(identity,'_source',return_value=object()), \
             patch.object(pipeline,'production_source_from_audio_first',return_value=production), \
             patch('meetingintel_staging_cache._validate_stage',side_effect=AssertionError('production cache contract must remain independent')):
            self.assertIs(canary._source(ref),production)
            (stage/'audio.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'reference artifact changed'):
                canary._source(ref)


if __name__ == '__main__':
    unittest.main()
