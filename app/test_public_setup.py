"""Public onboarding contracts: no model calls, real accounts or recordings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_lock
from mi_config import load_settings
import mi_paths
import meetingintel_pipeline as pipeline
import meetingintel_queue as queue


class PublicSetupTests(unittest.TestCase):
    def test_settings_are_data_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json'
            path.write_text(json.dumps({'MI_TRANSCRIPTION_BACKEND':'whisper','MI_GPU_MODE':'standalone'}))
            env={'MI_CONFIG_FILE':str(path),'MI_GPU_MODE':'shared'}
            load_settings(env)
            self.assertEqual('shared',env['MI_GPU_MODE'])
            self.assertEqual('whisper',env['MI_TRANSCRIPTION_BACKEND'])
            path.write_text(json.dumps({'PATH':'do not accept arbitrary environment overrides'}))
            with self.assertRaises(ValueError):load_settings({'MI_CONFIG_FILE':str(path)})

    def test_symlink_and_nonstring_settings_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json';path.write_text('{"MI_GPU_MODE": true}')
            with self.assertRaises(ValueError):load_settings({'MI_CONFIG_FILE':str(path)})
            link=Path(directory)/'link.json';link.symlink_to(path)
            with self.assertRaises(ValueError):load_settings({'MI_CONFIG_FILE':str(link)})

    def test_standalone_needs_one_endpoint_and_shared_keeps_both(self):
        with patch.dict(os.environ, {'MI_GPU_MODE':'standalone'}):
            self.assertEqual(gpu_lock.SINGLE_ENDPOINTS,gpu_lock.runtime_endpoints())
            self.assertEqual(list(gpu_lock.SINGLE_ENDPOINTS),gpu_lock.Session('meetingintel').config['endpoints'])
        with patch.dict(os.environ, {'MI_GPU_MODE':'shared'}):
            self.assertEqual(gpu_lock.DEFAULT_ENDPOINTS,gpu_lock.runtime_endpoints())
        with patch.dict(os.environ, {'MI_GPU_MODE':'invalid'}):
            with self.assertRaises(ValueError):gpu_lock.runtime_endpoints()
        self.assertEqual({'http://127.0.0.1:11435':{'meetingintel','meetingintel:latest'}},
            gpu_lock._owned({'owner':'meetingintel','endpoints':list(gpu_lock.SINGLE_ENDPOINTS)}))

    def test_voice_review_and_automatic_matching_share_asset_layout(self):
        with patch.dict(os.environ, {'MI_VOICE_ROOT':'/private/synthetic-voice-root'}):
            state=mi_paths.public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate5')
            report=mi_paths.public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate4/voiceprint_owner_bakeoff_report.json')
            model=mi_paths.public_path('home/Library/Application Support/MeetingIntel/voiceprint-gate4/models/community1-embedding.bin')
        self.assertEqual(state.parent / 'voiceprint-gate4',report.parent)
        self.assertEqual(report.parent / 'models',model.parent)

    def test_backend_choice_and_timezone_are_explicit(self):
        with patch.dict(os.environ, {'MI_TRANSCRIPTION_BACKEND':'whisper','MI_TIMEZONE':'Asia/Kolkata'}):
            self.assertEqual('legacy',mi_paths.transcription_backend())
            self.assertEqual('Asia/Kolkata',str(mi_paths.configured_timezone()))
        with patch.dict(os.environ, {'MI_TIMEZONE':'America/New_York'}):
            with self.assertRaisesRegex(ValueError, 'source contracts'):
                mi_paths.configured_timezone()
        with patch.dict(os.environ, {'MI_TRANSCRIPTION_BACKEND':'silent-fallback'}):
            with self.assertRaises(ValueError):mi_paths.transcription_backend()

    def test_launcher_accepts_backend_and_rejects_duplicates_before_processing(self):
        launcher=Path(__file__).with_name('mi')
        env={**os.environ,'MI_CONFIG_FILE':'/nonexistent/mi-public-settings.json'}
        good=subprocess.run([sys.executable,str(launcher),'--transcription-backend','whisper','help'],env=env,capture_output=True,text=True)
        self.assertEqual(0,good.returncode,good.stderr)
        bad=subprocess.run([sys.executable,str(launcher),'--transcription-backend','qwen','--transcription-backend=whisper','help'],env=env,capture_output=True,text=True)
        self.assertEqual(2,bad.returncode)

    def test_laptop_queue_resume_uses_frozen_backend_and_restores_default(self):
        original=pipeline.laptop_transcription_backend()
        job={'lane':'laptop','backend':'legacy'}
        with patch('meetingintel_cli._run_exact_laptop_source',side_effect=lambda *a,**k: (self.assertEqual('legacy',pipeline.laptop_transcription_backend()) or 0)), patch.object(queue,'completed',return_value=True):
            self.assertTrue(queue.execute(job,object()))
        self.assertEqual(original,pipeline.laptop_transcription_backend())

    def test_setup_inventory_demo_never_invokes_subprocess_or_network(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/setup_check.py'
        spec=importlib.util.spec_from_file_location('public_setup_check',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module.subprocess,'run',side_effect=AssertionError('no process needed')), patch('urllib.request.urlopen',side_effect=AssertionError('no network needed')):
            result=module.check(mode='demo')
        self.assertTrue(result['ready'])
