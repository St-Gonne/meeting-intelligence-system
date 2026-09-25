"""Integration at real selection boundaries with synthetic workers/results."""
import argparse
import contextlib
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_lock
import meetingintel_gpu as gpu
import meetingintel_pipeline as pipeline
from test_gpu_lock import FakeOllama

class MeetingGPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.a,cls.b=FakeOllama(),FakeOllama()
    @classmethod
    def tearDownClass(cls): cls.a.close();cls.b.close()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ,{'MI_RUNTIME_ROOT':str(self.root/'runtime'),'MI_LEARNING_ROOT':str(self.root/'learning')}).start()
        factory=functools.partial(gpu_lock.Session, wait_seconds=.2, lock_path=self.root/'gpu.lock', endpoints=(self.a.url,self.b.url))
        self.factory=patch.object(gpu,'Session',side_effect=factory).start()
        for server in (self.a,self.b): server.models=[];server.error=False;server.unloads=[]
    def test_whole_meeting_transcription_analysis_cleanup_one_acquisition(self):
        args=argparse.Namespace(dry_run=False)
        sequence=[]
        def stage(*a,**kw):
            sequence.append('transcription')
            result=pipeline.run_captured_command([sys.executable,'-c','from gpu_lock import assert_child; assert_child(); print("staged")'],'test')
            self.assertEqual(result.stdout,'staged\n')
            self.assertIsNotNone(gpu._current.get())
            return self.root,self.root,{}
        def flat(*a,**kw):
            sequence.append('analysis');self.b.models=['meetingintel']
            self.assertIsNotNone(gpu._current.get());return 'result'
        with patch.object(pipeline,'stage_diarized_eligibility',side_effect=stage), patch.object(pipeline,'validate_eligibility_summary',return_value=False), patch.object(pipeline,'routing_decision_snapshot',return_value={}), patch.object(pipeline,'run_flat_meeting',side_effect=flat):
            for _ in range(2):
                result=pipeline.select_meeting_result(args,{},None,'synthetic','synthetic','fixture','l2','l3',self.root)
                self.assertEqual(result,'result');self.assertEqual(self.b.models,[])
        self.assertEqual(sequence,['transcription','analysis']*2)
        self.assertEqual(self.factory.call_count,2)
        self.assertEqual(self.b.unloads,['meetingintel']*2)
    def test_busy_defers_before_transcription_or_fallback(self):
        self.a.models=['qwen3.8:27b-mlx']
        with patch.object(pipeline,'stage_diarized_eligibility') as stage, patch.object(pipeline,'run_flat_meeting') as fallback:
            with self.assertRaises(gpu_lock.Deferred):
                pipeline.select_meeting_result(argparse.Namespace(dry_run=False),{},None,'fixture','fixture','fp','l2','l3',self.root)
            stage.assert_not_called();fallback.assert_not_called()
    def test_dry_run_does_not_acquire_gpu(self):
        with patch.object(pipeline,'run_flat_meeting',return_value='dry'):
            result=pipeline.select_meeting_result(argparse.Namespace(dry_run=True),{},None,'fixture','fixture','fp','l2','l3',self.root)
        self.assertEqual(result,'dry');self.factory.assert_not_called()
    def test_nested_same_process_work_reuses_session(self):
        @gpu.job
        def inner(): return id(gpu._current.get())
        @gpu.job
        def outer(): return inner(),inner()
        a,b=outer();self.assertEqual(a,b);self.assertEqual(self.factory.call_count,1)
    def test_forged_marker_cannot_skip_admission(self):
        @gpu.job
        def worker(): self.fail('executed')
        with patch.dict(os.environ,{'LOCALAI_GPU_LOCK_HELD':'1'}):
            with self.assertRaises(gpu_lock.Deferred):worker()
    def test_managed_port_and_keepalive_context_preserved(self):
        from ollama_endpoint import DEFAULT_OLLAMA_URL,validate_ollama_endpoint,InvalidOllamaEndpoint
        from meetingintel_model_policy import ollama_payload
        self.assertEqual(DEFAULT_OLLAMA_URL,'http://127.0.0.1:11435/api/generate')
        with self.assertRaises(InvalidOllamaEndpoint):validate_ollama_endpoint('http://127.0.0.1:11434/api/generate')
        body=ollama_payload('meetingintel','synthetic')
        self.assertEqual(body['keep_alive'],'2m');self.assertEqual(body['options']['num_ctx'],65536)

class UIDeferralTests(unittest.TestCase):
    def test_deferred_worker_writes_truthful_receipt_and_returns75(self):
        import meetingintel_actions as actions
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'result.json'
            args=['worker','process','synthetic-id','--result-file',str(path)]
            with patch.object(sys,'argv',['worker.py',*args]), patch.object(actions,'perform',side_effect=gpu_lock.Deferred('synthetic busy')), patch.object(actions.learning,'emit'):
                self.assertEqual(actions.main(),75)
            result=json.loads(path.read_text())
            self.assertEqual(result['outcome'],'deferred');self.assertFalse(result['ok'])

if __name__=='__main__':unittest.main()
