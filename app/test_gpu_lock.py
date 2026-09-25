"""Real OS locks/processes, temporary files and fake Ollama; no model inference."""
import contextlib
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import gpu_lock as gpu

HERE = Path(__file__).resolve().parent

class FakeOllama:
    def __init__(self):
        self.models = []
        self.error = False
        self.sticky = False
        self.unloads = []
        fake = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                self.send_response(500 if fake.error else 200); self.end_headers()
                self.wfile.write(json.dumps({'models': [{'name': n} for n in fake.models]}).encode())
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body.get('keep_alive') == 0:
                    fake.unloads.append(body['model'])
                    if not fake.sticky:
                        fake.models = [n for n in fake.models if n != body['model']]
                else:
                    fake.models = [body['model']]
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"done":true,"message":{"content":"synthetic answer"}}')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
    def close(self):
        self.server.shutdown(); self.server.server_close()

class GPUContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.a, cls.b = FakeOllama(), FakeOllama()
    @classmethod
    def tearDownClass(cls): cls.a.close(); cls.b.close()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.lock = self.root/'gpu.lock'
        for fake in (self.a, self.b):
            fake.models=[]; fake.error=False; fake.sticky=False; fake.unloads=[]
    def session(self, **kw):
        return gpu.Session('meetingintel', .3, lock_path=self.lock, endpoints=(self.a.url,self.b.url), **kw)
    def cli(self, code, **kw):
        return subprocess.Popen([sys.executable, str(HERE/'gpu_lock.py'), 'run', '--owner','meetingintel',
            '--wait-seconds','1','--lock-path',str(self.lock),'--endpoints',self.a.url,self.b.url,
            '--',sys.executable,'-c',code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    def free(self):
        fd=os.open(self.lock,os.O_RDWR|os.O_CREAT,0o600)
        try:
            try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); return True
            except BlockingIOError: return False
        finally: os.close(fd)
    def wait_for(self, fn, seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if fn(): return
            time.sleep(.03)
        self.fail('condition timeout')
    def test_normal_handover_and_stable_inode(self):
        with self.session() as session:
            before=self.lock.stat().st_ino
            self.assertFalse(self.free())
            result=session.run([sys.executable,'-c',"print('ok')"])
            self.assertEqual((result.returncode,result.stdout),(0,'ok\n'))
        self.assertTrue(self.free())
        with self.session(): self.assertEqual(before,self.lock.stat().st_ino)
    def test_stale_metadata_does_not_own_lock(self):
        (self.root/'gpu.owner.json').write_text('{"pid":1,"owner":"foreign"}')
        with self.session(): self.assertFalse(self.free())
    def test_resident_model_rejected_without_eviction(self):
        self.a.models=['foreign']
        with self.assertRaises(gpu.Deferred):
            with self.session(): self.fail('admitted')
        self.assertEqual(self.a.models,['foreign']); self.assertEqual(self.a.unloads,[])
    def test_same_model_at_admission_is_not_ours(self):
        self.b.models=['meetingintel']
        with self.assertRaises(gpu.Deferred):
            with self.session(): self.fail('admitted')
        self.assertEqual(self.b.unloads,[])
    def test_endpoint_error_defers(self):
        self.b.error=True
        with self.assertRaises(gpu.Deferred):
            with self.session(): self.fail('admitted')
    def test_connection_refused_defers(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); url='http://127.0.0.1:'+str(sock.getsockname()[1])
        self.assertIsNone(gpu.loaded(url))
        with self.assertRaises(gpu.Deferred):
            with gpu.Session('meetingintel',0,lock_path=self.lock,endpoints=(self.a.url,url)): pass
    def test_only_owned_models_unloaded(self):
        with self.session(): self.b.models=['meetingintel:latest']
        self.assertEqual(self.b.unloads,['meetingintel:latest'])
    def test_uncertain_cleanup_returns75_and_retains_actual_lock(self):
        try:
            with self.assertRaises(gpu.Deferred):
                with self.session(): self.a.models=['foreign']
            self.assertFalse(self.free()); self.assertEqual(self.a.unloads,[])
        finally:
            self.a.models=[]
            self.wait_for(self.free)
    def test_cleanup_defer_closes_cli_pipes_while_guardian_keeps_lock(self):
        self.b.sticky=True
        code=("import urllib.request,json;"
              f"urllib.request.urlopen(urllib.request.Request({self.b.url + '/api/generate'!r},data=json.dumps({{'model':'meetingintel'}}).encode())).read()")
        p=self.cli(code)
        try:
            out,err=p.communicate(timeout=5)
            self.assertEqual(p.returncode,75,err);self.assertFalse(self.free())
        finally:
            self.b.models=[];self.b.sticky=False
            self.wait_for(self.free)

    def test_wait_timeout_does_not_execute(self):
        fd=os.open(self.lock,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX)
        sentinel=self.root/'started'
        try:
            p=self.cli(f"from pathlib import Path; Path({str(sentinel)!r}).touch()")
            out,err=p.communicate(timeout=5)
            self.assertEqual(p.returncode,75,err); self.assertFalse(sentinel.exists())
        finally: os.close(fd)
    def test_marker_alone_never_authorizes_child(self):
        p=subprocess.run([sys.executable,str(HERE/'gpu_lock.py'),'assert-child'],
            env={**os.environ,'LOCALAI_GPU_LOCK_HELD':'1'},capture_output=True)
        self.assertEqual(p.returncode,75)
    def test_real_inherited_descriptor_authorizes_child(self):
        with self.session() as session:
            result=session.run([sys.executable,str(HERE/'gpu_lock.py'),'assert-child'])
            self.assertEqual(result.returncode,0,result.stderr)
    def test_child75_and_nonzero_preserved(self):
        for code in (75,23):
            p=self.cli(f'raise SystemExit({code})');out,err=p.communicate(timeout=8)
            self.assertEqual(p.returncode,code,err)
    def test_parent_sigkill_stops_child_before_unlock(self):
        sentinel=self.root/'child'
        p=self.cli(f"import os,time; from pathlib import Path; Path({str(sentinel)!r}).write_text(str(os.getpid())); time.sleep(30)")
        self.wait_for(sentinel.exists); child=int(sentinel.read_text()); self.assertFalse(self.free())
        p.kill(); p.wait(timeout=3)
        self.wait_for(self.free)
        with self.assertRaises(ProcessLookupError): os.kill(child,0)
        p.stdout.close();p.stderr.close()
    def test_child_crash_cleanup_and_unlock(self):
        p=self.cli('import os,signal; os.kill(os.getpid(),signal.SIGKILL)')
        out,err=p.communicate(timeout=8); self.assertEqual(p.returncode,137,err);self.assertTrue(self.free())
    def test_lingering_foreground_descendant_is_stopped_and_defers(self):
        sentinel=self.root/'grandchild'
        code=("import subprocess,sys;from pathlib import Path;"
              "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
              f"Path({str(sentinel)!r}).write_text(str(p.pid))")
        p=self.cli(code);out,err=p.communicate(timeout=10)
        self.assertEqual(p.returncode,75,err)
        self.assertTrue(self.free())
        child=int(sentinel.read_text())
        status=subprocess.run(['/bin/ps','-o','stat=','-p',str(child)],capture_output=True,text=True).stdout.strip()
        self.assertTrue(not status or status.startswith('Z'),status)

    def test_cli_sigterm_cleans_up(self):
        sentinel=self.root/'child'
        p=self.cli(f"import os,time; from pathlib import Path; Path({str(sentinel)!r}).write_text(str(os.getpid())); time.sleep(30)")
        self.wait_for(sentinel.exists);p.terminate()
        out,err=p.communicate(timeout=8);self.assertNotEqual(p.returncode,0,err);self.assertTrue(self.free())

class ServerLimitTests(unittest.TestCase):
    def fake(self, missing=False, changed=False, stopped=False):
        counts = {}
        def run(command, **kwargs):
            if 'lsof' in command[0]:
                port = next(v.split(':')[1] for v in command if v.startswith('-iTCP:'))
                counts[port] = counts.get(port, 0) + 1
                if stopped: raise subprocess.CalledProcessError(1, command)
                pid = int(port) + (1 if changed and counts[port] > 1 else 0)
                return subprocess.CompletedProcess(command, 0, f'p{pid}\nn127.0.0.1:{port}\n', '')
            return subprocess.CompletedProcess(command, 0, 'ollama serve ' + ('' if missing else 'OLLAMA_MAX_LOADED_MODELS=1'), '')
        return run
    def test_current_both_listener_environments_required(self):
        with patch.object(gpu.subprocess, 'run', side_effect=self.fake()):
            self.assertTrue(gpu.limits_verified())
    def test_app_started_before_setenv_is_not_verified(self):
        with patch.object(gpu.subprocess, 'run', side_effect=self.fake(missing=True)):
            self.assertFalse(gpu.limits_verified())
    def test_stopped_server_is_not_verified(self):
        with patch.object(gpu.subprocess, 'run', side_effect=self.fake(stopped=True)):
            self.assertFalse(gpu.limits_verified())
    def test_restarted_listener_identity_is_not_verified(self):
        with patch.object(gpu.subprocess, 'run', side_effect=self.fake(changed=True)):
            self.assertFalse(gpu.limits_verified())

if __name__=='__main__': unittest.main(verbosity=2)
