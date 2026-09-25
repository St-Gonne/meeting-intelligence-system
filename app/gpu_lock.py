#!/usr/bin/env python3
"""LocalAI exclusive GPU admission and supervised cleanup (stdlib only).

CLI: gpu_lock.py run --owner OWNER [--wait-seconds 5400] -- COMMAND ...
Python: with Session(owner='meetingintel') as session: session.run(command)
The separate guardian owns flock, survives caller death, and supervises foreground
process groups. No command may detach workers. Never wrap dealgraph's own lock.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

DEFAULT_LOCK = Path.home() / 'Library/Application Support/LocalAI/gpu.lock'
DEFAULT_ENDPOINTS = ('http://127.0.0.1:11434', 'http://127.0.0.1:11435')
DEFER = 75

class Deferred(SystemExit):
    def __init__(self, reason='GPU busy or cleanup uncertain'):
        self.reason = reason
        super().__init__(DEFER)

def _send(sock, obj):
    sock.sendall(json.dumps(obj).encode() + b'\n')

def _recv(sock, timeout=None):
    # One request/reply at a time; no buffered read-ahead across select().
    data = bytearray()
    while True:
        if not select.select([sock], [], [], timeout)[0]:
            raise TimeoutError('guardian response timeout')
        b = sock.recv(1)
        if not b:
            raise EOFError('guardian connection closed')
        if b == b'\n':
            return json.loads(data)
        data.extend(b)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError('guardian message too large')

def loaded(url, timeout=2):
    """Refusal, malformed data and every other error mean uncertain, never empty."""
    try:
        with urllib.request.urlopen(url + '/api/ps', timeout=timeout) as response:
            body = json.load(response)
        rows = body['models']
        if not isinstance(rows, list):
            return None
        names = [row['name'] for row in rows]
        if any(not isinstance(name, str) or not name for name in names):
            return None
        return names
    except Exception:
        return None

def _empty(endpoints, timeout=2, deadline=None):
    for url in endpoints:
        budget = timeout if deadline is None else min(timeout, deadline - time.monotonic())
        if budget <= 0 or loaded(url, budget) != []:
            return False
    return deadline is None or time.monotonic() <= deadline

def server_limits():
    """Read current listener identities and their actual inherited environments.

    Never print or retain the environment itself. launchctl's environment and
    historical logs alone cannot establish the setting on a running process.
    """
    result = {}
    for port in (11434, 11435):
        try:
            def listener():
                response = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-iTCP:' + str(port),
                    '-sTCP:LISTEN', '-Fpn'], capture_output=True, text=True, timeout=3, check=True)
                pids = {int(line[1:]) for line in response.stdout.splitlines() if line.startswith('p')}
                addresses = {line[1:] for line in response.stdout.splitlines() if line.startswith('n')}
                if len(pids) != 1 or addresses != {'127.0.0.1:' + str(port)}:
                    raise ValueError('listener identity/scope uncertain')
                return pids.pop()
            pid = listener()
            response = subprocess.run(['/bin/ps', 'eww', '-p', str(pid), '-o', 'command='],
                capture_output=True, text=True, timeout=3, check=True)
            configured = re.search(r'(?:^|\s)OLLAMA_MAX_LOADED_MODELS=1(?:\s|$)', response.stdout) is not None
            if listener() != pid:
                raise ValueError('listener restarted during verification')
            result[str(port)] = {'pid': pid, 'max_loaded_models_one': configured}
        except Exception:
            result[str(port)] = {'pid': None, 'max_loaded_models_one': False}
    return result


def limits_verified():
    return all(row['max_loaded_models_one'] for row in server_limits().values())


def _peer_gone(sock):
    if select.select([sock], [], [], 0)[0]:
        return sock.recv(1, socket.MSG_PEEK) == b''
    return False

def _open_lock(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        os.close(fd)
        raise Deferred('unsafe lock file')
    return fd

def assert_child():
    """Validate the inherited lock descriptor, never trust the env marker alone.

    A separate open must be blocked, while reasserting the inherited open-file
    description succeeds. Do NOT unlock it: the guardian shares that description.
    """
    try:
        fd = int(os.environ['LOCALAI_GPU_LOCK_FD'])
        path = Path(os.environ['LOCALAI_GPU_LOCK_PATH'])
        before, actual = os.stat(path, follow_symlinks=False), os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError('descriptor mismatch')
        probe = os.open(path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('no existing owner')
        finally:
            os.close(probe)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except Exception as exc:
        raise Deferred('no valid inherited GPU lock') from exc

def _group_members(pgid):
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat='],
                            capture_output=True, text=True, check=True)
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and int(fields[1]) == pgid and not fields[2].startswith('Z'):
            members.append(int(fields[0]))
    return members


def _group_exists(pgid):
    return bool(_group_members(pgid))


def _signal_group(pgid, sig):
    # Enumerate only this newly-created foreground group. macOS may reject
    # killpg around exited/zombie groups; individual verified members are exact.
    for pid in _group_members(pgid):
        with contextlib.suppress(ProcessLookupError):
            if os.getpgid(pid) == pgid:
                os.kill(pid, sig)


def _stop_group(child):
    child.poll()
    if _group_exists(child.pid):
        _signal_group(child.pid, signal.SIGTERM)
        until = time.monotonic() + 2
        while _group_exists(child.pid) and time.monotonic() < until:
            child.poll()
            time.sleep(.05)
        if _group_exists(child.pid):
            _signal_group(child.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=2)
    until = time.monotonic() + 2
    while _group_exists(child.pid) and time.monotonic() < until:
        time.sleep(.05)
    return not _group_exists(child.pid)


def _owned(config):
    # Explicit endpoint + model allowlist: never evict an unrelated model.
    owner = config['owner']
    if owner == 'meetingintel':
        return {config['endpoints'][1]: {'meetingintel', 'meetingintel:latest'}}
    if owner.startswith(('qwen-delegate', 'moneymoney')):
        return {config['endpoints'][0]: {'qwen3.8:27b-mlx'}}
    return {}

def _cleanup(config, children):
    try:
        groups_ok = True
        for child in list(children):
            if _stop_group(child):
                children.remove(child)
            else:
                groups_ok = False
    except (OSError, subprocess.SubprocessError) as error:
        with contextlib.suppress(OSError):
            print("GPU child cleanup: " + repr(error), file=sys.stderr, flush=True)
        return False
    for url, allowed in _owned(config).items():
        models = loaded(url)
        if models is None:
            continue
        for model in models:
            if model in allowed:
                try:
                    request = urllib.request.Request(url + '/api/generate',
                        data=json.dumps({'model': model, 'keep_alive': 0, 'stream': False}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=3) as response:
                        response.read()
                except Exception:
                    pass
    return groups_ok and _empty(config['endpoints'])

def _detach_stdio():
    # A quarantine guardian may outlive the CLI. Do not hold the caller's stdout
    # pipes open after returning75 (Node/Aider wrappers wait for EOF as well).
    sink = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            os.dup2(sink, fd)
    finally:
        if sink > 2:
            os.close(sink)


def _guard(sock, config):
    fd = None
    acquired = False
    children = []
    started = None
    acquisition = uuid.uuid4().hex
    stop = False
    def signal_stop(*_):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal_stop)
    try:
        path = Path(config['lock_path'])
        fd = _open_lock(path)
        deadline = time.monotonic() + config['wait_seconds']
        while not stop:
            if _peer_gone(sock):
                return DEFER
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            if acquired:
                if tuple(config['endpoints']) == DEFAULT_ENDPOINTS and not limits_verified():
                    _send(sock, {'code': DEFER, 'reason': 'live server model limits not verified; run gpu_lock.py check'})
                    return DEFER
                remaining = max(.05, min(2, deadline - time.monotonic()))
                if _empty(config['endpoints'], remaining, deadline if config['wait_seconds'] > 0 else time.monotonic() + .25):
                    break
                fcntl.flock(fd, fcntl.LOCK_UN)
                acquired = False
            if time.monotonic() >= deadline:
                _send(sock, {'code': DEFER, 'reason': 'GPU busy or endpoints uncertain'})
                return DEFER
            time.sleep(min(config['poll_seconds'], max(0, deadline - time.monotonic())))
        if not acquired or stop:
            return DEFER
        started = time.monotonic()
        metadata = {'owner': config['owner'], 'pid': os.getpid(), 'acquisition_id': acquisition,
                    'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        # Sidecar is content-free diagnostics only; never read for admission.
        tmp = path.with_name('gpu.owner.' + acquisition + '.tmp')
        tmp.write_text(json.dumps(metadata)); tmp.chmod(0o600)
        os.replace(tmp, path.with_name('gpu.owner.json'))
        _send(sock, {'code': 0, 'guardian_pid': os.getpid()})
        while not stop:
            if _peer_gone(sock):
                break
            if not select.select([sock], [], [], .1)[0]:
                continue
            request = _recv(sock)
            if request['op'] == 'finish':
                break
            if request['op'] != 'run':
                raise ValueError('unknown guardian operation')
            env = request.get('env') or dict(os.environ)
            env.update(LOCALAI_GPU_LOCK_HELD='1', LOCALAI_GPU_LOCK_FD=str(fd),
                       LOCALAI_GPU_LOCK_PATH=str(path))
            # Temporary output files avoid pipe-buffer deadlocks. They are unlinked
            # immediately by TemporaryFile and never become retained meeting evidence.
            capture = request.get('capture', True)
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                child = subprocess.Popen(request['command'], env=env, cwd=request.get('cwd'),
                    pass_fds=(fd,), start_new_session=True,
                    stdout=stdout if capture else None, stderr=stderr if capture else None,
                    stdin=subprocess.DEVNULL if capture else None)
                children.append(child)
                while child.poll() is None and not stop and not _peer_gone(sock):
                    if select.select([sock], [], [], .05)[0] and not _peer_gone(sock):
                        message = _recv(sock)
                        if message.get('op') == 'finish':
                            stop = True
                        else:
                            raise ValueError('unexpected request during child execution')
                if stop or _peer_gone(sock):
                    break
                residual = _group_exists(child.pid)
                if not residual or _stop_group(child):
                    # Retire a verified-empty group now; never signal an old PID
                    # at the end of a long meeting after the OS could reuse it.
                    children.remove(child)
                stdout.seek(0); stderr.seek(0)
                _send(sock, {'code': DEFER if residual else child.returncode,
                    'stdout': stdout.read().decode(errors='replace') if capture else '',
                    'stderr': stderr.read().decode(errors='replace') if capture else ''})
        safe = _cleanup(config, children)
        with contextlib.suppress(OSError):
            _send(sock, {'code': 0 if safe else DEFER, 'reason': 'cleanup uncertain' if not safe else ''})
        # If cleanup is uncertain, the guardian RETAINS the actual flock. Metadata
        # is not used as a substitute lock. The caller can exit75 promptly.
        if not safe:
            _detach_stdio()
        while not safe:
            with contextlib.suppress(OSError):
                print('LocalAI guardian retains gpu.lock: cleanup uncertain', file=sys.stderr, flush=True)
            time.sleep(5)
            safe = _cleanup(config, children)
        return 0
    except (EOFError, BrokenPipeError, ConnectionResetError):
        _detach_stdio()
        if acquired:
            while not _cleanup(config, children):
                time.sleep(5)
        return DEFER
    except BaseException as exc:
        with contextlib.suppress(Exception):
            _send(sock, {'code': DEFER, 'reason': type(exc).__name__})
        _detach_stdio()
        if acquired:
            while not _cleanup(config, children):
                time.sleep(5)
        return DEFER
    finally:
        # Even an unexpected guardian error must not release around live workers.
        while acquired and started is not None and not _cleanup(config, children):
            time.sleep(5)
        if acquired and started is not None:
            with contextlib.suppress(OSError):
                print(json.dumps({'localai_gpu_hold': {'owner': config['owner'],
                    'acquisition_id': acquisition, 'seconds': round(time.monotonic()-started, 3)}}), file=sys.stderr)
        if fd is not None:
            # Close only after cleanup; children sharing the description have exited.
            os.close(fd)
        sock.close()

class Session:
    def __init__(self, owner, wait_seconds=5400, *, lock_path=DEFAULT_LOCK,
                 endpoints=DEFAULT_ENDPOINTS, poll_seconds=.25):
        if not re.fullmatch(r'[a-z0-9_-]{1,64}', owner):
            raise ValueError('invalid owner label')
        if not math.isfinite(wait_seconds) or wait_seconds < 0 or not 0 < poll_seconds <= 5:
            raise ValueError('invalid wait/poll time')
        if len(endpoints) != 2:
            raise ValueError('two endpoints required')
        for url in endpoints:
            if not re.fullmatch(r'http://(?:127\.0\.0\.1|localhost):[0-9]+', url):
                raise ValueError('only local HTTP endpoints admitted')
        self.config = dict(owner=owner, wait_seconds=wait_seconds, lock_path=str(lock_path),
                           endpoints=list(endpoints), poll_seconds=poll_seconds)
        self.sock = self.guardian = None
    def __enter__(self):
        if os.environ.get('LOCALAI_GPU_LOCK_HELD'):
            raise Deferred('nested acquisition refused')
        parent, child = socket.socketpair()
        self.sock = parent
        self.guardian = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_guard',
            str(child.fileno()), json.dumps(self.config)], pass_fds=(child.fileno(),), start_new_session=True)
        child.close()
        try:
            reply = _recv(parent, self.config['wait_seconds'] + 10)
            if reply['code']:
                raise Deferred(reply.get('reason'))
        except BaseException:
            parent.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.guardian.wait(timeout=3)
            raise
        return self
    def run(self, command, *, capture_output=True, text=True, env=None, cwd=None):
        _send(self.sock, dict(op='run', command=[str(c) for c in command],
                             capture=capture_output, env=env, cwd=str(cwd) if cwd else None))
        reply = _recv(self.sock)
        code = reply['code']
        if code == DEFER:
            raise Deferred(reply.get('reason', 'child deferred or left workers running'))
        return subprocess.CompletedProcess(command, code,
            reply.get('stdout', '') if text else reply.get('stdout', '').encode(),
            reply.get('stderr', '') if text else reply.get('stderr', '').encode())
    def __exit__(self, exc_type, exc, tb):
        try:
            _send(self.sock, {'op': 'finish'})
            reply = _recv(self.sock, 20)
            if reply['code']:
                threading.Thread(target=self.guardian.wait, daemon=True).start()
                raise Deferred(reply.get('reason'))
            self.guardian.wait(timeout=5)
        except (OSError, EOFError, TimeoutError, subprocess.TimeoutExpired) as error:
            raise Deferred('guardian cleanup not confirmed') from error
        finally:
            self.sock.close()
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_guard':
        return _guard(socket.socket(fileno=int(sys.argv[2])), json.loads(sys.argv[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    run = subs.add_parser('run')
    run.add_argument('--owner', required=True)
    run.add_argument('--wait-seconds', type=float, default=5400)
    run.add_argument('--lock-path', type=Path, default=DEFAULT_LOCK)
    run.add_argument('--endpoints', nargs=2, default=DEFAULT_ENDPOINTS)
    run.add_argument('command', nargs=argparse.REMAINDER)
    subs.add_parser('assert-child')
    subs.add_parser('check')
    args = parser.parse_args()
    if args.action == 'check':
        status = server_limits()
        print(json.dumps(status, sort_keys=True))
        return 0 if all(row['max_loaded_models_one'] for row in status.values()) else DEFER
    if args.action == 'assert-child':
        assert_child(); return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required')
    # A nested session is a configuration error, never authority to bypass flock.
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with Session(args.owner, args.wait_seconds, lock_path=args.lock_path, endpoints=args.endpoints) as session:
        result = session.run(command, capture_output=False)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Deferred as error:
        print('GPU deferred: ' + error.reason, file=sys.stderr)
        raise SystemExit(DEFER)
