"""Best-effort detached native interruption alert; capture never waits for it."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time

import meetingintel_runtime as runtime

SCHEMA_VERSION = 1
EXPIRY_SECONDS = 600
COMPILE_TIMEOUT = 30
_BUNDLE_PREFIX = 'MeetingIntel Capture Alert-'
_EXECUTABLE = 'MeetingIntelCaptureAlert'
_SOURCE = Path(__file__).resolve().parent / 'scripts/capture_alert.swift'
_ID = re.compile(r'(?:[a-fA-F0-9]{16,64}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})\Z')
_REASONS = {'source_lost', 'output_stalled', 'disk_critical', 'battery_critical',
            'startup_failed', 'startup_battery_critical',
            'recorder_process_died', 'control_lost', 'recording_stopped_externally',
            'terminal_closed', 'termination_requested', 'stop_unconfirmed',
            'recorder_exit_unconfirmed', 'capture_permission_failed', 'cleanup_failed',
            'manifest_write_failed', 'operator_cancelled_before_start', 'capture_interrupted'}
_ENV_KEYS = {'HOME', 'PATH', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
             'MI_RUNTIME_ROOT', 'MI_LEARNING_ROOT', 'MI_ORIGIN', 'MI_ACTOR', 'MI_OPERATION_ID'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _private_file(path, *, executable=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('unsafe alert file')
    if executable and not info.st_mode & stat.S_IXUSR:
        raise ValueError('alert is not executable')
    return path


def _private_directory(path, *, create=False):
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe alert directory')
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('alert directory must be private')
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, value):
    if path.is_symlink():
        raise ValueError('unsafe receipt')
    if path.exists():
        _private_file(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _cache():
    return _private_directory(runtime._root(create=True) / 'capture-alert', create=True)


def _bundle_info():
    return {'CFBundleIdentifier': 'com.meetingintel.capture-alert',
            'CFBundleName': 'MeetingIntelCaptureAlert', 'CFBundleExecutable': _EXECUTABLE,
            'CFBundlePackageType': 'APPL', 'CFBundleVersion': '1',
            'CFBundleShortVersionString': '1.0', 'LSUIElement': False,
            'NSHighResolutionCapable': True}


def _bundle_paths(cache, key):
    bundle = cache / (_BUNDLE_PREFIX + key + '.app')
    contents = bundle / 'Contents'
    return bundle, contents / 'MacOS' / _EXECUTABLE, contents / 'Info.plist', contents / 'build.json'


def _verified_binary(path):
    cache = _cache()
    if path.name != _EXECUTABLE or path.parent.name != 'MacOS' or path.parent.parent.name != 'Contents':
        raise ValueError('unowned alert executable')
    bundle = path.parent.parent.parent
    match = re.fullmatch(re.escape(_BUNDLE_PREFIX) + r'([a-f0-9]{64})\.app', bundle.name)
    if bundle.parent != cache or not match:
        raise ValueError('unowned alert bundle')
    for directory in (bundle, bundle / 'Contents', path.parent):
        _private_directory(directory)
    _private_file(path, executable=True)
    info_path = _private_file(bundle / 'Contents/Info.plist')
    if info_path.stat().st_size > 4096 or plistlib.loads(info_path.read_bytes()) != _bundle_info():
        raise ValueError('alert bundle identity changed')
    receipt_path = _private_file(bundle / 'Contents/build.json')
    if receipt_path.stat().st_size > 4096:
        raise ValueError('invalid executable receipt')
    receipt = json.loads(receipt_path.read_text())
    if receipt != {'schema_version': SCHEMA_VERSION, 'source_key': match.group(1), 'binary_sha256': _hash(path)}:
        raise ValueError('alert executable changed')
    return path


def _terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def prepare():
    """Compile once before capture, returning a verified cached helper or None."""
    process = None
    try:
        if sys.platform != 'darwin' or _SOURCE.is_symlink():
            return None
        source = _SOURCE.read_bytes()
        key = hashlib.sha256(source + platform.machine().encode() + platform.mac_ver()[0].encode()).hexdigest()
        cache = _cache()
        bundle, target, info_path, receipt_path = _bundle_paths(cache, key)
        if bundle.exists() or bundle.is_symlink():
            return _verified_binary(target)
        modules = _private_directory(cache / 'modules', create=True)
        with tempfile.TemporaryDirectory(prefix='.compile-', dir=cache) as temporary:
            temp = Path(temporary)
            source_path = temp / 'alert.swift'
            source_path.write_bytes(source)
            source_path.chmod(0o600)
            output = temp / 'alert'
            process = subprocess.Popen(
                ['/usr/bin/xcrun', 'swiftc', '-framework', 'AppKit', '-module-cache-path', str(modules), str(source_path), '-o', str(output)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, close_fds=True)
            if process.wait(timeout=COMPILE_TIMEOUT) != 0:
                return None
            output.chmod(0o700)
            _private_file(output, executable=True)
            receipt = {'schema_version': SCHEMA_VERSION, 'source_key': key, 'binary_sha256': _hash(output)}
            # Install a complete app bundle atomically. Its fixed identity makes
            # the real native window addressable by macOS accessibility tools.
            staged_bundle = temp / bundle.name
            staged_contents = staged_bundle / 'Contents'
            staged_macos = staged_contents / 'MacOS'
            staged_macos.mkdir(parents=True, mode=0o700)
            staged_bundle.chmod(0o700)
            staged_contents.chmod(0o700)
            os.replace(output, staged_macos / _EXECUTABLE)
            staged_info = staged_contents / 'Info.plist'
            staged_info.write_bytes(plistlib.dumps(_bundle_info()))
            staged_info.chmod(0o600)
            _atomic_json(staged_contents / 'build.json', receipt)
            os.replace(staged_bundle, bundle)
        return _verified_binary(target)
    except Exception:
        return None
    finally:
        if process is not None:
            _terminate(process)


def _session(session_root, capture_id):
    if not isinstance(capture_id, str) or not _ID.fullmatch(capture_id):
        raise ValueError('invalid capture identity')
    session_root = _private_directory(Path(session_root))
    manifest = _private_file(session_root / 'capture_manifest.json')
    if manifest.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('manifest too large')
    if json.loads(manifest.read_text()).get('capture_id') != capture_id:
        raise ValueError('capture identity mismatch')
    return session_root


def notify(session_root: Path, capture_id: str, reason: str, prepared_path: Path | None) -> bool:
    """True means detached worker spawned, never that an alert was seen."""
    eligible = False
    safe_reason = reason if isinstance(reason, str) and reason in _REASONS else 'capture_interrupted'
    try:
        session_root = _session(session_root, capture_id)
        if (session_root / 'capture_alert.json').exists() or (session_root / 'capture_alert.json').is_symlink():
            return False
        eligible = True
        if prepared_path is None:
            raise ValueError('helper unavailable')
        executable = _verified_binary(Path(prepared_path))
        environment = {key: value for key, value in os.environ.items() if key in _ENV_KEYS}
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker', str(session_root), capture_id, safe_reason, str(executable)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, env=environment)
        return True
    except Exception:
        if eligible:
            _emit('attempted', capture_id, outcome='started', reason=safe_reason)
            _emit('failed', capture_id, outcome='failure', reason=safe_reason)
        return False


def _emit(kind, capture_id, *, outcome, reason):
    try:
        import meetingintel_learning as learning
        learning.emit('notification.' + kind, source_id=capture_id, source_kind='laptop_capture',
                      stage='notification', outcome=outcome,
                      error_code=('source_lost' if reason == 'source_lost' else 'capture_interrupted') if kind == 'failed' else None)
    except Exception:
        pass


def _worker(session_root, capture_id, reason, executable, *, expiry=EXPIRY_SECONDS):
    process = None
    receipt_path = None
    receipt = None
    try:
        session_root = _session(session_root, capture_id)
        reason = reason if isinstance(reason, str) and reason in _REASONS else 'capture_interrupted'
        receipt_path = session_root / 'capture_alert.json'
        # Exactly one worker may claim a session. Repeated requests never make
        # duplicate alerts and never overwrite earlier acknowledgement evidence.
        fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        receipt = {'schema_version': SCHEMA_VERSION, 'capture_id': capture_id, 'status': 'attempted',
                   'reason': reason, 'attempted_at': _now(), 'updated_at': _now()}
        with os.fdopen(fd, 'w') as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        _emit('attempted', capture_id, outcome='started', reason=reason)
        executable = _verified_binary(Path(executable))
        mode = ('--startup-battery-critical' if reason == 'startup_battery_critical'
                else '--startup-failed' if reason == 'startup_failed' else '--interrupted')
        process = subprocess.Popen([str(executable), mode], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        deadline = time.monotonic() + expiry
        presented = False
        failure = 'expired_unacknowledged'
        pending = b''
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                ready = selector.select(timeout=min(0.25, max(0, deadline-time.monotonic())))
                if not ready:
                    continue
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    failure = 'dismissed_unacknowledged' if presented else 'presentation_unconfirmed'
                    break
                pending += chunk
                if len(pending) > 128:
                    failure = 'invalid_protocol'
                    break
                while b'\n' in pending:
                    token, pending = pending.split(b'\n', 1)
                    if token == b'presented' and not presented:
                        presented = True
                        receipt.update(status='presented', presented_at=_now(), updated_at=_now())
                        _atomic_json(receipt_path, receipt)
                        _emit('delivered', capture_id, outcome='success', reason=reason)
                    elif token == b'acknowledged' and presented:
                        receipt.update(status='acknowledged', acknowledged_at=_now(), updated_at=_now())
                        _atomic_json(receipt_path, receipt)
                        _emit('acknowledged', capture_id, outcome='success', reason=reason)
                        return True
                    else:
                        failure = 'invalid_protocol'
                        raise ValueError('invalid alert protocol')
        receipt.update(status='failed', failure_reason=failure, updated_at=_now())
        _atomic_json(receipt_path, receipt)
        _emit('failed', capture_id, outcome='failure', reason=reason)
        return False
    except FileExistsError:
        return False
    except Exception:
        if receipt is not None and receipt_path is not None:
            receipt.update(status='failed', failure_reason='helper_failed', updated_at=_now())
            try:
                _atomic_json(receipt_path, receipt)
            except Exception:
                pass
            _emit('failed', capture_id, outcome='failure', reason=reason)
        return False
    finally:
        if process is not None:
            _terminate(process)
            if process.stdout is not None:
                process.stdout.close()


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--worker':
        raise SystemExit(2)
    raise SystemExit(0 if _worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])) else 1)
