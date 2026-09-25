#!/usr/bin/env python3
"""Explicit user-scoped due checks for MeetingIntel's local usage review."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile

LABEL = 'com.meetingintel.learning-review'


def _root():
    return Path(os.environ.get('MI_LEARNING_ROOT', str(Path.home() / 'Library/Application Support/MeetingIntel/learning'))).expanduser()


def _safe(path, create=False, private=True):
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise OSError('unsafe scheduler path')
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()):
        raise OSError('unsafe scheduler owner')
    if create and private:
        path.chmod(0o700)
    return path


def _atomic(path, data, mode=0o600):
    if path.is_symlink():
        raise OSError('unsafe scheduler file')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _receipt(stage, outcome, category):
    root = _safe(_root(), create=True)
    _atomic(root / 'scheduler-receipt.json', json.dumps({
        'schema_version': 1, 'at': datetime.now(timezone.utc).isoformat(),
        'stage': stage, 'outcome': outcome, 'category': category,
    }, sort_keys=True).encode())


def _plist_path():
    return Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')


def _launchctl(arguments):
    return subprocess.run(['/bin/launchctl', *arguments], capture_output=True, text=True, timeout=15, check=False)


def due():
    """Quiet on unchanged/non-actionable state; never invokes processing."""
    try:
        _receipt('started', 'unknown', 'due_check_started')
        project = os.environ.get('MI_PROJECT_ROOT')
        if project:
            project_path = Path(project)
            if not project_path.is_absolute() or project_path.is_symlink():
                raise OSError('invalid project root')
            sys.path.insert(0, str(project_path))
        import meetingintel_learning as learning
        result = learning.review()
        category = result['status']
        _receipt('completed', 'failure' if category == 'failed' else 'success', category)
        return result
    except Exception:
        try:
            _receipt('failed', 'failure', 'review_unavailable')
        except Exception:
            pass
        return {'status': 'failed', 'error_code': 'scheduler_unavailable'}


def status():
    try:
        root = _safe(_root())
        plist_path = _plist_path()
        if plist_path.is_symlink():
            raise OSError('unsafe scheduler plist')
        installed = plist_path.is_file()
        receipt = None
        receipt_path = root / 'scheduler-receipt.json'
        if receipt_path.is_symlink():
            raise OSError('unsafe scheduler receipt')
        if receipt_path.exists():
            raw = json.loads(receipt_path.read_text())
            if isinstance(raw, dict):
                receipt = {k: raw.get(k) for k in ('at', 'stage', 'outcome', 'category')}
        loaded = False
        if installed:
            result = _launchctl(['print', f'gui/{os.getuid()}/{LABEL}'])
            loaded = result.returncode == 0
        return {'status': 'installed' if installed and loaded else 'not_loaded' if installed else 'not_installed',
                'installed': installed, 'loaded': loaded, 'receipt': receipt,
                'cadence': 'daily_and_login_due_check', 'review_window_days': 14}
    except Exception:
        return {'status': 'failed', 'error_code': 'scheduler_status_unavailable'}


def install():
    """Explicit activation; import, status and normal review never install."""
    try:
        if sys.platform != 'darwin':
            return {'status': 'failed', 'error_code': 'macos_required'}
        root = _safe(_root(), create=True)
        runtime = _safe(root / 'scheduler', create=True)
        plist_path = _plist_path()
        _safe(plist_path.parent, create=True, private=False)
        if plist_path.is_symlink():
            raise OSError('unsafe scheduler plist')
        if plist_path.exists():
            existing = plistlib.loads(plist_path.read_bytes())
            if existing.get('Label') != LABEL or existing.get('ProgramArguments', [None, None])[1] != str(runtime / 'runner.py'):
                raise OSError('existing scheduler not owned')
            return status()
        # Runtime runner can record startup even when the project becomes unavailable.
        _atomic(runtime / 'runner.py', Path(__file__).read_bytes())
        for name in ('stdout.log', 'stderr.log'):
            path = runtime / name
            if path.is_symlink():
                raise OSError('unsafe scheduler log')
            if not path.exists():
                _atomic(path, b'')
            path.chmod(0o600)
        project = Path(__file__).resolve().parent
        payload = {
            'Label': LABEL,
            'ProgramArguments': [sys.executable, str(runtime / 'runner.py'), '--due'],
            'RunAtLoad': True, 'StartInterval': 86400, 'ProcessType': 'Background',
            'EnvironmentVariables': {'MI_LEARNING_ROOT': str(root), 'MI_PROJECT_ROOT': str(project),
                                     'MI_ORIGIN': 'scheduled', 'MI_ACTOR': 'scheduler',
                                     'PYTHONDONTWRITEBYTECODE': '1'},
            'StandardOutPath': str(runtime / 'stdout.log'),
            'StandardErrorPath': str(runtime / 'stderr.log'),
        }
        _atomic(plist_path, plistlib.dumps(payload))
        result = _launchctl(['bootstrap', f'gui/{os.getuid()}', str(plist_path)])
        if result.returncode:
            return {'status': 'failed', 'error_code': 'scheduler_bootstrap_failed', 'installed': True}
        return status()
    except Exception:
        return {'status': 'failed', 'error_code': 'scheduler_install_failed'}


def uninstall():
    """Remove only this schedule; preserve learning data and runtime receipts."""
    try:
        path = _plist_path()
        if path.is_symlink():
            raise OSError('unsafe scheduler plist')
        if not path.exists():
            return {'status': 'not_installed'}
        payload = plistlib.loads(path.read_bytes())
        expected = str(_root() / 'scheduler/runner.py')
        if payload.get('Label') != LABEL or payload.get('ProgramArguments', [None, None])[1] != expected:
            raise OSError('unowned scheduler')
        result = _launchctl(['bootout', f'gui/{os.getuid()}', str(path)])
        if result.returncode:
            still_loaded = _launchctl(['print', f'gui/{os.getuid()}/{LABEL}'])
            if still_loaded.returncode == 0:
                return {'status': 'failed', 'error_code': 'scheduler_bootout_failed'}
        path.unlink()
        return {'status': 'not_installed', 'learning_data_preserved': True}
    except Exception:
        return {'status': 'failed', 'error_code': 'scheduler_uninstall_failed'}


if __name__ == '__main__':
    if sys.argv[1:] != ['--due']:
        raise SystemExit('Use mi learning scheduler install|status|uninstall|due')
    result = due()
    if result.get('status') == 'failed':
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1)
    if any(review.get('findings') for review in result.get('reviews', [])):
        print(json.dumps({'status': 'actionable_review', 'review_count': len(result['reviews'])}))
