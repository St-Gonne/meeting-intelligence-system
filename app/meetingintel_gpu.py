"""Per-meeting LocalAI admission, independent of capture and fetching locks."""
from contextvars import ContextVar
from functools import wraps
import os
import signal
import subprocess
import sys
import time

from gpu_lock import Deferred, Session, assert_child
import meetingintel_progress as progress

_current = ContextVar('meetingintel_gpu_session', default=None)


def job(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if args and getattr(args[0], 'dry_run', False):
            return function(*args, **kwargs)
        if _current.get() is not None:
            return function(*args, **kwargs)
        if os.environ.get('LOCALAI_GPU_LOCK_HELD'):
            assert_child()
            started = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                import meetingintel_run_health as health
                health.gpu_timing(0, int((time.monotonic() - started) * 1000))
        print('Waiting for the local GPU; processing starts when it is free.', flush=True)
        try:
            progress.update_current('gpu_wait')
        except (progress.ProgressError, OSError):
            pass
        started = time.monotonic()
        acquired = None
        try:
            with Session('meetingintel', wait_seconds=600) as session:
                acquired = time.monotonic()
                token = _current.set(session)
                try:
                    return function(*args, **kwargs)
                finally:
                    _current.reset(token)
        except Deferred as error:
            print('Processing deferred: ' + error.reason, file=sys.stderr)
            raise
        finally:
            import meetingintel_run_health as health
            ended = time.monotonic()
            health.gpu_timing(int(((acquired or ended) - started) * 1000),
                              int((ended - acquired) * 1000) if acquired is not None else 0)
    return wrapped


def run(command, **kwargs):
    session = _current.get()
    if session is not None:
        return session.run(command, **kwargs)
    return subprocess.run(command, **kwargs, **child_options())


def child_options():
    if os.environ.get('LOCALAI_GPU_LOCK_HELD'):
        return {'pass_fds': (assert_child(),)}
    return {}


def supervised_main(callback, *, inspect_flags=('--help', '-h', '--dry-run', '--eligibility-only', '--check')):
    """Standalone GPU scripts get the same guardian; child descriptor proves nesting."""
    if any(flag in sys.argv[1:] for flag in inspect_flags):
        return callback()
    if os.environ.get('LOCALAI_GPU_LOCK_HELD'):
        assert_child()
        return callback()
    with Session('meetingintel') as session:
        result = session.run([sys.executable, *sys.argv], capture_output=False)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode
