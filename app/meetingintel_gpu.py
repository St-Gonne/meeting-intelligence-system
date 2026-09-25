"""Per-meeting LocalAI admission, independent of capture and fetching locks."""
from contextvars import ContextVar
from functools import wraps
import os
import signal
import subprocess
import sys

from gpu_lock import Deferred, Session, assert_child

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
            return function(*args, **kwargs)
        print('Waiting for the local GPU; processing starts when it is free.', flush=True)
        try:
            with Session('meetingintel') as session:
                token = _current.set(session)
                try:
                    return function(*args, **kwargs)
                finally:
                    _current.reset(token)
        except Deferred as error:
            print('Processing deferred: ' + error.reason, file=sys.stderr)
            raise
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
