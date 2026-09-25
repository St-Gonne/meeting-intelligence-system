"""Small, content-free instrumentation around existing operation boundaries."""
from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import time
import uuid
from pathlib import Path

import meetingintel_learning as learning

_source = contextvars.ContextVar("mi_source", default=None)
_kind = contextvars.ContextVar("mi_source_kind", default=None)
_operation = contextvars.ContextVar("mi_operation", default=None)


def meetily_id(folder):
    return hashlib.sha256(("meetily:" + str(Path(folder).absolute())).encode()).hexdigest()


def safe_error(exc):
    # Never forward exception text, paths or model output to measurement storage.
    value = str(exc)
    for prefix, code in (("flat_layer2:invalid_owner_field", "invalid_owner_field"),
                         ("transcription", "transcription_failed"),
                         ("diarization", "diarization_failed"),
                         ("brief persistence", "brief_write_failed"),
                         ("ledger persistence", "ledger_write_failed"),
                         ("capture_incomplete", "capture_incomplete")):
        if value.startswith(prefix):
            return code
    return "operation_failed"


def emit(event_type, **fields):
    fields.setdefault("source_id", _source.get())
    fields.setdefault("operation_id", _operation.get())
    fields.setdefault("source_kind", _kind.get())
    return learning.emit(event_type, **fields)


def production_attempt(function):
    """Give CLI attempts one identity through preflight and persistence."""
    @functools.wraps(function)
    def wrapped(selected, *args, **kwargs):
        import os
        if os.environ.get("MI_ORIGIN") == "gui":
            return function(selected, *args, **kwargs)
        operation_id = uuid.uuid4().hex
        token = _operation.set(operation_id)
        sources = [candidate for candidate in selected if kwargs.get("refresh_existing") or
                   candidate.fingerprint not in kwargs.get("processed", set())]
        folder = kwargs.get("explicit_meeting_folder")
        started = time.monotonic()
        for candidate in sources:
            emit("operation.started", source_id=meetily_id(folder) if folder else candidate.fingerprint,
                 source_kind=candidate.source_kind, outcome="started")
        try:
            result = function(selected, *args, **kwargs)
        except BaseException as exc:
            for candidate in sources:
                emit("operation.failed", source_id=meetily_id(folder) if folder else candidate.fingerprint,
                     source_kind=candidate.source_kind, outcome="failed", error_code=safe_error(exc))
            raise
        else:
            for candidate in sources:
                emit("operation.succeeded" if result == 0 else "operation.failed",
                     source_id=meetily_id(folder) if folder else candidate.fingerprint,
                     source_kind=candidate.source_kind, outcome="success" if result == 0 else "failure",
                     elapsed_ms=int((time.monotonic() - started)*1000))
            return result
        finally:
            _operation.reset(token)
    return wrapped


def observed(stage, *, source_scope=False):
    """Annotate real function outcomes, never infer success from console output."""
    def decorate(function):
        signature = inspect.signature(function)
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            source = bound.get("source") or bound.get("audio_first_source")
            source_id = getattr(source, "fingerprint", None)
            if not source_id and bound.get("folder_path"):
                source_id = meetily_id(bound["folder_path"])
            if not source_id and bound.get("folder"):
                source_id = meetily_id(bound["folder"])
            source_token = _source.set(source_id or _source.get())
            kind_token = _kind.set(getattr(source, "source_kind", None) or ("meetily" if bound.get("folder_path") else _kind.get()))
            operation_token = None
            if source_scope:
                import os
                operation_token = _operation.set(os.environ.get("MI_OPERATION_ID") or _operation.get() or str(uuid.uuid4()))
            started = time.monotonic()
            backend = bound.get("transcription_backend") or "legacy"
            fields = {"details": {"backend": backend}} if stage == "transcription" else {}
            emit("stage.started", stage=stage, outcome="started", **fields)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                emit("stage.failed", stage=stage, outcome="failed", error_code=safe_error(exc),
                     elapsed_ms=int((time.monotonic()-started)*1000), **fields)
                raise
            else:
                emit("stage.succeeded", stage=stage, outcome="succeeded",
                     elapsed_ms=int((time.monotonic()-started)*1000), **fields)
                return result
            finally:
                _source.reset(source_token)
                _kind.reset(kind_token)
                if operation_token is not None:
                    _operation.reset(operation_token)
        return wrapped
    return decorate
