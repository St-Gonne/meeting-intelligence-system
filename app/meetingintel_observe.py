"""Small, content-free instrumentation around existing operation boundaries."""
from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import re
import time
import uuid
from pathlib import Path

from gpu_lock import Deferred
from meetingintel_runtime import BusyError, RuntimeSafetyError
import meetingintel_learning as learning

_source = contextvars.ContextVar("mi_source", default=None)
_kind = contextvars.ContextVar("mi_source_kind", default=None)
_operation = contextvars.ContextVar("mi_operation", default=None)


def meetily_id(folder):
    return hashlib.sha256(("meetily:" + str(Path(folder).absolute())).encode()).hexdigest()


def safe_error(exc):
    # Never forward exception text, paths or model output to measurement storage.
    if isinstance(exc, Deferred):
        return "deferred"
    if isinstance(exc, BusyError):
        return "busy"
    if isinstance(exc, RuntimeSafetyError) and str(exc) in {
        "runtime directory unavailable",
        "runtime file unavailable",
    }:
        return "runtime_unavailable"

    value = str(exc)
    if value == "diarization_helper:exit_1:transcription_repetition_rejected":
        return "transcription_repetition_rejected"
    if value in {
        "transcription_failed",
        "transcription_quality:repetition_loop; review source audio",
        "transcription_backend:unsupported",
        "transcription_backend:non_phone_qwen_forbidden",
        "transcription_qwen:preflight_failed",
        "transcription_qwen:window_did_not_finish",
    }:
        return "transcription_failed"
    if value in {
        "invalid_owner_field",
        "flat_layer2:invalid_owner_field",
        "flat_layer3:invalid_owner_field",
    }:
        return "invalid_owner_field"
    # Helper errors arrive here only through the pipeline's sanitized fixed
    # wrapper codes. Keep dependency/manifest failures coarse; preserve the
    # one explicitly reviewed repetition code above.
    helper_codes = {
        "diarization_helper:missing_manifest_after_staging",
        "diarization_helper:malformed_manifest",
        "diarization_helper:unsuccessful_manifest",
        "diarization_helper:missing_lineage",
        "diarization_helper:source_lineage_mismatch",
        "diarization_helper:audio_first_lineage_mismatch",
        "diarization_helper:output_lineage_mismatch",
        "diarization_helper:qwen_output_lineage_mismatch",
        "diarization_helper:qwen_provenance_mismatch",
        "diarization_helper:legacy_provenance_mismatch",
        "diarization_helper:unknown_transcription_mode",
        "diarization_helper:missing_qwen_provenance",
        "diarization_helper:invalid_qwen_provenance",
    }
    if value in helper_codes or re.fullmatch(
        r"diarization_helper:exit_[0-9]+(?::(?:transcription_repetition_rejected|hf_token_not_set))?",
        value,
    ):
        return "diarization_failed"
    for message, code in (
        ("brief persistence failed", "brief_write_failed"),
        ("ledger persistence failed", "ledger_write_failed"),
        ("transcript persistence failed", "transcript_persistence_failed"),
        ("capture_incomplete", "capture_incomplete"),
        ("capture_incomplete:operator_correction_requires_review", "capture_incomplete"),
    ):
        if value == message:
            return code
    return "operation_failed"


def emit(event_type, **fields):
    fields.setdefault("source_id", _source.get())
    fields.setdefault("operation_id", _operation.get())
    fields.setdefault("source_kind", _kind.get())
    import meetingintel_run_health as health
    health.event(event_type, **fields)
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
                source_id = meetily_id(folder) if folder else candidate.fingerprint
                previous = learning.source_state(source_id) if result != 0 else {}
                # Keep the accepted stage reason when this adapter reports a
                # nonzero return rather than raising the underlying failure.
                code = previous.get("error_code") if previous.get("outcome") in {"failed", "failure", "interrupted"} else None
                emit("operation.succeeded" if result == 0 else "operation.failed",
                     source_id=source_id,
                     source_kind=candidate.source_kind, outcome="success" if result == 0 else "failure",
                     error_code=code,
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
                if not getattr(bound.get('args'), 'dry_run', False):
                    import meetingintel_run_health as health
                    health.begin(getattr(source, 'fingerprint', None) or bound.get('fingerprint'),
                                 getattr(source, 'source_kind', None) or 'meetily', select=True,
                                 source_identity=getattr(source, 'source_identity', None),
                                 source_folder=getattr(source, 'source_folder', None) or bound.get('folder_path'))
            started = time.monotonic()
            backend = bound.get("transcription_backend") or "legacy"
            if (stage == "transcription" and bound.get("transcription_backend") is None
                    and getattr(source, "source_kind", None) == "laptop_capture"):
                import meetingintel_pipeline as pipeline
                backend = pipeline.LAPTOP_TRANSCRIPTION_BACKEND_DEFAULT
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
