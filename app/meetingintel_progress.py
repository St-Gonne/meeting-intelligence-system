"""Private, content-free progress snapshots for one MeetingIntel operation.

This module only records coarse stages, timestamps, an optional source
fingerprint, and optional byte counts. It never reads command output or meeting
artifacts. Callers may treat write failures as best-effort; ``read`` fails
closed and returns ``None`` for missing or unsafe state.
"""
from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import uuid


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024
MAX_COUNTER = (1 << 63) - 1

STAGE_LABELS = {
    "gpu_wait": "Waiting for GPU",
    "transcription": "Transcribing",
    "eligibility": "Checking eligibility",
    "analysis": "Preparing analysis",
    "identity": "Checking identity",
    "persistence": "Saving result",
    "complete": "Complete",
    "failed": "Stopped",
    "deferred": "Deferred",
}
TERMINAL_STAGES = frozenset({"complete", "failed", "deferred"})
ACTIVE_STAGES = frozenset(STAGE_LABELS) - TERMINAL_STAGES

_SAFE_OPERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
_SAFE_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_current_operation = contextvars.ContextVar("mi_progress_operation_id", default=None)


class ProgressError(RuntimeError):
    """Progress state could not be safely written or updated."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_OPERATION_ID.fullmatch(value) or value in {".", ".."}:
        raise ProgressError("invalid progress operation id")
    return value


def _fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SAFE_FINGERPRINT.fullmatch(value):
        raise ProgressError("invalid progress fingerprint")
    return value


def _counter(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0 or value > MAX_COUNTER:
        raise ProgressError("invalid progress byte counter")
    return value


def _stage(value: str, *, terminal: bool | None = None) -> str:
    if not isinstance(value, str) or value not in STAGE_LABELS:
        raise ProgressError("invalid progress stage")
    if terminal is True and value not in TERMINAL_STAGES:
        raise ProgressError("finish requires a terminal progress stage")
    if terminal is False and value in TERMINAL_STAGES:
        raise ProgressError("use finish for a terminal progress stage")
    return value


def _data_root(root: str | os.PathLike | None) -> Path:
    try:
        candidate = Path(os.environ.get("MI_PROGRESS_ROOT", ROOT) if root is None else root)
        candidate = Path(os.path.abspath(os.fspath(candidate)))
        info = candidate.lstat()
    except (OSError, TypeError, ValueError):
        raise ProgressError("progress root is unavailable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProgressError("progress root is unsafe")
    return candidate


def _check_directory(path: Path, *, create: bool) -> bool:
    """Check one directory boundary; return False when a read path is absent."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            return False
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError:
            raise ProgressError("progress directory could not be created") from None
        try:
            info = path.lstat()
        except OSError:
            raise ProgressError("progress directory is unavailable") from None
    except OSError:
        raise ProgressError("progress directory is unavailable") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ProgressError("progress directory is unsafe")
    return True


def _progress_directory(root: str | os.PathLike | None, *, create: bool) -> Path | None:
    data_root = _data_root(root)
    state = data_root / "state"
    if not _check_directory(state, create=create):
        return None
    progress = state / "progress"
    if not _check_directory(progress, create=create):
        return None
    # These exact parent relationships keep the generated state beneath root.
    if state.parent != data_root or progress.parent != state:
        raise ProgressError("progress path escaped its root")
    return progress


def _open_private_file(path: Path, flags: int) -> int:
    fd = -1
    try:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise ProgressError("progress file is unsafe")
        return fd
    except FileNotFoundError:
        raise
    except (OSError, ProgressError):
        if fd >= 0:
            os.close(fd)
        raise


def _read_payload(directory: Path, operation_id: str) -> dict | None:
    path = directory / (operation_id + ".json")
    fd = -1
    try:
        fd = _open_private_file(path, os.O_RDONLY)
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(MAX_RECORD_BYTES + 1)
        if len(raw) > MAX_RECORD_BYTES:
            raise ProgressError("progress record is too large")
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError, ProgressError):
        raise ProgressError("progress record is invalid") from None
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(payload, dict):
        raise ProgressError("progress record is invalid")
    return payload


def _reject_json_constant(_value: str):
    raise ValueError("invalid JSON constant")


def _parse_timestamp(value) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProgressError("invalid progress timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ProgressError("invalid progress timestamp") from None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProgressError("invalid progress timestamp")
    return parsed.astimezone(timezone.utc)


def _validated_record(payload: dict, operation_id: str) -> dict:
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != SCHEMA_VERSION:
        raise ProgressError("invalid progress schema")
    if payload.get("operation_id") != operation_id:
        raise ProgressError("invalid progress record")
    stage = payload.get("stage")
    if not isinstance(stage, str) or stage not in STAGE_LABELS:
        raise ProgressError("invalid progress stage")

    start_at = _parse_timestamp(payload.get("start_at"))
    stage_started_at = _parse_timestamp(payload.get("stage_started_at"))
    last_activity_at = _parse_timestamp(payload.get("last_activity_at"))
    finished_raw = payload.get("finished_at")
    finished_at = _parse_timestamp(finished_raw) if finished_raw is not None else None
    if stage_started_at < start_at or last_activity_at < stage_started_at:
        raise ProgressError("invalid progress stage times")
    if (stage in TERMINAL_STAGES) != (finished_at is not None):
        raise ProgressError("invalid progress completion state")
    if finished_at is not None and (finished_at != last_activity_at or finished_at < stage_started_at):
        raise ProgressError("invalid progress completion time")

    fingerprint64hex = payload.get("fingerprint64hex")
    if fingerprint64hex is not None:
        _fingerprint(fingerprint64hex)
    stdout_bytes = payload.get("stdout_bytes")
    stderr_bytes = payload.get("stderr_bytes")
    if stdout_bytes is not None:
        _counter(stdout_bytes, "stdout_bytes")
    if stderr_bytes is not None:
        _counter(stderr_bytes, "stderr_bytes")

    # Build a new allowlisted object. Unknown fields in old or damaged files
    # are never copied into a UI/API response.
    result = {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation_id,
        "stage": stage,
        "start_at": _stamp(start_at),
        "stage_started_at": _stamp(stage_started_at),
        "last_activity_at": _stamp(last_activity_at),
    }
    if fingerprint64hex is not None:
        result["fingerprint64hex"] = fingerprint64hex
    if stdout_bytes is not None:
        result["stdout_bytes"] = stdout_bytes
    if stderr_bytes is not None:
        result["stderr_bytes"] = stderr_bytes
    if finished_at is not None:
        result["finished_at"] = _stamp(finished_at)
    return result


def _snapshot(record: dict, now: datetime) -> dict:
    end = _parse_timestamp(record.get("finished_at")) if record.get("finished_at") else now.astimezone(timezone.utc)
    last_activity = _parse_timestamp(record["last_activity_at"])
    started = _parse_timestamp(record["start_at"])
    result = dict(record)
    result["stage_label"] = STAGE_LABELS[record["stage"]]
    result["elapsed_seconds"] = max(0, int((end - started).total_seconds()))
    result["seconds_since_activity"] = max(0, int((now.astimezone(timezone.utc) - last_activity).total_seconds()))
    return result


def _persist(directory: Path, record: dict) -> None:
    operation_id = record["operation_id"]
    destination = directory / (operation_id + ".json")
    temporary = directory / ("." + operation_id + "." + uuid.uuid4().hex + ".tmp")
    fd = -1
    try:
        # Do not silently replace a symlink, broad file, or another user's file.
        try:
            existing = _open_private_file(destination, os.O_RDONLY)
        except FileNotFoundError:
            pass
        else:
            os.close(existing)

        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > MAX_RECORD_BYTES:
            raise ProgressError("progress record is too large")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except ProgressError:
        raise
    except OSError:
        raise ProgressError("progress state could not be saved") from None
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _select_operation_id(operation_id: str | None) -> str:
    if operation_id is not None:
        return _identifier(operation_id)
    if "MI_OPERATION_ID" in os.environ:
        return _identifier(os.environ["MI_OPERATION_ID"])
    return uuid.uuid4().hex


def current_operation_id() -> str | None:
    """Return the operation id established by ``start`` in this context."""
    return _current_operation.get()


def worker_activity(stage: str) -> None:
    """Record an actual child-work event in the parent's existing snapshot."""
    operation_id = os.environ.get("MI_OPERATION_ID") or current_operation_id()
    if operation_id:
        try:
            update(stage, operation_id=operation_id)
        except (ProgressError, OSError):
            pass


def start(
    stage: str = "gpu_wait",
    *,
    operation_id: str | None = None,
    fingerprint64hex: str | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    root: str | os.PathLike | None = None,
) -> dict:
    """Create or replace one operation snapshot and bind its id to this context."""
    operation_id = _select_operation_id(operation_id)
    stage = _stage(stage, terminal=False)
    fingerprint64hex = _fingerprint(fingerprint64hex)
    stdout_bytes = _counter(stdout_bytes, "stdout_bytes")
    stderr_bytes = _counter(stderr_bytes, "stderr_bytes")
    now = _utc_now()
    timestamp = _stamp(now)
    record = {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation_id,
        "stage": stage,
        "start_at": timestamp,
        "stage_started_at": timestamp,
        "last_activity_at": timestamp,
    }
    if fingerprint64hex is not None:
        record["fingerprint64hex"] = fingerprint64hex
    if stdout_bytes is not None:
        record["stdout_bytes"] = stdout_bytes
    if stderr_bytes is not None:
        record["stderr_bytes"] = stderr_bytes
    directory = _progress_directory(root, create=True)
    _persist(directory, record)
    _current_operation.set(operation_id)
    return _snapshot(record, now)


def _load_for_update(operation_id: str, root: str | os.PathLike | None) -> tuple[Path, dict]:
    directory = _progress_directory(root, create=False)
    if directory is None:
        raise ProgressError("progress operation is unavailable")
    payload = _read_payload(directory, operation_id)
    if payload is None:
        raise ProgressError("progress operation is unavailable")
    record = _validated_record(payload, operation_id)
    if record["stage"] in TERMINAL_STAGES:
        raise ProgressError("progress operation is already finished")
    return directory, record


def _write_update(
    stage: str,
    *,
    operation_id: str | None,
    fingerprint64hex: str | None,
    stdout_bytes: int | None,
    stderr_bytes: int | None,
    root: str | os.PathLike | None,
    terminal: bool,
) -> dict:
    selected_id = _identifier(operation_id if operation_id is not None else current_operation_id())
    stage = _stage(stage, terminal=True if terminal else False)
    fingerprint64hex = _fingerprint(fingerprint64hex)
    stdout_bytes = _counter(stdout_bytes, "stdout_bytes")
    stderr_bytes = _counter(stderr_bytes, "stderr_bytes")
    directory, record = _load_for_update(selected_id, root)
    now = _utc_now()
    previous_activity = _parse_timestamp(record["last_activity_at"])
    if now < previous_activity:
        now = previous_activity
    previous_stage = record["stage"]
    record["stage"] = stage
    if stage != previous_stage:
        record["stage_started_at"] = _stamp(now)
    record["last_activity_at"] = _stamp(now)
    if fingerprint64hex is not None:
        record["fingerprint64hex"] = fingerprint64hex
    if stdout_bytes is not None:
        if stdout_bytes < record.get("stdout_bytes", 0):
            raise ProgressError("progress byte counters cannot decrease")
        record["stdout_bytes"] = stdout_bytes
    if stderr_bytes is not None:
        if stderr_bytes < record.get("stderr_bytes", 0):
            raise ProgressError("progress byte counters cannot decrease")
        record["stderr_bytes"] = stderr_bytes
    if terminal:
        record["finished_at"] = _stamp(now)
    _persist(directory, record)
    return _snapshot(record, now)


def update(
    stage: str,
    *,
    operation_id: str | None = None,
    fingerprint64hex: str | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    root: str | os.PathLike | None = None,
) -> dict:
    """Record a stage or activity update for an active operation."""
    return _write_update(
        stage,
        operation_id=operation_id,
        fingerprint64hex=fingerprint64hex,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        root=root,
        terminal=False,
    )


def update_current(
    stage: str,
    *,
    fingerprint64hex: str | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    root: str | os.PathLike | None = None,
) -> dict:
    """Update the operation id bound to the current context."""
    return update(
        stage,
        operation_id=current_operation_id(),
        fingerprint64hex=fingerprint64hex,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        root=root,
    )


def finish(
    stage: str = "complete",
    *,
    operation_id: str | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    root: str | os.PathLike | None = None,
) -> dict:
    """Finish an active operation with a safe terminal stage."""
    return _write_update(
        stage,
        operation_id=operation_id,
        fingerprint64hex=None,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        root=root,
        terminal=True,
    )


def read(operation_id: str | None = None, *, root: str | os.PathLike | None = None) -> dict | None:
    """Read one allowlisted snapshot; missing, malformed, or unsafe state is absent."""
    try:
        selected_id = _identifier(operation_id if operation_id is not None else
                                  (current_operation_id() or os.environ.get("MI_OPERATION_ID")))
        directory = _progress_directory(root, create=False)
        if directory is None:
            return None
        payload = _read_payload(directory, selected_id)
        if payload is None:
            return None
        record = _validated_record(payload, selected_id)
        return _snapshot(record, _utc_now())
    except Exception:
        # Reader is a display surface. Never pass malformed content or paths to
        # a caller, and never let a damaged optional sidecar interrupt work.
        return None
