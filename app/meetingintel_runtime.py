"""Small local operation boundary and authoritative source corrections.

The operation lock is independent of best-effort usage telemetry. A failure to
establish this boundary must stop work, not silently allow conflicting writes.
No recording, transcript, credential, or model output is stored here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Iterator
import uuid


class RuntimeSafetyError(RuntimeError):
    """The private operational state cannot safely be established or read."""


class BusyError(RuntimeSafetyError):
    """Another thread or process owns the operation boundary."""


_guard = threading.Lock()
_held: dict[str, dict] = {}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_KINDS = {"processing", "recording", "capture", "review"}
_MAX_JSON_BYTES = 4 * 1024 * 1024


def _after_fork() -> None:
    # A child must not acquire reentrancy through inherited Python state or
    # retain an extra descriptor that extends the parent's lock lifetime.
    global _guard
    for held in _held.values():
        try:
            os.close(held["fd"])
        except OSError:
            pass
    _held.clear()
    _guard = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise RuntimeSafetyError("invalid runtime identifier")
    return value


def _root(*, create: bool) -> Path:
    configured = os.environ.get("MI_RUNTIME_ROOT")
    root = Path(configured) if configured else (
        Path.home() / "Library" / "Application Support" / "MeetingIntel" / "runtime"
    )
    if not root.is_absolute() or ".." in root.parts:
        raise RuntimeSafetyError("runtime root must be an absolute contained path")
    current = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            current = current / component
            try:
                info = current.lstat()
            except FileNotFoundError:
                if not create:
                    return root
                try:
                    current.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeSafetyError("runtime path contains an unsafe directory")
        info = root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeSafetyError("runtime root must be private and user-owned")
        return root
    except OSError as error:
        raise RuntimeSafetyError("runtime directory unavailable") from error


def _open_file(root: Path, name: str, *, create: bool = False) -> int:
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    fd = -1
    try:
        fd = os.open(root / name, flags, 0o600)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_nlink != 1
        ):
            raise RuntimeSafetyError("runtime file must be private, regular and user-owned")
        return fd
    except FileNotFoundError:
        raise
    except (OSError, RuntimeSafetyError) as error:
        if fd >= 0:
            os.close(fd)
        if isinstance(error, RuntimeSafetyError):
            raise
        raise RuntimeSafetyError("runtime file unavailable") from error


def _read_json(root: Path, name: str) -> dict | None:
    try:
        fd = _open_file(root, name)
    except FileNotFoundError:
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            text = handle.read(_MAX_JSON_BYTES + 1)
        if len(text) > _MAX_JSON_BYTES:
            raise RuntimeSafetyError("runtime state exceeds size limit")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeSafetyError("runtime state must be an object")
        return payload
    except (OSError, ValueError) as error:
        raise RuntimeSafetyError("runtime state is unreadable or malformed") from error


def _write_json(root: Path, name: str, payload: dict) -> None:
    temporary = root / (".write-" + uuid.uuid4().hex)
    fd = -1
    try:
        # Validate existing destinations before replacing them. Never follow a
        # symlink or silently overwrite broadly readable operational state.
        try:
            old = _open_file(root, name)
        except FileNotFoundError:
            pass
        else:
            os.close(old)
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > _MAX_JSON_BYTES:
            raise RuntimeSafetyError("runtime state exceeds size limit")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / name)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise RuntimeSafetyError("runtime state could not be saved") from error
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def operation_lock(kind: str = "processing", operation_id: str | None = None) -> Iterator[dict]:
    """Own the single recording/processing boundary without blocking.

    Only the owning thread may nest this context; nested callers retain the
    outer owner and operation ID. A forked or separately launched child must
    acquire independently. No environment variable bypasses the boundary.
    """
    if kind not in _KINDS:
        raise RuntimeSafetyError("unsupported operation kind")
    if operation_id is not None:
        _identifier(operation_id)
    root = _root(create=True)
    key = str(root)
    thread_id = threading.get_ident()
    with _guard:
        held = _held.get(key)
        if held is not None:
            if held["pid"] != os.getpid() or held["thread"] != thread_id:
                raise BusyError("MeetingIntel is busy in another operation")
            held["depth"] += 1
        else:
            fd = _open_file(root, "operation.lock", create=True)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                os.close(fd)
                raise BusyError("MeetingIntel is busy in another operation") from error
            except OSError as error:
                os.close(fd)
                raise RuntimeSafetyError("operation lock could not be acquired") from error
            owner = {
                "kind": kind,
                "pid": os.getpid(),
                "operation_id": operation_id or uuid.uuid4().hex,
                "started_at": _now(),
            }
            try:
                _write_json(root, "operation.json", owner)
            except BaseException:
                os.close(fd)
                raise
            held = {"fd": fd, "pid": os.getpid(), "thread": thread_id, "depth": 1, "owner": owner}
            _held[key] = held
    try:
        yield dict(held["owner"])
    finally:
        with _guard:
            # A fork can unwind inherited frames after its at-fork cleanup.
            if held["pid"] == os.getpid() and _held.get(key) is held:
                held["depth"] -= 1
                if held["depth"] == 0:
                    del _held[key]
                    # The last metadata remains diagnostic history only;
                    # status derives active ownership from flock, never PID.
                    os.close(held["fd"])


def exclusive_operation(kind: str = "processing"):
    """Decorator for direct entrypoints as well as nested CLI adapters."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with operation_lock(kind):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def _owner(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    if (
        set(payload) != {"kind", "pid", "operation_id", "started_at"}
        or payload.get("kind") not in _KINDS
        or type(payload.get("pid")) is not int
        or payload["pid"] <= 0
    ):
        raise RuntimeSafetyError("invalid operation owner metadata")
    _identifier(payload["operation_id"])
    try:
        timestamp = datetime.fromisoformat(payload["started_at"])
    except (TypeError, ValueError) as error:
        raise RuntimeSafetyError("invalid operation timestamp") from error
    if timestamp.tzinfo is None:
        raise RuntimeSafetyError("operation timestamp must include timezone")
    return payload


def status() -> dict:
    """Read safe active-owner facts; missing runtime state creates nothing."""
    root = _root(create=False)
    with _guard:
        held = _held.get(str(root))
        if held is not None:
            return {"active": True, "owner": dict(held["owner"])}
        try:
            fd = _open_file(root, "operation.lock")
        except FileNotFoundError:
            return {"active": False, "owner": None}
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"active": True, "owner": _owner(_read_json(root, "operation.json"))}
            except OSError as error:
                raise RuntimeSafetyError("operation status is unavailable") from error
            return {"active": False, "owner": None}
        finally:
            os.close(fd)


def busy_status() -> dict:
    return status()


def _annotations(root: Path) -> dict:
    payload = _read_json(root, "source-annotations.json")
    if payload is None:
        return {}
    if set(payload) != {"schema_version", "sources"} or type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or not isinstance(payload["sources"], dict):
        raise RuntimeSafetyError("invalid source annotation state")
    for key, value in payload["sources"].items():
        _identifier(key)
        if not isinstance(value, dict) or set(value) != {"source_id", "completeness", "reason", "provenance", "updated_at"}:
            raise RuntimeSafetyError("invalid source annotation")
        if value["source_id"] != key or value["completeness"] != "incomplete" or value["reason"] != "meeting_continued_after_capture" or value["provenance"] != "operator":
            raise RuntimeSafetyError("invalid source annotation")
        try:
            timestamp = datetime.fromisoformat(value["updated_at"])
        except (ValueError, TypeError) as error:
            raise RuntimeSafetyError("invalid source annotation timestamp") from error
        if timestamp.tzinfo is None:
            raise RuntimeSafetyError("source annotation timestamp must include timezone")
    return payload["sources"]


def annotate_source(
    source_id: str,
    completeness: str = "incomplete",
    reason: str = "meeting_continued_after_capture",
    provenance: str = "operator",
) -> dict:
    """Record the explicit correction supported in v1; never rewrite media.

    This intentionally has no API to relabel a recording complete or clear a
    correction. Adding that capability needs its own evidence/consent rule.
    """
    _identifier(source_id)
    if completeness != "incomplete" or reason != "meeting_continued_after_capture" or provenance != "operator":
        raise RuntimeSafetyError("unsupported source completeness correction")
    root = _root(create=True)
    fd = _open_file(root, "annotations.lock", create=True)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BusyError("Source corrections are being updated") from error
        except OSError as error:
            raise RuntimeSafetyError("source correction lock unavailable") from error
        sources = _annotations(root)
        annotation = {"source_id": source_id, "completeness": completeness, "reason": reason, "provenance": provenance, "updated_at": _now()}
        sources[source_id] = annotation
        _write_json(root, "source-annotations.json", {"schema_version": 1, "sources": sources})
        return dict(annotation)
    finally:
        os.close(fd)


def source_annotation(source_id: str) -> dict | None:
    """Missing is None; unreadable/unsafe annotation evidence raises, never hides."""
    _identifier(source_id)
    root = _root(create=False)
    annotation = _annotations(root).get(source_id)
    return dict(annotation) if annotation is not None else None
