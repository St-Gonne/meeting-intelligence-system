#!/usr/bin/env python3
"""Fetch finalized phone recordings from one Google Drive folder with rclone.

This is an upstream delivery step only. It never normalizes, transcribes,
diarizes, or processes recordings into MeetingIntel production state.
"""

from __future__ import annotations
from mi_paths import public_path

import argparse
import hashlib
import json
import os
import plistlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(str(public_path('project')))
DEFAULT_REMOTE_ROOT = "meetingintel-drive:com.nll.asr/MeetingIntel/phone-recordings"
DEFAULT_DOWNLOAD_ROOT = public_path("project/staging/phone-drive")
DEFAULT_LEDGER_PATH = public_path("project/state/phone_drive_fetch_ledger.json")
DEFAULT_LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / "com.meetingintel.phone-drive-fetch.plist"
# The managed LaunchAgent infrastructure is deliberately outside the protected
# project tree.  This root contains only scheduler code copies, logs, receipts,
# and small runtime metadata; fetch state and recordings retain their existing
# project paths.
DEFAULT_SCHEDULER_RUNTIME_ROOT = (
    Path.home() / "Library" / "Application Support" / "MeetingIntel" / "phone-fetch"
)
DEFAULT_FETCH_LOG_PATH = DEFAULT_SCHEDULER_RUNTIME_ROOT / "phone_drive_fetch.log"
DEFAULT_FETCH_ERROR_LOG_PATH = DEFAULT_SCHEDULER_RUNTIME_ROOT / "phone_drive_fetch.error.log"
DEFAULT_SCHEDULER_RECEIPT_PATH = DEFAULT_SCHEDULER_RUNTIME_ROOT / "phone_fetch_scheduler_receipt.json"
DEFAULT_SCHEDULER_WRAPPER_PATH = PROJECT_ROOT / "phone_fetch_launchd_wrapper.sh"
DEFAULT_SCHEDULER_RUNNER_PATH = PROJECT_ROOT / "phone_fetch_scheduler_runner.py"
DEFAULT_SCHEDULER_RUNTIME_WRAPPER_PATH = DEFAULT_SCHEDULER_RUNTIME_ROOT / DEFAULT_SCHEDULER_WRAPPER_PATH.name
DEFAULT_SCHEDULER_RUNTIME_RUNNER_PATH = DEFAULT_SCHEDULER_RUNTIME_ROOT / DEFAULT_SCHEDULER_RUNNER_PATH.name
LEDGER_SCHEMA_VERSION = 1
SCHEDULER_RECEIPT_SCHEMA_VERSION = 2
DEFAULT_SETTLE_INTERVAL_SECONDS = 10 * 60
DEFAULT_LOCK_STALE_SECONDS = 60 * 60
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_SECONDS = 1.0
DEFAULT_RETRY_MAX_SECONDS = 30.0
DEFAULT_RETRY_JITTER_SECONDS = 0.25
SCHEDULER_LABEL = "com.meetingintel.phone-drive-fetch"
SCHEDULER_INTERVAL_SECONDS = 2 * 60 * 60
SCHEDULER_MODES = ("sync", "probe")
SCHEDULER_STAGES = ("wrapper_started", "python_started", "completed")
SCHEDULER_CONTROLLED_PATH = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'
CONTROLLED_RCLONE_PATHS = (
    str(public_path('home/homebrew/opt/rclone/bin')),
    "/opt/homebrew/bin",
    "/usr/local/bin",
)
TRUSTED_INTERPRETER_ROOTS = (
    Path(str(public_path('home/homebrew'))),
    Path("/opt/homebrew"),
    Path("/usr/local"),
)
SCHEDULER_RUNTIME_ENV = "MEETINGINTEL_SCHEDULER_INSTALLED_AT"
IST = ZoneInfo("Asia/Kolkata")


class FetchError(RuntimeError):
    pass


class FetchAlreadyRunning(FetchError):
    pass


class RcloneCommandError(FetchError):
    def __init__(self, args: Sequence[str], returncode: int, detail: str):
        self.args_list = tuple(args)
        self.returncode = returncode
        self.detail = detail.strip() or "rclone command failed"
        super().__init__(self.detail)


@dataclass(frozen=True)
class RemoteRecording:
    relative_path: Path
    remote_id: str
    size: int
    modified_time: str
    md5: str | None
    drive_revision: str | None = None

    @property
    def revision(self) -> str:
        payload = {
            "drive_revision": self.drive_revision,
            "md5": self.md5,
            "modified_time": self.modified_time,
            "size": self.size,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FetchConfig:
    remote_root: str = DEFAULT_REMOTE_ROOT
    destination_root: Path = DEFAULT_DOWNLOAD_ROOT
    ledger_path: Path = DEFAULT_LEDGER_PATH
    dry_run: bool = False
    settle_interval_seconds: int = DEFAULT_SETTLE_INTERVAL_SECONDS
    lock_path: Path | None = None
    lock_stale_seconds: int = DEFAULT_LOCK_STALE_SECONDS
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS
    retry_jitter_seconds: float = DEFAULT_RETRY_JITTER_SECONDS

    def __post_init__(self) -> None:
        if self.settle_interval_seconds < 0:
            raise ValueError("settle interval must not be negative")
        if self.lock_stale_seconds <= 0:
            raise ValueError("lock stale interval must be positive")
        if self.retry_attempts < 1 or self.retry_base_seconds < 0 or self.retry_max_seconds < 0 or self.retry_jitter_seconds < 0:
            raise ValueError("retry configuration is invalid")

    @property
    def effective_lock_path(self) -> Path:
        return self.lock_path or self.ledger_path.with_name(self.ledger_path.name + ".lock")


@dataclass
class FetchSummary:
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    changed: int = 0
    failed: int = 0
    settled_ready: int = 0
    pending_settlement: int = 0
    requires_operator_decision: int = 0
    already_running: bool = False

    @property
    def exit_code(self) -> int:
        return 1 if self.failed or self.requires_operator_decision else 0


@dataclass(frozen=True)
class FetchStatus:
    settled_ready: int
    pending_settlement: int
    failed: int
    requires_operator_decision: int
    references: tuple[str, ...] = ()


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def find_rclone() -> str:
    for candidate in (
        Path(str(public_path('home/homebrew/opt/rclone/bin/rclone'))),
        Path("/opt/homebrew/bin/rclone"),
        Path("/usr/local/bin/rclone"),
    ):
        if candidate.is_file():
            return str(candidate)
    executable = shutil.which("rclone")
    if executable is None:
        raise RuntimeError("rclone is not installed; install and authorize it before using mi sync")
    return executable


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe remote path: {value!r}")
    if len(path.parts) != 4 or not all(part.isdigit() for part in path.parts[:3]):
        raise ValueError("remote recording must use YYYY/MM/DD/filename.m4a layout")
    year, month, day = path.parts[:3]
    if len(year) != 4 or len(month) != 2 or len(day) != 2:
        raise ValueError("remote recording must use YYYY/MM/DD/filename.m4a layout")
    if path.suffix.lower() != ".m4a":
        raise ValueError("remote recording is not an .m4a file")
    return path


def _recording_from_rclone(payload: dict[str, Any]) -> RemoteRecording | None:
    if payload.get("IsDir") or Path(str(payload.get("Path", ""))).suffix.lower() != ".m4a":
        return None
    relative_path = _safe_relative_path(str(payload.get("Path", "")))
    remote_id = payload.get("ID")
    size = payload.get("Size")
    modified_time = payload.get("ModTime")
    if not isinstance(remote_id, str) or not remote_id:
        raise ValueError(f"remote recording has no immutable ID: {relative_path}")
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"remote recording has invalid size: {relative_path}")
    if not isinstance(modified_time, str) or not modified_time:
        raise ValueError(f"remote recording has no modification time: {relative_path}")
    hashes = payload.get("Hashes")
    md5 = hashes.get("md5") if isinstance(hashes, dict) else None
    if md5 is not None and not isinstance(md5, str):
        raise ValueError(f"remote recording has invalid MD5: {relative_path}")
    drive_revision = payload.get("Revision", payload.get("Version"))
    if drive_revision is not None and not isinstance(drive_revision, str):
        raise ValueError(f"remote recording has invalid revision: {relative_path}")
    return RemoteRecording(relative_path, remote_id, size, modified_time, md5, drive_revision)


def _rclone_call(
    args: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(list(args), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RcloneCommandError(args, int(exc.returncode or 1), detail) from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise RcloneCommandError(args, result.returncode, detail)
    return result


def list_remote_recordings(
    remote_root: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[RemoteRecording]:
    result = _rclone_call(
        [find_rclone(), "lsjson", remote_root, "--recursive", "--files-only", "--hash"],
        runner=runner,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("rclone returned a non-list directory response")
    recordings: list[RemoteRecording] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("rclone returned a non-object directory entry")
        recording = _recording_from_rclone(item)
        if recording is not None:
            recordings.append(recording)
    ids = [recording.remote_id for recording in recordings]
    if len(ids) != len(set(ids)):
        raise ValueError("rclone returned duplicate Drive IDs")
    paths = [str(recording.relative_path) for recording in recordings]
    if len(paths) != len(set(paths)):
        raise ValueError("rclone returned duplicate destination paths")
    return sorted(recordings, key=lambda item: str(item.relative_path))


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": LEDGER_SCHEMA_VERSION, "records": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError(f"unsupported fetch ledger: {path}")
    if not isinstance(payload.get("records"), dict):
        raise ValueError(f"fetch ledger records are invalid: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")


def remote_file_path(remote_root: str, relative_path: Path) -> str:
    return f"{remote_root.rstrip('/')}/{relative_path.as_posix()}"


class FetchRunLock:
    """A user-local O_EXCL lock with bounded stale recovery."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime],
        stale_seconds: int,
        process_alive: Callable[[int], bool] = lambda pid: _process_alive(pid),
    ):
        self.path = path
        self.clock = clock
        self.stale_seconds = stale_seconds
        self.process_alive = process_alive
        self.acquired = False

    def _payload(self) -> bytes:
        return json.dumps({"pid": os.getpid(), "created_at": self.clock().astimezone(IST).isoformat()}).encode()

    def _stale(self) -> bool:
        if self.path.is_symlink() or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = payload.get("pid")
            created_at = datetime.fromisoformat(payload["created_at"])
            age = (self.clock() - created_at).total_seconds()
            return age > self.stale_seconds and (not isinstance(pid, int) or not self.process_alive(pid))
        except (OSError, ValueError, TypeError, KeyError):
            age = self.clock().timestamp() - self.path.stat().st_mtime
            return age > self.stale_seconds

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink(self.path, "fetch lock")
        for attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if attempt == 0 and self._stale():
                    _reject_symlink(self.path, "fetch lock")
                    self.path.unlink()
                    continue
                raise FetchAlreadyRunning("fetch already running")
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self._payload())
                handle.flush()
                os.fsync(handle.fileno())
            self.acquired = True
            return
        raise FetchAlreadyRunning("fetch already running")

    def release(self) -> None:
        if self.acquired:
            _reject_symlink(self.path, "fetch lock")
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "FetchRunLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_transient_rclone_error(exc: BaseException) -> bool:
    if not isinstance(exc, RcloneCommandError):
        return False
    detail = exc.detail.lower()
    permanent = ("unauthorized", "invalid_grant", "permission denied", "forbidden", "access denied", "http 401", "http 403")
    if any(token in detail for token in permanent):
        return False
    transient = (
        "timeout", "timed out", "connection reset", "connection refused", "temporarily unavailable",
        "rate limit", "too many requests", "http 429", "http 500", "http 502", "http 503", "http 504",
        "server error", "transport endpoint", "network is unreachable", "no route to host",
        "temporary failure in name resolution",
    )
    return any(token in detail for token in transient) or exc.returncode in {7, 10, 11, 18, 28, 429, 500, 502, 503, 504}


def _retry(
    operation: Callable[[], Any],
    *,
    attempts: int,
    base_seconds: float,
    max_seconds: float,
    jitter_seconds: float,
    sleep: Callable[[float], None],
    random_fn: Callable[[], float],
) -> Any:
    if attempts < 1:
        raise ValueError("retry attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient_rclone_error(exc) or attempt + 1 >= attempts:
                raise
            delay = min(max_seconds, base_seconds * (2**attempt)) + max(0.0, random_fn()) * jitter_seconds
            sleep(delay)
    raise AssertionError("retry loop did not return or raise")


def _download(
    recording: RemoteRecording,
    config: FetchConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    _reject_symlink(config.destination_root, "destination root")
    destination = config.destination_root / recording.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = config.destination_root
    for part in recording.relative_path.parts[:-1]:
        current /= part
        _reject_symlink(current, "destination directory")
    _reject_symlink(destination, "destination file")
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        _rclone_call(
            [find_rclone(), "copyto", remote_file_path(config.remote_root, recording.relative_path), str(temporary)],
            runner=runner,
        )
        if not temporary.is_file() or temporary.stat().st_size != recording.size:
            raise RuntimeError(f"download size mismatch for {recording.relative_path}")
        if recording.md5 is not None and md5_file(temporary).lower() != recording.md5.lower():
            raise RuntimeError(f"download checksum mismatch for {recording.relative_path}")
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata(recording: RemoteRecording, clock: Callable[[], datetime], *, status: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "last_attempt_at": clock().astimezone(IST).isoformat(),
        "local_relative_path": recording.relative_path.as_posix(),
        "drive_file_id": recording.remote_id,
        "drive_revision": recording.drive_revision,
        "modified_time": recording.modified_time,
        "size": recording.size,
        "md5": recording.md5,
        "revision": recording.revision,
    }
    result.update(extra)
    return result


def _safe_failure(exc: BaseException, remote_root: str) -> str:
    return f"{type(exc).__name__}: {str(exc).replace(remote_root, '<configured-phone-folder>')}"


def classify_fetch_failure(ledger_path: Path) -> str:
    """Return a privacy-safe scheduler category for the latest failed fetch.

    The ledger keeps the full local diagnostic.  LaunchAgent receipts and
    stderr logs must remain useful without copying remote paths, tokens, or
    provider responses into another durable surface.
    """
    try:
        ledger = load_ledger(ledger_path)
    except Exception:
        return "unknown"
    details: list[str] = []
    listing_failure = ledger.get("last_listing_failure")
    if isinstance(listing_failure, dict) and isinstance(listing_failure.get("failure"), str):
        details.append(listing_failure["failure"])
    records = ledger.get("records")
    if isinstance(records, dict):
        for record in records.values():
            if isinstance(record, dict) and record.get("status") == "failed" and isinstance(record.get("failure"), str):
                details.append(record["failure"])
    detail = "\n".join(details).lower()
    if any(token in detail for token in ("rate limit", "too many requests", "http 429", "quota", "storage limit")):
        return "drive_quota_or_rate_limit"
    if any(token in detail for token in ("unauthorized", "invalid_grant", "permission denied", "forbidden", "access denied", "http 401", "http 403")):
        return "drive_auth_or_permission"
    if any(token in detail for token in ("timed out", "timeout", "connection", "network", "no route", "name resolution", "http 500", "http 502", "http 503", "http 504")):
        return "network_or_provider_transient"
    if any(token in detail for token in ("symlink", "untracked_destination", "checksum", "size mismatch", "remote_path_changed")):
        return "local_or_source_safety"
    return "unknown"


def _settlement_state(previous: dict[str, Any] | None, recording: RemoteRecording, now: datetime, interval: int) -> tuple[str, str]:
    if not previous or previous.get("settlement_signature") != recording.revision:
        return "pending_settlement", now.astimezone(IST).isoformat()
    first = previous.get("first_observed_at")
    if previous.get("status") == "settled_ready":
        return "settled_ready", str(first or now.astimezone(IST).isoformat())
    try:
        observed = datetime.fromisoformat(str(first))
        elapsed = (now - observed).total_seconds()
    except (TypeError, ValueError):
        return "pending_settlement", now.astimezone(IST).isoformat()
    if elapsed >= interval:
        return "settled_ready", observed.astimezone(IST).isoformat()
    return "pending_settlement", observed.astimezone(IST).isoformat()


def fetch_recordings(
    config: FetchConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], datetime] = now_ist,
    sleep: Callable[[float], None] = lambda seconds: __import__("time").sleep(seconds),
    random_fn: Callable[[], float] = random.random,
) -> FetchSummary:
    summary = FetchSummary()
    try:
        with FetchRunLock(
            config.effective_lock_path,
            clock=clock,
            stale_seconds=config.lock_stale_seconds,
        ):
            ledger = load_ledger(config.ledger_path)
            records = ledger["records"]
            try:
                recordings = _retry(
                    lambda: list_remote_recordings(config.remote_root, runner=runner),
                    attempts=config.retry_attempts,
                    base_seconds=config.retry_base_seconds,
                    max_seconds=config.retry_max_seconds,
                    jitter_seconds=config.retry_jitter_seconds,
                    sleep=sleep,
                    random_fn=random_fn,
                )
            except Exception as exc:
                summary.failed = 1
                failure = _safe_failure(exc, config.remote_root)
                ledger["last_listing_failure"] = {"failure": failure, "at": clock().astimezone(IST).isoformat()}
                if not config.dry_run:
                    atomic_write_json(config.ledger_path, ledger)
                print(f"FAILED_LISTING error={failure}")
                return summary
            summary.discovered = len(recordings)
            for recording in recordings:
                previous = records.get(recording.remote_id)
                if isinstance(previous, dict) and previous.get("local_relative_path") not in (None, recording.relative_path.as_posix()):
                    summary.requires_operator_decision += 1
                    records[recording.remote_id] = _metadata(
                        recording, clock, status="requires_operator_decision", failure="remote_path_changed",
                        prior_local_relative_path=previous.get("local_relative_path"),
                    )
                    if not config.dry_run:
                        atomic_write_json(config.ledger_path, ledger)
                    print(f"REQUIRES_OPERATOR_DECISION {recording.relative_path} reason=remote_path_changed")
                    continue
                if isinstance(previous, dict) and previous.get("status") == "requires_operator_decision":
                    summary.requires_operator_decision += 1
                    print(f"REQUIRES_OPERATOR_DECISION {recording.relative_path}")
                    continue
                if isinstance(previous, dict) and previous.get("revision") != recording.revision:
                    summary.changed += 1

                if isinstance(previous, dict) and previous.get("revision") == recording.revision and previous.get("status") == "downloaded":
                    first_observed = previous.get("first_observed_at") or previous.get("downloaded_at")
                    previous = dict(previous, status="pending_settlement", first_observed_at=first_observed, settlement_signature=recording.revision)
                    records[recording.remote_id] = previous

                destination = config.destination_root / recording.relative_path
                tracked_path = previous.get("local_relative_path") if isinstance(previous, dict) else None
                destination_is_untracked = destination.exists() and (
                    tracked_path != recording.relative_path.as_posix()
                    or (isinstance(previous, dict) and previous.get("failure") == "untracked_destination")
                )
                if destination_is_untracked:
                    summary.failed += 1
                    records[recording.remote_id] = _metadata(recording, clock, status="failed", failure="untracked_destination")
                    if not config.dry_run:
                        atomic_write_json(config.ledger_path, ledger)
                    print(f"FAILED_UNTRACKED_DESTINATION {recording.relative_path}")
                    continue
                try:
                    _reject_symlink(destination, "destination file")
                except Exception as exc:
                    summary.failed += 1
                    records[recording.remote_id] = _metadata(recording, clock, status="failed", failure=f"{type(exc).__name__}: {exc}")
                    if not config.dry_run:
                        atomic_write_json(config.ledger_path, ledger)
                    print(f"FAILED {recording.relative_path} error={type(exc).__name__}: {exc}")
                    continue

                if isinstance(previous, dict) and previous.get("revision") == recording.revision and previous.get("status") in {"pending_settlement", "settled_ready"}:
                    state, first_observed = _settlement_state(previous, recording, clock(), config.settle_interval_seconds)
                    if state == "settled_ready":
                        records[recording.remote_id] = dict(previous, status="settled_ready", first_observed_at=first_observed)
                        summary.settled_ready += 1
                        summary.skipped += 1
                        if not config.dry_run:
                            atomic_write_json(config.ledger_path, ledger)
                        print(f"SKIP_UNCHANGED {recording.relative_path} settled_ready")
                    else:
                        summary.pending_settlement += 1
                        summary.skipped += 1
                        print(f"PENDING_SETTLEMENT {recording.relative_path}")
                    continue

                if config.dry_run:
                    summary.downloaded += 1
                    summary.pending_settlement += 1
                    print(f"WOULD_DOWNLOAD {recording.relative_path} pending_settlement")
                    continue
                try:
                    digest = _retry(
                        lambda: _download(recording, config, runner=runner),
                        attempts=config.retry_attempts,
                        base_seconds=config.retry_base_seconds,
                        max_seconds=config.retry_max_seconds,
                        jitter_seconds=config.retry_jitter_seconds,
                        sleep=sleep,
                        random_fn=random_fn,
                    )
                    first_observed = clock().astimezone(IST).isoformat()
                    records[recording.remote_id] = _metadata(
                        recording, clock, status="pending_settlement", downloaded_at=first_observed,
                        first_observed_at=first_observed, settlement_signature=recording.revision, sha256=digest,
                    )
                    atomic_write_json(config.ledger_path, ledger)
                    summary.downloaded += 1
                    summary.pending_settlement += 1
                    print(f"DOWNLOADED {recording.relative_path} pending_settlement")
                except Exception as exc:
                    summary.failed += 1
                    records[recording.remote_id] = _metadata(recording, clock, status="failed", failure=_safe_failure(exc, config.remote_root))
                    atomic_write_json(config.ledger_path, ledger)
                    print(f"FAILED {recording.relative_path} error={_safe_failure(exc, config.remote_root)}")
            print(
                "SUMMARY "
                f"discovered={summary.discovered} downloaded={summary.downloaded} skipped={summary.skipped} "
                f"changed={summary.changed} settled_ready={summary.settled_ready} "
                f"pending_settlement={summary.pending_settlement} failed={summary.failed} "
                f"requires_operator_decision={summary.requires_operator_decision} dry_run={str(config.dry_run).lower()}"
            )
            return summary
    except FetchAlreadyRunning:
        summary.already_running = True
        print("FETCH_ALREADY_RUNNING fetch already running")
        return summary


def summarize_fetch_status(ledger_path: Path) -> FetchStatus:
    ledger = load_ledger(ledger_path)
    counts = {"settled_ready": 0, "pending_settlement": 0, "failed": 0, "requires_operator_decision": 0}
    references: list[str] = []
    for record in ledger["records"].values():
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if status == "downloaded":
            status = "pending_settlement"
        if status in counts:
            counts[status] += 1
        if status == "requires_operator_decision" and isinstance(record.get("local_relative_path"), str):
            references.append(record["local_relative_path"])
    return FetchStatus(**counts, references=tuple(sorted(references)))


def print_fetch_status(ledger_path: Path) -> FetchStatus:
    status = summarize_fetch_status(ledger_path)
    print(
        "PHONE_FETCH_STATUS "
        f"settled_ready={status.settled_ready} pending_settlement={status.pending_settlement} "
        f"failed={status.failed} requires_operator_decision={status.requires_operator_decision}"
    )
    for reference in status.references:
        print(f"DECISION_REFERENCE {reference}")
    return status


def _lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _within_trusted_root(path: Path, trusted_roots: Sequence[Path]) -> bool:
    candidate = _lexical_path(path)
    return any(candidate == root or root in candidate.parents for root in map(_lexical_path, trusted_roots))


def validate_interpreter_path(
    path: Path,
    *,
    trusted_roots: Sequence[Path] = TRUSTED_INTERPRETER_ROOTS,
    access: Callable[[str, int], bool] = os.access,
    max_hops: int = 64,
) -> tuple[bool, str]:
    """Validate a configured interpreter and every symlink in its chain.

    The stable Homebrew ``opt`` path is intentionally accepted, but only when
    each link remains within an explicit trusted local interpreter root.
    """
    current = _lexical_path(path)
    if not _within_trusted_root(current, trusted_roots):
        return False, "interpreter is outside trusted local roots"
    seen: set[Path] = set()
    for _ in range(max_hops):
        if current in seen:
            return False, "interpreter symlink chain is cyclic"
        seen.add(current)
        try:
            if current.is_symlink():
                link_target = Path(os.readlink(current))
                target = link_target if link_target.is_absolute() else current.parent / link_target
                target = _lexical_path(target)
                if not _within_trusted_root(target, trusted_roots):
                    return False, "interpreter symlink escapes trusted local roots"
                current = target
                continue
            if not current.exists():
                return False, "interpreter target is dangling"
            if not current.is_file():
                return False, "interpreter target is not a regular file"
            if not access(str(current), os.R_OK | os.X_OK):
                return False, "interpreter target is not executable"
            return True, "trusted executable"
        except OSError:
            return False, "interpreter target is unreadable"
    return False, "interpreter symlink chain is too long"


def write_scheduler_receipt(
    path: Path = DEFAULT_SCHEDULER_RECEIPT_PATH,
    *,
    success: bool | None,
    category: str,
    exit_code: int | None,
    stage: str = "completed",
    mode: str = "sync",
    clock: Callable[[], datetime] = now_ist,
) -> None:
    """Atomically write only safe lifecycle facts for one scheduler run."""
    if stage not in SCHEDULER_STAGES:
        raise ValueError("invalid scheduler receipt stage")
    if mode not in SCHEDULER_MODES:
        raise ValueError("invalid scheduler receipt mode")
    if stage == "completed" and success not in {True, False}:
        raise ValueError("completed scheduler receipts require a boolean outcome")
    if stage != "completed" and success is not None:
        raise ValueError("started scheduler receipts cannot have an outcome")
    _reject_symlink(path, "scheduler receipt")
    _reject_symlink(path.parent, "scheduler receipt directory")
    payload = {
        "schema_version": SCHEDULER_RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "mode": mode,
        "updated_at": clock().astimezone(IST).isoformat(),
        "outcome": "success" if success is True else "failure" if success is False else "unknown",
        "category": category,
        "exit_code": exit_code,
    }
    atomic_write_json(path, payload)
    path.chmod(0o600)


def load_scheduler_receipt(path: Path = DEFAULT_SCHEDULER_RECEIPT_PATH) -> dict[str, Any]:
    if path.is_symlink():
        return {"status": "invalid", "reason": "symlink"}
    if not path.exists():
        return {"status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "invalid", "reason": "unreadable_or_malformed"}
    if not isinstance(payload, dict) or not isinstance(payload.get("category"), str):
        return {"status": "invalid", "reason": "schema"}
    if payload.get("schema_version") == 1:
        if payload.get("outcome") not in {"success", "failure"} or not isinstance(payload.get("exit_code"), int) or not isinstance(payload.get("completed_at"), str):
            return {"status": "invalid", "reason": "schema"}
        return {
            "status": "valid",
            **payload,
            "stage": "completed",
            "mode": "sync",
            "updated_at": payload["completed_at"],
        }
    if (
        payload.get("schema_version") != SCHEDULER_RECEIPT_SCHEMA_VERSION
        or payload.get("stage") not in SCHEDULER_STAGES
        or payload.get("mode") not in SCHEDULER_MODES
        or payload.get("outcome") not in {"unknown", "success", "failure"}
        or not isinstance(payload.get("updated_at"), str)
        or (payload.get("exit_code") is not None and not isinstance(payload.get("exit_code"), int))
        or (payload.get("stage") == "completed" and payload.get("outcome") not in {"success", "failure"})
        or (payload.get("stage") != "completed" and payload.get("outcome") != "unknown")
    ):
        return {"status": "invalid", "reason": "schema"}
    return {"status": "valid", **payload}


def _private_path(path: Path, mode: int, *, directory: bool = False) -> None:
    _reject_symlink(path, "managed scheduler path")
    if directory:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        path.touch()
    if path.exists():
        path.chmod(mode)


def _path_has_symlink_component(path: Path) -> bool:
    """Return whether any existing component of *path* is a symlink."""
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for index, part in enumerate(parts):
        current = current / part
        # macOS commonly exposes /var and /tmp as compatibility symlinks.  The
        # managed root itself and every component below that system prefix are
        # still checked; the fixed user Application Support path is not
        # weakened by ignoring only the first system-level component.
        if index > 0 and current.is_symlink():
            return True
    return False


def _validate_runtime_root(path: Path) -> None:
    """Validate the private scheduler root without following links.

    The default root is fixed below the user's Application Support directory.
    An explicitly injected root is accepted only as a test seam when it is an
    absolute, user-owned, private directory outside the project tree.
    """
    if not path.is_absolute():
        raise ValueError("scheduler runtime root must be absolute")
    if path == PROJECT_ROOT or PROJECT_ROOT in path.parents:
        raise ValueError("scheduler runtime root must not be inside the project")
    if _path_has_symlink_component(path):
        raise RuntimeError("scheduler runtime root must not contain symlinks")
    if path.exists() and path.is_symlink():
        raise RuntimeError("scheduler runtime root must not be a symlink")
    if path.exists():
        stat = path.stat()
        if stat.st_uid != os.getuid():
            raise PermissionError("scheduler runtime root is not user-owned")
        if stat.st_mode & 0o077:
            raise PermissionError("scheduler runtime root must be user-only")


def _ensure_runtime_root(path: Path) -> None:
    _validate_runtime_root(path)
    path.mkdir(parents=True, exist_ok=True)
    _private_path(path, 0o700, directory=True)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise PermissionError("scheduler runtime root must be user-owned and private")


def _atomic_copy_private(source: Path, destination: Path, mode: int) -> None:
    """Copy a managed source file into the runtime root atomically."""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"managed scheduler source is not a regular file: {source.name}")
    _reject_symlink(destination, "managed scheduler runtime file")
    _reject_symlink(destination.parent, "managed scheduler runtime directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _private_path(destination.parent, 0o700, directory=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.chmod(mode)
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
        destination.chmod(mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scheduler_runtime_paths(runtime_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        runtime_root,
        runtime_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name,
        runtime_root / DEFAULT_SCHEDULER_RUNNER_PATH.name,
        runtime_root / "phone_drive_fetch.log",
        runtime_root / "phone_drive_fetch.error.log",
    )


def _resolve_runtime_root(
    runtime_root: Path | None,
    *,
    log_path: Path | None = None,
    error_log_path: Path | None = None,
    receipt_path: Path | None = None,
) -> Path:
    if runtime_root is not None:
        return runtime_root
    # Explicit temporary paths are an injectable test seam.  Production calls
    # with no paths always use the fixed Application Support root.
    for candidate in (log_path, error_log_path, receipt_path):
        if candidate is not None and candidate.parent.parent != candidate.parent:
            return candidate.parent.parent
    return DEFAULT_SCHEDULER_RUNTIME_ROOT


def build_scheduler_command(
    *,
    mode: str = "sync",
    python_path: Path = Path(sys.executable),
    project_root: Path = PROJECT_ROOT,
    wrapper_path: Path | None = None,
    runner_path: Path | None = None,
    receipt_path: Path = DEFAULT_SCHEDULER_RECEIPT_PATH,
) -> list[str]:
    if mode not in SCHEDULER_MODES:
        raise ValueError("invalid scheduler mode")
    wrapper = wrapper_path or DEFAULT_SCHEDULER_RUNTIME_WRAPPER_PATH
    runner = runner_path or DEFAULT_SCHEDULER_RUNTIME_RUNNER_PATH
    cli_path = project_root / "meetingintel_cli.py"
    paths = (python_path, wrapper, runner, cli_path, receipt_path)
    if any(not path.is_absolute() for path in paths):
        raise ValueError("scheduler command requires absolute paths")
    return [
        "/bin/sh",
        str(wrapper),
        str(python_path),
        str(runner),
        str(cli_path),
        str(receipt_path),
        mode,
    ]


def build_daily_launch_agent(
    hour: int,
    minute: int,
    *,
    log_path: Path | None = None,
    error_log_path: Path | None = None,
    receipt_path: Path | None = None,
    mode: str = "sync",
    project_root: Path = PROJECT_ROOT,
    python_path: Path = Path(sys.executable),
    runtime_root: Path | None = None,
    wrapper_path: Path | None = None,
    runner_path: Path | None = None,
    installed_at: str | None = None,
) -> bytes:
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("daily schedule must use a valid local hour and minute")
    if mode not in SCHEDULER_MODES:
        raise ValueError("invalid scheduler mode")
    runtime_root = _resolve_runtime_root(
        runtime_root, log_path=log_path, error_log_path=error_log_path, receipt_path=receipt_path,
    )
    _validate_runtime_root(runtime_root)
    default_wrapper = runtime_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name
    default_runner = runtime_root / DEFAULT_SCHEDULER_RUNNER_PATH.name
    effective_log = log_path or runtime_root / "phone_drive_fetch.log"
    effective_error_log = error_log_path or runtime_root / "phone_drive_fetch.error.log"
    effective_receipt = receipt_path or runtime_root / "phone_fetch_scheduler_receipt.json"
    payload = {
        "Label": SCHEDULER_LABEL,
        "ProgramArguments": build_scheduler_command(
            mode=mode,
            python_path=python_path,
            project_root=project_root,
            wrapper_path=wrapper_path or default_wrapper,
            runner_path=runner_path or default_runner,
            receipt_path=effective_receipt,
        ),
        "WorkingDirectory": str(runtime_root),
        "RunAtLoad": True,
        "StartInterval": SCHEDULER_INTERVAL_SECONDS,
        "StandardOutPath": str(effective_log),
        "StandardErrorPath": str(effective_error_log),
        "EnvironmentVariables": {
            "PATH": SCHEDULER_CONTROLLED_PATH,
            "HOME": str(Path.home()),
            "MEETINGINTEL_PHONE_FETCH_SCHEDULED": "1",
            SCHEDULER_RUNTIME_ENV: installed_at or "unknown",
        },
    }
    return plistlib.dumps(payload, sort_keys=True)


def _launchd_snapshot(
    agent_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    try:
        result = runner(
            ["launchctl", "print", f"gui/{os.getuid()}/{SCHEDULER_LABEL}"],
            check=False, capture_output=True, text=True,
        )
    except Exception as exc:
        return {"loaded": False, "error": f"launchd query failed: {type(exc).__name__}"}
    return {
        "loaded": result.returncode == 0,
        "output": result.stdout or "",
        "error": result.stderr.strip() if result.returncode else "",
    }


def _launchd_facts(output: str) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "run_count": None,
        "last_exit_code": None,
        "last_exit_label": None,
        "state": "unknown",
    }
    patterns = {
        "run_count": r"^\s*runs\s*=\s*(\d+)\s*$",
        "last_exit_code": r"^\s*last exit code\s*=\s*(-?\d+)(?:\s*:\s*([A-Za-z0-9_.-]+))?\s*$",
        "state": r"^\s*state\s*=\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            if key == "last_exit_code":
                facts[key] = int(match.group(1))
                facts["last_exit_label"] = match.group(2)
            else:
                facts[key] = int(match.group(1)) if key != "state" else match.group(1).strip()
    return facts


def _scheduler_check(category: str, ok: bool | None, detail: str) -> dict[str, Any]:
    return {
        "category": category,
        "ok": ok is True,
        "status": "pass" if ok is True else "not_checked" if ok is None else "fail",
        "detail": detail,
    }


def _receipt_is_stale(receipt: dict[str, Any], installed_at: str | None) -> bool:
    if receipt.get("status") != "valid" or not installed_at or installed_at == "unknown":
        return False
    try:
        receipt_time = datetime.fromisoformat(str(receipt.get("updated_at")).replace("Z", "+00:00"))
        install_time = datetime.fromisoformat(installed_at.replace("Z", "+00:00"))
        if receipt_time.tzinfo is None or install_time.tzinfo is None:
            return True
        return receipt_time < install_time
    except (TypeError, ValueError):
        return True


def _safe_path_check(path: Path, *, directory: bool, private: bool, access: Callable[[str, int], bool]) -> tuple[bool, str]:
    try:
        if path.is_symlink():
            return False, "symlink rejected"
        if not path.exists():
            return False, "missing"
        if directory and not path.is_dir():
            return False, "not a directory"
        if not directory and not path.is_file():
            return False, "not a regular file"
        if private and path.stat().st_mode & 0o077:
            return False, "permissions are not user-only"
        if not access(str(path), os.R_OK | os.X_OK if directory else os.R_OK):
            return False, "not readable by the scheduler user"
        return True, "accessible"
    except OSError:
        return False, "unreadable"


def _runtime_path_check(
    path: Path,
    *,
    directory: bool,
    access: Callable[[str, int], bool],
) -> tuple[bool, str]:
    if _path_has_symlink_component(path):
        return False, "symlink rejected"
    ok, detail = _safe_path_check(path, directory=directory, private=True, access=access)
    if not ok:
        return ok, detail
    try:
        if path.stat().st_uid != os.getuid():
            return False, "not user-owned"
    except OSError:
        return False, "unreadable"
    return True, "private and user-owned"


def scheduler_diagnose(
    *,
    agent_path: Path = DEFAULT_LAUNCH_AGENT_PATH,
    project_root: Path = PROJECT_ROOT,
    runtime_root: Path | None = None,
    log_path: Path | None = None,
    error_log_path: Path | None = None,
    receipt_path: Path | None = None,
    receipt: dict[str, Any] | None = None,
    trusted_roots: Sequence[Path] = TRUSTED_INTERPRETER_ROOTS,
    launchd_snapshot: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    access: Callable[[str, int], bool] = os.access,
    which: Callable[..., str | None] = shutil.which,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    installed = agent_path.exists() and not agent_path.is_symlink()
    payload: dict[str, Any] | None = None
    if agent_path.is_symlink():
        checks.append(_scheduler_check("launchd_configuration", False, "managed plist is a symlink"))
    elif not agent_path.exists():
        checks.append(_scheduler_check("launchd_configuration", False, "managed plist is missing"))
    else:
        try:
            candidate = plistlib.loads(agent_path.read_bytes())
            if not isinstance(candidate, dict):
                raise ValueError("plist root is not a dictionary")
            payload = candidate
        except (OSError, ValueError, plistlib.InvalidFileException):
            checks.append(_scheduler_check("launchd_configuration", False, "managed plist is unreadable or malformed"))

    if payload is not None:
        args = payload.get("ProgramArguments")
        if runtime_root is None:
            configured_working_directory = payload.get("WorkingDirectory")
            runtime_root = Path(configured_working_directory) if isinstance(configured_working_directory, str) else DEFAULT_SCHEDULER_RUNTIME_ROOT
        if isinstance(args, list) and len(args) == 7:
            if receipt_path is None and isinstance(args[5], str):
                receipt_path = Path(args[5])
            if log_path is None and isinstance(payload.get("StandardOutPath"), str):
                log_path = Path(payload["StandardOutPath"])
            if error_log_path is None and isinstance(payload.get("StandardErrorPath"), str):
                error_log_path = Path(payload["StandardErrorPath"])
    if runtime_root is None:
        runtime_root = DEFAULT_SCHEDULER_RUNTIME_ROOT
    log_path = log_path or runtime_root / "phone_drive_fetch.log"
    error_log_path = error_log_path or runtime_root / "phone_drive_fetch.error.log"
    receipt_path = receipt_path or runtime_root / "phone_fetch_scheduler_receipt.json"

    if payload is not None:
        args = payload.get("ProgramArguments")
        env = payload.get("EnvironmentVariables")
        expected_script = str(project_root / "meetingintel_cli.py")
        expected_wrapper = str(runtime_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name)
        expected_runner = str(runtime_root / DEFAULT_SCHEDULER_RUNNER_PATH.name)
        expected_receipt = str(receipt_path)
        shape_ok = (
            isinstance(args, list) and len(args) == 7 and args[0] == "/bin/sh"
            and args[1] == expected_wrapper and isinstance(args[2], str)
            and args[3:6] == [expected_runner, expected_script, expected_receipt]
            and args[6] in SCHEDULER_MODES
            and payload.get("WorkingDirectory") == str(runtime_root)
            and payload.get("RunAtLoad") is True
            and payload.get("StartInterval") == SCHEDULER_INTERVAL_SECONDS
            and payload.get("StandardOutPath") == str(log_path)
            and payload.get("StandardErrorPath") == str(error_log_path)
            and isinstance(env, dict)
            and isinstance(env.get("PATH"), str) and bool(env.get("PATH"))
            and env.get("HOME") == str(Path.home())
            and env.get("MEETINGINTEL_PHONE_FETCH_SCHEDULED") == "1"
        )
        checks.append(_scheduler_check("launchd_configuration", shape_ok, "expected command and schedule shape" if shape_ok else "command, path, environment, or interval shape mismatch"))
        executable = Path(args[2]) if isinstance(args, list) and len(args) > 2 and isinstance(args[2], str) else None
        path_value = env.get("PATH", "") if isinstance(env, dict) else ""
    else:
        executable = None
        path_value = ""

    if executable is None:
        checks.append(_scheduler_check("executable", False, "executable is not configured"))
    else:
        executable_ok, executable_detail = validate_interpreter_path(
            executable, trusted_roots=trusted_roots, access=access,
        )
        checks.append(_scheduler_check("executable", executable_ok, executable_detail))
    shell_ok, shell_detail = _safe_path_check(Path("/bin/sh"), directory=False, private=False, access=access)
    checks.append(_scheduler_check("executable", shell_ok, f"launchd wrapper shell {shell_detail}"))

    runtime_ok, runtime_detail = _runtime_path_check(runtime_root, directory=True, access=access)
    checks.append(_scheduler_check("runtime_root", runtime_ok, f"runtime root {runtime_detail}"))
    for label, path in (
        ("scheduler wrapper", runtime_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name),
        ("scheduler runner", runtime_root / DEFAULT_SCHEDULER_RUNNER_PATH.name),
    ):
        script_ok, script_detail = _runtime_path_check(path, directory=False, access=access)
        checks.append(_scheduler_check("runtime_root", script_ok, f"{label} {script_detail}"))
    for label, path in (("stdout log", log_path), ("stderr log", error_log_path)):
        log_ok, log_detail = _runtime_path_check(path, directory=False, access=access)
        checks.append(_scheduler_check("runtime_root", log_ok, f"{label} {log_detail}"))
    project_ok, project_detail = _safe_path_check(project_root, directory=True, private=False, access=access)
    checks.append(_scheduler_check("protected_path", project_ok, f"project root {project_detail}"))
    cli_ok, cli_detail = _safe_path_check(project_root / "meetingintel_cli.py", directory=False, private=False, access=access)
    checks.append(_scheduler_check("protected_path", cli_ok, f"MeetingIntel CLI {cli_detail}"))
    plist_ok, plist_detail = _safe_path_check(agent_path, directory=False, private=True, access=access)
    checks.append(_scheduler_check("protected_path", plist_ok, f"managed plist {plist_detail}"))
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt_ok, receipt_detail = _runtime_path_check(receipt_path, directory=False, access=access)
        checks.append(_scheduler_check("runtime_root", receipt_ok, f"scheduler receipt {receipt_detail}"))

    controlled_path_present = any(entry in path_value.split(os.pathsep) for entry in CONTROLLED_RCLONE_PATHS)
    checks.append(_scheduler_check("environment", controlled_path_present, "controlled rclone PATH is present" if controlled_path_present else "controlled rclone PATH is missing"))
    rclone = which("rclone", path=path_value) if path_value else None
    checks.append(_scheduler_check("rclone", bool(rclone), "rclone resolves from controlled PATH" if rclone else "rclone does not resolve from controlled PATH"))

    snapshot = launchd_snapshot or _launchd_snapshot(agent_path, runner=runner)
    loaded = bool(snapshot.get("loaded"))
    facts = _launchd_facts(str(snapshot.get("output", "")))
    receipt_data = receipt if receipt is not None else load_scheduler_receipt(receipt_path)
    receipt_status = receipt_data.get("status")
    installed_at = (
        payload.get("EnvironmentVariables", {}).get(SCHEDULER_RUNTIME_ENV)
        if isinstance(payload, dict) and isinstance(payload.get("EnvironmentVariables"), dict)
        else None
    )
    if _receipt_is_stale(receipt_data, installed_at):
        receipt_status = "stale"
    configured_mode = (
        payload.get("ProgramArguments", [None] * 7)[6]
        if isinstance(payload, dict)
        and isinstance(payload.get("ProgramArguments"), list)
        and len(payload["ProgramArguments"]) == 7
        else None
    )
    runtime_outcome = "unknown"
    receipt_at = receipt_data.get("updated_at") if receipt_status == "valid" else None
    if receipt_at is None and receipt_status == "valid":
        receipt_at = receipt_data.get("completed_at")
    if snapshot.get("error"):
        checks.append(_scheduler_check("launchd_runtime", False, "launchd status query failed"))
    elif not loaded:
        checks.append(_scheduler_check("launchd_runtime", False, "LaunchAgent is not loaded"))
        runtime_outcome = "unknown"
    elif receipt_status == "stale":
        runtime_outcome = "unknown"
        checks.append(_scheduler_check("launchd_runtime", False, "scheduler receipt predates the current installation"))
    elif receipt_status == "valid" and configured_mode in SCHEDULER_MODES and receipt_data.get("mode") != configured_mode:
        runtime_outcome = "unknown"
        checks.append(_scheduler_check("launchd_runtime", False, "scheduler receipt mode does not match the managed command mode"))
    elif receipt_status == "valid" and receipt_data.get("stage") == "wrapper_started":
        runtime_outcome = "wrapper_started"
        checks.append(_scheduler_check("launchd_runtime", False, "wrapper started; Python outcome is unknown"))
    elif receipt_status == "valid" and receipt_data.get("stage") == "python_started":
        runtime_outcome = "python_started"
        checks.append(_scheduler_check("launchd_runtime", False, "Python started; MeetingIntel completion is unknown"))
    elif receipt_status == "valid" and receipt_data.get("mode") == "probe" and receipt_data.get("outcome") == "failure":
        runtime_outcome = "probe_failure"
        checks.append(_scheduler_check("launchd_runtime", False, f"scheduler probe failed: {receipt_data.get('category')}"))
    elif receipt_status == "valid" and receipt_data.get("mode") == "probe" and receipt_data.get("outcome") == "success":
        runtime_outcome = "confirmed_probe"
        checks.append(_scheduler_check("launchd_runtime", False, "scheduler probe succeeded; real fetch has not been confirmed"))
    elif receipt_status == "valid" and receipt_data.get("category") == "python_exec_failed":
        runtime_outcome = "python_exec_failure"
        checks.append(_scheduler_check("launchd_runtime", False, "wrapper started but the configured Python could not execute"))
    elif receipt_status == "valid" and receipt_data.get("category") == "project_cli_failure":
        runtime_outcome = "project_cli_failure"
        checks.append(_scheduler_check("launchd_runtime", False, "Python started but could not execute the absolute MeetingIntel CLI probe"))
    elif receipt_status == "valid" and receipt_data.get("category") == "meetingintel_failure":
        runtime_outcome = "meetingintel_failure"
        checks.append(_scheduler_check("launchd_runtime", False, "Python started but MeetingIntel failed before completion"))
    elif receipt_status == "valid" and receipt_data.get("outcome") == "failure":
        runtime_outcome = "confirmed_failure"
        checks.append(_scheduler_check("launchd_runtime", False, f"confirmed failed receipt: {receipt_data.get('category')}"))
    elif receipt_status == "valid" and receipt_data.get("outcome") == "success" and facts["last_exit_code"] not in (None, 0):
        runtime_outcome = "confirmed_failure"
        checks.append(_scheduler_check("launchd_runtime", False, f"last run exited {facts['last_exit_code']}"))
    elif receipt_status == "valid" and receipt_data.get("outcome") == "success":
        runtime_outcome = "confirmed_success"
        detail = "confirmed successful scheduler receipt"
        if facts["state"].lower() == "not running":
            detail += "; idle between interval executions"
        checks.append(_scheduler_check("launchd_runtime", True, detail))
    elif receipt_status == "missing" and facts["last_exit_code"] not in (None, 0):
        runtime_outcome = "launchd_launch_failure"
        checks.append(_scheduler_check("launchd_runtime", False, f"LaunchAgent exited {facts['last_exit_code']} before the wrapper-start receipt"))
    elif receipt_status == "missing" and facts["run_count"] in (None, 0):
        runtime_outcome = "never_run"
        checks.append(_scheduler_check("launchd_runtime", False, "no completed scheduler receipt; no run confirmed"))
    elif receipt_status == "missing":
        runtime_outcome = "unknown"
        checks.append(_scheduler_check("launchd_runtime", False, "runtime outcome unknown; completed receipt is missing"))
    else:
        runtime_outcome = "unknown"
        checks.append(_scheduler_check("launchd_runtime", False, "runtime outcome unknown; receipt is invalid"))
    checks.append(_scheduler_check("drive_authorization", None, "not checked by read-only scheduler diagnostics"))
    healthy = all(check["ok"] for check in checks if check["status"] != "not_checked")
    return {
        "healthy": healthy,
        "installed": installed,
        "loaded": loaded,
        "run_count": facts["run_count"],
        "last_exit_code": facts["last_exit_code"],
        "last_exit_label": facts["last_exit_label"],
        "state": facts["state"],
        "mode": configured_mode or "unknown",
        "runtime_outcome": runtime_outcome,
        "receipt_at": receipt_at,
        "checks": tuple(checks),
    }


def format_scheduler_report(report: dict[str, Any]) -> str:
    def value(item: Any) -> str:
        return "unknown" if item is None else str(item)

    lines = [
        "PHONE_FETCH_SCHEDULER "
        f"installed={str(report['installed']).lower()} loaded={str(report['loaded']).lower()} "
        f"healthy={str(report['healthy']).lower()} runs={value(report['run_count'])} "
        f"last_exit_code={value(report['last_exit_code'])} state={value(report['state'])} "
        f"last_exit_reason={value(report.get('last_exit_label'))} "
        f"mode={value(report.get('mode'))} "
        f"runtime_outcome={value(report['runtime_outcome'])} receipt_at={value(report['receipt_at'])}"
    ]
    for check in report["checks"]:
        lines.append(f"SCHEDULER_CHECK category={check['category']} status={check['status']} detail={check['detail']}")
    return "\n".join(lines)


def install_daily_schedule(
    hour: int,
    minute: int,
    *,
    agent_path: Path = DEFAULT_LAUNCH_AGENT_PATH,
    log_path: Path | None = None,
    error_log_path: Path | None = None,
    receipt_path: Path | None = None,
    mode: str = "sync",
    project_root: Path = PROJECT_ROOT,
    python_path: Path = Path(sys.executable),
    runtime_root: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    loaded_before: bool | None = None,
) -> Path:
    if mode not in SCHEDULER_MODES:
        raise ValueError("invalid scheduler mode")
    _reject_symlink(agent_path, "managed LaunchAgent")
    runtime_root = _resolve_runtime_root(
        runtime_root, log_path=log_path, error_log_path=error_log_path, receipt_path=receipt_path,
    )
    _ensure_runtime_root(runtime_root)
    effective_log = log_path or runtime_root / "phone_drive_fetch.log"
    effective_error_log = error_log_path or runtime_root / "phone_drive_fetch.error.log"
    effective_receipt = receipt_path or runtime_root / "phone_fetch_scheduler_receipt.json"
    for path, label in (
        (effective_log, "scheduler stdout log"),
        (effective_error_log, "scheduler stderr log"),
        (effective_receipt, "scheduler receipt"),
    ):
        if not _within_trusted_root(path, (runtime_root,)):
            raise ValueError(f"{label} must remain inside the scheduler runtime root")
        _reject_symlink(path, label)
        _reject_symlink(path.parent, f"{label} directory")
    source_wrapper = project_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name
    source_runner = project_root / DEFAULT_SCHEDULER_RUNNER_PATH.name
    runtime_wrapper = runtime_root / DEFAULT_SCHEDULER_WRAPPER_PATH.name
    runtime_runner = runtime_root / DEFAULT_SCHEDULER_RUNNER_PATH.name
    _atomic_copy_private(source_wrapper, runtime_wrapper, 0o700)
    _atomic_copy_private(source_runner, runtime_runner, 0o700)
    prior = agent_path.read_bytes() if agent_path.exists() else None
    effective_log.parent.mkdir(parents=True, exist_ok=True)
    _private_path(effective_log.parent, 0o700, directory=True)
    _private_path(effective_error_log.parent, 0o700, directory=True)
    _private_path(effective_receipt.parent, 0o700, directory=True)
    for path in (effective_log, effective_error_log):
        _private_path(path, 0o600)
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = agent_path.with_name(f".{agent_path.name}.tmp")
    _reject_symlink(temporary, "managed LaunchAgent temporary file")
    try:
        installed_at = now_ist().isoformat()
        temporary.write_bytes(build_daily_launch_agent(
            hour,
            minute,
            log_path=effective_log,
            error_log_path=effective_error_log,
            receipt_path=effective_receipt,
            mode=mode,
            project_root=project_root,
            python_path=python_path,
            runtime_root=runtime_root,
            wrapper_path=runtime_wrapper,
            runner_path=runtime_runner,
            installed_at=installed_at,
        ))
        temporary.chmod(0o600)
        os.replace(temporary, agent_path)
        agent_path.chmod(0o600)
        domain = f"gui/{os.getuid()}"
        runner(["launchctl", "bootout", domain, str(agent_path)], check=False, capture_output=True, text=True)
        runner(["launchctl", "bootstrap", domain, str(agent_path)], check=True, capture_output=True, text=True)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if prior is None:
            if agent_path.exists() and not agent_path.is_symlink():
                agent_path.unlink()
        else:
            agent_path.write_bytes(prior)
            agent_path.chmod(0o600)
            if loaded_before:
                try:
                    runner(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(agent_path)], check=True, capture_output=True, text=True)
                except Exception:
                    pass
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return agent_path


def run_scheduler_probe(
    *,
    agent_path: Path = DEFAULT_LAUNCH_AGENT_PATH,
    log_path: Path | None = None,
    error_log_path: Path | None = None,
    receipt_path: Path | None = None,
    runtime_root: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Load the same managed job in no-Drive probe mode and kick it once.

    This deliberately uses the existing single LaunchAgent label.  The
    operator restores normal fetch mode with ``scheduler install`` after the
    probe receipt is verified.
    """
    install_daily_schedule(
        7,
        0,
        agent_path=agent_path,
        log_path=log_path,
        error_log_path=error_log_path,
        receipt_path=receipt_path,
        mode="probe",
        runtime_root=runtime_root,
        runner=runner,
    )
    result = runner(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{SCHEDULER_LABEL}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.returncode


def remove_daily_schedule(
    *,
    agent_path: Path = DEFAULT_LAUNCH_AGENT_PATH,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    domain = f"gui/{os.getuid()}"
    runner(["launchctl", "bootout", domain, str(agent_path)], check=False, capture_output=True, text=True)
    if agent_path.exists() or agent_path.is_symlink():
        # Unlinking a symlink at the exact managed path does not follow it and
        # permits explicit cleanup of a malformed managed plist safely.
        agent_path.unlink()


def scheduler_status(
    *,
    agent_path: Path = DEFAULT_LAUNCH_AGENT_PATH,
    receipt_path: Path | None = None,
    runtime_root: Path | None = None,
    trusted_roots: Sequence[Path] = TRUSTED_INTERPRETER_ROOTS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    snapshot = _launchd_snapshot(agent_path, runner=runner)
    return scheduler_diagnose(
        agent_path=agent_path,
        runtime_root=runtime_root,
        receipt_path=receipt_path,
        trusted_roots=trusted_roots,
        launchd_snapshot=snapshot,
        runner=runner,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = fetch_recordings(FetchConfig(args.remote_root, args.destination_root, args.ledger_path, args.dry_run))
    except Exception as exc:
        print(f"FAILED fetch_setup error={type(exc).__name__}: {exc}")
        return 1
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
