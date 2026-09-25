#!/usr/bin/env python3
"""Private, bounded runner for the managed phone-fetch LaunchAgent.

The shell wrapper proves that launchd reached the wrapper.  This runner proves
that the configured interpreter started, then invokes the exact MeetingIntel
CLI command without a shell or profile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 2


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_receipt(
    path: Path,
    *,
    stage: str,
    mode: str,
    outcome: str,
    category: str,
    exit_code: int | None,
    clock: Callable[[], str] = _timestamp,
) -> None:
    if not path.is_absolute() or path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("unsafe scheduler receipt path")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "mode": mode,
        "updated_at": clock(),
        "outcome": outcome,
        "category": category,
        "exit_code": exit_code,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.chmod(0o600)
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_command(python_path: str, cli_path: str, mode: str) -> list[str]:
    if mode == "sync":
        args = ["sync"]
    elif mode == "probe":
        args = ["sync", "scheduler", "probe-command"]
    else:
        raise ValueError("invalid scheduler mode")
    if not all(Path(value).is_absolute() for value in (python_path, cli_path)):
        raise ValueError("scheduler runner requires absolute command paths")
    return [python_path, cli_path, *args]


def run(
    argv: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], str] = _timestamp,
) -> int:
    if len(argv) != 4:
        return 78
    python_path, cli_path, receipt_value, mode = argv
    receipt = Path(receipt_value)
    try:
        _write_receipt(
            receipt,
            stage="python_started",
            mode=mode,
            outcome="unknown",
            category="python_started",
            exit_code=None,
            clock=clock,
        )
        command = build_command(python_path, cli_path, mode)
        result = runner(
            command,
            check=False,
            # The LaunchAgent supplies the private runtime root as cwd.  Keep
            # the runner from re-entering the protected project tree while the
            # absolute CLI path remains explicit and importable.
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        success = result.returncode == 0
        category = (
            "probe_success" if mode == "probe" and success
            else "success" if success
            else "project_cli_failure" if mode == "probe"
            else "meetingintel_failure"
        )
        _write_receipt(
            receipt,
            stage="completed",
            mode=mode,
            outcome="success" if success else "failure",
            category=category,
            exit_code=result.returncode,
            clock=clock,
        )
        return result.returncode
    except FileNotFoundError:
        # A missing interpreter is normally caught by the POSIX wrapper as
        # exit 127.  If Python did start but the absolute CLI path cannot be
        # opened, preserve that boundary in the safe receipt.
        category = "python_exec_failed" if not Path(python_path).exists() else "project_cli_failure"
        try:
            _write_receipt(
                receipt,
                stage="completed",
                mode=mode,
                outcome="failure",
                category=category,
                exit_code=127,
                clock=clock,
            )
        except Exception:
            pass
        return 127
    except Exception:
        # The shell wrapper has already established wrapper_started.  A Python
        # process that cannot complete its own bookkeeping is a command-stage
        # failure, but never exposes the exception or command contents.
        try:
            _write_receipt(
                receipt,
                stage="completed",
                mode=mode,
                outcome="failure",
                category="meetingintel_failure",
                exit_code=1,
                clock=clock,
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
