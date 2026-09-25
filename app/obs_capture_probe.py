"""Isolated OBS installation and control preflight for laptop-capture testing.

This module does not admit OBS output to MeetingIntel. It prepares only the
local authenticated control boundary needed by the synthetic recorder gate.
"""

from __future__ import annotations
from mi_paths import public_path

import configparser
import os
import plistlib
import secrets
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_OBS_APP = Path(str(public_path('home/Applications/OBS.app')))
DEFAULT_OBS_CONFIG_ROOT = Path(str(public_path('home/Library/Application Support/obs-studio')))
DEFAULT_PRIVATE_ROOT = Path(
    str(public_path('home/Library/Application Support/MeetingIntel/laptop-capture'))
)
DEFAULT_WEBSOCKET_PORT = 4455
DEFAULT_CONTROL_SCRIPT = Path(__file__).parent / "scripts" / "obs_websocket_control.mjs"


@dataclass(frozen=True)
class ObsInstallStatus:
    installed: bool
    version: str | None
    architecture: str | None


@dataclass(frozen=True)
class ObsPermissionStatus:
    microphone: str
    screen_capture: str
    video_device: str
    input_monitoring: str


def _atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, mode)
    temporary.replace(path)


def inspect_obs_install(app_path: Path = DEFAULT_OBS_APP) -> ObsInstallStatus:
    executable = app_path / "Contents" / "MacOS" / "OBS"
    info_path = app_path / "Contents" / "Info.plist"
    if not executable.is_file() or not info_path.is_file():
        return ObsInstallStatus(False, None, None)
    try:
        with info_path.open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException):
        return ObsInstallStatus(False, None, None)
    version = info.get("CFBundleShortVersionString")
    architecture = "arm64" if executable.read_bytes()[:4] in {
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
    } else "unknown"
    return ObsInstallStatus(
        True,
        version if isinstance(version, str) else None,
        architecture,
    )


def latest_obs_log(config_root: Path = DEFAULT_OBS_CONFIG_ROOT) -> Path | None:
    logs_root = config_root / "logs"
    if not logs_root.is_dir():
        return None
    candidates = sorted(
        (path for path in logs_root.iterdir() if path.is_file() and path.suffix == ".txt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    return candidates[-1] if candidates else None


def permission_status_from_log(log_path: Path | None) -> ObsPermissionStatus:
    text = ""
    if log_path is not None:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""

    def state(denied_phrase: str, granted_phrase: str) -> str:
        if denied_phrase in text:
            return "denied"
        if granted_phrase in text:
            return "granted"
        return "unknown"

    return ObsPermissionStatus(
        microphone=state(
            "Permission for audio device access denied",
            "Permission for audio device access granted",
        ),
        screen_capture=state(
            "Permission for screen capture denied",
            "Permission for screen capture granted",
        ),
        video_device=state(
            "Permission for video device access denied",
            "Permission for video device access granted",
        ),
        input_monitoring=state(
            "Permission for input monitoring denied",
            "Permission for input monitoring granted",
        ),
    )


def websocket_reachable(
    host: str = "127.0.0.1",
    port: int = DEFAULT_WEBSOCKET_PORT,
    timeout_seconds: float = 0.5,
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def configure_authenticated_websocket(
    *,
    global_ini: Path = DEFAULT_OBS_CONFIG_ROOT / "global.ini",
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    password_factory: Callable[[int], str] = secrets.token_urlsafe,
    port: int = DEFAULT_WEBSOCKET_PORT,
) -> Path:
    """Enable password-protected local OBS control with an atomic backup.

    OBS must be stopped by the caller. The returned private password file is
    user-only and must never be committed or printed.
    """

    if port < 1024 or port > 65535:
        raise ValueError("websocket port must be between 1024 and 65535")
    if not global_ini.is_file():
        raise FileNotFoundError(f"OBS global config not found: {global_ini}")

    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)
    password_path = private_root / "obs-websocket-password"
    if password_path.exists():
        password = password_path.read_text(encoding="utf-8").strip()
        if not password:
            raise ValueError("existing OBS WebSocket password file is empty")
    else:
        password = password_factory(32)
        if not isinstance(password, str) or len(password) < 24:
            raise ValueError("generated OBS WebSocket password is too short")
        _atomic_write(password_path, password + "\n")

    backup = global_ini.with_name("global.ini.before-meetingintel-capture")
    if not backup.exists():
        shutil.copy2(global_ini, backup)
        os.chmod(backup, 0o600)

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with global_ini.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if not parser.has_section("OBSWebSocket"):
        parser.add_section("OBSWebSocket")
    parser.set("OBSWebSocket", "ServerEnabled", "true")
    parser.set("OBSWebSocket", "ServerPort", str(port))
    parser.set("OBSWebSocket", "AuthRequired", "true")
    parser.set("OBSWebSocket", "ServerPassword", password)
    parser.set("OBSWebSocket", "AlertsEnabled", "false")

    from io import StringIO

    output = StringIO()
    parser.write(output, space_around_delimiters=False)
    _atomic_write(global_ini, output.getvalue())
    return password_path


def set_websocket_enabled(
    enabled: bool,
    *,
    global_ini: Path = DEFAULT_OBS_CONFIG_ROOT / "global.ini",
) -> None:
    """Atomically enable or disable the already-configured control server."""

    if not global_ini.is_file():
        raise FileNotFoundError(f"OBS global config not found: {global_ini}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with global_ini.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if not parser.has_section("OBSWebSocket"):
        raise ValueError("OBS WebSocket is not configured")
    parser.set("OBSWebSocket", "ServerEnabled", "true" if enabled else "false")

    from io import StringIO

    output = StringIO()
    parser.write(output, space_around_delimiters=False)
    _atomic_write(global_ini, output.getvalue())


def run_obs_request(
    request_type: str,
    request_data: str = "{}",
    *,
    password_path: Path = DEFAULT_PRIVATE_ROOT / "obs-websocket-password",
    control_script: Path = DEFAULT_CONTROL_SCRIPT,
    node_executable: str = "node",
    timeout_seconds: float = 10.0,
) -> str:
    """Run one authenticated local request without placing the secret in argv."""

    if not request_type or any(character.isspace() for character in request_type):
        raise ValueError("request type must be one non-empty token")
    password = password_path.read_text(encoding="utf-8").strip()
    if not password:
        raise ValueError("OBS WebSocket password file is empty")
    environment = os.environ.copy()
    environment["MI_OBS_WS_PASSWORD"] = password
    completed = subprocess.run(
        [node_executable, str(control_script), request_type, request_data],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
    )
    return completed.stdout.strip()
