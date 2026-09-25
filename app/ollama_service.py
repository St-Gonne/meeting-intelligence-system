"""Bounded, user-scoped Ollama service management for MeetingIntel.

This module owns only the MeetingIntel LaunchAgent configuration. It never
creates, pulls, removes, or generates with an Ollama model. All operating-system
and network seams are injectable so tests never touch a user's LaunchAgents,
ports, logs, or process table.
"""

from __future__ import annotations

import os
import json
import plistlib
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from ollama_endpoint import (
    OLLAMA_APPROVED_DIGEST,
    OLLAMA_MODEL,
    OllamaEndpointError,
    OllamaModelDigestMismatch,
    OllamaModelMissing,
    OllamaUnreachable,
    preflight_ollama,
    validate_ollama_endpoint,
)


SERVICE_LABEL = "com.meetingintel.ollama"
MANAGED_HOST = "127.0.0.1"
MANAGED_PORT = 11435
MANAGED_HOST_VALUE = f"{MANAGED_HOST}:{MANAGED_PORT}"
MANAGED_ENDPOINT = f"http://{MANAGED_HOST_VALUE}/api/generate"
RESTART_THROTTLE_SECONDS = 10
RECOVERY_WINDOW_SECONDS = 20


class OllamaServiceError(RuntimeError):
    """Base class for safe service lifecycle failures."""


class ServiceNotInstalled(OllamaServiceError):
    """The MeetingIntel LaunchAgent is not installed."""


class ServiceConflict(OllamaServiceError):
    """Another listener or Ollama owner conflicts with the managed service."""


class ServiceConfigInvalid(OllamaServiceError):
    """The managed plist is missing, malformed, or does not match policy."""


class ServiceRecoveryFailed(OllamaServiceError):
    """The bounded managed-service recovery did not restore health."""


@dataclass(frozen=True)
class LaunchctlSnapshot:
    loaded: bool
    pid: int | None


@dataclass(frozen=True)
class ServicePaths:
    home: Path

    @property
    def launch_agents(self) -> Path:
        return self.home / "Library" / "LaunchAgents"

    @property
    def plist(self) -> Path:
        return self.launch_agents / f"{SERVICE_LABEL}.plist"

    @property
    def log_root(self) -> Path:
        return self.home / "Library" / "Logs" / "MeetingIntel"

    @property
    def stdout_log(self) -> Path:
        return self.log_root / "ollama.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.log_root / "ollama.stderr.log"


@dataclass(frozen=True)
class ListenerInfo:
    host: str
    port: int
    owner: str
    pid: int | None = None
    command: str = ""

    @property
    def wildcard(self) -> bool:
        return self.host in {"0.0.0.0", "::", "*", "[::]"}

    @property
    def exact_loopback(self) -> bool:
        return self.host == MANAGED_HOST and self.port == MANAGED_PORT


@dataclass(frozen=True)
class ServiceStatus:
    configured_endpoint: str
    expected_listener: str
    installed: bool
    loaded: bool
    port_responds: bool
    ollama_version: str | None
    alias_present: bool
    alias_digest_matches: bool
    model_loaded: bool
    listener_scope: str
    listener_conflict: bool
    default_port_conflict: bool
    managed_pid_owns_listener: bool
    managed_pid: int | None
    ollama_app_conflict: bool
    managed_settings_valid: bool
    managed_endpoint_matches: bool
    healthy: bool
    warnings: tuple[str, ...]


def default_service_paths() -> ServicePaths:
    return ServicePaths(Path.home())


def default_ollama_binary() -> str:
    return shutil.which("ollama") or "/opt/homebrew/bin/ollama"


def render_launch_agent(
    *,
    ollama_binary: str,
    paths: ServicePaths,
) -> dict[str, Any]:
    """Render the only supported server owner and its exact loopback policy."""
    if not Path(ollama_binary).is_absolute():
        raise ServiceConfigInvalid("Ollama CLI path must be absolute")
    return {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [ollama_binary, "serve"],
        "EnvironmentVariables": {
            "OLLAMA_HOST": MANAGED_HOST_VALUE,
            "OLLAMA_MAX_LOADED_MODELS": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": RESTART_THROTTLE_SECONDS,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.stdout_log),
        "StandardErrorPath": str(paths.stderr_log),
    }


def _parse_endpoint_name(name: str) -> tuple[str, int] | None:
    value = name.strip()
    if value.startswith("["):
        closing = value.find("]:")
        if closing < 0:
            return None
        host = value[1:closing]
        port_text = value[closing + 2 :]
    else:
        if ":" not in value:
            return None
        host, port_text = value.rsplit(":", 1)
    if not port_text.isdigit():
        return None
    port = int(port_text)
    if not 1 <= port <= 65535:
        return None
    return host, port


def parse_lsof_f_output(output: str) -> tuple[ListenerInfo, ...]:
    """Parse sanitized ``lsof -FpcfnPT`` listener records deterministically."""
    entries: list[ListenerInfo] = []
    pid: int | None = None
    command = ""
    listening = False
    name: str | None = None

    def emit_if_listener() -> None:
        nonlocal name
        if not listening or name is None or pid is None:
            return
        endpoint = _parse_endpoint_name(name)
        if endpoint is None:
            name = None
            return
        host, port = endpoint
        entries.append(ListenerInfo(host, port, command, pid, command))
        name = None

    for raw_line in output.splitlines():
        if not raw_line:
            continue
        field, value = raw_line[0], raw_line[1:]
        if field == "p":
            emit_if_listener()
            try:
                pid = int(value)
            except ValueError:
                pid = None
            listening = False
            name = None
        elif field == "c":
            command = value
        elif field == "f":
            emit_if_listener()
            listening = False
            name = None
        elif field == "T":
            listening = value == "ST=LISTEN" or value.endswith("=LISTEN")
            emit_if_listener()
        elif field == "n":
            emit_if_listener()
            name = value
    emit_if_listener()
    return tuple(entries)


class LaunchctlAdapter:
    """Small injectable wrapper around the user-scoped launchctl commands."""

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None, uid: int | None = None):
        self.runner = runner or subprocess.run
        self.uid = os.getuid() if uid is None else uid
        self.domain = f"gui/{self.uid}"

    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return self.runner(
            ["launchctl", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def is_loaded(self, label: str = SERVICE_LABEL) -> bool:
        return self.snapshot(label).loaded

    def snapshot(self, label: str = SERVICE_LABEL) -> LaunchctlSnapshot:
        result = self._run(["print", f"{self.domain}/{label}"])
        if result.returncode != 0:
            return LaunchctlSnapshot(False, None)
        match = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", result.stdout or "")
        return LaunchctlSnapshot(True, int(match.group(1)) if match else None)

    def bootstrap(self, plist_path: Path) -> None:
        result = self._run(["bootstrap", self.domain, str(plist_path)])
        if result.returncode != 0:
            raise OllamaServiceError("launchctl could not load the managed service")

    def bootout(self, label: str = SERVICE_LABEL) -> None:
        result = self._run(["bootout", f"{self.domain}/{label}"])
        if result.returncode != 0:
            raise OllamaServiceError("launchctl could not unload the managed service")

    def kickstart(self, label: str = SERVICE_LABEL, *, kill: bool = False) -> None:
        arguments = ["kickstart"]
        if kill:
            arguments.append("-k")
        arguments.append(f"{self.domain}/{label}")
        result = self._run(arguments)
        if result.returncode != 0:
            raise OllamaServiceError("launchctl could not restart the managed service")


class ProcessInspector:
    """Read-only listener inspection; output is never exposed to operators."""

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None):
        self.runner = runner or subprocess.run

    def listeners(self, port: int = MANAGED_PORT) -> tuple[ListenerInfo, ...]:
        return self.snapshot((port,))

    def snapshot(self, ports: Sequence[int] = (MANAGED_PORT, 11434)) -> tuple[ListenerInfo, ...]:
        arguments = ["lsof", "-nP", "-FpcfnPT"]
        arguments.extend(f"-iTCP:{port}" for port in ports)
        arguments.extend(["-sTCP:LISTEN"])
        result = self.runner(
            arguments,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            return ()
        return parse_lsof_f_output(result.stdout or "")


def _default_http_opener(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _api_endpoint(endpoint: str, path: str) -> str:
    parsed = urlsplit(validate_ollama_endpoint(endpoint))
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _get_json(
    endpoint: str,
    path: str,
    *,
    opener: Callable[..., Any],
    timeout: float,
) -> tuple[bool, int | None, Mapping[str, Any] | None]:
    request = urllib.request.Request(_api_endpoint(endpoint, path), method="GET")
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, None, None
    try:
        decoded = json.loads(payload)
    except (ValueError, TypeError):
        return True, status, None
    return True, status, decoded if isinstance(decoded, Mapping) else None


def _read_managed_plist(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = plistlib.loads(path.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _managed_settings(payload: Mapping[str, Any] | None, *, ollama_binary: str, paths: ServicePaths) -> tuple[bool, bool]:
    if payload is None:
        return False, False
    expected = render_launch_agent(ollama_binary=ollama_binary, paths=paths)
    return (
        payload.get("Label") == SERVICE_LABEL
        and payload.get("ProgramArguments") == expected["ProgramArguments"],
        payload.get("EnvironmentVariables") == expected["EnvironmentVariables"],
    )


def _listener_is_managed(listener: ListenerInfo, managed_pid: int | None) -> bool:
    """Only launchd's exact service PID can establish managed ownership."""
    return (
        managed_pid is not None
        and listener.exact_loopback
        and listener.pid == managed_pid
    )


def _listener_is_ollama(listener: ListenerInfo) -> bool:
    owner = f"{listener.owner} {listener.command}".casefold()
    return "ollama" in owner


def _conflict_message(conflicts: Sequence[ListenerInfo]) -> str:
    ports = sorted({listener.port for listener in conflicts})
    if ports == [11434]:
        return (
            "another Ollama listener is using default port 11434; "
            "quit or disable Ollama.app background startup before using the "
            "MeetingIntel-managed port"
        )
    if MANAGED_PORT in ports:
        return (
            "another listener owns configured managed port 11435; "
            "quit or disable the other Ollama owner before using the managed service"
        )
    return "another Ollama listener conflicts with the managed service"


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ServiceConfigInvalid(f"managed log directory must not be a symlink: {path.name}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _ensure_private_file(path: Path) -> None:
    if path.is_symlink():
        raise ServiceConfigInvalid(f"managed log file must not be a symlink: {path.name}")
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    path.chmod(0o600)


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{SERVICE_LABEL}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if path.exists() and not path.is_symlink():
            path.chmod(0o600)


class OllamaServiceManager:
    def __init__(
        self,
        *,
        paths: ServicePaths | None = None,
        ollama_binary: str | None = None,
        launchctl: Any | None = None,
        process_inspector: Any | None = None,
        opener: Callable[..., Any] | None = None,
        version_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.paths = paths or default_service_paths()
        self.ollama_binary = ollama_binary or default_ollama_binary()
        self.launchctl = launchctl or LaunchctlAdapter()
        self.process_inspector = process_inspector or ProcessInspector()
        self.opener = opener or _default_http_opener
        self.version_runner = version_runner or subprocess.run
        self.environ = os.environ if environ is None else environ

    def _payload(self) -> dict[str, Any]:
        return render_launch_agent(ollama_binary=self.ollama_binary, paths=self.paths)

    def _validate_binary(self) -> None:
        if not Path(self.ollama_binary).is_file():
            raise ServiceConfigInvalid("installed Ollama CLI was not found")

    def _validate_installed(self) -> Mapping[str, Any]:
        payload = _read_managed_plist(self.paths.plist)
        config_ok, settings_ok = _managed_settings(
            payload, ollama_binary=self.ollama_binary, paths=self.paths
        )
        if not config_ok or not settings_ok:
            raise ServiceConfigInvalid("managed LaunchAgent configuration is missing or invalid")
        return payload or {}

    def _launch_snapshot(self) -> LaunchctlSnapshot:
        snapshot = getattr(self.launchctl, "snapshot", None)
        if callable(snapshot):
            return snapshot(SERVICE_LABEL)
        return LaunchctlSnapshot(bool(self.launchctl.is_loaded(SERVICE_LABEL)), None)

    def _listener_snapshot(self) -> tuple[ListenerInfo, ...]:
        snapshot = getattr(self.process_inspector, "snapshot", None)
        if callable(snapshot):
            return tuple(snapshot((MANAGED_PORT, 11434)))
        return tuple(self.process_inspector.listeners(MANAGED_PORT)) + tuple(
            self.process_inspector.listeners(11434)
        )

    @staticmethod
    def _conflicting_listeners(
        listeners: Sequence[ListenerInfo], managed_pid: int | None
    ) -> tuple[ListenerInfo, ...]:
        conflicts: list[ListenerInfo] = []
        for listener in listeners:
            if listener.port == MANAGED_PORT and not _listener_is_managed(listener, managed_pid):
                conflicts.append(listener)
            elif listener.port == 11434 and _listener_is_ollama(listener) and listener.host != "127.0.0.1":
                conflicts.append(listener)
        return tuple(conflicts)

    def install(self) -> None:
        self._validate_binary()
        launch_before = self._launch_snapshot()
        listeners = self._listener_snapshot()
        conflicts = self._conflicting_listeners(listeners, launch_before.pid)
        if conflicts:
            raise ServiceConflict(_conflict_message(conflicts))
        self.paths.launch_agents.mkdir(parents=True, exist_ok=True)
        _ensure_private_directory(self.paths.log_root)
        _ensure_private_file(self.paths.stdout_log)
        _ensure_private_file(self.paths.stderr_log)
        data = plistlib.dumps(self._payload(), fmt=plistlib.FMT_XML, sort_keys=False)
        if self.paths.plist.is_symlink():
            raise ServiceConfigInvalid("managed plist must not be a symlink")
        if self.paths.plist.exists() and not self.paths.plist.is_file():
            raise ServiceConfigInvalid("managed plist path is not a regular file")
        existing = self.paths.plist.read_bytes() if self.paths.plist.is_file() else None
        if self.paths.plist.is_file():
            self.paths.plist.chmod(0o600)
        changed = existing != data
        try:
            if changed:
                if launch_before.loaded:
                    self.launchctl.bootout(SERVICE_LABEL)
                _atomic_replace_bytes(self.paths.plist, data)
            if changed or not launch_before.loaded:
                self.launchctl.bootstrap(self.paths.plist)
        except Exception:
            try:
                if changed:
                    if existing is None:
                        if self.paths.plist.exists() and not self.paths.plist.is_symlink():
                            self.paths.plist.unlink()
                    else:
                        _atomic_replace_bytes(self.paths.plist, existing)
                current = self._launch_snapshot()
                if current.loaded and not launch_before.loaded:
                    self.launchctl.bootout(SERVICE_LABEL)
                elif launch_before.loaded and not current.loaded:
                    self.launchctl.bootstrap(self.paths.plist)
            except Exception:
                pass
            raise

    def start(self) -> None:
        self._validate_binary()
        self._validate_installed()
        launch = self._launch_snapshot()
        conflicts = self._conflicting_listeners(self._listener_snapshot(), launch.pid)
        if conflicts:
            raise ServiceConflict(_conflict_message(conflicts))
        if not launch.loaded:
            self.launchctl.bootstrap(self.paths.plist)

    def restart(self) -> None:
        self._validate_binary()
        self._validate_installed()
        launch = self._launch_snapshot()
        conflicts = self._conflicting_listeners(self._listener_snapshot(), launch.pid)
        if conflicts:
            raise ServiceConflict(_conflict_message(conflicts))
        self.launchctl.kickstart(SERVICE_LABEL, kill=True)

    def stop(self) -> None:
        self._validate_installed()
        if self._launch_snapshot().loaded:
            self.launchctl.bootout(SERVICE_LABEL)

    def remove(self) -> None:
        if self.paths.plist.is_symlink():
            raise ServiceConfigInvalid("managed plist must not be a symlink")
        if not self.paths.plist.exists():
            raise ServiceNotInstalled("managed LaunchAgent plist is not present")
        if self._launch_snapshot().loaded:
            self.launchctl.bootout(SERVICE_LABEL)
        if not self.paths.plist.is_file():
            raise ServiceConfigInvalid("managed plist path is not a regular file")
        self.paths.plist.unlink()

    def status(self, endpoint: str = MANAGED_ENDPOINT) -> ServiceStatus:
        try:
            endpoint = validate_ollama_endpoint(endpoint)
            endpoint_valid = True
        except OllamaEndpointError:
            endpoint_valid = False
        payload = _read_managed_plist(self.paths.plist)
        config_ok, settings_ok = _managed_settings(
            payload, ollama_binary=self.ollama_binary, paths=self.paths
        )
        installed = config_ok
        launch = self._launch_snapshot()
        loaded = launch.loaded
        listeners = self._listener_snapshot()
        managed_listeners = tuple(
            listener for listener in listeners if listener.port == MANAGED_PORT
        )
        default_listeners = tuple(
            listener for listener in listeners if listener.port == 11434
        )
        wildcard = any(listener.wildcard for listener in managed_listeners)
        exact = any(listener.exact_loopback for listener in managed_listeners)
        managed_pid_owns_listener = any(
            _listener_is_managed(listener, launch.pid) for listener in managed_listeners
        )
        conflicts = self._conflicting_listeners(listeners, launch.pid)
        default_port_conflict = any(_listener_is_ollama(listener) and listener.host != "127.0.0.1" for listener in default_listeners)
        app_conflict = any(
            "ollama.app" in f"{listener.owner} {listener.command}".casefold()
            and (listener.port != 11434 or listener.host != "127.0.0.1")
            for listener in listeners
        )
        if wildcard:
            listener_scope = "wildcard"
        elif exact:
            listener_scope = "exact-loopback"
        else:
            listener_scope = "none"

        port_responds = False
        alias_present = False
        digest_matches = False
        if endpoint_valid:
            try:
                result = preflight_ollama(MANAGED_ENDPOINT, opener=self.opener)
            except OllamaModelMissing:
                port_responds = True
                alias_present = False
            except OllamaModelDigestMismatch:
                port_responds = True
                alias_present = True
            except OllamaUnreachable:
                pass
            else:
                port_responds = True
                alias_present = bool(result.get("model_available"))
                digest_matches = bool(result.get("model_digest_matches"))

        ps_ok, _ps_status, ps_payload = (
            _get_json(MANAGED_ENDPOINT, "/api/ps", opener=self.opener, timeout=3.0)
            if endpoint_valid and port_responds
            else (False, None, None)
        )
        model_loaded = bool(ps_ok and isinstance(ps_payload, Mapping) and ps_payload.get("models"))
        warnings: list[str] = []
        if self.environ.get("OLLAMA_KEEP_ALIVE") == "-1":
            warnings.append("terminal OLLAMA_KEEP_ALIVE=-1 is configured")
        if self.environ.get("OLLAMA_HOST") != MANAGED_HOST_VALUE:
            if self.environ.get("OLLAMA_HOST"):
                warnings.append("terminal OLLAMA_HOST differs from the managed endpoint")
            else:
                warnings.append("terminal OLLAMA_HOST is not set to the managed endpoint")
        if app_conflict:
            warnings.append("Ollama.app or another GUI owner conflicts; quit or disable its background startup")
        if wildcard:
            warnings.append("wildcard Ollama listener is unsafe; require exact loopback binding")
        if conflicts and not app_conflict:
            if any(listener.port == MANAGED_PORT for listener in conflicts):
                warnings.append("another listener owns configured managed port 11435")
        if default_port_conflict:
            warnings.append("another Ollama listener competes on default port 11434")
        if loaded and not managed_pid_owns_listener:
            warnings.append("loaded managed service PID does not own exact loopback port 11435")
        managed_endpoint = None
        if payload is not None:
            environment = payload.get("EnvironmentVariables")
            if isinstance(environment, Mapping) and isinstance(environment.get("OLLAMA_HOST"), str):
                managed_endpoint = f"http://{environment['OLLAMA_HOST']}/api/generate"
        endpoint_matches = managed_endpoint == endpoint == MANAGED_ENDPOINT
        healthy = all(
            (
                endpoint_valid,
                installed,
                loaded,
                settings_ok,
                endpoint_matches,
                port_responds,
                alias_present,
                digest_matches,
                exact,
                not wildcard,
                not conflicts,
                not app_conflict,
                not default_port_conflict,
                managed_pid_owns_listener,
            )
        )
        return ServiceStatus(
            configured_endpoint=endpoint,
            expected_listener=f"{MANAGED_HOST}:{MANAGED_PORT}",
            installed=installed,
            loaded=loaded,
            port_responds=port_responds,
            ollama_version=self._version(),
            alias_present=alias_present,
            alias_digest_matches=digest_matches,
            model_loaded=model_loaded,
            listener_scope=listener_scope,
            listener_conflict=any(listener.port == MANAGED_PORT for listener in conflicts),
            default_port_conflict=default_port_conflict,
            managed_pid_owns_listener=managed_pid_owns_listener,
            managed_pid=launch.pid,
            ollama_app_conflict=app_conflict,
            managed_settings_valid=config_ok and settings_ok,
            managed_endpoint_matches=endpoint_matches,
            healthy=healthy,
            warnings=tuple(warnings),
        )

    def _version(self) -> str | None:
        try:
            result = self.version_runner(
                [self.ollama_binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        first_line = (result.stdout or "").splitlines()[0:1]
        return first_line[0][:120] if first_line else None

    def recover(
        self,
        endpoint: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> bool:
        """Kick only the installed managed service and poll /api/tags once."""
        self._validate_binary()
        payload = self._validate_installed()
        environment = payload.get("EnvironmentVariables")
        stored_endpoint = None
        if isinstance(environment, Mapping) and isinstance(environment.get("OLLAMA_HOST"), str):
            stored_endpoint = f"http://{environment['OLLAMA_HOST']}/api/generate"
        if stored_endpoint != endpoint or endpoint != MANAGED_ENDPOINT:
            raise ServiceConfigInvalid("managed endpoint does not match the resolved MeetingIntel endpoint")
        launch_before = self._launch_snapshot()
        initial_listeners = self._listener_snapshot()
        initial_conflicts = self._conflicting_listeners(initial_listeners, launch_before.pid)
        if initial_conflicts:
            raise ServiceConflict(_conflict_message(initial_conflicts))
        if launch_before.loaded:
            self.launchctl.kickstart(SERVICE_LABEL, kill=True)
        else:
            self.launchctl.bootstrap(self.paths.plist)
        deadline = monotonic() + RECOVERY_WINDOW_SECONDS
        for attempt in range(21):
            if monotonic() > deadline:
                break
            launch = self._launch_snapshot()
            listeners = self._listener_snapshot()
            conflicts = self._conflicting_listeners(listeners, launch.pid)
            managed_listener = any(
                _listener_is_managed(listener, launch.pid)
                for listener in listeners
                if listener.port == MANAGED_PORT
            )
            wildcard = any(
                listener.wildcard
                for listener in listeners
                if listener.port == MANAGED_PORT
            )
            ownership_ok = (
                launch.loaded
                and launch.pid is not None
                and managed_listener
                and not conflicts
                and not wildcard
            )
            preflight_ok = False
            try:
                preflight_ollama(endpoint, opener=self.opener)
            except OllamaEndpointError:
                pass
            else:
                preflight_ok = True
            if ownership_ok and preflight_ok:
                return True
            if attempt < 20:
                sleep(1.0)
        raise ServiceRecoveryFailed("managed Ollama recovery did not pass endpoint, alias, and digest checks")


def format_status(status: ServiceStatus) -> str:
    values = (
        ("configured_generate_endpoint", status.configured_endpoint),
        ("expected_listener", status.expected_listener),
        ("managed_launchagent_installed", status.installed),
        ("managed_launchagent_loaded", status.loaded),
        ("expected_port_responds", status.port_responds),
        ("ollama_version", status.ollama_version or "unavailable"),
        ("meetingintel_alias_present", status.alias_present),
        ("approved_alias_digest_matches", status.alias_digest_matches),
        ("model_currently_loaded", status.model_loaded),
        ("listener_scope", status.listener_scope),
        ("listener_conflict", status.listener_conflict),
        ("default_port_11434_conflict", status.default_port_conflict),
        ("managed_pid_owns_listener", status.managed_pid_owns_listener),
        ("ollama_app_conflict", status.ollama_app_conflict),
        ("managed_settings_valid", status.managed_settings_valid),
        ("managed_endpoint_matches", status.managed_endpoint_matches),
        ("healthy", status.healthy),
    )
    lines = [f"{key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in values]
    lines.extend(f"warning: {warning}" for warning in status.warnings)
    return "\n".join(lines)
