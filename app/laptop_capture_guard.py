"""Foreground, fail-closed OBS guard for bounded laptop capture.

This is not a MeetingIntel production source or a general recorder UI. It
keeps one OBS recording awake, watches explicit recorder/source health,
preserves 15-minute MKV chunks, and writes a recorder-neutral manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from meetingintel_runtime import exclusive_operation
import meetingintel_observe as observe
import laptop_capture_contract as contract
import obs_capture_probe as obs_probe
import meetingintel_capture_alert as capture_alert


PROOF_ROOT = Path("/private/tmp")
DEFAULT_OBS_EXECUTABLE = (
    obs_probe.DEFAULT_OBS_APP / "Contents" / "MacOS" / "OBS"
)
DEFAULT_PROFILE = "MeetingIntel Capture Guard"
DEFAULT_SCENE = "MeetingIntel Capture Guard"
DEFAULT_SEGMENT_SECONDS = 15 * 60
DEFAULT_POLL_SECONDS = 2.0
OBS_START_TIMEOUT_SECONDS = 20.0
RECORD_START_TIMEOUT_SECONDS = 10.0
SOURCE_LOSS_PATTERNS = (
    "Stream stopped as no capture source",
    "Failed to create source 'macOS System Audio'",
    "init_audio_screen_stream: Failed to start capture",
)
PERMISSION_FAILURE_PATTERNS = (
    "Permission for screen capture denied",
    "Permission for audio device access denied",
)
OUTPUT_PATH_PATTERN = re.compile(
    r"(?:Writing (?:Hybrid MP4/MOV )?file|Changing output file to) '([^']+)'"
)
EXPECTED_PROFILE_PARAMETERS = (
    ("Output", "Mode", "Advanced"),
    ("AdvOut", "RecType", "Standard"),
    ("AdvOut", "RecFormat2", "mkv"),
    ("AdvOut", "RecSplitFile", "true"),
    ("AdvOut", "RecSplitFileType", "Time"),
)
OBS_SENTINEL_NAME_PATTERN = re.compile(
    r"run_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DEFAULT_OBS_SENTINEL_ROOT = obs_probe.DEFAULT_OBS_CONFIG_ROOT / ".sentinel"
DEFAULT_OBS_SENTINEL_ARCHIVE_ROOT = (
    obs_probe.DEFAULT_PRIVATE_ROOT / "sentinel-archive"
)


class GuardError(RuntimeError):
    """Expected, operator-visible capture-guard failure."""


class StopRequested(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


@dataclass(frozen=True)
class GuardConfig:
    session_root: Path
    allowed_root: Path = PROOF_ROOT
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    obs_executable: Path = DEFAULT_OBS_EXECUTABLE
    profile_name: str = DEFAULT_PROFILE
    scene_name: str = DEFAULT_SCENE


@dataclass(frozen=True)
class GuardLogEvent:
    type: str
    message: str
    output_path: Path | None = None


@dataclass
class GuardState:
    capture_id: str
    started_at: str
    output_paths: list[Path] = field(default_factory=list)
    source_events: list[dict[str, str]] = field(default_factory=list)
    last_output_bytes: int = 0
    seconds_since_output_growth: float = 0.0
    log_offset: int = 0
    stop_reason: str | None = None
    stop_signal: int | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    recording_ready: bool = False


@contextmanager
def capture_signal_scope(state: GuardState):
    """Handlers only record intent; ordinary code stops at safe boundaries.

    Leave these handlers installed through finalization so a second Control-C
    cannot interrupt checksum persistence or control/sleep-assertion cleanup.
    """
    previous = {}

    def requested(signum, _frame):
        if state.stop_signal is None or signum in (signal.SIGTERM, signal.SIGHUP):
            state.stop_signal = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, requested)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def check_stop_requested(state: GuardState) -> None:
    if state.stop_signal is not None:
        raise StopRequested(state.stop_signal)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_session_root(path: Path, allowed_root: Path = PROOF_ROOT) -> Path:
    """Restrict capture to a new directory below one explicit trusted root."""

    if path.is_symlink() or allowed_root.is_symlink():
        raise GuardError("session root must not be a symlink")
    try:
        resolved_allowed_root = allowed_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise GuardError("allowed recording root must already exist") from error
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise GuardError(
            f"session root must be under {resolved_allowed_root}"
        ) from error
    try:
        resolved_parent.relative_to(resolved_allowed_root)
    except ValueError as error:
        raise GuardError(
            f"session root must be under {resolved_allowed_root}"
        ) from error
    if path.exists() and any(path.iterdir()):
        raise GuardError("session root must be new or empty")
    return path


def validate_proof_session_root(path: Path) -> Path:
    """Backward-compatible proof-root validator."""

    return validate_session_root(path, PROOF_ROOT)


def build_keep_awake_command(obs_executable: Path) -> list[str]:
    if not obs_executable.is_file():
        raise GuardError(f"OBS executable is unavailable: {obs_executable}")
    obs_app = obs_executable.parents[2]
    if not (obs_app / "Contents" / "Info.plist").is_file():
        raise GuardError(f"OBS application bundle is unavailable: {obs_app}")
    return [
        "/usr/bin/caffeinate",
        "-dimsu",
        "/usr/bin/open",
        "-n",
        "-W",
        "-j",
        str(obs_app),
        "--args",
        "--websocket_ipv4_only",
        "--minimize-to-tray",
        "--disable-updater",
        "--disable-missing-files-check",
    ]


def obs_app_is_running(obs_executable: Path = DEFAULT_OBS_EXECUTABLE) -> bool:
    """Check for the exact OBS executable without relying on macOS process lists."""

    try:
        result = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                "-a",
                "-c",
                "OBS",
                "-d",
                "txt",
                "-Fpn",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GuardError("OBS process state could not be checked") from error
    if result.returncode not in (0, 1):
        raise GuardError("OBS process state could not be checked")
    executable_record = f"n{obs_executable}"
    return any(
        line == executable_record
        for line in result.stdout.splitlines()
    )


def archive_stale_obs_sentinels(
    *,
    sentinel_root: Path = DEFAULT_OBS_SENTINEL_ROOT,
    archive_root: Path = DEFAULT_OBS_SENTINEL_ARCHIVE_ROOT,
) -> tuple[Path, ...]:
    """Preserve empty OBS startup markers so they cannot block a guarded launch."""

    if sentinel_root.is_symlink():
        raise GuardError("OBS sentinel root must not be a symlink")
    if not sentinel_root.exists():
        return ()
    if not sentinel_root.is_dir():
        raise GuardError("OBS sentinel root is not a directory")
    stale = tuple(sorted(sentinel_root.iterdir(), key=lambda path: path.name))
    if not stale:
        return ()
    for path in stale:
        if (
            path.is_symlink()
            or not path.is_file()
            or OBS_SENTINEL_NAME_PATTERN.fullmatch(path.name) is None
            or path.stat().st_uid != os.getuid()
            or path.stat().st_size != 0
        ):
            raise GuardError("OBS sentinel state is unexpected; no markers were moved")

    if archive_root.is_symlink():
        raise GuardError("OBS sentinel archive root must not be a symlink")
    archive_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(archive_root, 0o700)
    batch = archive_root / f"stale-{uuid.uuid4()}"
    batch.mkdir(mode=0o700)
    archived: list[Path] = []
    try:
        for source in stale:
            target = batch / source.name
            source.replace(target)
            os.chmod(target, 0o600)
            archived.append(target)
    except OSError as error:
        for target in reversed(archived):
            try:
                target.replace(sentinel_root / target.name)
            except OSError:
                pass
        try:
            batch.rmdir()
        except OSError:
            pass
        raise GuardError("OBS stale startup markers could not be archived") from error
    return tuple(archived)


def prepare_obs_startup(config: GuardConfig) -> tuple[Path, ...]:
    """Require exclusive OBS ownership and recover only validated stale markers."""

    if obs_probe.websocket_reachable():
        raise GuardError("OBS control port is already in use")
    if obs_app_is_running(config.obs_executable):
        raise GuardError("OBS is already running; quit OBS and retry")
    return archive_stale_obs_sentinels()


def parse_obs_log_updates(text: str) -> tuple[GuardLogEvent, ...]:
    events: list[GuardLogEvent] = []
    for line in text.splitlines():
        output_match = OUTPUT_PATH_PATTERN.search(line)
        if output_match:
            events.append(
                GuardLogEvent(
                    "output_path",
                    line,
                    Path(output_match.group(1)),
                )
            )
        if any(pattern in line for pattern in SOURCE_LOSS_PATTERNS):
            events.append(GuardLogEvent("source_lost", line))
        if any(pattern in line for pattern in PERMISSION_FAILURE_PATTERNS):
            events.append(GuardLogEvent("permission_failed", line))
    return tuple(events)


def read_log_updates(log_path: Path, offset: int) -> tuple[str, int]:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            text = handle.read()
            return text, handle.tell()
    except OSError as error:
        raise GuardError(f"cannot read OBS log: {log_path}") from error


def battery_snapshot() -> tuple[bool, int | None]:
    try:
        result = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    text = result.stdout
    match = re.search(r"(\d+)%", text)
    percent = int(match.group(1)) if match else None
    return "AC Power" in text, percent


def request_json(
    request_type: str,
    request_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    raw = obs_probe.run_obs_request(
        request_type,
        json.dumps(request_data or {}, separators=(",", ":")),
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GuardError("OBS control returned invalid JSON") from error
    if not payload.get("ok"):
        raise GuardError(f"OBS request failed: {request_type}")
    response_data = payload.get("responseData", {})
    return dict(response_data) if isinstance(response_data, Mapping) else {}


def _set_profile_parameter(category: str, name: str, value: str) -> None:
    request_json(
        "SetProfileParameter",
        {
            "parameterCategory": category,
            "parameterName": name,
            "parameterValue": value,
        },
    )


def _get_profile_parameter(category: str, name: str) -> str | None:
    response = request_json(
        "GetProfileParameter",
        {
            "parameterCategory": category,
            "parameterName": name,
        },
    )
    value = response.get("parameterValue")
    return str(value) if value is not None else None


def configure_guard_profile(config: GuardConfig) -> None:
    profile_response = request_json("GetProfileList")
    profiles = profile_response.get("profiles", [])
    previous_profile = profile_response.get("currentProfileName")
    if config.profile_name not in profiles:
        request_json("CreateProfile", {"profileName": config.profile_name})
        profiles = [*profiles, config.profile_name]
    request_json("SetCurrentProfile", {"profileName": config.profile_name})

    segments_root = config.session_root / "segments"
    parameters = (
        ("Output", "Mode", "Advanced"),
        ("Output", "FilenameFormatting", "MI-GUARD-%CCYY-%MM-%DD_%hh-%mm-%ss"),
        ("AdvOut", "RecType", "Standard"),
        ("AdvOut", "RecFilePath", str(segments_root)),
        ("AdvOut", "RecFormat2", "mkv"),
        (
            "AdvOut",
            "RecEncoder",
            "com.apple.videotoolbox.videoencoder.ave.avc",
        ),
        ("AdvOut", "RecAudioEncoder", "CoreAudio_AAC"),
        ("AdvOut", "RecTracks", "1"),
        ("AdvOut", "RecSplitFile", "true"),
        ("AdvOut", "RecSplitFileType", "Time"),
        ("AdvOut", "RecSplitFileTime", str(max(1, config.segment_seconds // 60))),
    )
    for category, name, value in parameters:
        _set_profile_parameter(category, name, value)

    # OBS persists SetProfileParameter immediately but does not reliably rebuild
    # the active output handler. Reload the profile before StartRecord.
    reload_profile = next(
        (
            profile
            for profile in profiles
            if profile != config.profile_name
        ),
        None,
    )
    if previous_profile != config.profile_name:
        reload_profile = previous_profile or reload_profile
    if reload_profile is None:
        reload_profile = f"{config.profile_name} Reload"
        request_json("CreateProfile", {"profileName": reload_profile})
    request_json("SetCurrentProfile", {"profileName": reload_profile})
    request_json("SetCurrentProfile", {"profileName": config.profile_name})

    collections = request_json("GetSceneCollectionList").get(
        "sceneCollections", []
    )
    if config.scene_name not in collections:
        request_json(
            "CreateSceneCollection",
            {"sceneCollectionName": config.scene_name},
        )
    request_json(
        "SetCurrentSceneCollection",
        {"sceneCollectionName": config.scene_name},
    )

    scenes = request_json("GetSceneList").get("scenes", [])
    scene_names = {
        item.get("sceneName")
        for item in scenes
        if isinstance(item, Mapping)
    }
    if config.scene_name not in scene_names:
        request_json("CreateScene", {"sceneName": config.scene_name})

    inputs = request_json("GetInputList").get("inputs", [])
    input_names = {
        item.get("inputName")
        for item in inputs
        if isinstance(item, Mapping)
    }
    desired_inputs = (
        (
            "Capture Guard Background",
            "color_source_v3",
            {"color": 4278190335, "width": 1280, "height": 720},
        ),
        (
            "macOS System Audio",
            "sck_audio_capture",
            {"application": "", "type": 0},
        ),
        (
            "Capture Guard Microphone",
            "coreaudio_input_capture",
            {"device_id": "default", "enable_downmix": True},
        ),
    )
    for input_name, input_kind, settings in desired_inputs:
        if input_name not in input_names:
            request_json(
                "CreateInput",
                {
                    "sceneName": config.scene_name,
                    "inputName": input_name,
                    "inputKind": input_kind,
                    "inputSettings": settings,
                    "sceneItemEnabled": True,
                },
            )
    if "Mic/Aux" in input_names:
        request_json(
            "SetInputMute",
            {"inputName": "Mic/Aux", "inputMuted": True},
        )
    request_json(
        "SetCurrentProgramScene",
        {"sceneName": config.scene_name},
    )


def preflight(config: GuardConfig, log_path: Path) -> tuple[str, ...]:
    findings: list[str] = []
    install = obs_probe.inspect_obs_install(config.obs_executable.parents[2])
    if not install.installed:
        findings.append("obs_not_installed")
    permissions = obs_probe.permission_status_from_log(log_path)
    if permissions.screen_capture != "granted":
        findings.append("screen_capture_permission_not_granted")
    if permissions.microphone != "granted":
        findings.append("microphone_permission_not_granted")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        findings.append("obs_log_unreadable")
    else:
        log_event_types = {
            event.type for event in parse_obs_log_updates(log_text)
        }
        if "source_lost" in log_event_types:
            findings.append("source_not_reporting")
        if "permission_failed" in log_event_types:
            findings.append("capture_permission_failed")
    if not os.access(config.session_root / "segments", os.W_OK):
        findings.append("output_not_writable")
    disk_free = shutil.disk_usage(config.session_root).free
    if disk_free <= contract.CRITICAL_DISK_BYTES:
        findings.append("disk_critical")
    on_ac_power, battery_percent = battery_snapshot()
    if (
        not on_ac_power
        and battery_percent is not None
        and battery_percent <= contract.CRITICAL_BATTERY_PERCENT
    ):
        findings.append("battery_critical")
    record_directory = request_json("GetRecordDirectory").get("recordDirectory")
    if record_directory != str(config.session_root / "segments"):
        findings.append("output_path_mismatch")
    for category, name, expected in EXPECTED_PROFILE_PARAMETERS:
        actual = _get_profile_parameter(category, name)
        if actual is None or actual.lower() != expected.lower():
            findings.append(
                f"profile_parameter_mismatch:{category}/{name}"
            )
    split_minutes = _get_profile_parameter("AdvOut", "RecSplitFileTime")
    expected_minutes = str(max(1, config.segment_seconds // 60))
    if split_minutes != expected_minutes:
        findings.append("profile_parameter_mismatch:AdvOut/RecSplitFileTime")
    status = request_json("GetRecordStatus")
    if status.get("outputActive"):
        findings.append("recording_already_active")
    inputs = request_json("GetInputList").get("inputs", [])
    input_names = {
        item.get("inputName")
        for item in inputs
        if isinstance(item, Mapping)
    }
    for expected_input in ("macOS System Audio", "Capture Guard Microphone"):
        if expected_input not in input_names:
            findings.append(f"source_missing:{expected_input}")
    return tuple(findings)


def create_session(config: GuardConfig) -> GuardState:
    validate_session_root(config.session_root, config.allowed_root)
    config.session_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(config.session_root, 0o700)
    (config.session_root / "segments").mkdir(mode=0o700, exist_ok=True)
    os.chmod(config.session_root / "segments", 0o700)
    lock_path = config.session_root / ".recording.lock"
    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    state = GuardState(capture_id=str(uuid.uuid4()), started_at=utc_now())
    write_guard_manifest(config, state, "recording")
    return state


def _segment_entry(
    session_root: Path,
    path: Path,
    index: int,
    finalized: bool,
) -> dict[str, object]:
    if path.exists():
        os.chmod(path, 0o600)
    probe = contract.ffprobe_media(path)
    entry: dict[str, object] = {
        "index": index,
        "path": str(path.relative_to(session_root)),
        "state": "finalized" if finalized else "active",
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "duration_seconds": probe.duration_seconds,
    }
    if finalized and path.exists():
        entry["sha256"] = contract._sha256(path)
    return entry


def write_guard_manifest(
    config: GuardConfig,
    state: GuardState,
    status: str,
) -> None:
    finalized = status != "recording" and not state.cleanup_errors
    segments = [
        _segment_entry(config.session_root, path, index, finalized)
        for index, path in enumerate(state.output_paths)
        if path.exists()
    ]
    if status == "recording" and segments:
        for segment in segments[:-1]:
            segment["state"] = "finalized"
            segment["sha256"] = contract._sha256(
                config.session_root / str(segment["path"])
            )
    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "capture_id": state.capture_id,
        "source_kind": contract.SOURCE_KIND,
        "status": status,
        "capture_phase": ("recording" if state.recording_ready else "starting") if status == "recording" else "stopped",
        "started_at": state.started_at,
        "ended_at": utc_now() if status != "recording" else None,
        "segment_seconds": config.segment_seconds,
        "source_events": state.source_events,
        "stop_reason": state.stop_reason,
        "finalizing": False,
        "cleanup_errors": list(state.cleanup_errors),
        "segments": segments,
    }
    contract.write_manifest_atomically(
        config.session_root / "capture_manifest.json",
        manifest,
    )
    os.chmod(config.session_root / "capture_manifest.json", 0o600)


def write_interruption_checkpoint(config: GuardConfig, state: GuardState) -> None:
    """Publish failure before slow shutdown, without finalizing active media."""
    path = config.session_root / "capture_manifest.json"
    manifest = contract.load_manifest(path)
    if manifest.get("capture_id") != state.capture_id:
        raise GuardError("capture_manifest_identity_changed")
    manifest.update(status="interrupted", capture_phase="stopping", finalizing=True, stop_reason=state.stop_reason,
                    source_events=state.source_events, interrupted_at=utc_now(),
                    cleanup_errors=list(state.cleanup_errors))
    contract.write_manifest_atomically(path, manifest)
    os.chmod(path, 0o600)


def process_log_events(
    config: GuardConfig,
    state: GuardState,
    events: Iterable[GuardLogEvent],
) -> str | None:
    reason = None
    for event in events:
        if event.type == "output_path" and event.output_path is not None:
            try:
                event.output_path.resolve().relative_to(
                    (config.session_root / "segments").resolve()
                )
            except ValueError:
                reason = "output_path_escaped_session"
                continue
            if event.output_path not in state.output_paths:
                state.output_paths.append(event.output_path)
            if event.output_path.exists():
                os.chmod(event.output_path, 0o600)
        elif event.type == "source_lost":
            state.source_events.append({"type": "source_lost", "at": utc_now()})
            reason = reason or "source_lost"
        elif event.type == "permission_failed":
            reason = reason or "capture_permission_failed"
    return reason


def stop_recording_safely() -> bool:
    try:
        status = request_json("GetRecordStatus")
        if status.get("outputActive"):
            request_json("StopRecord")
        return True
    except (GuardError, subprocess.SubprocessError, OSError):
        return False


def resolve_controlled_obs_pid(
    obs_executable: Path,
    port: int = obs_probe.DEFAULT_WEBSOCKET_PORT,
) -> int | None:
    try:
        listeners = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                "-a",
                f"-iTCP:{port}",
                "-sTCP:LISTEN",
                "-Fp",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = {
        int(line[1:])
        for line in listeners.stdout.splitlines()
        if re.fullmatch(r"p\d+", line)
    }
    if listeners.returncode != 0 or len(pids) != 1:
        return None
    pid = next(iter(pids))
    try:
        process_info = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    command = process_info.stdout.strip()
    executable_text = str(obs_executable)
    if process_info.returncode != 0 or not (
        command == executable_text
        or command.startswith(executable_text + " ")
    ):
        return None
    return pid


def exit_obs_safely(
    process: subprocess.Popen[bytes],
    obs_pid: int,
) -> bool:
    try:
        subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                'tell application id "com.obsproject.obs-studio" to quit',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        process.wait(timeout=8)
        return True
    except (
        OSError,
        subprocess.SubprocessError,
        subprocess.TimeoutExpired,
    ):
        pass
    try:
        os.kill(obs_pid, signal.SIGTERM)
        process.wait(timeout=8)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def resolve_final_outcome(
    status: str,
    stop_reason: str | None,
    stop_confirmed: bool,
    output_paths: Sequence[Path],
) -> tuple[str, str | None, int]:
    if status == "complete" and not stop_confirmed:
        return "interrupted", "stop_unconfirmed", 2
    if status == "complete" and not any(path.exists() for path in output_paths):
        return "interrupted", "no_recorded_segments", 2
    return status, stop_reason, 0 if status == "complete" else 2


def wait_for_obs(timeout_seconds: float = OBS_START_TIMEOUT_SECONDS) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        log_path = obs_probe.latest_obs_log()
        if log_path is not None and obs_probe.websocket_reachable():
            try:
                request_json("GetVersion")
            except (GuardError, subprocess.SubprocessError, OSError):
                pass
            else:
                return log_path
        time.sleep(0.25)
    raise GuardError("OBS did not become controllable before the startup deadline")


def wait_for_recording_started(
    config: GuardConfig,
    state: GuardState,
    log_path: Path,
    process: subprocess.Popen[bytes],
    timeout_seconds: float = RECORD_START_TIMEOUT_SECONDS,
) -> None:
    """Wait for both OBS active state and an in-root output path."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        check_stop_requested(state)
        if process.poll() is not None:
            raise GuardError("recorder_process_died_during_start")
        text, state.log_offset = read_log_updates(log_path, state.log_offset)
        reason = process_log_events(
            config,
            state,
            parse_obs_log_updates(text),
        )
        if reason:
            raise GuardError(reason)
        status = request_json("GetRecordStatus")
        if status.get("outputActive") and state.output_paths:
            return
        time.sleep(0.1)
    raise GuardError("recording_did_not_start")


def monitor_recording(
    config: GuardConfig,
    state: GuardState,
    log_path: Path,
    process: subprocess.Popen[bytes],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    previous_check = monotonic()
    while True:
        check_stop_requested(state)
        if process.poll() is not None:
            return "recorder_process_died"
        text, state.log_offset = read_log_updates(log_path, state.log_offset)
        reason = process_log_events(
            config,
            state,
            parse_obs_log_updates(text),
        )
        if reason:
            return reason

        try:
            record_status = request_json("GetRecordStatus")
        except (GuardError, subprocess.SubprocessError, OSError):
            return "control_lost"
        if not record_status.get("outputActive"):
            return "recording_stopped_externally"

        now = monotonic()
        elapsed = max(0.0, now - previous_check)
        previous_check = now
        output_bytes = int(record_status.get("outputBytes", 0))
        if output_bytes > state.last_output_bytes:
            state.seconds_since_output_growth = 0.0
            state.last_output_bytes = output_bytes
        else:
            state.seconds_since_output_growth += elapsed
        if (
            state.seconds_since_output_growth
            >= contract.OUTPUT_STALL_SECONDS
        ):
            return "output_stalled"

        disk_free = shutil.disk_usage(config.session_root).free
        if disk_free <= contract.CRITICAL_DISK_BYTES:
            return "disk_critical"
        on_ac_power, battery_percent = battery_snapshot()
        if (
            not on_ac_power
            and battery_percent is not None
            and battery_percent <= contract.CRITICAL_BATTERY_PERCENT
        ):
            return "battery_critical"

        write_guard_manifest(config, state, "recording")
        sleep(config.poll_seconds)


@exclusive_operation("capture")
def run_guard(config: GuardConfig) -> int:
    state = create_session(config)
    with capture_signal_scope(state):
        return _run_guard(config, state)


def _run_guard(config: GuardConfig, state: GuardState) -> int:
    obs_process: subprocess.Popen[bytes] | None = None
    active_log_path: Path | None = None
    final_status = "interrupted"
    recording_started = False
    obs_pid: int | None = None
    exit_code = 2
    alert_path = None
    alerted = False
    control_enabled = False

    def publish_interruption(*, refresh=False):
        nonlocal alerted
        if alerted and not refresh:
            return
        try:
            write_interruption_checkpoint(config, state)
        except (OSError, ValueError, GuardError):
            state.cleanup_errors.append("interruption_checkpoint_failed")
        if alerted:
            return
        alerted = True
        if not recording_started:
            saved_audio = (
                "No audio was saved by this attempt."
                if state.stop_reason == "preflight failed: battery_critical" and not state.output_paths
                else "Review this attempt's manifest before relying on any media."
            )
            print("\aRECORDING DID NOT START. " + saved_audio +
                  " Check the failure above, then run `mi record` again if the meeting is continuing.",
                  file=sys.stderr, flush=True)
            alert_reason = (
                "startup_battery_critical"
                if state.stop_reason == "preflight failed: battery_critical"
                else "startup_failed"
            )
        else:
            print("\aRecording interrupted. Preserving available audio; later conversation may be missing. "
                  "Wait for shutdown, then run `mi record` again if the meeting is continuing.",
                  file=sys.stderr, flush=True)
            alert_reason = state.stop_reason or "capture_interrupted"
        try:
            launched = capture_alert.notify(config.session_root, state.capture_id,
                                            alert_reason, alert_path)
        except Exception:
            launched = False
        if not launched:
            print("Desktop alert unavailable. This Terminal and the MeetingIntel inbox retain the warning.",
                  file=sys.stderr, flush=True)

    try:
        alert_path = capture_alert.prepare()
        if alert_path is None:
            print("Desktop interruption alert unavailable; keep this Terminal visible.", flush=True)
        check_stop_requested(state)
        archived_sentinels = prepare_obs_startup(config)
        if archived_sentinels:
            print(
                "Recovered stale OBS startup state: "
                f"{len(archived_sentinels)} marker"
                f"{'s' if len(archived_sentinels) != 1 else ''} archived.",
                flush=True,
            )
        obs_probe.configure_authenticated_websocket()
        control_enabled = True
        command = build_keep_awake_command(config.obs_executable)
        obs_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log_path = wait_for_obs()
        active_log_path = log_path
        obs_pid = resolve_controlled_obs_pid(config.obs_executable)
        if obs_pid is None:
            raise GuardError("controlled OBS process could not be verified")
        check_stop_requested(state)
        configure_guard_profile(config)
        check_stop_requested(state)
        findings = preflight(config, log_path)
        if findings:
            raise GuardError("preflight failed: " + ", ".join(findings))

        state.log_offset = log_path.stat().st_size
        request_json("StartRecord")
        wait_for_recording_started(
            config,
            state,
            log_path,
            obs_process,
        )
        check_stop_requested(state)
        recording_started = True
        state.recording_ready = True
        write_guard_manifest(config, state, "recording")
        observe.emit("capture.started", source_id=state.capture_id, source_kind="laptop_capture",
                     stage="capture", outcome="started")
        print(
            "Capture guard active. Keep this terminal open; press Control-C to stop.",
            flush=True,
        )
        reason = monitor_recording(config, state, log_path, obs_process)
        state.stop_reason = reason
        publish_interruption()
    except StopRequested as request:
        if request.signum == signal.SIGINT:
            final_status = "complete" if recording_started else "interrupted"
            state.stop_reason = "operator_stop" if recording_started else "operator_cancelled_before_start"
            exit_code = 0 if recording_started else 2
        else:
            state.stop_reason = "terminal_closed" if request.signum == signal.SIGHUP else "termination_requested"
            publish_interruption()
    except KeyboardInterrupt:
        # Preserve this path for callers that raise KeyboardInterrupt directly.
        final_status = "complete" if recording_started else "interrupted"
        state.stop_reason = (
            "operator_stop" if recording_started else "operator_cancelled_before_start"
        )
        exit_code = 0 if recording_started else 2
    except (GuardError, OSError, subprocess.SubprocessError) as error:
        state.stop_reason = str(error)
        print(f"Capture guard failed: {error}", file=sys.stderr)
    finally:
        if final_status == "interrupted" and state.stop_reason != "operator_cancelled_before_start":
            publish_interruption()
        print("Finishing recording and saving evidence. Please wait; another Control-C will not skip cleanup.", flush=True)
        try:
            stop_confirmed = stop_recording_safely() if obs_pid is not None else False
        except Exception:
            stop_confirmed = False
            state.cleanup_errors.append("stop_failed")
        time.sleep(0.25)
        if active_log_path is not None:
            try:
                text, state.log_offset = read_log_updates(
                    active_log_path,
                    state.log_offset,
                )
                final_reason = process_log_events(
                    config,
                    state,
                    parse_obs_log_updates(text),
                )
                if final_reason:
                    final_status = "interrupted"
                    state.stop_reason = final_reason
                    publish_interruption()
            except (GuardError, OSError):
                final_status = "interrupted"
                if state.stop_reason in (None, "operator_stop"):
                    state.stop_reason = "obs_log_unreadable"
                state.cleanup_errors.append("obs_log_unreadable")
        final_status, state.stop_reason, exit_code = resolve_final_outcome(
            final_status,
            state.stop_reason,
            stop_confirmed,
            state.output_paths,
        )
        try:
            obs_exited = (
                exit_obs_safely(obs_process, obs_pid)
                if (obs_process is not None and obs_process.poll() is None and obs_pid is not None)
                else obs_process is None or obs_process.poll() is not None
            )
        except Exception:
            obs_exited = False
        if obs_process is not None and not obs_exited:
            try:
                terminate_process_group(obs_process)
            except Exception:
                state.cleanup_errors.append("recorder_termination_failed")
            final_status = "interrupted"
            state.cleanup_errors.append("recorder_exit_unconfirmed")
            if state.stop_reason in (None, "operator_stop"):
                state.stop_reason = "recorder_exit_unconfirmed"
            exit_code = 2
        if control_enabled:
            try:
                obs_probe.set_websocket_enabled(False)
            except (OSError, ValueError):
                state.cleanup_errors.append("control_disable_failed")
        if state.stop_signal in (signal.SIGTERM, signal.SIGHUP) and final_status == "complete":
            final_status = "interrupted"
            state.stop_reason = "terminal_closed" if state.stop_signal == signal.SIGHUP else "termination_requested"
        if state.cleanup_errors:
            final_status = "interrupted"
            if state.stop_reason in (None, "operator_stop"):
                state.stop_reason = "cleanup_failed"
        try:
            (config.session_root / ".recording.lock").unlink(missing_ok=True)
        except OSError:
            state.cleanup_errors.append("recording_lock_cleanup_failed")
            final_status = "interrupted"
            if state.stop_reason in (None, "operator_stop"):
                state.stop_reason = "cleanup_failed"
            print("Capture lock cleanup failed; review the saved evidence before restarting.", file=sys.stderr)
        if final_status == "interrupted":
            exit_code = 2
            if state.stop_reason != "operator_cancelled_before_start":
                publish_interruption()
        try:
            write_guard_manifest(config, state, final_status)
        except (OSError, ValueError, GuardError, subprocess.SubprocessError):
            final_status = "interrupted"
            state.stop_reason = "manifest_write_failed"
            state.cleanup_errors.append("manifest_write_failed")
            exit_code = 2
            publish_interruption(refresh=True)
            print("Final capture evidence could not be saved. Audio files were retained; review the capture before use.",
                  file=sys.stderr, flush=True)
        observe.emit("capture.stopped" if final_status == "complete" else "capture.interrupted",
                     source_id=state.capture_id, source_kind="laptop_capture", stage="capture",
                     outcome="success" if final_status == "complete" else "interrupted",
                     error_code=None if final_status == "complete" else "capture_interrupted")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated MeetingIntel OBS capture guard",
    )
    parser.add_argument(
        "--session-root",
        required=True,
        type=Path,
        help="new or empty proof directory under /private/tmp",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=DEFAULT_SEGMENT_SECONDS,
        help="test override; production candidate is fixed at 900 seconds",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.segment_seconds < 60 or args.segment_seconds % 60:
        raise SystemExit("--segment-seconds must be a whole number of minutes")
    if args.poll_seconds <= 0 or args.poll_seconds > 10:
        raise SystemExit("--poll-seconds must be greater than 0 and at most 10")
    return run_guard(
        GuardConfig(
            session_root=args.session_root,
            segment_seconds=args.segment_seconds,
            poll_seconds=args.poll_seconds,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
