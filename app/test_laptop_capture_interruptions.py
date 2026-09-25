"""Synthetic interruption/cleanup regressions; never starts OBS or captures audio."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import laptop_capture_guard as guard


class SignalScopeTests(unittest.TestCase):
    def test_signals_only_flag_and_restore_previous_handlers(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        state = guard.GuardState("synthetic", guard.utc_now())
        with guard.capture_signal_scope(state):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(signal.SIGINT, state.stop_signal)
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(signal.SIGTERM, state.stop_signal)
            with self.assertRaises(guard.StopRequested) as caught:
                guard.check_stop_requested(state)
            self.assertEqual(signal.SIGTERM, caught.exception.signum)
        self.assertEqual(previous, {s: signal.getsignal(s) for s in previous})

    def test_real_child_term_and_hup_are_caught_at_safe_boundary(self):
        # Actual OS signal delivery, isolated from the test runner and no recorder.
        script = """
import os, signal, sys
import laptop_capture_guard as guard
signum = int(sys.argv[1])
state = guard.GuardState('synthetic', guard.utc_now())
previous = signal.getsignal(signum)
with guard.capture_signal_scope(state):
    os.kill(os.getpid(), signum)
    assert state.stop_signal == signum
    try:
        guard.check_stop_requested(state)
    except guard.StopRequested as stop:
        assert stop.signum == signum
    else:
        raise AssertionError('signal was not delivered')
assert signal.getsignal(signum) == previous
print('signal safely handled')
"""
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum):
                result = subprocess.run([sys.executable, "-c", script, str(signum)],
                                        cwd=Path(__file__).resolve().parent,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual("signal safely handled", result.stdout.strip())


class GuardInterruptionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/private/tmp")))
        self.config = guard.GuardConfig(self.root / "capture", allowed_root=self.root)
        self.state = guard.create_session(self.config)
        self.segment = self.config.session_root / "segments" / "synthetic.mkv"
        self.segment.write_bytes(b"synthetic media fixture")
        self.state.output_paths.append(self.segment)
        self.log = self.root / "obs.log"
        self.log.write_text("", encoding="utf-8")
        self.process = mock.Mock()
        self.process.poll.return_value = None
        self.mocks = {}
        replacements = {
            "prepare_obs_startup": [], "obs_probe.configure_authenticated_websocket": None,
            "build_keep_awake_command": ["never-executed"], "subprocess.Popen": self.process,
            "wait_for_obs": self.log, "resolve_controlled_obs_pid": 12345,
            "configure_guard_profile": None, "preflight": (), "request_json": {},
            "wait_for_recording_started": None, "stop_recording_safely": True,
            "read_log_updates": ("", 0), "exit_obs_safely": True,
            "terminate_process_group": None, "obs_probe.set_websocket_enabled": None,
            "time.sleep": None, "observe.emit": True,
            "capture_alert.prepare": None, "capture_alert.notify": False,
            "contract.ffprobe_media": SimpleNamespace(duration_seconds=12.0),
        }
        for name, value in replacements.items():
            self.mocks[name] = self.stack.enter_context(mock.patch.object(
                self._owner(name), name.split(".")[-1], return_value=value))
        self.monitor = self.stack.enter_context(mock.patch.object(guard, "monitor_recording", side_effect=self.stop))

    @staticmethod
    def _owner(name):
        owner = guard
        for part in name.split(".")[:-1]:
            owner = getattr(owner, part)
        return owner

    def stop(self, *_args):
        signal.raise_signal(signal.SIGINT)
        guard.check_stop_requested(self.state)

    def run_capture(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), guard.capture_signal_scope(self.state):
            result = guard._run_guard(self.config, self.state)
        self.assertFalse((self.config.session_root / ".recording.lock").exists())
        return result

    def manifest(self):
        return json.loads((self.config.session_root / "capture_manifest.json").read_text())

    def events(self):
        return [call.args[0] for call in self.mocks["observe.emit"].call_args_list]

    def assert_interrupted(self, reason=None):
        value = self.manifest()
        self.assertEqual("interrupted", value["status"])
        self.assertFalse(value["finalizing"])
        if reason is not None:
            self.assertEqual(reason, value["stop_reason"])
        self.assertIn("capture.interrupted", self.events())
        self.assertNotIn("capture.stopped", self.events())

    def test_repeated_ctrl_c_during_cleanup_still_saves_and_releases(self):
        def stop_again():
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            return True
        self.mocks["stop_recording_safely"].side_effect = stop_again
        self.assertEqual(0, self.run_capture())
        self.assertEqual("complete", self.manifest()["status"])
        self.assertEqual("finalized", self.manifest()["segments"][0]["state"])
        self.mocks["exit_obs_safely"].assert_called_once()
        self.mocks["obs_probe.set_websocket_enabled"].assert_called_once_with(False)
        self.assertEqual(["capture.started", "capture.stopped"], self.events())

    def test_cancel_before_confirmed_readiness_never_emits_started(self):
        self.mocks["wait_for_recording_started"].side_effect = self.stop
        self.assertEqual(2, self.run_capture())
        self.monitor.assert_not_called()
        self.assertNotIn("capture.started", self.events())
        self.assert_interrupted("operator_cancelled_before_start")

    def test_source_loss_tail_overrides_operator_stop(self):
        self.mocks["read_log_updates"].return_value = ("Stream stopped as no capture source\n", 50)
        self.assertEqual(2, self.run_capture())
        self.assert_interrupted("source_lost")
        self.assertEqual("source_lost", self.manifest()["source_events"][0]["type"])
        self.mocks["capture_alert.notify"].assert_called_once()

    def test_lost_log_tail_cannot_report_success(self):
        self.mocks["read_log_updates"].side_effect = guard.GuardError("log vanished")
        self.assertEqual(2, self.run_capture())
        self.assert_interrupted("obs_log_unreadable")
        self.mocks["exit_obs_safely"].assert_called_once()
        self.mocks["obs_probe.set_websocket_enabled"].assert_called_once_with(False)

    def test_lock_cleanup_failure_cannot_persist_or_emit_success(self):
        original_unlink = Path.unlink
        lock = self.config.session_root / ".recording.lock"

        def cannot_unlink(path, *args, **kwargs):
            if path == lock:
                raise OSError("synthetic lock cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", new=cannot_unlink), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             guard.capture_signal_scope(self.state):
            self.assertEqual(2, guard._run_guard(self.config, self.state))
        self.assertTrue(lock.exists())
        self.assert_interrupted()
        self.assertTrue(any("lock" in error for error in self.manifest()["cleanup_errors"]))
        self.mocks["exit_obs_safely"].assert_called_once()
        self.mocks["obs_probe.set_websocket_enabled"].assert_called_once_with(False)

    def test_term_during_cleanup_overrides_ordinary_stop(self):
        def terminated():
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            return True
        self.mocks["stop_recording_safely"].side_effect = terminated
        self.assertEqual(2, self.run_capture())
        self.assert_interrupted("termination_requested")

    def test_hup_during_monitor_preserves_interruption(self):
        def hangup(*_args):
            signal.raise_signal(signal.SIGHUP)
            guard.check_stop_requested(self.state)
        self.monitor.side_effect = hangup
        self.assertEqual(2, self.run_capture())
        self.assert_interrupted("terminal_closed")

    def test_cleanup_failures_still_attempt_exit_control_and_manifest(self):
        self.mocks["stop_recording_safely"].side_effect = OSError("synthetic stop failure")
        self.mocks["exit_obs_safely"].side_effect = OSError("synthetic quit failure")
        self.mocks["terminate_process_group"].side_effect = OSError("synthetic terminate failure")
        self.mocks["obs_probe.set_websocket_enabled"].side_effect = OSError("synthetic control failure")
        self.assertEqual(2, self.run_capture())
        self.assert_interrupted()
        self.mocks["exit_obs_safely"].assert_called_once()
        self.mocks["terminate_process_group"].assert_called_once()
        self.mocks["obs_probe.set_websocket_enabled"].assert_called_once_with(False)
        self.assertEqual({"stop_failed", "recorder_termination_failed", "recorder_exit_unconfirmed", "control_disable_failed"},
                         set(self.manifest()["cleanup_errors"]))
        self.assertEqual("active", self.manifest()["segments"][0]["state"])
        self.assertNotIn("sha256", self.manifest()["segments"][0])

    def test_foreign_obs_startup_rejection_never_controls_existing_recorder(self):
        self.mocks["prepare_obs_startup"].side_effect = guard.GuardError("OBS is already running")
        self.assertEqual(2, self.run_capture())
        for name in ("subprocess.Popen", "stop_recording_safely", "exit_obs_safely",
                     "obs_probe.configure_authenticated_websocket", "obs_probe.set_websocket_enabled"):
            self.mocks[name].assert_not_called()
        self.assert_interrupted("OBS is already running")

    def test_critical_battery_preflight_says_recording_never_started(self):
        self.state.output_paths.clear()
        self.segment.unlink()
        self.mocks["preflight"].return_value = ("battery_critical",)
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr), guard.capture_signal_scope(self.state):
            self.assertEqual(2, guard._run_guard(self.config, self.state))
        self.assert_interrupted("preflight failed: battery_critical")
        self.assertEqual([], self.manifest()["segments"])
        self.assertIn("RECORDING DID NOT START", stderr.getvalue())
        self.assertIn("No audio was saved", stderr.getvalue())
        self.assertNotIn("Preserving available audio", stderr.getvalue())
        self.assertNotIn("StartRecord", [call.args[0] for call in self.mocks["request_json"].call_args_list])
        self.mocks["capture_alert.notify"].assert_called_once()
        self.assertEqual("startup_battery_critical", self.mocks["capture_alert.notify"].call_args.args[2])

    def test_manifest_failure_leaves_checkpoint_incomplete_and_cleans_up(self):
        actual_write = guard.write_guard_manifest
        def fail_final_write(config, state, status):
            if status != "recording":
                raise OSError("synthetic write failure")
            return actual_write(config, state, status)
        with mock.patch.object(guard, "write_guard_manifest", side_effect=fail_final_write):
            self.assertEqual(2, self.run_capture())
        value = self.manifest()
        self.assertEqual("interrupted", value["status"])
        self.assertTrue(value["finalizing"])
        self.assertEqual("manifest_write_failed", value["stop_reason"])
        self.assertNotIn("capture.stopped", self.events())
        self.mocks["exit_obs_safely"].assert_called_once()
        self.mocks["obs_probe.set_websocket_enabled"].assert_called_once_with(False)

    def test_checkpoint_preserves_prior_segments_without_media_work(self):
        guard.write_guard_manifest(self.config, self.state, "recording")
        previous = self.manifest()
        self.state.stop_reason = "source_lost"
        with mock.patch.object(guard.contract, "ffprobe_media", side_effect=AssertionError("must not probe")), \
             mock.patch.object(guard.contract, "_sha256", side_effect=AssertionError("must not hash")):
            guard.write_interruption_checkpoint(self.config, self.state)
        value = self.manifest()
        self.assertEqual(previous["segments"], value["segments"])
        self.assertEqual("interrupted", value["status"])
        self.assertTrue(value["finalizing"])
        self.assertIsNone(value["ended_at"])
        self.assertEqual(0o600, (self.config.session_root / "capture_manifest.json").stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main()
