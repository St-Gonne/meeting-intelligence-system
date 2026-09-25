from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laptop_capture_guard as guard


class GuardPureFunctionTests(unittest.TestCase):
    def test_proof_root_must_be_private_tmp(self) -> None:
        with self.assertRaisesRegex(guard.GuardError, "under /private/tmp"):
            guard.validate_proof_session_root(Path("/Users/example/capture"))

    def test_nonempty_session_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            (root / "existing").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(guard.GuardError, "new or empty"):
                guard.validate_proof_session_root(root)

    def test_keep_awake_wraps_exact_obs_executable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            app = Path(temporary) / "OBS.app"
            executable = app / "Contents" / "MacOS" / "OBS"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"synthetic")
            (app / "Contents" / "Info.plist").write_bytes(b"synthetic")
            command = guard.build_keep_awake_command(executable)
        self.assertEqual("/usr/bin/caffeinate", command[0])
        self.assertIn("-dimsu", command)
        self.assertEqual("/usr/bin/open", command[2])
        self.assertIn(str(app), command)
        self.assertIn("-W", command)
        self.assertIn("-j", command)
        self.assertIn("--websocket_ipv4_only", command)
        self.assertIn("--minimize-to-tray", command)

    def test_obs_exit_uses_graceful_macos_quit(self) -> None:
        process = mock.Mock()
        with mock.patch(
            "laptop_capture_guard.subprocess.run"
        ) as run, mock.patch("laptop_capture_guard.os.kill") as kill:
            self.assertTrue(guard.exit_obs_safely(process, 123))
        self.assertEqual("/usr/bin/osascript", run.call_args.args[0][0])
        self.assertIn("com.obsproject.obs-studio", run.call_args.args[0][2])
        process.wait.assert_called_once_with(timeout=8)
        kill.assert_not_called()

    def test_obs_exit_falls_back_to_sigterm(self) -> None:
        process = mock.Mock()
        with mock.patch(
            "laptop_capture_guard.subprocess.run",
            side_effect=OSError("osascript unavailable"),
        ), mock.patch("laptop_capture_guard.os.kill") as kill:
            self.assertTrue(guard.exit_obs_safely(process, 123))
        kill.assert_called_once_with(123, guard.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=8)

    def test_obs_app_running_requires_exact_executable_record(self) -> None:
        executable = Path("/Applications/OBS.app/Contents/MacOS/OBS")
        result = mock.Mock(
            returncode=0,
            stdout=f"p42\nn{executable}\nn/usr/lib/libSystem.B.dylib\n",
        )
        with mock.patch(
            "laptop_capture_guard.subprocess.run",
            return_value=result,
        ) as run:
            self.assertTrue(guard.obs_app_is_running(executable))
        self.assertEqual("/usr/sbin/lsof", run.call_args.args[0][0])
        self.assertIn("-Fpn", run.call_args.args[0])

    def test_obs_app_running_accepts_no_exact_match(self) -> None:
        with mock.patch(
            "laptop_capture_guard.subprocess.run",
            return_value=mock.Mock(
                returncode=1,
                stdout="p42\nn/Applications/Other.app/Contents/MacOS/Other\n",
            ),
        ):
            self.assertFalse(
                guard.obs_app_is_running(
                    Path("/Applications/OBS.app/Contents/MacOS/OBS")
                )
            )

    def test_stale_obs_sentinels_are_preserved_in_private_archive(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            sentinel_root = root / ".sentinel"
            archive_root = root / "archive"
            sentinel_root.mkdir()
            marker = sentinel_root / "run_2c41fce6-8393-4da6-b56d-25d78a481248"
            marker.touch()
            archived = guard.archive_stale_obs_sentinels(
                sentinel_root=sentinel_root,
                archive_root=archive_root,
            )
            self.assertEqual(1, len(archived))
            self.assertFalse(marker.exists())
            self.assertTrue(archived[0].is_file())
            self.assertEqual(0o700, archive_root.stat().st_mode & 0o777)
            self.assertEqual(0o600, archived[0].stat().st_mode & 0o777)

    def test_unexpected_obs_sentinel_is_not_moved(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            sentinel_root = root / ".sentinel"
            sentinel_root.mkdir()
            marker = sentinel_root / "unexpected"
            marker.touch()
            with self.assertRaisesRegex(guard.GuardError, "unexpected"):
                guard.archive_stale_obs_sentinels(
                    sentinel_root=sentinel_root,
                    archive_root=root / "archive",
                )
            self.assertTrue(marker.exists())
            self.assertFalse((root / "archive").exists())

    def test_obs_startup_refuses_running_uncontrolled_app(self) -> None:
        config = guard.GuardConfig(Path("/private/tmp/session"))
        with mock.patch(
            "laptop_capture_guard.obs_probe.websocket_reachable",
            return_value=False,
        ), mock.patch(
            "laptop_capture_guard.obs_app_is_running",
            return_value=True,
        ), mock.patch(
            "laptop_capture_guard.archive_stale_obs_sentinels"
        ) as archive:
            with self.assertRaisesRegex(guard.GuardError, "already running"):
                guard.prepare_obs_startup(config)
        archive.assert_not_called()

    def test_session_may_use_one_explicit_durable_root(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            allowed_root = Path(temporary)
            session = allowed_root / "Meeting 2026-07-28_12-00-00"
            self.assertEqual(
                session,
                guard.validate_session_root(session, allowed_root),
            )

    def test_log_parser_reports_output_and_source_loss(self) -> None:
        events = guard.parse_obs_log_updates(
            "Writing file '/private/tmp/proof/segments/one.mkv'...\n"
            "[ mac-screencapture ]: Stream stopped as no capture source was not found.\n"
        )
        self.assertEqual(["output_path", "source_lost"], [item.type for item in events])
        self.assertEqual(
            Path("/private/tmp/proof/segments/one.mkv"),
            events[0].output_path,
        )

    def test_log_parser_reports_hybrid_mov_output_path(self) -> None:
        events = guard.parse_obs_log_updates(
            "[mov output: 'simple_file_output'] Writing Hybrid MP4/MOV file "
            "'/Users/example/Movies/escaped.mov'...\n"
        )
        self.assertEqual(("output_path",), tuple(item.type for item in events))
        self.assertEqual(
            Path("/Users/example/Movies/escaped.mov"),
            events[0].output_path,
        )

    def test_permission_denial_is_a_distinct_log_failure(self) -> None:
        events = guard.parse_obs_log_updates(
            "[macOS] Permission for screen capture denied.\n"
        )
        self.assertEqual(("permission_failed",), tuple(item.type for item in events))

    def test_output_path_must_remain_in_session(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            config = guard.GuardConfig(root)
            state = guard.GuardState("capture", "2026-07-28T00:00:00Z")
            reason = guard.process_log_events(
                config,
                state,
                [
                    guard.GuardLogEvent(
                        "output_path",
                        "escaped",
                        Path("/private/tmp/outside.mkv"),
                    )
                ],
            )
        self.assertEqual("output_path_escaped_session", reason)

    def test_source_loss_is_persisted_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            config = guard.GuardConfig(root)
            state = guard.GuardState("capture", "2026-07-28T00:00:00Z")
            with mock.patch("laptop_capture_guard.utc_now", return_value="2026-07-28T00:01:00Z"):
                reason = guard.process_log_events(
                    config,
                    state,
                    [guard.GuardLogEvent("source_lost", "stream stopped")],
                )
        self.assertEqual("source_lost", reason)
        self.assertEqual(
            [{"type": "source_lost", "at": "2026-07-28T00:01:00Z"}],
            state.source_events,
        )


class GuardManifestTests(unittest.TestCase):
    def test_complete_without_a_segment_fails_closed(self) -> None:
        outcome = guard.resolve_final_outcome(
            "complete",
            "operator_stop",
            True,
            (),
        )
        self.assertEqual(
            ("interrupted", "no_recorded_segments", 2),
            outcome,
        )

    def test_unconfirmed_stop_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            segment = Path(temporary) / "one.mkv"
            segment.write_bytes(b"synthetic")
            outcome = guard.resolve_final_outcome(
                "complete",
                "operator_stop",
                False,
                (segment,),
            )
        self.assertEqual(("interrupted", "stop_unconfirmed", 2), outcome)

    def test_session_creation_is_private_locked_and_atomic(self) -> None:
        parent = Path(tempfile.mkdtemp(dir="/private/tmp"))
        root = parent / "session"
        config = guard.GuardConfig(root)
        with mock.patch("laptop_capture_guard.utc_now", return_value="2026-07-28T00:00:00Z"):
            state = guard.create_session(config)
        self.assertTrue((root / ".recording.lock").is_file())
        self.assertEqual(0o700, root.stat().st_mode & 0o777)
        payload = json.loads((root / "capture_manifest.json").read_text())
        self.assertEqual("recording", payload["status"])
        self.assertEqual(state.capture_id, payload["capture_id"])
        self.assertEqual(
            0o600,
            (root / "capture_manifest.json").stat().st_mode & 0o777,
        )
        self.assertFalse((root / ".capture_manifest.json.tmp").exists())

    def test_discovered_segment_is_made_private(self) -> None:
        parent = Path(tempfile.mkdtemp(dir="/private/tmp"))
        root = parent / "session"
        (root / "segments").mkdir(parents=True)
        segment = root / "segments" / "one.mkv"
        segment.write_bytes(b"synthetic")
        segment.chmod(0o644)
        state = guard.GuardState("capture", "2026-07-28T00:00:00Z")
        reason = guard.process_log_events(
            guard.GuardConfig(root),
            state,
            [guard.GuardLogEvent("output_path", "output", segment)],
        )
        self.assertIsNone(reason)
        self.assertEqual(0o600, segment.stat().st_mode & 0o777)

    def test_manifest_refresh_makes_active_segment_private(self) -> None:
        parent = Path(tempfile.mkdtemp(dir="/private/tmp"))
        root = parent / "session"
        config = guard.GuardConfig(root)
        state = guard.create_session(config)
        segment = root / "segments" / "one.mkv"
        segment.write_bytes(b"synthetic")
        segment.chmod(0o644)
        state.output_paths.append(segment)
        with mock.patch(
            "laptop_capture_guard.contract.ffprobe_media",
            return_value=guard.contract.MediaProbe(True, True, 1.0),
        ):
            guard.write_guard_manifest(config, state, "recording")
        self.assertEqual(0o600, segment.stat().st_mode & 0o777)

    def test_interrupted_manifest_rejects_unrecovered_source(self) -> None:
        parent = Path(tempfile.mkdtemp(dir="/private/tmp"))
        root = parent / "session"
        config = guard.GuardConfig(root)
        with mock.patch("laptop_capture_guard.utc_now", return_value="2026-07-28T00:00:00Z"):
            state = guard.create_session(config)
        segment = root / "segments" / "one.mkv"
        segment.write_bytes(b"synthetic")
        state.output_paths.append(segment)
        state.source_events.append(
            {"type": "source_lost", "at": "2026-07-28T00:01:00Z"}
        )
        state.stop_reason = "source_lost"
        fake_probe = mock.patch(
            "laptop_capture_guard.contract.ffprobe_media",
            return_value=guard.contract.MediaProbe(True, True, 10.0),
        )
        with fake_probe, mock.patch(
            "laptop_capture_guard.utc_now",
            return_value="2026-07-28T00:02:00Z",
        ):
            guard.write_guard_manifest(config, state, "interrupted")
        payload = json.loads((root / "capture_manifest.json").read_text())
        with mock.patch(
            "laptop_capture_contract.ffprobe_media",
            return_value=guard.contract.MediaProbe(True, True, 10.0),
        ):
            audit = guard.contract.audit_capture_session(root, payload)
        self.assertIn("source_not_recovered", audit.codes())
        self.assertFalse(audit.recovery_ready)


class GuardMonitorTests(unittest.TestCase):
    def config_and_state(self) -> tuple[guard.GuardConfig, guard.GuardState, Path]:
        parent = Path(tempfile.mkdtemp(dir="/private/tmp"))
        root = parent / "session"
        root.mkdir()
        (root / "segments").mkdir()
        log = root / "obs.txt"
        log.write_text("", encoding="utf-8")
        return (
            guard.GuardConfig(root, poll_seconds=0.01),
            guard.GuardState("capture", "2026-07-28T00:00:00Z"),
            log,
        )

    def test_process_death_interrupts(self) -> None:
        config, state, log = self.config_and_state()
        process = mock.Mock()
        process.poll.return_value = 1
        reason = guard.monitor_recording(config, state, log, process)
        self.assertEqual("recorder_process_died", reason)

    def test_controlled_pid_requires_exact_obs_executable(self) -> None:
        executable = Path("/Applications/OBS.app/Contents/MacOS/OBS")
        listener = mock.Mock(returncode=0, stdout="p123\n")
        exact_process = mock.Mock(
            returncode=0,
            stdout=str(executable) + " --minimize-to-tray\n",
        )
        with mock.patch(
            "laptop_capture_guard.subprocess.run",
            side_effect=[listener, exact_process],
        ):
            self.assertEqual(
                123,
                guard.resolve_controlled_obs_pid(executable),
            )

    def test_controlled_pid_rejects_wrong_executable(self) -> None:
        listener = mock.Mock(returncode=0, stdout="p123\n")
        wrong_process = mock.Mock(
            returncode=0,
            stdout="/Applications/Other.app/Contents/MacOS/Other\n",
        )
        with mock.patch(
            "laptop_capture_guard.subprocess.run",
            side_effect=[listener, wrong_process],
        ):
            self.assertIsNone(
                guard.resolve_controlled_obs_pid(
                    Path("/Applications/OBS.app/Contents/MacOS/OBS")
                )
            )

    def test_start_waits_for_active_status_and_in_root_output(self) -> None:
        config, state, log = self.config_and_state()
        segment = config.session_root / "segments" / "one.mkv"
        log.write_text(
            f"Writing file '{segment}'...\n",
            encoding="utf-8",
        )
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch(
            "laptop_capture_guard.request_json",
            return_value={"outputActive": True},
        ):
            guard.wait_for_recording_started(
                config,
                state,
                log,
                process,
                timeout_seconds=0.1,
            )
        self.assertEqual([segment], state.output_paths)

    def test_start_rejects_escaped_simple_output(self) -> None:
        config, state, log = self.config_and_state()
        log.write_text(
            "[mov output: 'simple_file_output'] Writing Hybrid MP4/MOV file "
            "'/Users/example/Movies/escaped.mov'...\n",
            encoding="utf-8",
        )
        process = mock.Mock()
        process.poll.return_value = None
        with self.assertRaisesRegex(
            guard.GuardError,
            "output_path_escaped_session",
        ):
            guard.wait_for_recording_started(
                config,
                state,
                log,
                process,
                timeout_seconds=0.1,
            )

    def test_control_loss_interrupts(self) -> None:
        config, state, log = self.config_and_state()
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch(
            "laptop_capture_guard.request_json",
            side_effect=guard.GuardError("lost"),
        ):
            reason = guard.monitor_recording(config, state, log, process)
        self.assertEqual("control_lost", reason)

    def test_explicit_source_loss_interrupts_before_file_growth_check(self) -> None:
        config, state, log = self.config_and_state()
        log.write_text(
            "[ mac-screencapture ]: Stream stopped as no capture source was not found.\n",
            encoding="utf-8",
        )
        process = mock.Mock()
        process.poll.return_value = None
        reason = guard.monitor_recording(config, state, log, process)
        self.assertEqual("source_lost", reason)

    def test_output_stall_interrupts(self) -> None:
        config, state, log = self.config_and_state()
        state.seconds_since_output_growth = guard.contract.OUTPUT_STALL_SECONDS
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch(
            "laptop_capture_guard.request_json",
            return_value={"outputActive": True, "outputBytes": 0},
        ), mock.patch(
            "laptop_capture_guard.battery_snapshot",
            return_value=(True, 100),
        ):
            reason = guard.monitor_recording(
                config,
                state,
                log,
                process,
                monotonic=mock.Mock(side_effect=[0.0, 1.0]),
                sleep=lambda _: None,
            )
        self.assertEqual("output_stalled", reason)

    def test_critical_battery_interrupts(self) -> None:
        config, state, log = self.config_and_state()
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch(
            "laptop_capture_guard.request_json",
            return_value={"outputActive": True, "outputBytes": 1},
        ), mock.patch(
            "laptop_capture_guard.battery_snapshot",
            return_value=(False, guard.contract.CRITICAL_BATTERY_PERCENT),
        ):
            reason = guard.monitor_recording(
                config,
                state,
                log,
                process,
                monotonic=mock.Mock(side_effect=[0.0, 1.0]),
                sleep=lambda _: None,
            )
        self.assertEqual("battery_critical", reason)


class GuardPreflightTests(unittest.TestCase):
    @staticmethod
    def profile_parameter_response(request_data: dict[str, object]) -> dict[str, str]:
        key = (
            str(request_data["parameterCategory"]),
            str(request_data["parameterName"]),
        )
        values = {
            ("Output", "Mode"): "Advanced",
            ("AdvOut", "RecType"): "Standard",
            ("AdvOut", "RecFormat2"): "mkv",
            ("AdvOut", "RecSplitFile"): "true",
            ("AdvOut", "RecSplitFileType"): "Time",
            ("AdvOut", "RecSplitFileTime"): "15",
        }
        return {"parameterValue": values[key]}

    @staticmethod
    def add_hotkey_response(
        request: str,
        response: dict[str, object],
    ) -> dict[str, object]:
        if request == "GetHotkeyList":
            return {"hotkeys": ["OBSBasic.Exit"]}
        return response

    def test_missing_required_source_blocks_start(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            (root / "segments").mkdir()
            log = root / "obs.txt"
            log.write_text(
                "[macOS] Permission for audio device access granted.\n"
                "[macOS] Permission for screen capture granted.\n",
                encoding="utf-8",
            )
            responses = {
                "GetRecordDirectory": {"recordDirectory": str(root / "segments")},
                "GetRecordStatus": {"outputActive": False},
                "GetInputList": {
                    "inputs": [{"inputName": "macOS System Audio"}]
                },
            }
            def response(request: str, data=None):
                if request == "GetProfileParameter":
                    return self.profile_parameter_response(data)
                return self.add_hotkey_response(request, responses.get(request, {}))

            with mock.patch(
                "laptop_capture_guard.obs_probe.inspect_obs_install",
                return_value=guard.obs_probe.ObsInstallStatus(True, "32.2.1", "arm64"),
            ), mock.patch(
                "laptop_capture_guard.request_json",
                side_effect=response,
            ), mock.patch(
                "laptop_capture_guard.battery_snapshot",
                return_value=(True, 100),
            ):
                findings = guard.preflight(guard.GuardConfig(root), log)
        self.assertIn("source_missing:Capture Guard Microphone", findings)

    def test_source_creation_failure_in_log_blocks_start(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            (root / "segments").mkdir()
            log = root / "obs.txt"
            log.write_text(
                "[macOS] Permission for audio device access granted.\n"
                "[macOS] Permission for screen capture granted.\n"
                "Failed to create source 'macOS System Audio'!\n",
                encoding="utf-8",
            )
            responses = {
                "GetRecordDirectory": {"recordDirectory": str(root / "segments")},
                "GetRecordStatus": {"outputActive": False},
                "GetInputList": {
                    "inputs": [
                        {"inputName": "macOS System Audio"},
                        {"inputName": "Capture Guard Microphone"},
                    ]
                },
            }
            def response(request: str, data=None):
                if request == "GetProfileParameter":
                    return self.profile_parameter_response(data)
                return self.add_hotkey_response(request, responses.get(request, {}))

            with mock.patch(
                "laptop_capture_guard.obs_probe.inspect_obs_install",
                return_value=guard.obs_probe.ObsInstallStatus(True, "32.2.1", "arm64"),
            ), mock.patch(
                "laptop_capture_guard.request_json",
                side_effect=response,
            ), mock.patch(
                "laptop_capture_guard.battery_snapshot",
                return_value=(True, 100),
            ):
                findings = guard.preflight(guard.GuardConfig(root), log)
        self.assertIn("source_not_reporting", findings)

    def test_simple_output_mode_blocks_start_even_when_directory_matches(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            (root / "segments").mkdir()
            log = root / "obs.txt"
            log.write_text(
                "[macOS] Permission for audio device access granted.\n"
                "[macOS] Permission for screen capture granted.\n",
                encoding="utf-8",
            )
            responses = {
                "GetRecordDirectory": {"recordDirectory": str(root / "segments")},
                "GetRecordStatus": {"outputActive": False},
                "GetInputList": {
                    "inputs": [
                        {"inputName": "macOS System Audio"},
                        {"inputName": "Capture Guard Microphone"},
                    ]
                },
            }

            def response(request: str, data=None):
                if request == "GetProfileParameter":
                    parameter = self.profile_parameter_response(data)
                    if data["parameterName"] == "Mode":
                        parameter["parameterValue"] = "Simple"
                    return parameter
                return self.add_hotkey_response(request, responses.get(request, {}))

            with mock.patch(
                "laptop_capture_guard.obs_probe.inspect_obs_install",
                return_value=guard.obs_probe.ObsInstallStatus(True, "32.2.1", "arm64"),
            ), mock.patch(
                "laptop_capture_guard.request_json",
                side_effect=response,
            ), mock.patch(
                "laptop_capture_guard.battery_snapshot",
                return_value=(True, 100),
            ):
                findings = guard.preflight(guard.GuardConfig(root), log)
        self.assertIn("profile_parameter_mismatch:Output/Mode", findings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
