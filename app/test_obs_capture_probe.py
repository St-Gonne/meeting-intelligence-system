from __future__ import annotations

import configparser
import os
import plistlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import obs_capture_probe as probe


class ObsCaptureProbeTests(unittest.TestCase):
    def test_missing_install_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status = probe.inspect_obs_install(Path(temporary) / "OBS.app")
        self.assertFalse(status.installed)

    def test_install_version_and_arm64_shape_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary) / "OBS.app"
            executable = app / "Contents" / "MacOS" / "OBS"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"\xcf\xfa\xed\xfe" + b"synthetic")
            info = app / "Contents" / "Info.plist"
            with info.open("wb") as handle:
                plistlib.dump({"CFBundleShortVersionString": "32.2.1"}, handle)
            status = probe.inspect_obs_install(app)
        self.assertEqual(probe.ObsInstallStatus(True, "32.2.1", "arm64"), status)

    def test_permission_log_parsing_is_explicit_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "obs.txt"
            log.write_text(
                "[macOS] Permission for audio device access denied.\n"
                "[macOS] Permission for screen capture denied.\n",
                encoding="utf-8",
            )
            status = probe.permission_status_from_log(log)
        self.assertEqual("denied", status.microphone)
        self.assertEqual("denied", status.screen_capture)
        self.assertEqual("unknown", status.video_device)
        self.assertEqual("unknown", status.input_monitoring)

    def test_latest_log_uses_mtime_not_filename_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            newer_name = logs / "2099-newer-name.txt"
            older_name = logs / "2000-older-name.txt"
            newer_name.write_text("first", encoding="utf-8")
            older_name.write_text("second", encoding="utf-8")
            os.utime(newer_name, (1, 1))
            os.utime(older_name, (2, 2))
            self.assertEqual(older_name, probe.latest_obs_log(root))

    def test_websocket_configuration_is_authenticated_atomic_and_backed_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            global_ini = root / "obs" / "global.ini"
            global_ini.parent.mkdir()
            global_ini.write_text("[General]\nFirstRun=true\n", encoding="utf-8")
            private_root = root / "private"
            password_path = probe.configure_authenticated_websocket(
                global_ini=global_ini,
                private_root=private_root,
                password_factory=lambda _: "x" * 32,
            )

            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
            parser.read(global_ini)
            self.assertEqual("true", parser.get("OBSWebSocket", "ServerEnabled"))
            self.assertEqual("true", parser.get("OBSWebSocket", "AuthRequired"))
            self.assertEqual("4455", parser.get("OBSWebSocket", "ServerPort"))
            self.assertEqual("x" * 32, parser.get("OBSWebSocket", "ServerPassword"))
            self.assertEqual("x" * 32, password_path.read_text().strip())
            self.assertEqual(0o600, password_path.stat().st_mode & 0o777)
            self.assertEqual(0o700, private_root.stat().st_mode & 0o777)
            backup = global_ini.with_name("global.ini.before-meetingintel-capture")
            self.assertIn("FirstRun=true", backup.read_text())

    def test_existing_password_is_reused_not_rotated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            global_ini = root / "global.ini"
            global_ini.write_text("[General]\n", encoding="utf-8")
            private_root = root / "private"
            private_root.mkdir()
            password_path = private_root / "obs-websocket-password"
            password_path.write_text("existing-password-of-safe-length\n", encoding="utf-8")
            probe.configure_authenticated_websocket(
                global_ini=global_ini,
                private_root=private_root,
                password_factory=lambda _: "new-password-that-must-not-be-used",
            )
            self.assertEqual(
                "existing-password-of-safe-length",
                password_path.read_text().strip(),
            )

    def test_unsafe_port_is_rejected_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            global_ini = root / "global.ini"
            global_ini.write_text("[General]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.configure_authenticated_websocket(
                    global_ini=global_ini,
                    private_root=root / "private",
                    port=80,
                )
            self.assertFalse((root / "private").exists())

    def test_request_secret_is_environment_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            password = root / "password"
            password.write_text("private-secret\n", encoding="utf-8")
            with mock.patch("obs_capture_probe.subprocess.run") as run:
                run.return_value.stdout = '{"ok":true}\n'
                result = probe.run_obs_request(
                    "GetVersion",
                    password_path=password,
                    control_script=root / "control.mjs",
                )
            self.assertEqual(result, '{"ok":true}')
            argv = run.call_args.args[0]
            self.assertNotIn("private-secret", argv)
            self.assertEqual(
                run.call_args.kwargs["env"]["MI_OBS_WS_PASSWORD"],
                "private-secret",
            )

    def test_request_type_must_be_one_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "one non-empty token"):
            probe.run_obs_request("Get Version")

    def test_control_server_can_be_disabled_at_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            global_ini = Path(temporary) / "global.ini"
            global_ini.write_text(
                "[OBSWebSocket]\nServerEnabled=true\nAuthRequired=true\n",
                encoding="utf-8",
            )
            probe.set_websocket_enabled(False, global_ini=global_ini)
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
            parser.read(global_ini)
            self.assertEqual("false", parser.get("OBSWebSocket", "ServerEnabled"))
            self.assertEqual("true", parser.get("OBSWebSocket", "AuthRequired"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
