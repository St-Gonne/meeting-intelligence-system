from __future__ import annotations

import json
import os
import stat
import plistlib
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock

from ollama_endpoint import OLLAMA_APPROVED_DIGEST, OLLAMA_MODEL
from ollama_service import (
    MANAGED_ENDPOINT,
    MANAGED_HOST_VALUE,
    MANAGED_PORT,
    SERVICE_LABEL,
    ListenerInfo,
    LaunchctlSnapshot,
    OllamaServiceManager,
    ProcessInspector,
    parse_lsof_f_output,
    ServiceConfigInvalid,
    ServiceConflict,
    ServiceRecoveryFailed,
    ServicePaths,
    render_launch_agent,
)


class FakeResponse:
    def __init__(self, payload: object, status: int = 200):
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class FakeLaunchctl:
    def __init__(self, loaded: bool = False, pid: int = 4321):
        self.loaded = loaded
        self.pid = pid
        self.bootstrap_failures = 0
        self.calls: list[tuple[str, ...]] = []

    def is_loaded(self, label=SERVICE_LABEL):
        self.calls.append(("is_loaded", label))
        return self.loaded

    def snapshot(self, label=SERVICE_LABEL):
        self.calls.append(("snapshot", label))
        return LaunchctlSnapshot(self.loaded, self.pid if self.loaded else None)

    def bootstrap(self, path):
        self.calls.append(("bootstrap", str(path)))
        if self.bootstrap_failures:
            self.bootstrap_failures -= 1
            raise RuntimeError("synthetic bootstrap failure")
        self.loaded = True

    def bootout(self, label=SERVICE_LABEL):
        self.calls.append(("bootout", label))
        self.loaded = False

    def kickstart(self, label=SERVICE_LABEL, *, kill=False):
        self.calls.append(("kickstart", label, str(kill)))
        self.loaded = True


class FakeInspector:
    def __init__(self, listeners=()):
        self.current = tuple(listeners)

    def listeners(self, port=MANAGED_PORT):
        return self.current


class OllamaServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name).resolve()
        self.paths = ServicePaths(self.home)
        self.binary = str((self.home / "bin" / "ollama").resolve())
        Path(self.binary).parent.mkdir()
        Path(self.binary).write_text("synthetic CLI fixture\n", encoding="utf-8")
        self.launchctl = FakeLaunchctl()
        self.inspector = FakeInspector()
        self.environ = {"OLLAMA_HOST": MANAGED_HOST_VALUE}
        self.version = Mock(return_value=Mock(returncode=0, stdout="ollama version 0.32.1\n"))
        self.opener = Mock(
            side_effect=[
                FakeResponse({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_APPROVED_DIGEST}]}),
                FakeResponse({"models": []}),
            ]
        )
        self.manager = OllamaServiceManager(
            paths=self.paths,
            ollama_binary=self.binary,
            launchctl=self.launchctl,
            process_inspector=self.inspector,
            opener=self.opener,
            version_runner=self.version,
            environ=self.environ,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def install(self):
        self.manager.install()
        self.assertTrue(self.paths.plist.is_file())

    def test_launchagent_is_exact_loopback_single_serve_owner(self):
        payload = render_launch_agent(ollama_binary=self.binary, paths=self.paths)
        self.assertEqual([self.binary, "serve"], payload["ProgramArguments"])
        self.assertEqual(
            {"OLLAMA_HOST": MANAGED_HOST_VALUE, "OLLAMA_MAX_LOADED_MODELS": "1"},
            payload["EnvironmentVariables"],
        )
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual({"SuccessfulExit": False}, payload["KeepAlive"])
        self.assertEqual(10, payload["ThrottleInterval"])
        self.assertNotIn("create", json.dumps(payload))
        self.assertNotIn("pull", json.dumps(payload))
        self.assertNotIn("generate", json.dumps(payload))

    def test_install_is_atomic_and_idempotent(self):
        self.install()
        first = self.paths.plist.read_bytes()
        self.manager.install()
        self.assertEqual(first, self.paths.plist.read_bytes())
        self.assertEqual(1, sum(call[0] == "bootstrap" for call in self.launchctl.calls))
        self.assertFalse(any(path.name.endswith(".tmp") for path in self.paths.launch_agents.iterdir()))
        parsed = plistlib.loads(first)
        self.assertEqual(SERVICE_LABEL, parsed["Label"])

    def test_install_enforces_user_only_permissions_on_exact_managed_paths(self):
        self.paths.log_root.mkdir(parents=True)
        self.paths.log_root.chmod(0o755)
        self.paths.stdout_log.write_text("old\n", encoding="utf-8")
        self.paths.stderr_log.write_text("old\n", encoding="utf-8")
        self.paths.stdout_log.chmod(0o644)
        self.paths.stderr_log.chmod(0o666)
        self.install()
        self.assertEqual(0o700, stat.S_IMODE(self.paths.log_root.stat().st_mode))
        self.assertEqual(0o600, stat.S_IMODE(self.paths.stdout_log.stat().st_mode))
        self.assertEqual(0o600, stat.S_IMODE(self.paths.stderr_log.stat().st_mode))
        self.assertEqual(0o600, stat.S_IMODE(self.paths.plist.stat().st_mode))

    def test_install_rolls_back_plist_and_loaded_state_when_bootstrap_fails(self):
        self.install()
        prior = self.paths.plist.read_bytes()
        replacement_binary = self.home / "bin" / "ollama-replacement"
        replacement_binary.write_text("synthetic CLI fixture\n", encoding="utf-8")
        self.manager.ollama_binary = str(replacement_binary)
        self.launchctl.bootstrap_failures = 1
        with self.assertRaises(RuntimeError):
            self.manager.install()
        self.assertEqual(prior, self.paths.plist.read_bytes())
        self.assertTrue(self.launchctl.loaded)
        self.assertFalse(any(path.name.endswith(".tmp") for path in self.paths.launch_agents.iterdir()))

    def test_remove_is_narrow_and_keeps_logs_and_unrelated_files(self):
        self.install()
        self.paths.log_root.mkdir(parents=True, exist_ok=True)
        log = self.paths.log_root / "ollama.stderr.log"
        log.write_text("synthetic log\n", encoding="utf-8")
        unrelated = self.home / "unrelated-model-marker"
        unrelated.write_text("keep", encoding="utf-8")
        self.manager.remove()
        self.assertFalse(self.paths.plist.exists())
        self.assertTrue(log.exists())
        self.assertTrue(unrelated.exists())
        lifecycle_calls = [call[0] for call in self.launchctl.calls if call[0] != "is_loaded"]
        self.assertEqual("bootout", lifecycle_calls[-1])

    def test_remove_allows_malformed_exact_managed_plist_but_not_symlink(self):
        self.paths.launch_agents.mkdir(parents=True)
        self.paths.plist.write_bytes(b"malformed plist")
        self.manager.remove()
        self.assertFalse(self.paths.plist.exists())

        target = self.paths.launch_agents / "unrelated.plist"
        target.write_bytes(b"keep")
        self.paths.plist.symlink_to(target)
        with self.assertRaises(ServiceConfigInvalid):
            self.manager.remove()
        self.assertTrue(target.exists())

    def test_status_healthy_requires_exact_listener_alias_digest_and_settings(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 4321),)
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertTrue(status.healthy)
        self.assertEqual("exact-loopback", status.listener_scope)
        self.assertTrue(status.alias_present)
        self.assertTrue(status.alias_digest_matches)
        self.assertFalse(status.model_loaded)
        self.assertEqual("ollama version 0.32.1", status.ollama_version)

    def test_status_rejects_wildcard_and_duplicate_or_gui_listener(self):
        self.install()
        self.inspector.current = (
            ListenerInfo("0.0.0.0", MANAGED_PORT, "Ollama.app", 8765),
            ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 4321),
        )
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(status.healthy)
        self.assertEqual("wildcard", status.listener_scope)
        self.assertTrue(status.listener_conflict)
        self.assertTrue(status.ollama_app_conflict)

    def test_loaded_service_pid_mismatch_is_unmanaged_even_with_correct_alias_digest(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 9999),)
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(status.healthy)
        self.assertFalse(status.managed_pid_owns_listener)
        self.assertTrue(status.listener_conflict)

    def test_exact_loopback_gui_on_11434_is_allowed_by_coordination(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", 11434, "Ollama.app", 4321),)
        self.manager.start()

    def test_status_reports_default_port_competition_separately_from_managed_port(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", 11434, "ollama", 9876),)
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(status.healthy)
        self.assertFalse(status.listener_conflict)
        self.assertFalse(status.default_port_conflict)

    def test_status_reports_missing_alias_and_wrong_digest_as_unhealthy(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 4321),)
        self.manager.opener = Mock(return_value=FakeResponse({"models": [{"name": "other", "digest": "x"}]}))
        missing = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(missing.healthy)
        self.assertFalse(missing.alias_present)

        self.manager.opener = Mock(return_value=FakeResponse({"models": [{"name": OLLAMA_MODEL, "digest": "wrong"}]}))
        wrong = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(wrong.healthy)
        self.assertTrue(wrong.alias_present)
        self.assertFalse(wrong.alias_digest_matches)

    def test_status_is_unhealthy_when_uninstalled_or_stopped(self):
        self.inspector.current = ()
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(status.healthy)
        self.assertFalse(status.installed)
        self.install()
        self.launchctl.loaded = False
        self.manager.opener = Mock(
            return_value=FakeResponse({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_APPROVED_DIGEST}]}),
        )
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertFalse(status.healthy)
        self.assertFalse(status.loaded)

    def test_status_warns_on_terminal_drift_without_printing_values(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 4321),)
        self.manager.environ = {"OLLAMA_KEEP_ALIVE": "-1"}
        status = self.manager.status(MANAGED_ENDPOINT)
        self.assertIn("terminal OLLAMA_KEEP_ALIVE=-1 is configured", status.warnings)
        self.assertIn("terminal OLLAMA_HOST is not set", " ".join(status.warnings))

    def test_recovery_kickstarts_once_and_polls_until_digest_valid(self):
        self.install()
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 4321),)
        self.manager.opener = Mock(
            side_effect=[
                urllib.error.URLError("connection refused"),
                FakeResponse({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_APPROVED_DIGEST}]}),
            ]
        )
        self.assertTrue(self.manager.recover(MANAGED_ENDPOINT, sleep=lambda _seconds: None, monotonic=lambda: 0.0))
        self.assertEqual(1, sum(call[0] == "kickstart" for call in self.launchctl.calls))

    def test_recovery_is_refused_for_endpoint_mismatch_or_conflict(self):
        self.install()
        with self.assertRaises(ServiceConfigInvalid):
            self.manager.recover("http://localhost:11434/api/generate")
        self.inspector.current = (ListenerInfo("127.0.0.1", MANAGED_PORT, "ollama", 8765),)
        with self.assertRaises(ServiceConflict):
            self.manager.recover(MANAGED_ENDPOINT)

    def test_recovery_is_bounded_after_one_kickstart_attempt(self):
        self.install()
        self.inspector.current = ()
        self.manager.opener = Mock(side_effect=urllib.error.URLError("connection refused"))
        with self.assertRaises(ServiceRecoveryFailed):
            self.manager.recover(
                MANAGED_ENDPOINT,
                sleep=lambda _seconds: None,
                monotonic=lambda: 0.0,
            )
        self.assertEqual(1, sum(call[0] == "kickstart" for call in self.launchctl.calls))
        self.assertEqual(21, self.manager.opener.call_count)


class LsofParserTests(unittest.TestCase):
    def test_machine_format_parses_ipv4_ipv6_wildcard_and_multiple_listeners(self):
        output = """p100
collama
f3
TST=LISTEN
n127.0.0.1:11434
p101
collama
f4
n127.0.0.1:11435
TST=LISTEN
p102
collama
f5
TST=LISTEN
n*:11435
p103
collama
f6
TST=LISTEN
n[::]:11435
p104
collama
f7
TST=LISTEN
n[::1]:11435
"""
        listeners = parse_lsof_f_output(output)
        self.assertEqual(
            [
                (100, "127.0.0.1", 11434),
                (101, "127.0.0.1", 11435),
                (102, "*", 11435),
                (103, "::", 11435),
                (104, "::1", 11435),
            ],
            [(item.pid, item.host, item.port) for item in listeners],
        )

    def test_machine_format_ignores_malformed_empty_and_non_listening_records(self):
        output = """pbad
collama
f1
TST=LISTEN
n127.0.0.1:11435
p200
collama
f2
TST=ESTABLISHED
n127.0.0.1:11435
p201
collama
f3
TST=LISTEN
nnot-an-endpoint
p202
collama
f4
TST=LISTEN
n127.0.0.1:not-a-port
"""
        self.assertEqual((), parse_lsof_f_output(""))
        self.assertEqual((), parse_lsof_f_output(output))

    def test_process_inspector_uses_machine_format_and_one_command(self):
        completed = Mock(returncode=0, stdout="p300\ncollama\nf1\nTST=LISTEN\nn127.0.0.1:11435\n")
        runner = Mock(return_value=completed)
        inspector = ProcessInspector(runner=runner)
        listeners = inspector.snapshot((11435, 11434))
        self.assertEqual((11435,), tuple(item.port for item in listeners))
        command = runner.call_args.args[0]
        self.assertIn("-FpcfnPT", command)
        self.assertNotIn("-nP", command[3:])

    def test_launchctl_snapshot_extracts_user_domain_pid_without_exposing_output(self):
        from ollama_service import LaunchctlAdapter

        runner = Mock(
            return_value=Mock(
                returncode=0,
                stdout="gui/501/com.meetingintel.ollama = {\n\tpid = 777\n}\n",
            )
        )
        snapshot = LaunchctlAdapter(runner=runner, uid=501).snapshot()
        self.assertEqual(LaunchctlSnapshot(True, 777), snapshot)
        self.assertEqual(
            ["launchctl", "print", "gui/501/com.meetingintel.ollama"],
            runner.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
