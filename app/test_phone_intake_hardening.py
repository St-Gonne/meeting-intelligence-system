from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import phone_drive_fetch as fetch
import phone_fetch_scheduler_runner as scheduler_runner


class PhoneIntakeHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "staging"
        self.ledger = self.root / "state" / "ledger.json"
        self.lock = self.root / "state" / "fetch.lock"
        self.remote_root = "test-drive:phone-recordings"
        self.current = datetime(2026, 7, 16, 9, 0, tzinfo=fetch.IST)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def clock(self) -> datetime:
        return self.current

    def config(self, **kwargs) -> fetch.FetchConfig:
        return fetch.FetchConfig(
            self.remote_root, self.destination, self.ledger, False,
            lock_path=self.lock, **kwargs,
        )

    def listing(self, *, remote_id: str = "id-1", path: str = "2026/07/16/recording.m4a", modified: str = "2026-07-16T03:30:00Z") -> str:
        return json.dumps([{
            "Path": path,
            "ID": remote_id,
            "Size": 5,
            "ModTime": modified,
            "Revision": "rev-1",
            "Hashes": {"md5": "a5ca0b5894324f8bb54bb9fffad29d1e"},
        }])

    def runner(self, listing: str, *, calls=None):
        calls = [] if calls is None else calls

        def run(args, **kwargs):
            calls.append(args)
            if args[1] == "lsjson":
                return subprocess.CompletedProcess(args, 0, listing, "")
            if args[1] == "copyto":
                Path(args[-1]).write_bytes(b"audio")
                return subprocess.CompletedProcess(args, 0, "", "")
            raise AssertionError(args)

        return run, calls

    def scheduler_paths(self, name="scheduler"):
        base = self.root / name
        base.mkdir(parents=True)
        base.chmod(0o700)
        project = base / "project"
        project.mkdir()
        project.chmod(0o700)
        (project / "meetingintel_cli.py").write_text("# test")
        (project / "phone_fetch_launchd_wrapper.sh").write_text("#!/bin/sh\n")
        (project / "phone_fetch_scheduler_runner.py").write_text("# test\n")
        (project / "phone_fetch_launchd_wrapper.sh").chmod(0o700)
        (project / "phone_fetch_scheduler_runner.py").chmod(0o700)
        executable = base / "python"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
        logs = base / "logs"
        logs.mkdir()
        logs.chmod(0o700)
        output = logs / "fetch.log"
        error = logs / "fetch.error.log"
        output.write_text("")
        error.write_text("")
        output.chmod(0o600)
        error.chmod(0o600)
        agent = base / "LaunchAgents" / "managed.plist"
        agent.parent.mkdir()
        agent.write_bytes(fetch.build_daily_launch_agent(
            7,
            0,
            log_path=output,
            error_log_path=error,
            receipt_path=base / "receipt.json",
            project_root=project,
            python_path=executable,
            runtime_root=project,
        ))
        agent.chmod(0o600)
        return agent, project, executable, output, error

    def good_scheduler_diagnosis(self, **kwargs):
        agent, project, _executable, output, error = self.scheduler_paths()
        receipt = self.root / "scheduler" / "receipt.json"
        fetch.write_scheduler_receipt(receipt, success=True, category="success", exit_code=0, clock=self.clock)
        return fetch.scheduler_diagnose(
            agent_path=agent,
            project_root=project,
            log_path=output,
            error_log_path=error,
            receipt_path=receipt,
            trusted_roots=(self.root / "scheduler",),
            launchd_snapshot={"loaded": True, "output": "runs = 6\nlast exit code = 0\nstate = running\n"},
            which=lambda name, path=None: "/controlled/rclone" if name == "rclone" else None,
            **kwargs,
        )

    def test_first_seen_is_pending_then_settles_after_second_unchanged_observation(self):
        runner, _ = self.runner(self.listing())
        first = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertEqual((1, 0), (first.pending_settlement, first.settled_ready))
        self.current += timedelta(minutes=10)
        runner, calls = self.runner(self.listing())
        second = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertEqual((0, 1), (second.pending_settlement, second.settled_ready))
        self.assertEqual(1, len(calls))
        self.assertEqual("settled_ready", json.loads(self.ledger.read_text())["records"]["id-1"]["status"])

    def test_changed_revision_restarts_settlement(self):
        runner, _ = self.runner(self.listing())
        fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.current += timedelta(minutes=10)
        runner, _ = self.runner(self.listing(modified="2026-07-16T03:31:00Z"))
        summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertEqual(1, summary.changed)
        self.assertEqual(1, summary.pending_settlement)
        record = json.loads(self.ledger.read_text())["records"]["id-1"]
        self.assertEqual("pending_settlement", record["status"])
        self.assertEqual(self.current.astimezone(fetch.IST).isoformat(), record["first_observed_at"])

    def test_live_lock_excludes_second_run_without_ledger_change(self):
        before = {"schema_version": 1, "records": {}}
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text(json.dumps(before))
        with fetch.FetchRunLock(self.lock, clock=self.clock, stale_seconds=3600, process_alive=lambda pid: True):
            runner, calls = self.runner(self.listing())
            summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertTrue(summary.already_running)
        self.assertEqual([], calls)
        self.assertEqual(before, json.loads(self.ledger.read_text()))

    def test_stale_lock_is_recovered_once(self):
        self.lock.parent.mkdir(parents=True)
        stale = (self.current - timedelta(hours=2)).isoformat()
        self.lock.write_text(json.dumps({"pid": 12345, "created_at": stale}))
        runner, _ = self.runner(self.listing())
        summary = fetch.fetch_recordings(self.config(lock_stale_seconds=60), runner=runner, clock=self.clock)
        self.assertEqual(1, summary.downloaded)
        self.assertFalse(self.lock.exists())

    def test_same_drive_id_moved_to_new_path_is_fail_closed(self):
        runner, _ = self.runner(self.listing())
        fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        moved = self.listing(path="2026/07/16/moved.m4a")
        runner, calls = self.runner(moved)
        summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertEqual(1, summary.requires_operator_decision)
        self.assertEqual(1, len(calls))
        self.assertFalse((self.destination / "2026/07/16/moved.m4a").exists())
        record = json.loads(self.ledger.read_text())["records"]["id-1"]
        self.assertEqual("remote_path_changed", record["failure"])

    def test_transient_listing_retries_with_bounded_backoff(self):
        calls = []
        attempts = []

        def runner(args, **kwargs):
            calls.append(args)
            if len(calls) < 3:
                raise fetch.RcloneCommandError(args, 28, "connection timed out")
            return subprocess.CompletedProcess(args, 0, "[]", "")

        summary = fetch.fetch_recordings(
            self.config(retry_attempts=3, retry_base_seconds=2, retry_jitter_seconds=0),
            runner=runner, clock=self.clock, sleep=attempts.append,
        )
        self.assertEqual(0, summary.failed)
        self.assertEqual([2, 4], attempts)
        self.assertEqual(3, len(calls))

    def test_auth_failure_is_not_retried_and_is_recorded(self):
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            raise fetch.RcloneCommandError(args, 401, "unauthorized")

        summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock, sleep=lambda _: self.fail("slept"))
        self.assertEqual(1, summary.failed)
        self.assertEqual(1, len(calls))
        failure = json.loads(self.ledger.read_text())["last_listing_failure"]["failure"]
        self.assertIn("unauthorized", failure)

    def test_checksum_failure_is_not_retried(self):
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            if args[1] == "lsjson":
                return subprocess.CompletedProcess(args, 0, self.listing(), "")
            Path(args[-1]).write_bytes(b"wrong")
            return subprocess.CompletedProcess(args, 0, "", "")

        summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock, sleep=lambda _: self.fail("slept"))
        self.assertEqual(1, summary.failed)
        self.assertEqual(2, len(calls))

    def test_download_failure_details_do_not_expose_remote_root(self):
        def runner(args, **kwargs):
            if args[1] == "lsjson":
                return subprocess.CompletedProcess(args, 0, self.listing(), "")
            raise fetch.RcloneCommandError(args, 1, f"failed at {self.remote_root}/private-file")

        summary = fetch.fetch_recordings(self.config(), runner=runner, clock=self.clock)
        self.assertEqual(1, summary.failed)
        failure = json.loads(self.ledger.read_text())["records"]["id-1"]["failure"]
        self.assertNotIn(self.remote_root, failure)
        self.assertIn("<configured-phone-folder>", failure)

    def test_status_reports_only_safe_counts_and_decision_references(self):
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text(json.dumps({"schema_version": 1, "records": {
            "a": {"status": "settled_ready"},
            "b": {"status": "pending_settlement"},
            "c": {"status": "failed"},
            "d": {"status": "requires_operator_decision", "local_relative_path": "2026/07/16/opaque.m4a"},
        }}))
        status = fetch.summarize_fetch_status(self.ledger)
        self.assertEqual((1, 1, 1, 1), (status.settled_ready, status.pending_settlement, status.failed, status.requires_operator_decision))
        self.assertEqual(("2026/07/16/opaque.m4a",), status.references)

    def test_scheduler_renders_login_and_two_hour_schedule(self):
        payload = plistlib.loads(fetch.build_daily_launch_agent(7, 0, log_path=self.root / "logs" / "out.log", error_log_path=self.root / "logs" / "err.log"))
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(7200, payload["StartInterval"])
        self.assertNotIn("StartCalendarInterval", payload)
        self.assertEqual("/bin/sh", payload["ProgramArguments"][0])
        self.assertEqual("sync", payload["ProgramArguments"][-1])
        self.assertEqual(fetch.SCHEDULER_CONTROLLED_PATH, payload["EnvironmentVariables"]["PATH"])

    def test_scheduler_command_is_absolute_and_probe_is_no_drive(self):
        command = scheduler_runner.build_command("/trusted/python", "/project/meetingintel_cli.py", "probe")
        self.assertEqual(
            ["/trusted/python", "/project/meetingintel_cli.py", "sync", "scheduler", "probe-command"],
            command,
        )
        with patch("meetingintel_cli.fetch_recordings") as fetch_recordings, patch("meetingintel_cli.discover_candidates") as discover:
            import meetingintel_cli as cli
            self.assertEqual(0, cli.run_cli(["sync", "scheduler", "probe-command"]))
        fetch_recordings.assert_not_called()
        discover.assert_not_called()

    def test_scheduler_wrapper_receipts_cover_wrapper_python_failure_and_command_success(self):
        base = self.root / "wrapper-states"
        base.mkdir()
        project = base / "project"
        project.mkdir()
        cli_path = project / "meetingintel_cli.py"
        cli_path.write_text("# test")
        wrapper = project / "phone_fetch_launchd_wrapper.sh"
        wrapper.write_text(fetch.DEFAULT_SCHEDULER_WRAPPER_PATH.read_text())
        wrapper.chmod(0o600)
        runner_path = project / "phone_fetch_scheduler_runner.py"
        runner_path.write_text("# test")
        fake_python = base / "fake-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n")
        fake_python.chmod(0o700)
        env = {"PATH": fetch.SCHEDULER_CONTROLLED_PATH, "HOME": str(Path.home())}

        wrapper_receipt = base / "wrapper.json"
        started = subprocess.run(
            ["/bin/sh", str(wrapper), str(fake_python), str(runner_path), str(cli_path), str(wrapper_receipt), "sync"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, started.returncode)
        self.assertEqual("wrapper_started", fetch.load_scheduler_receipt(wrapper_receipt)["stage"])

        missing_receipt = base / "python-failure.json"
        missing = subprocess.run(
            ["/bin/sh", str(wrapper), str(base / "missing-python"), str(runner_path), str(cli_path), str(missing_receipt), "sync"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(127, missing.returncode)
        failed = fetch.load_scheduler_receipt(missing_receipt)
        self.assertEqual(("completed", "python_exec_failed"), (failed["stage"], failed["category"]))

        success_receipt = base / "success.json"
        calls = []

        def successful_runner(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "", "")

        self.assertEqual(
            0,
            scheduler_runner.run(
                [str(fake_python), str(cli_path), str(success_receipt), "probe"],
                runner=successful_runner,
                clock=lambda: "2026-07-23T00:00:00+00:00",
            ),
        )
        receipt = fetch.load_scheduler_receipt(success_receipt)
        self.assertEqual(("completed", "probe_success"), (receipt["stage"], receipt["category"]))
        self.assertEqual(
            {"schema_version", "stage", "mode", "updated_at", "outcome", "category", "exit_code"},
            set(json.loads(success_receipt.read_text())),
        )
        self.assertEqual([str(fake_python), str(cli_path), "sync", "scheduler", "probe-command"], calls[0][0])

        command_failure_receipt = base / "command-failure.json"
        self.assertEqual(
            1,
            scheduler_runner.run(
                [str(fake_python), str(cli_path), str(command_failure_receipt), "sync"],
                runner=lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", ""),
                clock=lambda: "2026-07-23T00:00:00+00:00",
            ),
        )
        command_failure = fetch.load_scheduler_receipt(command_failure_receipt)
        self.assertEqual(("completed", "meetingintel_failure"), (command_failure["stage"], command_failure["category"]))

    def test_scheduler_probe_uses_same_managed_label_with_injected_launchd(self):
        agent, project, executable, output, error = self.scheduler_paths("probe-install")
        receipt = self.root / "probe-install" / "receipt.json"
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        self.assertEqual(
            0,
            fetch.run_scheduler_probe(
                agent_path=agent,
                log_path=output,
                error_log_path=error,
                receipt_path=receipt,
                runner=runner,
            ),
        )
        payload = plistlib.loads(agent.read_bytes())
        self.assertEqual("probe", payload["ProgramArguments"][-1])
        self.assertEqual("kickstart", calls[-1][1])

    def test_scheduler_status_classifies_exit_78_without_wrapper_receipt(self):
        agent, project, _executable, output, error = self.scheduler_paths("exit-78")
        receipt = self.root / "exit-78" / "receipt.json"
        report = fetch.scheduler_diagnose(
            agent_path=agent,
            project_root=project,
            log_path=output,
            error_log_path=error,
            receipt_path=receipt,
            trusted_roots=(self.root / "exit-78",),
            launchd_snapshot={"loaded": True, "output": "runs = 3\nlast exit code = 78: EX_CONFIG\nstate = not running\n"},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("launchd_launch_failure", report["runtime_outcome"])
        self.assertEqual(78, report["last_exit_code"])
        self.assertEqual("EX_CONFIG", report["last_exit_label"])
        self.assertFalse(report["healthy"])
        runtime = next(item for item in report["checks"] if item["category"] == "launchd_runtime")
        self.assertIn("before the wrapper-start receipt", runtime["detail"])

    def test_scheduler_external_runtime_root_is_private_and_atomic(self):
        base = self.root / "external-install"
        base.mkdir()
        base.chmod(0o700)
        project = base / "project"
        project.mkdir()
        project.chmod(0o700)
        (project / "meetingintel_cli.py").write_text("# test")
        (project / "phone_fetch_launchd_wrapper.sh").write_text("#!/bin/sh\n")
        (project / "phone_fetch_scheduler_runner.py").write_text("# runner\n")
        (project / "phone_fetch_launchd_wrapper.sh").chmod(0o700)
        (project / "phone_fetch_scheduler_runner.py").chmod(0o700)
        runtime = base / "runtime"
        agent = base / "LaunchAgents" / "managed.plist"
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        fetch.install_daily_schedule(
            7,
            0,
            agent_path=agent,
            project_root=project,
            python_path=Path(sys.executable),
            runtime_root=runtime,
            runner=runner,
        )
        payload = plistlib.loads(agent.read_bytes())
        self.assertEqual(str(runtime), payload["WorkingDirectory"])
        self.assertTrue(payload["ProgramArguments"][1].startswith(str(runtime)))
        self.assertTrue(payload["ProgramArguments"][3].startswith(str(runtime)))
        self.assertTrue(payload["ProgramArguments"][5].startswith(str(runtime)))
        self.assertFalse((runtime / "phone_fetch_launchd_wrapper.sh").is_symlink())
        self.assertFalse((runtime / "phone_fetch_scheduler_runner.py").is_symlink())
        self.assertEqual(0o700, runtime.stat().st_mode & 0o777)
        self.assertEqual(0o700, (runtime / "phone_fetch_launchd_wrapper.sh").stat().st_mode & 0o777)
        self.assertEqual(0o700, (runtime / "phone_fetch_scheduler_runner.py").stat().st_mode & 0o777)
        self.assertFalse(list(runtime.glob("*.tmp")))
        self.assertEqual(["bootout", "bootstrap"], [call[1] for call in calls])

    def test_scheduler_runtime_root_rejects_symlink_and_broad_permissions(self):
        target = self.root / "runtime-target"
        target.mkdir()
        target.chmod(0o700)
        linked = self.root / "runtime-link"
        linked.symlink_to(target)
        with self.assertRaises(RuntimeError):
            fetch.build_daily_launch_agent(7, 0, runtime_root=linked)

        broad = self.root / "runtime-broad"
        broad.mkdir()
        broad.chmod(0o755)
        with self.assertRaises(PermissionError):
            fetch.build_daily_launch_agent(7, 0, runtime_root=broad)

    def test_scheduler_stale_receipt_cannot_pass_after_install_generation(self):
        agent, project, _executable, output, error = self.scheduler_paths("stale-receipt")
        receipt = self.root / "stale-receipt" / "receipt.json"
        fetch.write_scheduler_receipt(
            receipt,
            success=True,
            category="success",
            exit_code=0,
            clock=lambda: datetime(2026, 7, 16, 9, 0, tzinfo=fetch.IST),
        )
        payload = plistlib.loads(agent.read_bytes())
        payload["EnvironmentVariables"][fetch.SCHEDULER_RUNTIME_ENV] = "2026-07-23T00:00:00+05:30"
        agent.write_bytes(plistlib.dumps(payload))
        report = fetch.scheduler_diagnose(
            agent_path=agent,
            project_root=project,
            log_path=output,
            error_log_path=error,
            receipt_path=receipt,
            trusted_roots=(self.root / "stale-receipt",),
            launchd_snapshot={"loaded": True, "output": "runs = 1\nlast exit code = 0\nstate = not running\n"},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("unknown", report["runtime_outcome"])
        self.assertFalse(report["healthy"])
        self.assertIn("predates", next(item for item in report["checks"] if item["category"] == "launchd_runtime")["detail"])

    def test_scheduler_probe_failure_surfaces_project_cli_stage(self):
        receipt = self.root / "probe-project-cli.json"
        result = scheduler_runner.run(
            ["/trusted/python", "/project/meetingintel_cli.py", str(receipt), "probe"],
            runner=lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", ""),
            clock=lambda: "2026-07-23T00:00:00+00:00",
        )
        self.assertEqual(1, result)
        self.assertEqual("project_cli_failure", fetch.load_scheduler_receipt(receipt)["category"])

    def test_scheduler_status_surfaces_python_and_meetingintel_failure_stages(self):
        agent, project, _executable, output, error = self.scheduler_paths("stage-status")
        receipt = self.root / "stage-status" / "receipt.json"
        snapshot = {"loaded": True, "output": "runs = 1\nlast exit code = 1\nstate = not running\n"}
        fetch.write_scheduler_receipt(
            receipt, success=None, stage="wrapper_started", category="wrapper_started", exit_code=None, clock=self.clock,
        )
        wrapper = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, launchd_snapshot=snapshot, trusted_roots=(self.root / "stage-status",),
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("wrapper_started", wrapper["runtime_outcome"])

        fetch.write_scheduler_receipt(
            receipt, success=False, category="python_exec_failed", exit_code=127, clock=self.clock,
        )
        python_failure = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, launchd_snapshot=snapshot, trusted_roots=(self.root / "stage-status",),
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("python_exec_failure", python_failure["runtime_outcome"])

        fetch.write_scheduler_receipt(
            receipt, success=False, category="meetingintel_failure", exit_code=1, clock=self.clock,
        )
        meetingintel_failure = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, launchd_snapshot=snapshot, trusted_roots=(self.root / "stage-status",),
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("meetingintel_failure", meetingintel_failure["runtime_outcome"])

    def test_scheduler_does_not_reuse_receipt_from_a_different_mode(self):
        agent, project, _executable, output, error = self.scheduler_paths("mode-mismatch")
        receipt = self.root / "mode-mismatch" / "receipt.json"
        fetch.write_scheduler_receipt(receipt, success=True, category="success", exit_code=0, clock=self.clock)
        payload = plistlib.loads(agent.read_bytes())
        payload["ProgramArguments"][-1] = "probe"
        agent.write_bytes(plistlib.dumps(payload))
        report = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, launchd_snapshot={"loaded": True, "output": "runs = 1\nlast exit code = 0\nstate = not running\n"},
            trusted_roots=(self.root / "mode-mismatch",), which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("unknown", report["runtime_outcome"])
        self.assertFalse(report["healthy"])

    def test_scheduler_status_is_read_only_and_injectable(self):
        agent = self.root / "LaunchAgents" / "managed.plist"
        agent.parent.mkdir(parents=True)
        agent.write_bytes(b"plist")
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "loaded", "")

        state = fetch.scheduler_status(agent_path=agent, runner=runner)
        self.assertTrue(state["installed"])
        self.assertTrue(state["loaded"])
        self.assertIn("checks", state)
        self.assertEqual(b"plist", agent.read_bytes())
        self.assertEqual("print", calls[0][1])

    def test_scheduler_diagnose_good_configuration_is_healthy_without_external_calls(self):
        report = self.good_scheduler_diagnosis()
        self.assertTrue(report["healthy"])
        self.assertEqual(6, report["run_count"])
        self.assertEqual(0, report["last_exit_code"])
        self.assertEqual("running", report["state"])
        self.assertEqual("not_checked", next(item for item in report["checks"] if item["category"] == "drive_authorization")["status"])

    def test_scheduler_diagnose_rejects_malformed_plist_and_missing_executable(self):
        agent, project, _executable, output, error = self.scheduler_paths("missing-executable")
        agent.write_bytes(b"not a plist")
        malformed = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            launchd_snapshot={"loaded": True, "output": ""},
            which=lambda *args, **kwargs: None,
        )
        self.assertFalse(malformed["healthy"])
        self.assertEqual("fail", next(item for item in malformed["checks"] if item["category"] == "launchd_configuration")["status"])

        agent, project, _executable, output, error = self.scheduler_paths("missing-executable-valid")
        payload = plistlib.loads(agent.read_bytes())
        payload["ProgramArguments"][2] = str(self.root / "missing-python")
        agent.write_bytes(plistlib.dumps(payload))
        missing = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            trusted_roots=(self.root,),
            launchd_snapshot={"loaded": True, "output": ""},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertFalse(missing["healthy"])
        self.assertEqual("interpreter target is dangling", next(item for item in missing["checks"] if item["category"] == "executable")["detail"])

    def test_scheduler_diagnose_rejects_missing_working_directory_and_symlinks(self):
        agent, project, _executable, output, error = self.scheduler_paths("missing-workdir")
        (project / "meetingintel_cli.py").unlink()
        (project / "phone_fetch_launchd_wrapper.sh").unlink()
        (project / "phone_fetch_scheduler_runner.py").unlink()
        project.rmdir()
        missing_workdir = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            launchd_snapshot={"loaded": True, "output": ""},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertFalse(missing_workdir["healthy"])
        self.assertTrue(any(item["category"] == "protected_path" and item["status"] == "fail" for item in missing_workdir["checks"]))

        agent, project, _executable, output, error = self.scheduler_paths("symlinked-plist")
        target = self.root / "plist-target"
        target.write_bytes(agent.read_bytes())
        agent.unlink()
        agent.symlink_to(target)
        symlinked = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            launchd_snapshot={"loaded": True, "output": ""},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertFalse(symlinked["healthy"])
        self.assertEqual("managed plist is a symlink", symlinked["checks"][0]["detail"])
        target.write_bytes(b"not changed")
        self.assertEqual(b"not changed", target.read_bytes())

    def test_scheduler_diagnose_rejects_symlinked_runner(self):
        agent, project, _executable, output, error = self.scheduler_paths("symlinked-runner")
        runner = project / "phone_fetch_scheduler_runner.py"
        target = self.root / "unrelated-runner"
        target.write_text("# untouched")
        runner.unlink()
        runner.symlink_to(target)
        report = fetch.scheduler_diagnose(
            agent_path=agent,
            project_root=project,
            log_path=output,
            error_log_path=error,
            receipt_path=self.root / "symlinked-runner" / "receipt.json",
            trusted_roots=(self.root / "symlinked-runner",),
            launchd_snapshot={"loaded": True, "output": "runs = 0\nstate = not running\n"},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertFalse(report["healthy"])
        self.assertTrue(any(item["detail"] == "scheduler runner symlink rejected" for item in report["checks"]))
        self.assertEqual("# untouched", target.read_text())

    def test_scheduler_diagnose_names_environment_and_rclone_failures(self):
        agent, project, _executable, output, error = self.scheduler_paths()
        payload = plistlib.loads(agent.read_bytes())
        payload["EnvironmentVariables"]["PATH"] = "/usr/bin"
        agent.write_bytes(plistlib.dumps(payload))
        receipt = self.root / "scheduler" / "receipt.json"
        fetch.write_scheduler_receipt(receipt, success=True, category="success", exit_code=0, clock=self.clock)
        report = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, trusted_roots=(self.root / "scheduler",),
            launchd_snapshot={"loaded": True, "output": ""},
            which=lambda *args, **kwargs: None,
        )
        self.assertFalse(report["healthy"])
        self.assertEqual({"environment", "rclone"}, {item["category"] for item in report["checks"] if item["status"] == "fail"})

    def test_scheduler_trusted_homebrew_style_symlink_chain_is_allowed(self):
        agent, project, executable, output, error = self.scheduler_paths("trusted-symlink")
        real = executable.with_name("real-python")
        real.write_text("#!/bin/sh\n")
        real.chmod(0o700)
        executable.unlink()
        executable.symlink_to(real)
        receipt = self.root / "trusted-symlink" / "receipt.json"
        fetch.write_scheduler_receipt(receipt, success=True, category="success", exit_code=0, clock=self.clock)
        report = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=receipt, trusted_roots=(self.root / "trusted-symlink",),
            launchd_snapshot={"loaded": True, "output": "last exit code = 0\nstate = not running\n"},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertTrue(report["healthy"])
        self.assertEqual("pass", next(item for item in report["checks"] if item["category"] == "executable")["status"])

    def test_scheduler_rejects_dangling_escaping_cyclic_non_executable_and_non_regular_symlinks(self):
        cases = (
            ("dangling", "dangling"),
            ("escaping", "escaping"),
            ("cyclic", "cyclic"),
            ("non-executable", "non-executable"),
            ("non-regular", "non-regular"),
        )
        for name, expected in cases:
            with self.subTest(name=name):
                agent, project, executable, output, error = self.scheduler_paths(f"symlink-{name}")
                root = self.root / f"symlink-{name}"
                executable.unlink()
                if name == "dangling":
                    executable.symlink_to(root / "missing")
                elif name == "escaping":
                    outside = self.root / "outside-python"
                    outside.write_text("#!/bin/sh\n")
                    outside.chmod(0o700)
                    executable.symlink_to(outside)
                elif name == "cyclic":
                    other = root / "other-python"
                    executable.symlink_to(other)
                    other.symlink_to(executable)
                elif name == "non-executable":
                    target = root / "plain-python"
                    target.write_text("not executable")
                    target.chmod(0o600)
                    executable.symlink_to(target)
                else:
                    target = root / "directory-python"
                    target.mkdir()
                    executable.symlink_to(target)
                ok, detail = fetch.validate_interpreter_path(
                    executable, trusted_roots=(root,), access=os.access,
                )
                self.assertFalse(ok)
                expected_detail = {
                    "escaping": "escapes",
                    "non-executable": "not executable",
                    "non-regular": "not a regular",
                }.get(expected, expected)
                self.assertIn(expected_detail, detail)

    def test_scheduler_receipts_distinguish_success_failure_unknown_and_never_run(self):
        agent, project, _executable, output, error = self.scheduler_paths("receipt-states")
        root = self.root / "receipt-states"
        snapshot = {"loaded": True, "output": "runs = 1\nlast exit code = 0\nstate = not running\n"}
        missing = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=root / "receipt.json", trusted_roots=(root,), launchd_snapshot=snapshot,
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("unknown", missing["runtime_outcome"])
        self.assertFalse(missing["healthy"])

        success_path = root / "success.json"
        fetch.write_scheduler_receipt(success_path, success=True, category="success", exit_code=0, clock=self.clock)
        payload = plistlib.loads(agent.read_bytes())
        payload["ProgramArguments"][5] = str(success_path)
        agent.write_bytes(plistlib.dumps(payload))
        success = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=success_path, trusted_roots=(root,), launchd_snapshot=snapshot,
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("confirmed_success", success["runtime_outcome"])
        self.assertTrue(success["healthy"])

        failed_path = root / "failed.json"
        fetch.write_scheduler_receipt(failed_path, success=False, category="fetch_failure", exit_code=1, clock=self.clock)
        payload["ProgramArguments"][5] = str(failed_path)
        agent.write_bytes(plistlib.dumps(payload))
        failed = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=failed_path, trusted_roots=(root,), launchd_snapshot=snapshot,
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("confirmed_failure", failed["runtime_outcome"])
        self.assertFalse(failed["healthy"])

        never = dict(snapshot, output="runs = 0\nstate = not running\n")
        never_path = root / "never.json"
        payload["ProgramArguments"][5] = str(never_path)
        agent.write_bytes(plistlib.dumps(payload))
        never_run = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            receipt_path=never_path, trusted_roots=(root,), launchd_snapshot=never,
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertEqual("never_run", never_run["runtime_outcome"])
        self.assertFalse(never_run["healthy"])

    def test_scheduler_diagnose_reports_unreadable_paths(self):
        agent, project, _executable, output, error = self.scheduler_paths("unreadable")
        report = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            launchd_snapshot={"loaded": True, "output": ""},
            access=lambda _path, _mode: False,
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        self.assertFalse(report["healthy"])
        self.assertTrue(any(item["category"] == "protected_path" and item["detail"].endswith("not readable by the scheduler user") for item in report["checks"]))

    def test_scheduler_status_surfaces_launchd_exit_without_raw_paths_or_logs(self):
        agent, project, _executable, output, error = self.scheduler_paths()

        report = fetch.scheduler_diagnose(
            agent_path=agent, project_root=project, log_path=output, error_log_path=error,
            launchd_snapshot={"loaded": True, "output": "runs = 6\nlast exit code = 78\nstate = not running\n"},
            which=lambda *args, **kwargs: "/controlled/rclone",
        )
        rendered = fetch.format_scheduler_report(report)
        self.assertFalse(report["healthy"])
        self.assertIn("last_exit_code=78", rendered)
        self.assertIn("category=launchd_runtime status=fail", rendered)
        self.assertNotIn(str(project), rendered)
        self.assertNotIn("raw stderr", rendered)

    def test_scheduler_install_rollback_restores_prior_plist_and_private_paths(self):
        agent = self.root / "LaunchAgents" / "managed.plist"
        agent.parent.mkdir(parents=True)
        prior = b"prior plist"
        agent.write_bytes(prior)
        project = self.root / "install-project"
        project.mkdir()
        (project / "meetingintel_cli.py").write_text("# test")
        (project / "phone_fetch_launchd_wrapper.sh").write_text("#!/bin/sh\n")
        (project / "phone_fetch_scheduler_runner.py").write_text("# test\n")
        executable = self.root / "install-python"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
        out_log = self.root / "logs" / "out.log"
        err_log = self.root / "logs" / "err.log"
        receipt = self.root / "state" / "receipt.json"

        def failing_runner(args, **kwargs):
            if args[1] == "bootstrap":
                raise subprocess.CalledProcessError(1, args, stderr="bootstrap failed")
            return subprocess.CompletedProcess(args, 0, "", "")

        with self.assertRaises(subprocess.CalledProcessError):
            fetch.install_daily_schedule(
                7,
                0,
                agent_path=agent,
                log_path=out_log,
                error_log_path=err_log,
                receipt_path=receipt,
                project_root=project,
                python_path=executable,
                runner=failing_runner,
                loaded_before=True,
            )
        self.assertEqual(prior, agent.read_bytes())
        self.assertEqual(0o600, agent.stat().st_mode & 0o777)
        self.assertEqual(0o700, out_log.parent.stat().st_mode & 0o777)
        self.assertEqual(0o600, out_log.stat().st_mode & 0o777)
        self.assertEqual(0o600, err_log.stat().st_mode & 0o777)
        self.assertFalse(list(agent.parent.glob("*.tmp")))

    def test_scheduler_install_and_remove_are_injected_and_narrow(self):
        agent = self.root / "LaunchAgents" / "managed.plist"
        out_log = self.root / "logs" / "out.log"
        err_log = self.root / "logs" / "err.log"
        project = self.root / "install-project"
        project.mkdir()
        (project / "meetingintel_cli.py").write_text("# test")
        (project / "phone_fetch_launchd_wrapper.sh").write_text("#!/bin/sh\n")
        (project / "phone_fetch_scheduler_runner.py").write_text("# test\n")
        executable = self.root / "install-python"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
        receipt = self.root / "state" / "receipt.json"
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        fetch.install_daily_schedule(
            7,
            0,
            agent_path=agent,
            log_path=out_log,
            error_log_path=err_log,
            receipt_path=receipt,
            project_root=project,
            python_path=executable,
            runner=runner,
        )
        self.assertTrue(agent.is_file())
        fetch.remove_daily_schedule(agent_path=agent, runner=runner)
        self.assertFalse(agent.exists())
        self.assertEqual(["bootout", "bootstrap", "bootout"], [call[1] for call in calls])

    def test_scheduler_remove_unlinks_malformed_symlink_without_following_it(self):
        agent = self.root / "LaunchAgents" / "managed.plist"
        target = self.root / "unrelated"
        target.write_text("untouched")
        agent.parent.mkdir(parents=True)
        agent.symlink_to(target)
        fetch.remove_daily_schedule(agent_path=agent, runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""))
        self.assertFalse(agent.exists())
        self.assertEqual("untouched", target.read_text())


if __name__ == "__main__":
    unittest.main()
