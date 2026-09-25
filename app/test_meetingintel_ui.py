"""HTTP and subprocess-contract tests; all sources and workers are synthetic."""
import copy
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import meetingintel_ui as ui

SOURCE = "a" * 64


class Adapter:
    def __init__(self):
        self.item = {"id": SOURCE, "title": "Synthetic recording", "source_kind": "laptop",
                     "created_at": "2026-09-22T09:00:00+05:30", "status": "ready", "category": "ready",
                     "recording_state": "retained", "transcript_state": "saved", "report_state": "not saved",
                     "brief_state": "not saved", "can_process": True, "can_repair_brief": False,
                     "artifacts": [{"kind": "transcript"}], "handoff": "mi inbox"}
        self.snapshot_calls = 0
        self.artifact_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return {"items": [copy.deepcopy(self.item)], "warnings": []}

    def detail(self, source_id):
        if source_id != SOURCE:
            raise ValueError("missing")
        return copy.deepcopy(self.item)

    def artifact(self, source_id, kind):
        self.artifact_calls += 1
        return '<script>alert("untrusted transcript")</script>'


class Process:
    pid = 987654321
    result = None

    def poll(self):
        return self.result


class UIBase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mi-ui-test-")
        self.root = Path(self.temporary.name).resolve()
        self.adapter = Adapter()
        self.calls = []
        self.process = Process()
        self.process.result = None
        self.time = 100.0
        self.events = patch.object(ui, "_emit")
        self.events.start()

        def launch(command, **kwargs):
            self.calls.append((command, kwargs))
            return self.process
        self.state = ui.UIState(adapter=self.adapter, runtime_root=self.root, launcher=launch, clock=lambda: self.time)

    def tearDown(self):
        self.events.stop()
        self.temporary.cleanup()


class UITest(UIBase):
    def test_gpu_deferral_is_visible_and_not_completion(self):
        operation = self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        path = self.root / (operation["id"] + ".result.json")
        path.write_text(json.dumps({"ok": False, "outcome": "deferred"}))
        status = self.state.operation_status()
        self.assertEqual(status["status"], "deferred")
        self.assertEqual(status["outcome"], "deferred")
        self.assertEqual(len(self.calls), 1)

    def test_snapshot_never_opens_artifact_or_launches_worker(self):
        self.assertEqual(len(self.state.snapshot()["items"]), 1)
        self.assertEqual(self.adapter.artifact_calls, 0)
        self.assertEqual(self.calls, [])

    def test_preparation_never_starts_work_and_cancel_consumes_confirmation(self):
        confirmation = self.state.prepare("process", SOURCE)
        self.assertEqual(self.calls, [])
        self.state.cancel_confirmation(confirmation["confirmation_id"])
        with self.assertRaises(ui.UIError) as caught:
            self.state.execute(confirmation["confirmation_id"])
        self.assertEqual(caught.exception.code, "confirmation_expired_or_used")

    def test_worker_gets_exact_args_and_gui_origin(self):
        operation = self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        command, options = self.calls[0]
        self.assertEqual(command[2:5], ["worker", "process", SOURCE])
        self.assertEqual(command[5], "--result-file")
        self.assertEqual(Path(command[6]).parent, self.root)
        self.assertEqual(options["env"]["MI_ORIGIN"], "gui")
        self.assertEqual(options["env"]["MI_OPERATION_ID"], operation["id"])
        self.assertTrue(options["start_new_session"])
        self.assertNotIn("shell", options)

    def test_duplicate_submit_and_other_busy_action_do_not_start_second_worker(self):
        confirmation = self.state.prepare("process", SOURCE)["confirmation_id"]
        self.state.execute(confirmation)
        with self.assertRaises(ui.UIError):
            self.state.execute(confirmation)
        other = self.state.prepare("process", SOURCE)["confirmation_id"]
        with self.assertRaises(ui.UIError) as caught:
            self.state.execute(other)
        self.assertEqual(caught.exception.code, "operation_busy")
        self.assertEqual(len(self.calls), 1)

    def test_refresh_preserves_worker_and_never_restarts(self):
        self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        for _ in range(3):
            self.assertEqual(self.state.snapshot()["operation"]["status"], "running")
        self.assertEqual(len(self.calls), 1)

    def test_stale_selection_revalidated_before_start(self):
        confirmation = self.state.prepare("process", SOURCE)["confirmation_id"]
        self.adapter.item["can_process"] = False
        with self.assertRaises(ui.UIError) as caught:
            self.state.execute(confirmation)
        self.assertEqual(caught.exception.code, "action_no_longer_available")
        self.assertEqual(self.calls, [])

    def test_expired_confirmation_rejected(self):
        confirmation = self.state.prepare("process", SOURCE)["confirmation_id"]
        self.time = 221
        with self.assertRaises(ui.UIError):
            self.state.execute(confirmation)
        self.assertEqual(self.calls, [])

    def test_exit_zero_without_structured_result_is_not_completion(self):
        self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        self.process.result = 0
        self.assertEqual(self.state.operation_status()["status"], "interrupted")
        self.assertEqual(self.state.snapshot()["items"][0]["report_state"], "not saved")

    def test_result_restored_after_server_restart_but_does_not_invent_artifacts(self):
        operation = self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        ui._atomic_json(self.root / (operation["id"] + ".result.json"), {"ok": True, "outcome": "succeeded"})
        restored = ui.UIState(adapter=self.adapter, runtime_root=self.root)
        self.assertEqual(restored.operation_status()["status"], "finished")
        self.assertEqual(restored.snapshot()["items"][0]["report_state"], "not saved")

    def test_crashed_worker_restored_as_interrupted(self):
        self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        with patch.object(ui.os, "kill", side_effect=ProcessLookupError):
            restored = ui.UIState(adapter=self.adapter, runtime_root=self.root)
        self.assertEqual(restored.operation_status()["status"], "interrupted")

    def test_live_pid_after_restart_is_unknown_not_owned_running(self):
        self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        with patch.object(ui.os, "kill"):
            restored = ui.UIState(adapter=self.adapter, runtime_root=self.root)
            self.assertEqual(restored.operation_status()["status"], "unknown")

    def test_invalid_result_is_unknown_not_success(self):
        operation = self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        ui._atomic_json(self.root / (operation["id"] + ".result.json"), {"ok": "true"})
        self.assertEqual(self.state.operation_status()["status"], "unknown")

    def test_private_operation_record_permissions(self):
        self.state.execute(self.state.prepare("process", SOURCE)["confirmation_id"])
        self.assertEqual((self.root / "operation.json").stat().st_mode & 0o777, 0o600)

    def test_symlink_runtime_ancestor_rejected(self):
        target = self.root / "real"
        target.mkdir(mode=0o700)
        link = self.root / "linked"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            ui.UIState(adapter=self.adapter, runtime_root=link / "child")

    def test_action_and_source_path_injection_rejected(self):
        for action, source in [("rm -rf", SOURCE), ("process", "../../secret"), ("process", "a;touch")]:
            with self.subTest(action=action, source=source):
                with self.assertRaises(ui.UIError):
                    self.state.prepare(action, source)
        self.assertEqual(self.calls, [])


class HTTPTest(UIBase):
    def setUp(self):
        super().setUp()
        self.server = ui.UIServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        super().tearDown()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        current = {"X-MI-Token": self.state.token, "Origin": self.server.origin}
        if headers:
            current.update(headers)
        if body is not None:
            current["Content-Type"] = "application/json"
            body = json.dumps(body)
        connection.request(method, path, body=body, headers=current)
        response = connection.getresponse()
        status, data, response_headers = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return status, data, response_headers

    def test_local_static_shell_has_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"MeetingIntel", body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_api_requires_session_and_exact_host(self):
        for headers in [{"X-MI-Token": ""}, {"Host": "evil.example"}, {"Origin": "https://evil.example"}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/snapshot", headers=headers)[0], 403)

    def test_post_requires_origin_and_strict_body(self):
        body = {"action": "process", "source_id": SOURCE}
        self.assertEqual(self.request("POST", "/api/prepare", body, {"Origin": ""})[0], 403)
        self.assertEqual(self.request("POST", "/api/prepare", dict(body, command="anything"))[0], 400)
        self.assertEqual(self.request("POST", "/api/prepare", {"action": [], "source_id": SOURCE})[0], 400)

    def test_http_prepare_execute_replay_and_readonly_refresh(self):
        status, body, _ = self.request("POST", "/api/prepare", {"action": "process", "source_id": SOURCE})
        self.assertEqual(status, 200)
        confirmation = json.loads(body)["confirmation_id"]
        self.assertEqual(self.request("POST", "/api/action", {"confirmation_id": confirmation})[0], 200)
        self.assertEqual(self.request("POST", "/api/action", {"confirmation_id": confirmation})[0], 409)
        for _ in range(2):
            status, body, _ = self.request("GET", "/api/snapshot")
            self.assertEqual(json.loads(body)["operation"]["status"], "running")
        self.assertEqual(len(self.calls), 1)

    def test_traversal_and_arbitrary_artifact_rejected(self):
        for path in ["/../meetingintel_ui.py", "/%2e%2e/meetingintel_ui.py", "/api/detail?id=../../secret", "/api/artifact?id=" + SOURCE + "&kind=../../secret", "/api/artifact?id=" + SOURCE + "&kind=transcript&path=/etc/passwd"]:
            with self.subTest(path=path):
                self.assertIn(self.request("GET", path)[0], {400, 404})

    def test_artifact_is_data_not_html(self):
        status, body, headers = self.request("GET", "/api/artifact?id=" + SOURCE + "&kind=transcript")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertIn("<script>", json.loads(body)["text"])

    def test_second_server_cannot_share_operation_record(self):
        with self.assertRaises(BlockingIOError):
            ui.UIServer(("127.0.0.1", 0), self.state)

    def test_get_cannot_execute_action(self):
        self.assertEqual(self.request("GET", "/api/action")[0], 404)
        self.assertEqual(self.calls, [])

    def test_session_metadata_is_private_and_authenticates_existing_instance(self):
        path = self.root / "session.json"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        session = ui._existing_session(self.root, self.server.server_address[1])
        self.assertEqual(session["instance_id"], self.server.instance_id)
        self.assertEqual(session["token"], self.state.token)
        self.assertEqual(self.request("GET", "/api/session", headers={"X-MI-Token": ""})[0], 403)

    def test_cli_reopens_existing_session_without_new_adapter_or_worker(self):
        with patch.dict(os.environ, {"MI_UI_ROOT": str(self.root)}), patch.object(ui, "UIState", side_effect=AssertionError("must reuse")), patch.object(ui.webbrowser, "open", return_value=True) as opened, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ui.run_cli(["--port", str(self.server.server_address[1])]), 0)
        opened.assert_called_once_with(self.server.origin + "/#token=" + self.state.token)
        self.assertEqual(self.calls, [])

    def test_cli_no_browser_reuses_authenticated_session(self):
        with patch.dict(os.environ, {"MI_UI_ROOT": str(self.root)}), patch.object(ui.webbrowser, "open") as opened, patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(ui.run_cli(["--port", str(self.server.server_address[1]), "--no-browser"]), 0)
        opened.assert_not_called()
        self.assertIn(self.server.origin + "/#token=" + self.state.token, output.getvalue())
        self.assertEqual(self.calls, [])

    def test_reopen_rejects_unexpected_origin_without_network(self):
        path = self.root / "session.json"
        session = json.loads(path.read_text())
        for origin in ["https://evil.example", "http://127.0.0.1:1", self.server.origin + "/other", self.server.origin + "@evil.example"]:
            with self.subTest(origin=origin):
                session["origin"] = origin
                ui._atomic_json(path, session)
                with patch.object(ui.http.client, "HTTPConnection", side_effect=AssertionError("must not connect")):
                    self.assertIsNone(ui._existing_session(self.root, self.server.server_address[1]))

    def test_reopen_rejects_wrong_process_identity_and_session_token(self):
        path = self.root / "session.json"
        session = json.loads(path.read_text())
        altered = dict(session, pid=session["pid"] + 1)
        ui._atomic_json(path, altered)
        with patch.object(ui.os, "kill"):
            self.assertIsNone(ui._existing_session(self.root, self.server.server_address[1]))
        ui._atomic_json(path, dict(session, token="z" * 43))
        self.assertIsNone(ui._existing_session(self.root, self.server.server_address[1]))

    def test_stale_session_without_owned_lock_is_not_reopened(self):
        ui.fcntl.flock(self.server.lockfd, ui.fcntl.LOCK_UN)
        with patch.object(ui.http.client, "HTTPConnection", side_effect=AssertionError("must not connect")):
            self.assertIsNone(ui._existing_session(self.root, self.server.server_address[1]))

    def test_close_removes_only_own_session_metadata(self):
        self.server.shutdown()
        self.server.server_close()
        self.assertFalse((self.root / "session.json").exists())
        replacement = {"origin": self.server.origin, "token": "new-owner-token", "pid": os.getpid(), "instance_id": "b" * 32}
        ui._atomic_json(self.root / "session.json", replacement)
        self.server.server_close()
        self.assertEqual(json.loads((self.root / "session.json").read_text()), replacement)


if __name__ == "__main__":
    unittest.main()
