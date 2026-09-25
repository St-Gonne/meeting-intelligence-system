from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import meetingintel_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.root = self.parent / "runtime"
        self.environment = patch.dict(os.environ, {"MI_RUNTIME_ROOT": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def child(self, script):
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(runtime.__file__).parent,
            capture_output=True, text=True, timeout=10, check=False,
        )

    def test_missing_status_and_annotation_are_read_only(self):
        self.assertEqual({"active": False, "owner": None}, runtime.status())
        self.assertIsNone(runtime.source_annotation("test-source"))
        self.assertFalse(self.root.exists())

    def test_reentrant_lock_and_exception_release(self):
        with self.assertRaisesRegex(ValueError, "injected"):
            with runtime.operation_lock(operation_id="attempt-one") as outer:
                with runtime.operation_lock() as inner:
                    self.assertEqual(outer, inner)
                    self.assertEqual(outer, runtime.status()["owner"])
                self.assertTrue(runtime.status()["active"])
                raise ValueError("injected")
        self.assertFalse(runtime.status()["active"])
        with runtime.operation_lock("recording") as owner:
            self.assertEqual("recording", owner["kind"])
        self.assertEqual(0o700, self.root.stat().st_mode & 0o777)
        self.assertEqual(0o600, (self.root / "operation.lock").stat().st_mode & 0o777)

    def test_other_thread_cannot_enter_even_with_same_id(self):
        results = []
        def contender():
            try:
                with runtime.operation_lock(operation_id="same-id"):
                    results.append("entered")
            except runtime.BusyError:
                results.append("busy")
        with runtime.operation_lock(operation_id="same-id"):
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(["busy"], results)

    def test_other_process_cannot_enter_and_reports_owner(self):
        with runtime.operation_lock("recording", "capture-test"):
            result = self.child(
                "import meetingintel_runtime as r, json\n"
                "print(json.dumps(r.status()))\n"
                "try:\n"
                " with r.operation_lock(): pass\n"
                "except r.BusyError: print('busy')\n"
                "else: raise AssertionError('concurrent entry')\n"
            )
        self.assertEqual(0, result.returncode, result.stderr)
        lines = result.stdout.splitlines()
        self.assertTrue(json.loads(lines[0])["active"])
        self.assertEqual("capture-test", json.loads(lines[0])["owner"]["operation_id"])
        self.assertEqual("busy", lines[1])

    def test_abrupt_process_exit_releases_lock_without_trusting_stale_metadata(self):
        result = self.child(
            "import meetingintel_runtime as r, os\n"
            "with r.operation_lock(): os._exit(0)\n"
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((self.root / "operation.json").exists())
        self.assertFalse(runtime.status()["active"])
        with runtime.operation_lock():
            self.assertTrue(runtime.status()["active"])

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_forked_child_cannot_inherit_reentrancy(self):
        with runtime.operation_lock():
            pid = os.fork()
            if pid == 0:
                try:
                    with runtime.operation_lock():
                        os._exit(2)
                except runtime.BusyError:
                    os._exit(0)
                except BaseException:
                    os._exit(3)
            _, outcome = os.waitpid(pid, 0)
            self.assertEqual(0, os.waitstatus_to_exitcode(outcome))
            self.assertTrue(runtime.status()["active"])

    def test_decorator_shares_lock_with_outer_context(self):
        @runtime.exclusive_operation("processing")
        def action():
            return runtime.status()["owner"]["operation_id"]
        with runtime.operation_lock(operation_id="outer"):
            self.assertEqual("outer", action())

    def test_symlink_runtime_root_is_rejected(self):
        target = self.parent / "elsewhere"
        target.mkdir(mode=0o700)
        self.root.symlink_to(target, target_is_directory=True)
        with self.assertRaises(runtime.RuntimeSafetyError):
            with runtime.operation_lock():
                self.fail("entered symlink root")
        self.assertEqual([], list(target.iterdir()))

    def test_symlink_ancestor_is_rejected(self):
        target = self.parent / "elsewhere"
        target.mkdir(mode=0o700)
        link = self.parent / "linked"
        link.symlink_to(target, target_is_directory=True)
        with patch.dict(os.environ, {"MI_RUNTIME_ROOT": str(link / "runtime")}):
            with self.assertRaises(runtime.RuntimeSafetyError):
                runtime.status()

    def test_broad_root_and_lock_permissions_fail_closed(self):
        self.root.mkdir(mode=0o755)
        self.root.chmod(0o755)
        with self.assertRaises(runtime.RuntimeSafetyError):
            with runtime.operation_lock():
                pass
        self.root.chmod(0o700)
        lock = self.root / "operation.lock"
        lock.write_text("")
        lock.chmod(0o644)
        with self.assertRaises(runtime.RuntimeSafetyError):
            with runtime.operation_lock():
                pass

    def test_symlink_lock_does_not_touch_target(self):
        self.root.mkdir(mode=0o700)
        target = self.parent / "target"
        target.write_text("untouched")
        (self.root / "operation.lock").symlink_to(target)
        with self.assertRaises(runtime.RuntimeSafetyError):
            with runtime.operation_lock():
                pass
        self.assertEqual("untouched", target.read_text())

    def test_hardlinked_lock_is_rejected(self):
        self.root.mkdir(mode=0o700)
        target = self.parent / "target"
        target.write_text("")
        target.chmod(0o600)
        os.link(target, self.root / "operation.lock")
        with self.assertRaises(runtime.RuntimeSafetyError):
            with runtime.operation_lock():
                pass

    def test_failed_metadata_write_releases_lock(self):
        with patch.object(runtime, "_write_json", side_effect=runtime.RuntimeSafetyError("injected")):
            with self.assertRaises(runtime.RuntimeSafetyError):
                with runtime.operation_lock():
                    pass
        with runtime.operation_lock():
            self.assertTrue(runtime.status()["active"])

    def test_annotations_remain_separate_from_processing_ownership(self):
        with runtime.operation_lock():
            written = runtime.annotate_source("laptop:source-one")
            self.assertEqual(written, runtime.source_annotation("laptop:source-one"))
            self.assertTrue(runtime.status()["active"])
        runtime.annotate_source("phone:source-two")
        self.assertEqual(written, runtime.source_annotation("laptop:source-one"))
        self.assertEqual("incomplete", runtime.source_annotation("phone:source-two")["completeness"])
        self.assertIsNone(runtime.source_annotation("missing"))
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.annotate_source("laptop:source-one", completeness="complete")

    def test_corrupt_annotation_cannot_be_treated_as_absent_or_overwritten(self):
        runtime.annotate_source("test-source")
        path = self.root / "source-annotations.json"
        path.write_text('{"schema_version":1,"sources":{"test-source":{}}}')
        before = path.read_bytes()
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.source_annotation("test-source")
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.annotate_source("other-source")
        self.assertEqual(before, path.read_bytes())

    def test_annotation_symlink_is_rejected_for_reads_and_writes(self):
        self.root.mkdir(mode=0o700)
        target = self.parent / "target"
        target.write_text('{"schema_version":1,"sources":{}}')
        target.chmod(0o600)
        (self.root / "source-annotations.json").symlink_to(target)
        before = target.read_bytes()
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.source_annotation("test-source")
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.annotate_source("test-source")
        self.assertEqual(before, target.read_bytes())

    def test_failed_atomic_annotation_save_preserves_previous_correction(self):
        original = runtime.annotate_source("test-source")
        with patch.object(os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(runtime.RuntimeSafetyError):
                runtime.annotate_source("other-source")
        self.assertEqual(original, runtime.source_annotation("test-source"))
        self.assertIsNone(runtime.source_annotation("other-source"))
        self.assertFalse(list(self.root.glob(".write-*")))

    def test_unsafe_source_id_and_relative_root_fail_closed(self):
        for source in ("../source", "/absolute/path", "a\nsecret", "a b", ""):
            with self.assertRaises(runtime.RuntimeSafetyError):
                runtime.annotate_source(source)
        with patch.dict(os.environ, {"MI_RUNTIME_ROOT": "relative"}):
            with self.assertRaises(runtime.RuntimeSafetyError):
                runtime.status()


if __name__ == "__main__":
    unittest.main()
