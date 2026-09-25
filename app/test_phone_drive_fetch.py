from __future__ import annotations

import io
import json
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import phone_drive_fetch as fetch


class PhoneDriveFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests inject the command runner; no installed rclone is needed.
        executable = patch.object(fetch, "find_rclone", return_value="/synthetic/rclone")
        executable.start()
        self.addCleanup(executable.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "staging"
        self.ledger = self.root / "state" / "ledger.json"
        self.remote_root = "test-drive:phone-recordings"
        self.clock = lambda: datetime(2026, 7, 15, 9, 0, tzinfo=fetch.IST)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, dry_run: bool = False, **kwargs) -> fetch.FetchConfig:
        return fetch.FetchConfig(self.remote_root, self.destination, self.ledger, dry_run, **kwargs)

    def listing(self, *, remote_id: str = "drive-id-1", size: int = 5, modified: str = "2026-07-14T13:51:21.580Z") -> str:
        return json.dumps([
            {
                "Path": "2026/07/14/2026_07_14_19_01_19.m4a",
                "ID": remote_id,
                "Size": size,
                "ModTime": modified,
                "Hashes": {"md5": "a5ca0b5894324f8bb54bb9fffad29d1e"},
            }
        ])

    def runner(self, listing: str, body: bytes = b"audio"):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if args[1] == "lsjson":
                return subprocess.CompletedProcess(args, 0, listing, "")
            if args[1] == "copyto":
                Path(args[-1]).write_bytes(body)
                return subprocess.CompletedProcess(args, 0, "", "")
            raise AssertionError(args)

        return run, calls

    def execute(self, config: fetch.FetchConfig, runner):
        output = io.StringIO()
        with redirect_stdout(output):
            summary = fetch.fetch_recordings(config, runner=runner, clock=self.clock)
        return summary, output.getvalue()

    def test_downloads_once_records_immutable_drive_id_and_uses_atomic_ledger(self):
        runner, calls = self.runner(self.listing())
        summary, _ = self.execute(self.config(), runner)
        target = self.destination / "2026/07/14/2026_07_14_19_01_19.m4a"
        self.assertEqual((1, 0), (summary.downloaded, summary.failed))
        self.assertEqual(b"audio", target.read_bytes())
        ledger = json.loads(self.ledger.read_text())
        self.assertEqual(1, ledger["schema_version"])
        self.assertEqual(["drive-id-1"], list(ledger["records"]))
        self.assertEqual("rclone", Path(calls[0][0]).name)
        self.assertEqual("lsjson", calls[0][1])
        self.assertEqual("rclone", Path(calls[1][0]).name)
        self.assertEqual("copyto", calls[1][1])

    def test_unchanged_file_is_skipped_without_a_second_download(self):
        first_runner, _ = self.runner(self.listing())
        self.execute(self.config(settle_interval_seconds=0), first_runner)
        second_runner, calls = self.runner(self.listing())
        summary, output = self.execute(self.config(settle_interval_seconds=0), second_runner)
        self.assertEqual((1, 0, 0), (summary.skipped, summary.downloaded, summary.failed))
        self.assertEqual(1, len(calls))
        self.assertIn("SKIP_UNCHANGED", output)

    def test_changed_remote_is_downloaded_over_tracked_audio(self):
        first_runner, _ = self.runner(self.listing(), body=b"audio")
        self.execute(self.config(), first_runner)
        changed_runner, calls = self.runner(
            self.listing(modified="2026-07-15T01:00:00.000Z"), body=b"audio"
        )
        summary, output = self.execute(self.config(), changed_runner)
        target = self.destination / "2026/07/14/2026_07_14_19_01_19.m4a"
        self.assertEqual((1, 1, 0), (summary.changed, summary.downloaded, summary.failed))
        self.assertEqual(0, summary.exit_code)
        self.assertEqual(2, len(calls))
        self.assertEqual(b"audio", target.read_bytes())
        self.assertIn("DOWNLOADED", output)

    def test_failed_download_is_recorded_and_retried(self):
        runner, _ = self.runner(self.listing(), body=b"bad")
        summary, _ = self.execute(self.config(), runner)
        self.assertEqual(1, summary.failed)
        ledger = json.loads(self.ledger.read_text())
        self.assertEqual("failed", ledger["records"]["drive-id-1"]["status"])

    def test_failure_category_is_safe_and_distinguishes_auth_from_quota(self):
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text(json.dumps({
            "schema_version": 1,
            "records": {},
            "last_listing_failure": {"failure": "RcloneCommandError: unauthorized"},
        }))
        self.assertEqual("drive_auth_or_permission", fetch.classify_fetch_failure(self.ledger))
        self.ledger.write_text(json.dumps({
            "schema_version": 1,
            "records": {},
            "last_listing_failure": {"failure": "RcloneCommandError: HTTP 429 rate limit"},
        }))
        self.assertEqual("drive_quota_or_rate_limit", fetch.classify_fetch_failure(self.ledger))

    def test_checksum_mismatch_does_not_replace_destination(self):
        runner, _ = self.runner(self.listing(), body=b"other")
        summary, _ = self.execute(self.config(), runner)
        self.assertEqual(1, summary.failed)
        self.assertFalse((self.destination / "2026/07/14/2026_07_14_19_01_19.m4a").exists())

    def test_symlink_destination_is_rejected(self):
        self.destination.mkdir(parents=True)
        target = self.destination / "2026"
        target.symlink_to(self.root)
        runner, _ = self.runner(self.listing())
        summary, output = self.execute(self.config(), runner)
        self.assertEqual(1, summary.failed)
        self.assertIn("FAILED", output)

    def test_dry_run_lists_new_files_without_writing_destination_or_ledger(self):
        runner, calls = self.runner(self.listing())
        summary, output = self.execute(self.config(dry_run=True), runner)
        self.assertEqual((1, 0), (summary.downloaded, summary.failed))
        self.assertEqual(1, len(calls))
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.ledger.exists())
        self.assertIn("WOULD_DOWNLOAD", output)

    def test_untracked_existing_destination_is_never_overwritten(self):
        target = self.destination / "2026/07/14/2026_07_14_19_01_19.m4a"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing")
        runner, calls = self.runner(self.listing())
        summary, output = self.execute(self.config(), runner)
        self.assertEqual(1, summary.failed)
        self.assertEqual(1, len(calls))
        self.assertEqual(b"existing", target.read_bytes())
        self.assertIn("FAILED_UNTRACKED_DESTINATION", output)

    def test_rejects_paths_outside_expected_date_layout(self):
        listing = json.dumps([{"Path": "2026/07/recording.m4a", "ID": "x", "Size": 1, "ModTime": "now"}])
        runner, _ = self.runner(listing)
        with self.assertRaises(ValueError):
            fetch.list_remote_recordings(self.remote_root, runner=runner)

    def test_daily_agent_runs_sync_with_a_bounded_path(self):
        payload = plistlib.loads(fetch.build_daily_launch_agent(7, 5))
        self.assertEqual("/bin/sh", payload["ProgramArguments"][0])
        self.assertEqual(str(fetch.DEFAULT_SCHEDULER_RUNTIME_WRAPPER_PATH), payload["ProgramArguments"][1])
        self.assertEqual(str(fetch.DEFAULT_SCHEDULER_RUNTIME_RUNNER_PATH), payload["ProgramArguments"][3])
        self.assertEqual([str(fetch.PROJECT_ROOT / "meetingintel_cli.py"), str(fetch.DEFAULT_SCHEDULER_RECEIPT_PATH), "sync"], payload["ProgramArguments"][4:])
        self.assertEqual(str(fetch.DEFAULT_SCHEDULER_RUNTIME_ROOT), payload["WorkingDirectory"])
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(fetch.SCHEDULER_INTERVAL_SECONDS, payload["StartInterval"])
        self.assertIn("/opt/homebrew/bin", payload["EnvironmentVariables"]["PATH"])

    def test_daily_agent_rejects_invalid_clock_values(self):
        for hour, minute in ((-1, 0), (24, 0), (7, -1), (7, 60)):
            with self.subTest(hour=hour, minute=minute), self.assertRaises(ValueError):
                fetch.build_daily_launch_agent(hour, minute)


if __name__ == "__main__":
    unittest.main()
