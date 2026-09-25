from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import meetingintel_cli as cli
import meetingintel_inventory as inventory
import meetingintel_pipeline as pipeline
import phone_recording_ingest as ingest


IST = timezone(timedelta(hours=5, minutes=30))


class MeetingIntelInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.meetings = self.root / "meetings"
        self.staging = self.root / "staging"
        self.fetch_ledger = self.root / "state" / "phone_fetch.json"
        self.ingest_root = self.root / "ingest" / "phone"
        self.processing_ledger = self.root / "state" / "processed.json"
        self.now = datetime(2026, 7, 23, 12, 0, tzinfo=IST)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self) -> inventory.InventoryConfig:
        return inventory.InventoryConfig(
            meetings_root=self.meetings,
            phone_staging_root=self.staging,
            phone_fetch_ledger=self.fetch_ledger,
            phone_ingest_root=self.ingest_root,
            processing_ledger=self.processing_ledger,
        )

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def add_meetily(self, folder_name: str, created_at: str, name: str) -> Path:
        folder = self.meetings / folder_name
        folder.mkdir(parents=True)
        self.write_json(folder / "metadata.json", {"status": "completed", "created_at": created_at, "meeting_name": name})
        (folder / "transcripts.json").write_text("not inspected", encoding="utf-8")
        return folder

    def add_fetch_record(self, path: str, status: str, **extra: object) -> None:
        payload = json.loads(self.fetch_ledger.read_text()) if self.fetch_ledger.is_file() else {"schema_version": 1, "records": {}}
        record = {"status": status, "local_relative_path": path, "modified_time": "2026-07-23T06:00:00Z"}
        record.update(extra)
        payload["records"][path] = record
        self.write_json(self.fetch_ledger, payload)

    def add_normalized(self, filename: str, created_at: str, processed: bool, *, grouping: bool = False) -> str:
        raw_path = self.staging / "2026" / "07" / "23" / filename
        source_fingerprint = hashlib.sha256(filename.encode()).hexdigest()
        source_identity = hashlib.sha256(
            f"phone_recording|{raw_path}|{created_at}|{source_fingerprint}".encode()
        ).hexdigest()
        processing_fingerprint = pipeline.build_audio_first_fingerprint(source_identity)
        target = self.ingest_root / ("phone_group_" if grouping else "phone_") / filename
        metadata_path = target / ingest.METADATA_FILENAME
        self.write_json(
            metadata_path,
            {
                "schema_version": 1,
                "source_kind": "phone_recording",
                "source_path": str(raw_path),
                "source_filename": filename,
                "source_fingerprint_sha256": source_fingerprint,
                "created_at": created_at,
                "timezone": "Asia/Kolkata",
                "duration_seconds": 30.0,
                "ingested_at": created_at,
                "status": "normalized",
                "canonical_audio_path": str(target / ingest.CANONICAL_AUDIO_FILENAME),
                "audio_handling": "concat_copy" if grouping else "copy",
                "transcription_status": "not_started",
                **({
                    "source_group": {
                        "policy": ingest.GROUPING_POLICY,
                        "group_fingerprint_sha256": source_identity,
                        "segment_count": 2,
                    },
                    "source_segments": [],
                } if grouping else {}),
            },
        )
        payload = json.loads((self.ingest_root / ingest.LEDGER_FILENAME).read_text()) if (self.ingest_root / ingest.LEDGER_FILENAME).is_file() else {"schema_version": 1, "records": {}}
        payload["records"][source_identity] = {
            "status": "normalized",
            "source_path": str(raw_path),
            "source_filename": filename,
            "source_fingerprint_sha256": source_fingerprint,
            "created_at": created_at,
            "target_folder": str(target),
            "metadata_path": str(metadata_path),
            "canonical_audio_path": str(target / ingest.CANONICAL_AUDIO_FILENAME),
        }
        self.write_json(self.ingest_root / ingest.LEDGER_FILENAME, payload)
        if processed:
            processing = json.loads(self.processing_ledger.read_text()) if self.processing_ledger.is_file() else {"schema_version": 2, "records": {}}
            processing["records"][processing_fingerprint] = {
                "folder_path": str(target),
                "source_kind": "phone_recording",
                "source_identity": source_identity,
                "processing_mode": "flat_from_diarized_transcript",
            }
            self.write_json(self.processing_ledger, processing)
        return processing_fingerprint

    def test_inventory_shows_both_lanes_and_all_phone_states(self):
        meetily = self.add_meetily("meet-1", "2026-07-23T08:00:00+05:30", "Meetily ready")
        self.add_meetily("meet-2", "2026-07-23T09:00:00+05:30", "Meetily done")
        self.write_json(self.processing_ledger, {"schema_version": 2, "records": {"meet": {"folder_path": str(self.meetings / "meet-2"), "processing_mode": "diarized_1to1"}}})
        self.add_fetch_record("2026/07/23/settled/settled.m4a", "settled_ready")
        self.add_fetch_record("2026/07/23/pending/pending.m4a", "pending_settlement")
        self.add_fetch_record("2026/07/23/decision/decision.m4a", "requires_operator_decision")
        self.add_fetch_record("2026/07/23/processed.m4a", "settled_ready")
        self.add_normalized("processed.m4a", "2026-07-23T10:00:00+05:30", processed=True)
        self.add_normalized("ready.m4a", "2026-07-23T11:00:00+05:30", processed=False)
        report = inventory.inventory("all", config=self.config(), clock=lambda: self.now)
        statuses = [item.status for item in report.items]
        self.assertIn("meetily ready", statuses)
        self.assertIn("meetily already processed", statuses)
        self.assertIn("phone downloaded and settled (not normalized)", statuses)
        self.assertIn("phone pending settlement", statuses)
        self.assertIn("operator decision required", statuses)
        self.assertIn("phone normalized and ready", statuses)
        self.assertIn("phone already processed", statuses)
        self.assertEqual(1, sum(item.status == "phone already processed" for item in report.items))
        self.assertEqual(1, sum(item.display_name == "processed.m4a" for item in report.items))
        processed_item = next(item for item in report.items if item.display_name == "processed.m4a")
        self.assertEqual("flat_from_diarized_transcript", processed_item.processing_mode)

    def test_inventory_deduplicates_normalized_phone_and_groups_only_explicit_batches(self):
        self.add_normalized("grouped.m4a", "2026-07-23T08:00:00+05:30", processed=False, grouping=True)
        self.add_fetch_record("2026/07/23/grouped.m4a", "settled_ready")
        self.add_fetch_record("2026/07/23/group-a/segment-a.m4a", "settled_ready", grouping_status="deterministic_group")
        self.add_fetch_record("2026/07/23/group-a/segment-b.m4a", "settled_ready", grouping_status="deterministic_group")
        self.add_fetch_record("2026/07/23/group-b/ambiguous-a.m4a", "settled_ready")
        self.add_fetch_record("2026/07/23/group-b/ambiguous-b.m4a", "settled_ready")
        report = inventory.inventory("all", config=self.config(), clock=lambda: self.now)
        self.assertEqual(1, sum(item.display_name == "grouped.m4a" for item in report.items))
        grouped = next(item for item in report.items if "2 staged segments" in item.display_name and item.status == "phone pending normalization")
        self.assertEqual("2026/07/23/group-a", grouped.display_name.split(" ", 1)[0])
        self.assertTrue(any("2 staged segments" in item.display_name and item.status == "operator decision required" for item in report.items))

    def test_inventory_deduplicates_same_hashed_segment_from_legacy_staging_root(self):
        self.add_normalized("legacy.m4a", "2026-07-23T08:00:00+05:30", processed=False)
        ingest_ledger_path = self.ingest_root / ingest.LEDGER_FILENAME
        ingest_ledger = json.loads(ingest_ledger_path.read_text())
        record = next(iter(ingest_ledger["records"].values()))
        metadata_path = Path(record["metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        fingerprint = metadata["source_fingerprint_sha256"]
        legacy_path = self.root / "legacy-staging" / "legacy.m4a"
        record["source_path"] = str(legacy_path)
        record["source_segment_paths"] = [str(legacy_path)]
        metadata["source_path"] = str(legacy_path)
        metadata["source_segments"] = [
            {
                "source_path": str(legacy_path),
                "source_filename": "legacy.m4a",
                "fingerprint_sha256": fingerprint,
            }
        ]
        self.write_json(ingest_ledger_path, ingest_ledger)
        self.write_json(metadata_path, metadata)
        self.add_fetch_record("2026/07/23/legacy.m4a", "settled_ready", sha256=fingerprint)

        report = inventory.inventory("all", config=self.config(), clock=lambda: self.now)

        self.assertEqual(1, len(report.items))
        self.assertEqual("phone normalized and ready", report.items[0].status)

    def test_pending_multi_segment_batch_remains_waiting_until_all_segments_settle(self):
        self.add_fetch_record("2026/08/06/2026_08_06_13_50_00.m4a", "pending_settlement")
        self.add_fetch_record("2026/08/06/2026_08_06_13_50_01.m4a", "pending_settlement")
        report = inventory.inventory("all", config=self.config(), clock=lambda: self.now)
        pending = [item for item in report.items if item.source_kind == "phone_recording"]
        self.assertEqual(1, len(pending))
        self.assertEqual("phone pending settlement", pending[0].status)

    def test_failed_phone_attempt_is_visible_without_marking_it_processed(self):
        self.add_normalized("failed.m4a", "2026-07-23T11:00:00+05:30", processed=False)
        with patch.object(
            pipeline, "qwen_latest_failure_category", return_value="unsupported_detected_language"
        ):
            report = inventory.inventory("all", config=self.config(), clock=lambda: self.now)
        self.assertEqual(1, len(report.items))
        self.assertEqual(
            "phone processing stopped: ambiguous language tag", report.items[0].status
        )

    def test_scopes_and_audio_first_are_deterministic(self):
        self.add_meetily("old", "2026-07-22T08:00:00+05:30", "Old")
        self.add_fetch_record("2026/07/23/new.m4a", "settled_ready")
        self.add_normalized("ready.m4a", "2026-07-23T11:00:00+05:30", processed=False)
        self.add_normalized("done.m4a", "2026-07-23T12:00:00+05:30", processed=True)
        today = inventory.inventory("today", config=self.config(), clock=lambda: self.now)
        self.assertTrue(all(item.created_at.date() == self.now.date() for item in today.items))
        newest = inventory.inventory("last", config=self.config(), clock=lambda: self.now)
        self.assertEqual(("phone_recording", "phone already processed"), (newest.items[0].source_kind, newest.items[0].status))
        new = inventory.inventory("new", config=self.config(), clock=lambda: self.now)
        self.assertTrue(all("already processed" not in item.status for item in new.items))
        audio = inventory.inventory("all", config=self.config(), audio_first=True, clock=lambda: self.now)
        self.assertTrue(audio.items)
        self.assertTrue(all(item.source_kind == "phone_recording" and item.status in {"phone normalized and ready", "phone already processed"} for item in audio.items))
        self.assertNotIn("phone downloaded and settled (not normalized)", {item.status for item in audio.items})

    def test_empty_inventory_has_required_message_and_no_writes_or_audio_reads(self):
        before = sorted(path.relative_to(self.root) for path in self.root.rglob("*") if path.is_file())
        report = inventory.inventory("today", config=self.config(), clock=lambda: self.now)
        self.assertEqual((), report.items)
        rendered = inventory.format_inventory(report)
        self.assertIn("No matching meetings or phone recordings.", rendered)
        after = sorted(path.relative_to(self.root) for path in self.root.rglob("*") if path.is_file())
        self.assertEqual(before, after)

    def test_cli_inventory_syntax_is_read_only_and_bypasses_processing(self):
        report = inventory.InventoryReport("today", True, (), {}, ())
        with patch.object(cli, "build_inventory", return_value=report) as build, patch.object(cli, "discover_candidates") as discover, patch.object(cli, "load_secret_environment") as secrets, patch.object(cli, "fetch_recordings") as fetch:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, cli.run_cli(["--audio-first", "today", "list"]))
                self.assertEqual(0, cli.run_cli(["list"]))
        self.assertEqual(2, build.call_count)
        self.assertEqual(("today", "today"), tuple(call.args[0] for call in build.call_args_list))
        self.assertEqual((True, False), tuple(call.kwargs["audio_first"] for call in build.call_args_list))
        discover.assert_not_called()
        secrets.assert_not_called()
        fetch.assert_not_called()

    def test_all_inventory_scope_spellings_are_supported(self):
        report = inventory.InventoryReport("all", False, (), {}, ())
        commands = [
            [scope, "list"] for scope in ("today", "last", "new", "all")
        ] + [
            ["--audio-first", scope, "list"] for scope in ("today", "last", "new", "all")
        ]
        with patch.object(cli, "build_inventory", return_value=report) as build:
            with redirect_stdout(io.StringIO()):
                for command in commands:
                    self.assertEqual(0, cli.run_cli(command))
        self.assertEqual(8, build.call_count)


if __name__ == "__main__":
    unittest.main()
