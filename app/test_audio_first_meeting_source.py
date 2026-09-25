from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from unittest import mock

import audio_first_meeting_source as adapter
import phone_recording_ingest as ingest
from phone_recording_ingest import deterministic_folder_name, parse_source_timestamp


class AudioFirstMeetingSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ingest_root = self.root / "ingest"
        self.ingest_root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source(
        self,
        *,
        parent: Path | None = None,
        timestamp: str = "2026_06_26_12_12_01",
        body: bytes = b"audio",
        original_path: str | None = None,
        changes: dict | None = None,
    ) -> tuple[Path, dict]:
        parent = parent or self.ingest_root
        created = parse_source_timestamp(f"{timestamp}.m4a")
        fingerprint = hashlib.sha256(body).hexdigest()
        folder = parent / deterministic_folder_name(created, fingerprint)
        folder.mkdir(parents=True)
        audio = folder / "audio.m4a"
        audio.write_bytes(body)
        metadata = {
            "schema_version": 1,
            "source_kind": "phone_recording",
            "source_filename": f"{timestamp}.m4a",
            "source_path": original_path or f"/original/{timestamp}.m4a",
            "source_fingerprint_sha256": fingerprint,
            "created_at": created.isoformat(),
            "timezone": "Asia/Kolkata",
            "duration_seconds": 12.5,
            "ingested_at": "2026-07-07T12:00:00+05:30",
            "status": "normalized",
            "canonical_audio_path": str(audio.resolve()),
            "audio_handling": "copy",
            "transcription_status": "not_started",
        }
        metadata.update(changes or {})
        (folder / "meeting_source.json").write_text(json.dumps(metadata), encoding="utf-8")
        return folder, metadata

    def assert_rejected(self, folder: Path):
        with self.assertRaises(adapter.SourceValidationError):
            adapter.load_audio_first_source(folder)

    def make_group_source(self, *, parent=None, changes=None):
        parent = parent or self.ingest_root
        created_values = (
            "2026-06-26T12:12:01+05:30",
            "2026-06-26T12:27:03+05:30",
            "2026-06-26T12:42:04+05:30",
        )
        fingerprints = tuple(hashlib.sha256(str(index).encode()).hexdigest() for index in range(3))
        group_fingerprint = ingest.group_fingerprint_from_values(list(zip(created_values, fingerprints)))
        created = datetime.fromisoformat(created_values[0])
        body = b"canonical-group"
        canonical_fingerprint = hashlib.sha256(body).hexdigest()
        folder = parent / ingest.deterministic_group_folder_name(created, group_fingerprint)
        folder.mkdir(parents=True)
        audio = folder / "audio.m4a"
        audio.write_bytes(body)
        segments = []
        durations = (899.880522, 899.875420, 66.464853)
        for index, (created_at, fingerprint, duration) in enumerate(
            zip(created_values, fingerprints, durations)
        ):
            filename = datetime.fromisoformat(created_at).strftime("%Y_%m_%d_%H_%M_%S.m4a")
            segments.append(
                {
                    "source_path": f"/original/2026/06/26/{filename}",
                    "source_filename": filename,
                    "fingerprint_sha256": fingerprint,
                    "created_at": created_at,
                    "duration_seconds": duration,
                }
            )
        metadata = {
            "schema_version": 1,
            "source_kind": "phone_recording",
            "source_filename": segments[0]["source_filename"],
            "source_path": segments[0]["source_path"],
            "source_fingerprint_sha256": canonical_fingerprint,
            "created_at": created_values[0],
            "timezone": "Asia/Kolkata",
            "duration_seconds": sum(durations),
            "ingested_at": "2026-07-08T10:00:00+05:30",
            "status": "normalized",
            "canonical_audio_path": str(audio.resolve()),
            "audio_handling": "concat_copy",
            "transcription_status": "not_started",
            "source_group": {
                "policy": ingest.GROUPING_POLICY,
                "group_fingerprint_sha256": group_fingerprint,
                "segment_count": 3,
            },
            "source_segments": segments,
        }
        metadata.update(changes or {})
        (folder / "meeting_source.json").write_text(json.dumps(metadata), encoding="utf-8")
        return folder, metadata

    def test_valid_normalized_phone_source_loads_into_frozen_representation(self):
        folder, _ = self.make_source()
        source = adapter.load_audio_first_source(folder)
        self.assertEqual("phone_recording", source.source_kind)
        self.assertEqual(12.5, source.duration_seconds)
        with self.assertRaises(FrozenInstanceError):
            source.source_kind = "changed"  # type: ignore[misc]

    def test_recursive_discovery_and_deterministic_order(self):
        later, _ = self.make_source(parent=self.ingest_root / "z", timestamp="2026_06_26_12_12_02", body=b"z")
        earlier, _ = self.make_source(parent=self.ingest_root / "a", timestamp="2026_06_26_12_12_01", body=b"a")
        results = adapter.discover_audio_first_sources(self.ingest_root)
        self.assertEqual([earlier.resolve(), later.resolve()], [result.source_folder for result in results])
        self.assertTrue(all(result.valid for result in results))

    def test_malformed_candidate_does_not_hide_later_valid_source(self):
        bad = self.ingest_root / "a-bad"
        bad.mkdir()
        (bad / "meeting_source.json").write_text("not-json")
        good, _ = self.make_source(parent=self.ingest_root / "z", body=b"good")
        results = adapter.discover_audio_first_sources(self.ingest_root)
        self.assertEqual(2, len(results))
        self.assertFalse(results[0].valid)
        self.assertEqual(good.resolve(), results[1].source_folder)
        self.assertTrue(results[1].valid)

    def test_missing_metadata_rejected(self):
        folder = self.ingest_root / "empty"
        folder.mkdir()
        self.assert_rejected(folder)

    def test_malformed_json_and_non_object_rejected(self):
        for text in ("{broken", "[]"):
            with self.subTest(text=text):
                folder = self.ingest_root / hashlib.sha256(text.encode()).hexdigest()[:8]
                folder.mkdir()
                (folder / "meeting_source.json").write_text(text)
                self.assert_rejected(folder)

    def test_wrong_source_kind_rejected(self):
        folder, _ = self.make_source(changes={"source_kind": "meetily"})
        self.assert_rejected(folder)

    def test_incomplete_normalization_rejected(self):
        folder, _ = self.make_source(changes={"status": "pending"})
        self.assert_rejected(folder)

    def test_contradictory_transcription_status_rejected(self):
        folder, _ = self.make_source(changes={"transcription_status": "complete"})
        self.assert_rejected(folder)

    def test_unsupported_audio_handling_rejected(self):
        folder, _ = self.make_source(changes={"audio_handling": "symlink"})
        self.assert_rejected(folder)

    def test_missing_required_field_and_wrong_type_rejected(self):
        folder, metadata = self.make_source()
        del metadata["duration_seconds"]
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)
        metadata["duration_seconds"] = "12.5"
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)

    def test_canonical_audio_missing_rejected(self):
        folder, _ = self.make_source()
        (folder / "audio.m4a").unlink()
        self.assert_rejected(folder)

    def test_canonical_path_escape_and_symlink_rejected(self):
        folder, metadata = self.make_source()
        outside = self.root / "outside.m4a"
        outside.write_bytes(b"audio")
        metadata["canonical_audio_path"] = str(outside)
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)

        metadata["canonical_audio_path"] = str((folder / "audio.m4a").resolve())
        (folder / "audio.m4a").unlink()
        (folder / "audio.m4a").symlink_to(outside)
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)

    def test_source_and_audio_filename_contract_mismatches_rejected(self):
        folder, _ = self.make_source(changes={"source_filename": "other.m4a"})
        self.assert_rejected(folder)
        folder2, _ = self.make_source(timestamp="2026_06_26_12_12_02", body=b"two")
        metadata_path = folder2 / "meeting_source.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["canonical_audio_path"] = str(folder2 / "other.m4a")
        metadata_path.write_text(json.dumps(metadata))
        self.assert_rejected(folder2)

    def test_malformed_and_mismatched_fingerprints_rejected(self):
        folder, _ = self.make_source(changes={"source_fingerprint_sha256": "ABC"})
        self.assert_rejected(folder)
        folder2, metadata = self.make_source(timestamp="2026_06_26_12_12_02", body=b"two")
        metadata["source_fingerprint_sha256"] = "0" * 64
        (folder2 / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder2)

    def test_malformed_naive_and_unsupported_offset_created_at_rejected(self):
        for index, created_at in enumerate(("bad", "2026-06-26T12:12:01", "2026-06-26T12:12:01+00:00"), start=1):
            with self.subTest(created_at=created_at):
                folder, _ = self.make_source(timestamp=f"2026_06_26_12_12_0{index}", body=str(index).encode(), changes={"created_at": created_at})
                self.assert_rejected(folder)

    def test_unsupported_timezone_name_rejected(self):
        folder, _ = self.make_source(changes={"timezone": "UTC"})
        self.assert_rejected(folder)

    def test_invalid_duration_rejected(self):
        for index, duration in enumerate((0, -1, float("inf"), True), start=1):
            with self.subTest(duration=duration):
                folder, _ = self.make_source(timestamp=f"2026_06_26_12_12_0{index}", body=str(index).encode(), changes={"duration_seconds": duration})
                self.assert_rejected(folder)

    def test_missing_or_relative_original_provenance_rejected(self):
        folder, metadata = self.make_source()
        del metadata["source_path"]
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)
        metadata["source_path"] = "relative/file.m4a"
        (folder / "meeting_source.json").write_text(json.dumps(metadata))
        self.assert_rejected(folder)

    def test_deterministic_folder_name_metadata_mismatch_rejected(self):
        folder, _ = self.make_source()
        moved = folder.with_name("wrong-name")
        folder.rename(moved)
        metadata_path = moved / "meeting_source.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["canonical_audio_path"] = str((moved / "audio.m4a").resolve())
        metadata_path.write_text(json.dumps(metadata))
        self.assert_rejected(moved)

    def test_changed_content_and_identical_timestamp_sources_have_distinct_identity(self):
        first, _ = self.make_source(parent=self.ingest_root / "one", body=b"one", original_path="/phone/a/2026_06_26_12_12_01.m4a")
        second, _ = self.make_source(parent=self.ingest_root / "two", body=b"two", original_path="/phone/b/2026_06_26_12_12_01.m4a")
        one = adapter.load_audio_first_source(first)
        two = adapter.load_audio_first_source(second)
        self.assertNotEqual(one.source_fingerprint, two.source_fingerprint)
        self.assertNotEqual(one.source_identity, two.source_identity)

    def test_same_normalized_source_under_another_root_preserves_identity(self):
        original_path = "/phone/2026_06_26_12_12_01.m4a"
        first, _ = self.make_source(parent=self.ingest_root / "one", original_path=original_path)
        second_root = self.root / "moved"
        second_root.mkdir()
        second, _ = self.make_source(parent=second_root, original_path=original_path)
        self.assertEqual(
            adapter.load_audio_first_source(first).source_identity,
            adapter.load_audio_first_source(second).source_identity,
        )

    def test_adapter_performs_no_writes(self):
        folder, _ = self.make_source()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.iterdir() if path.is_file()}
        adapter.load_audio_first_source(folder)
        adapter.discover_audio_first_sources(self.ingest_root)
        after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_no_subprocess_model_transcription_or_diarization_invocation(self):
        folder, _ = self.make_source()
        with mock.patch.object(subprocess, "run") as run:
            adapter.load_audio_first_source(folder)
            adapter.discover_audio_first_sources(self.ingest_root)
        run.assert_not_called()

    def test_valid_grouped_source_is_accepted_with_namespaced_identity(self):
        folder, metadata = self.make_group_source()
        source = adapter.load_audio_first_source(folder)
        self.assertEqual("concat_copy", source.audio_handling)
        self.assertEqual(ingest.GROUPING_POLICY, source.grouping_policy)
        self.assertEqual(
            metadata["source_group"]["group_fingerprint_sha256"], source.source_identity
        )
        self.assertEqual(hashlib.sha256(b"canonical-group").hexdigest(), source.source_fingerprint)

    def test_grouped_source_identity_survives_canonical_parent_move(self):
        first, metadata = self.make_group_source(parent=self.ingest_root / "one")
        first_identity = adapter.load_audio_first_source(first).source_identity
        moved_parent = self.root / "moved"
        moved_parent.mkdir()
        moved = moved_parent / first.name
        first.rename(moved)
        metadata["canonical_audio_path"] = str((moved / "audio.m4a").resolve())
        (moved / "meeting_source.json").write_text(json.dumps(metadata))
        self.assertEqual(first_identity, adapter.load_audio_first_source(moved).source_identity)

    def test_grouped_policy_count_and_fingerprint_are_strict(self):
        mutations = (
            lambda metadata: metadata["source_group"].update(policy="wrong"),
            lambda metadata: metadata["source_group"].update(segment_count=2),
            lambda metadata: metadata["source_group"].update(group_fingerprint_sha256="0" * 64),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                parent = self.ingest_root / str(index)
                folder, metadata = self.make_group_source(parent=parent)
                mutate(metadata)
                (folder / "meeting_source.json").write_text(json.dumps(metadata))
                self.assert_rejected(folder)

    def test_grouped_segments_must_be_ordered_unique_and_match_primary(self):
        mutations = (
            lambda segments: segments.reverse(),
            lambda segments: segments[1].update(created_at=segments[0]["created_at"]),
            lambda segments: segments[1].update(fingerprint_sha256=segments[0]["fingerprint_sha256"]),
            lambda segments: segments[0].update(source_path="/wrong/first.m4a", source_filename="first.m4a"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                folder, metadata = self.make_group_source(parent=self.ingest_root / str(index))
                mutate(metadata["source_segments"])
                (folder / "meeting_source.json").write_text(json.dumps(metadata))
                self.assert_rejected(folder)

    def test_concat_copy_requires_grouped_provenance_and_copy_rejects_it(self):
        folder, metadata = self.make_source(changes={"audio_handling": "concat_copy"})
        self.assert_rejected(folder)
        grouped, grouped_metadata = self.make_group_source(parent=self.ingest_root / "group")
        grouped_metadata["audio_handling"] = "copy"
        (grouped / "meeting_source.json").write_text(json.dumps(grouped_metadata))
        self.assert_rejected(grouped)

    def test_grouped_canonical_audio_hash_mismatch_is_rejected(self):
        folder, _ = self.make_group_source()
        (folder / "audio.m4a").write_bytes(b"changed")
        self.assert_rejected(folder)


if __name__ == "__main__":
    unittest.main()
