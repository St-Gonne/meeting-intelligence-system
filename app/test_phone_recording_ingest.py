from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import phone_recording_ingest as ingest


IST = ZoneInfo("Asia/Kolkata")


class PhoneRecordingIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "ingest"
        self.source.mkdir()
        self.clock_value = datetime(2026, 7, 7, 12, 0, 0, tzinfo=IST)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_audio(self, relative: str = "2026/06/26/2026_06_26_12_12_01.m4a", body: bytes = b"audio") -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def run(self, result=None):  # type: ignore[override]
        return super().run(result)

    def execute(
        self,
        *,
        dry_run: bool = False,
        probe=None,
        stream_probe=None,
        concat_runner=None,
        operator_resolution=None,
    ):
        output = io.StringIO()
        with redirect_stdout(output):
            summary = ingest.run_ingest(
                ingest.IngestConfig(
                    self.source, self.destination, dry_run, operator_resolution
                ),
                duration_probe=probe or (lambda _path: 42.25),
                stream_probe=stream_probe or (lambda _path: ("aac", "LC", 44100, 2)),
                concat_runner=concat_runner or ingest.run_concat_copy,
                clock=lambda: self.clock_value,
            )
        return summary, output.getvalue()

    def records(self):
        return json.loads((self.destination / ingest.LEDGER_FILENAME).read_text())["records"]

    def write_operator_resolution(
        self, identities, *, resolution="separate", schema_version=None, payload_overrides=None
    ):
        payload = {
            "schema_version": schema_version or ingest.OPERATOR_RESOLUTION_SCHEMA_VERSION,
            "resolution": resolution,
            "source_fingerprints_sha256": list(identities),
        }
        if payload_overrides:
            payload.update(payload_overrides)
        path = self.root / "operator-resolution.json"
        path.write_text(json.dumps(payload))
        return path

    def only_target(self) -> Path:
        folders = [path for path in self.destination.iterdir() if path.is_dir()]
        self.assertEqual(1, len(folders))
        return folders[0]

    def test_recursive_m4a_discovery_and_other_formats_ignored(self):
        audio = self.add_audio()
        (self.source / "ignore.wav").write_bytes(b"wav")
        (self.source / "ignore.txt").write_text("text")
        self.assertEqual([audio], ingest.discover_sources(self.source))

    def test_filename_timestamp_is_parsed_as_asia_kolkata(self):
        parsed = ingest.parse_source_timestamp("2026_06_26_12_12_01.m4a")
        self.assertEqual("2026-06-26T12:12:01+05:30", parsed.isoformat())

    def test_malformed_and_invalid_timestamp_names_fail_without_guessing(self):
        for name in ("recording.m4a", "2026_13_26_12_12_01.m4a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ingest.parse_source_timestamp(name)

    def test_duration_and_provenance_are_captured(self):
        source = self.add_audio()
        summary, _ = self.execute(probe=lambda path: 12.75)
        self.assertEqual(1, summary.ingested)
        metadata = json.loads((self.only_target() / ingest.METADATA_FILENAME).read_text())
        self.assertEqual(12.75, metadata["duration_seconds"])
        self.assertEqual("phone_recording", metadata["source_kind"])
        self.assertEqual(str(source.resolve()), metadata["source_path"])
        self.assertEqual("not_started", metadata["transcription_status"])

    def test_duration_failure_isolated_and_later_source_is_ingested(self):
        bad = self.add_audio("2026_06_26_12_12_01.m4a", b"bad")
        good = self.add_audio("2026_06_26_12_12_02.m4a", b"good")

        def probe(path):
            if path == bad.resolve():
                raise RuntimeError("no duration")
            return 7.0

        summary, output = self.execute(probe=probe)
        self.assertEqual((1, 1), (summary.failed, summary.ingested))
        self.assertIn(f"FAILED {bad.resolve()}", output)
        self.assertIn(f"INGESTED {good.resolve()}", output)

    def test_fingerprint_changes_with_content(self):
        source = self.add_audio(body=b"one")
        first = ingest.sha256_file(source)
        source.write_bytes(b"two")
        self.assertNotEqual(first, ingest.sha256_file(source))

    def test_unchanged_source_rerun_skips(self):
        self.add_audio()
        self.execute()
        summary, output = self.execute()
        self.assertEqual((0, 1, 0), (summary.ingested, summary.skipped, summary.failed))
        self.assertIn("SKIP_ALREADY_INGESTED", output)
        self.assertEqual(1, len(self.records()))

    def test_same_filename_changed_content_is_a_new_source(self):
        source = self.add_audio(body=b"first")
        self.execute()
        source.write_bytes(b"second")
        summary, _ = self.execute()
        self.assertEqual(1, summary.ingested)
        self.assertEqual(2, len(self.records()))
        self.assertEqual(2, len([p for p in self.destination.iterdir() if p.is_dir()]))

    def test_target_name_is_deterministic_and_collision_safe(self):
        created = ingest.parse_source_timestamp("2026_06_26_12_12_01.m4a")
        first = ingest.deterministic_folder_name(created, "a" * 64)
        self.assertEqual(first, ingest.deterministic_folder_name(created, "a" * 64))
        self.assertNotEqual(first, ingest.deterministic_folder_name(created, "b" * 64))
        self.assertEqual("phone_2026-06-26_12-12-01_aaaaaaaaaaaaaaaa", first)

    def test_original_source_is_unchanged(self):
        source = self.add_audio(body=b"original")
        before = (source.read_bytes(), source.stat().st_mtime_ns)
        self.execute()
        self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns))

    def test_no_fake_transcript_is_created(self):
        self.add_audio()
        self.execute()
        names = {path.name for path in self.only_target().iterdir()}
        self.assertEqual({ingest.CANONICAL_AUDIO_FILENAME, ingest.METADATA_FILENAME}, names)

    def test_no_transcription_diarization_or_model_process_is_invoked(self):
        self.add_audio()
        with mock.patch.object(ingest.subprocess, "run") as process:
            self.execute(probe=lambda _path: 1.0)
        process.assert_not_called()

    def test_copied_audio_remains_after_source_is_removed(self):
        source = self.add_audio(body=b"durable")
        self.execute()
        canonical = self.only_target() / ingest.CANONICAL_AUDIO_FILENAME
        source.unlink()
        self.assertEqual(b"durable", canonical.read_bytes())

    def test_ledger_is_slim_and_contains_no_audio_or_transcript_body(self):
        self.add_audio(body=b"SECRET_AUDIO_BODY")
        self.execute()
        text = (self.destination / ingest.LEDGER_FILENAME).read_text()
        self.assertNotIn("SECRET_AUDIO_BODY", text)
        self.assertNotIn("transcript", text.lower())
        self.assertEqual(1, len(self.records()))

    def test_ledger_write_is_atomic_and_preserves_existing_file_on_replace_failure(self):
        ledger = self.destination / ingest.LEDGER_FILENAME
        self.destination.mkdir()
        ledger.write_text('{"old": true}\n')
        with mock.patch.object(ingest.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                ingest.atomic_write_json(ledger, {"new": True})
        self.assertEqual('{"old": true}\n', ledger.read_text())
        self.assertEqual([], list(self.destination.glob("*.tmp")))

    def test_run_recovers_verified_folder_after_ledger_write_interruption(self):
        self.add_audio()
        real_atomic_write = ingest.atomic_write_json

        def fail_ledger_only(path, payload):
            if path.name == ingest.LEDGER_FILENAME:
                raise OSError("interrupted")
            return real_atomic_write(path, payload)

        with mock.patch.object(ingest, "atomic_write_json", side_effect=fail_ledger_only):
            failed, _ = self.execute()
        self.assertEqual(1, failed.failed)
        self.assertFalse((self.destination / ingest.LEDGER_FILENAME).exists())

        recovered, _ = self.execute()
        self.assertEqual((1, 0), (recovered.ingested, recovered.failed))
        self.assertEqual(1, len(self.records()))

    def test_successful_entry_survives_later_source_failure(self):
        self.add_audio("2026_06_26_12_12_01.m4a", b"good")
        self.execute()
        self.add_audio("bad-name.m4a", b"bad")
        summary, _ = self.execute()
        self.assertEqual(1, summary.failed)
        self.assertEqual(1, len(self.records()))

    def test_dry_run_checks_duration_and_fingerprint_but_writes_nothing(self):
        source = self.add_audio()
        calls = []

        def probe(path):
            calls.append(path)
            return 9.5

        with mock.patch.object(ingest, "sha256_file", wraps=ingest.sha256_file) as fingerprint:
            summary, output = self.execute(dry_run=True, probe=probe)
        self.assertEqual(1, summary.discovered)
        self.assertEqual([source.resolve()], calls)
        fingerprint.assert_called_once_with(source.resolve())
        self.assertIn("dry_run=true", output)
        self.assertFalse(self.destination.exists())

    def test_dry_run_reports_already_ingested_without_writing(self):
        self.add_audio()
        self.execute()
        ledger_before = (self.destination / ingest.LEDGER_FILENAME).read_bytes()
        summary, output = self.execute(dry_run=True)
        self.assertEqual(1, summary.skipped)
        self.assertIn("SKIP_ALREADY_INGESTED", output)
        self.assertEqual(ledger_before, (self.destination / ingest.LEDGER_FILENAME).read_bytes())

    def test_bad_source_does_not_stop_evaluation_of_later_sorted_source(self):
        self.add_audio("2026_06_26_12_12_01.m4a", b"first")
        self.add_audio("2026_06_26_12_12_02.m4a", b"second")
        visited = []

        def probe(path):
            visited.append(path.name)
            if path.name.endswith("01.m4a"):
                raise RuntimeError("broken")
            return 1.0

        summary, _ = self.execute(probe=probe)
        self.assertEqual(["2026_06_26_12_12_01.m4a", "2026_06_26_12_12_02.m4a"], visited)
        self.assertEqual((1, 1), (summary.failed, summary.ingested))

    def test_fingerprint_is_sha256_content_hash(self):
        source = self.add_audio(body=b"content")
        self.assertEqual(hashlib.sha256(b"content").hexdigest(), ingest.sha256_file(source))


class SegmentedPhoneIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source" / "2026" / "06" / "26"
        self.destination = self.root / "ingest"
        self.source.mkdir(parents=True)
        self.clock_value = datetime(2026, 7, 8, 10, 0, tzinfo=IST)

    def tearDown(self):
        self.temporary.cleanup()

    def execute(
        self,
        *,
        dry_run: bool = False,
        probe=None,
        stream_probe=None,
        concat_runner=None,
        operator_resolution=None,
    ):
        output = io.StringIO()
        with redirect_stdout(output):
            summary = ingest.run_ingest(
                ingest.IngestConfig(
                    self.root / "source", self.destination, dry_run, operator_resolution
                ),
                duration_probe=probe or (lambda _path: 42.25),
                stream_probe=stream_probe or (lambda _path: ("aac", "LC", 44100, 2)),
                concat_runner=concat_runner or ingest.run_concat_copy,
                clock=lambda: self.clock_value,
            )
        return summary, output.getvalue()

    def write_operator_resolution(
        self, identities, *, resolution="separate", schema_version=None, payload_overrides=None
    ):
        payload = {
            "schema_version": schema_version or ingest.OPERATOR_RESOLUTION_SCHEMA_VERSION,
            "resolution": resolution,
            "source_fingerprints_sha256": list(identities),
        }
        if payload_overrides:
            payload.update(payload_overrides)
        path = self.root / "operator-resolution.json"
        path.write_text(json.dumps(payload))
        return path

    def records(self):
        return json.loads((self.destination / ingest.LEDGER_FILENAME).read_text())["records"]

    def segment(self, name, duration, body=None, parent=None):
        folder = parent or self.source
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.m4a"
        path.write_bytes(body or name.encode())
        return ingest.SourceSegment(
            path.resolve(), ingest.parse_source_timestamp(path.name), duration,
            ingest.sha256_file(path),
        )

    def real_shape(self):
        return [
            self.segment("2026_06_26_12_12_01", 899.880522),
            self.segment("2026_06_26_12_27_03", 899.875420),
            self.segment("2026_06_26_12_42_04", 66.464853),
        ]

    def test_two_full_chunks_start_group_and_tail_terminates(self):
        segments = self.real_shape()
        candidates = ingest.group_source_segments(list(reversed(segments)))
        self.assertEqual(["GROUP"], [candidate.kind for candidate in candidates])
        self.assertEqual(tuple(segments), candidates[0].segments)
        self.assertEqual([True, True, False], [item.is_full_chunk for item in candidates[0].segments])
        self.assertAlmostEqual(2.119478, ingest.continuity_gap_seconds(segments[0], segments[1]), places=6)
        self.assertAlmostEqual(1.124580, ingest.continuity_gap_seconds(segments[1], segments[2]), places=6)

    def test_three_full_chunks_form_group(self):
        segments = [
            self.segment("2026_06_26_12_00_00", 900),
            self.segment("2026_06_26_12_15_01", 900),
            self.segment("2026_06_26_12_30_02", 900),
        ]
        self.assertEqual("GROUP", ingest.group_source_segments(segments)[0].kind)

    def test_short_proximity_does_not_group(self):
        segments = [
            self.segment("2026_06_26_12_00_00", 60),
            self.segment("2026_06_26_12_01_01", 30),
        ]
        self.assertEqual(["SINGLETON", "SINGLETON"], [c.kind for c in ingest.group_source_segments(segments)])

    def test_one_full_plus_tail_is_ambiguous(self):
        segments = [
            self.segment("2026_06_26_12_00_00", 900),
            self.segment("2026_06_26_12_15_01", 20),
        ]
        candidate = ingest.group_source_segments(segments)[0]
        self.assertEqual(("AMBIGUOUS", "one_full_chunk_plus_tail"), (candidate.kind, candidate.reason))

    def test_operator_resolution_separates_exact_ambiguous_pair_and_records_provenance(self):
        grouped = self.real_shape()
        full = self.segment("2026_06_26_12_55_00", 900, body=b"operator-full")
        tail = self.segment("2026_06_26_13_10_01", 100, body=b"operator-tail")
        pair = (full, tail)
        resolution_path = self.write_operator_resolution(
            [item.fingerprint_sha256 for item in pair]
        )
        durations = {
            item.source_path: item.duration_seconds
            for item in list(grouped) + list(pair)
        }
        total = sum(item.duration_seconds for item in grouped)

        def probe(path):
            return durations.get(path.resolve(), total)

        summary, _ = self.execute(
            probe=probe,
            stream_probe=lambda _path: ("aac", "LC", "44100", 2),
            concat_runner=lambda _list, output: output.write_bytes(b"joined"),
            operator_resolution=resolution_path,
        )
        self.assertEqual((3, 1, 2, 0, 0), (
            summary.ingested, summary.grouped, summary.singleton,
            summary.ambiguous, summary.failed,
        ))
        resolved_records = [
            record for record in self.records().values()
            if record.get("operator_resolution") is not None
        ]
        self.assertEqual(2, len(resolved_records))
        for record in resolved_records:
            self.assertEqual(
                {
                    "schema_version": ingest.OPERATOR_RESOLUTION_SCHEMA_VERSION,
                    "resolution": "separate",
                    "source_fingerprints_sha256": [
                        item.fingerprint_sha256 for item in pair
                    ],
                },
                record["operator_resolution"],
            )

    def test_operator_resolution_rejects_duplicate_identity_without_writes(self):
        full = self.segment("2026_06_26_12_00_00", 900)
        resolution_path = self.write_operator_resolution(
            [full.fingerprint_sha256, full.fingerprint_sha256]
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.execute(operator_resolution=resolution_path)
        self.assertFalse(self.destination.exists())

    def test_operator_resolution_rejects_unlisted_identity_without_writes(self):
        full = self.segment("2026_06_26_12_00_00", 900)
        tail = self.segment("2026_06_26_12_15_01", 20)
        resolution_path = self.write_operator_resolution(
            [full.fingerprint_sha256, "0" * 64]
        )
        with self.assertRaisesRegex(ValueError, "unlisted"):
            self.execute(operator_resolution=resolution_path)
        self.assertFalse(self.destination.exists())

    def test_legacy_separate_resolution_schema_remains_accepted(self):
        full = self.segment("2026_06_26_12_00_00", 900)
        tail = self.segment("2026_06_26_12_15_01", 20)
        resolution_path = self.write_operator_resolution(
            [full.fingerprint_sha256, tail.fingerprint_sha256], schema_version=1
        )
        summary, _ = self.execute(
            probe=lambda path: {full.source_path: 900, tail.source_path: 20}[path.resolve()],
            operator_resolution=resolution_path,
        )
        self.assertEqual((2, 0, 0), (summary.ingested, summary.ambiguous, summary.failed))

    def test_discard_resolution_records_complete_candidates_without_normalizing_audio(self):
        grouped = self.real_shape()
        singleton = self.segment("2026_06_26_13_00_00", 20, body=b"discard-singleton")
        resolution_path = self.write_operator_resolution(
            [item.fingerprint_sha256 for item in [*grouped, singleton]],
            resolution="discard",
        )
        durations = {item.source_path: item.duration_seconds for item in [*grouped, singleton]}

        summary, output = self.execute(
            probe=lambda path: durations[path.resolve()],
            operator_resolution=resolution_path,
        )
        self.assertEqual((0, 2, 0), (summary.ingested, summary.discarded, summary.failed))
        self.assertIn("DISCARDED", output)
        self.assertEqual([], [path for path in self.destination.iterdir() if path.is_dir()])
        self.assertEqual(2, len(self.records()))
        self.assertTrue(all(record["status"] == "discarded" for record in self.records().values()))

        rerun, rerun_output = self.execute(probe=lambda path: durations[path.resolve()])
        self.assertEqual((0, 2, 0), (rerun.ingested, rerun.skipped, rerun.failed))
        self.assertIn("SKIP_ALREADY_DISCARDED", rerun_output)
        self.assertEqual([], [path for path in self.destination.iterdir() if path.is_dir()])

    def test_discard_resolution_rejects_partial_group_without_writes(self):
        grouped = self.real_shape()
        resolution_path = self.write_operator_resolution(
            [grouped[0].fingerprint_sha256], resolution="discard"
        )
        with self.assertRaisesRegex(ValueError, "every segment"):
            self.execute(
                probe=lambda path: {item.source_path: item.duration_seconds for item in grouped}[path.resolve()],
                operator_resolution=resolution_path,
            )
        self.assertFalse(self.destination.exists())

    def test_join_resolution_overrides_gap_only_with_explicit_chronological_order(self):
        first = self.segment("2026_06_26_12_00_00", 900, body=b"join-first")
        second = self.segment("2026_06_26_12_15_10", 120, body=b"join-second")
        resolution_path = self.write_operator_resolution(
            [first.fingerprint_sha256, second.fingerprint_sha256], resolution="join"
        )
        durations = {first.source_path: first.duration_seconds, second.source_path: second.duration_seconds}
        expected_duration = sum(durations.values())

        summary, _ = self.execute(
            probe=lambda path: durations.get(path.resolve(), expected_duration),
            stream_probe=lambda _path: ("aac", "LC", "44100", 2),
            concat_runner=lambda _list, output: output.write_bytes(b"joined"),
            operator_resolution=resolution_path,
        )
        self.assertEqual((1, 1, 0), (summary.ingested, summary.grouped, summary.failed))
        record = next(iter(self.records().values()))
        self.assertEqual("join", record["operator_resolution"]["resolution"])
        self.assertEqual(2, record["segment_count"])

    def test_join_resolution_rejects_non_chronological_order_without_writes(self):
        first = self.segment("2026_06_26_12_00_00", 900, body=b"join-first")
        second = self.segment("2026_06_26_12_15_10", 120, body=b"join-second")
        resolution_path = self.write_operator_resolution(
            [second.fingerprint_sha256, first.fingerprint_sha256], resolution="join"
        )
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.execute(
                probe=lambda path: {first.source_path: 900, second.source_path: 120}[path.resolve()],
                operator_resolution=resolution_path,
            )
        self.assertFalse(self.destination.exists())

    def test_continuity_outside_bounds_does_not_group_or_bridge(self):
        for second_start in ("2026_06_26_12_14_58", "2026_06_26_12_15_06", "2026_06_26_12_30_00"):
            with self.subTest(second_start=second_start):
                first = self.segment("2026_06_26_12_00_00", 900, body=(second_start + "a").encode())
                second = self.segment(second_start, 900, body=(second_start + "b").encode())
                self.assertEqual(2, len(ingest.group_source_segments([first, second])))
                first.source_path.unlink()
                second.source_path.unlink()

    def test_directory_and_date_change_stop_grouping(self):
        other = self.root / "source" / "2026" / "06" / "27"
        first = self.segment("2026_06_26_23_45_00", 900)
        second = self.segment("2026_06_27_00_00_01", 900, parent=other)
        self.assertEqual(2, len(ingest.group_source_segments([first, second])))

    def test_duplicate_evidence_fails_group_candidate(self):
        segments = self.real_shape()[:2]
        duplicate_hash = ingest.SourceSegment(
            segments[1].source_path, segments[1].created_at,
            segments[1].duration_seconds, segments[0].fingerprint_sha256,
        )
        self.assertEqual("FAILED", ingest.group_source_segments([segments[0], duplicate_hash])[0].kind)
        duplicate_time = ingest.SourceSegment(
            segments[1].source_path, segments[0].created_at,
            segments[1].duration_seconds, segments[1].fingerprint_sha256,
        )
        duplicate_candidate = ingest.group_source_segments([segments[0], duplicate_time])[0]
        self.assertEqual(("FAILED", "duplicate_timestamp"), (duplicate_candidate.kind, duplicate_candidate.reason))

    def test_second_tail_does_not_join_group(self):
        segments = self.real_shape()
        fourth = self.segment("2026_06_26_12_43_11", 10)
        candidates = ingest.group_source_segments(segments + [fourth])
        self.assertEqual(["GROUP", "SINGLETON"], [candidate.kind for candidate in candidates])

    def test_group_identity_is_namespaced_stable_and_segment_sensitive(self):
        segments = tuple(self.real_shape())
        first = ingest.group_fingerprint_sha256(segments)
        moved_values = [(item.created_at.isoformat(), item.fingerprint_sha256) for item in segments]
        self.assertEqual(first, ingest.group_fingerprint_from_values(moved_values))
        self.assertNotEqual(first, segments[0].fingerprint_sha256)
        changed = list(moved_values)
        changed[-1] = (changed[-1][0], "f" * 64)
        self.assertNotEqual(first, ingest.group_fingerprint_from_values(changed))
        self.assertNotEqual(first, ingest.group_fingerprint_sha256(segments[:-1]))
        added = segments + (self.segment("2026_06_26_12_43_11", 10),)
        self.assertNotEqual(first, ingest.group_fingerprint_sha256(added))
        segments[0].source_path.write_bytes(b"changed bytes")
        changed_segment = ingest.SourceSegment(
            segments[0].source_path, segments[0].created_at,
            segments[0].duration_seconds, ingest.sha256_file(segments[0].source_path),
        )
        self.assertNotEqual(
            first, ingest.group_fingerprint_sha256((changed_segment,) + segments[1:])
        )

    def test_concat_copy_command_is_exact(self):
        concat_list = self.root / "list.txt"
        output = self.root / "audio.m4a"
        with mock.patch.object(ingest, "find_ffmpeg", return_value="/ffmpeg"), mock.patch.object(
            ingest.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            ingest.run_concat_copy(concat_list, output)
        self.assertEqual(
            ["/ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-map", "0:a:0", "-c", "copy", "-y", str(output)],
            run.call_args.args[0],
        )
        self.assertTrue(run.call_args.kwargs["check"])

    def test_concat_list_quoting_supports_spaces_and_apostrophes_and_rejects_controls(self):
        path = Path("/source/Meeting files/Alex's recording.m4a")
        self.assertEqual(
            "file '/source/Meeting files/Alex'\\''s recording.m4a'\n",
            ingest._concat_list_line(path),
        )
        for unsafe in ("line\nbreak.m4a", "carriage\rreturn.m4a", "nul\x00byte.m4a"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                ingest._concat_list_line(Path("/source") / unsafe)

    def test_group_normalization_orders_concat_and_persists_canonical_facts(self):
        segments = tuple(self.real_shape())
        candidate = ingest.SourceCandidate("GROUP", segments)
        target = self.destination / ingest.deterministic_group_folder_name(candidate.created_at, candidate.identity)
        source_before = [item.source_path.read_bytes() for item in segments]
        captured = {}

        def concat_runner(concat_list, output):
            captured["list"] = concat_list.read_text()
            output.write_bytes(b"canonical-group-audio")

        metadata = ingest._normalize_group(
            candidate, target, self.clock_value,
            duration_probe=lambda path: sum(item.duration_seconds for item in segments),
            stream_probe=lambda path: ("aac", "LC", "44100", 2),
            concat_runner=concat_runner,
        )
        self.assertEqual([item.source_path.read_bytes() for item in segments], source_before)
        positions = [captured["list"].index(str(item.source_path)) for item in segments]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual("concat_copy", metadata["audio_handling"])
        self.assertEqual(candidate.identity, metadata["source_group"]["group_fingerprint_sha256"])
        self.assertEqual(len(segments), metadata["source_group"]["segment_count"])
        self.assertEqual(segments[0].created_at.isoformat(), metadata["created_at"])
        self.assertEqual(
            hashlib.sha256(b"canonical-group-audio").hexdigest(),
            metadata["source_fingerprint_sha256"],
        )
        self.assertEqual(
            {ingest.CANONICAL_AUDIO_FILENAME, ingest.METADATA_FILENAME},
            {path.name for path in target.iterdir()},
        )
        serialized = json.dumps(metadata).lower()
        self.assertNotIn("transcript_text", serialized)
        self.assertNotIn("transcript_body", serialized)

    def test_group_normalization_fails_closed_for_changed_stream_empty_or_duration(self):
        segments = tuple(self.real_shape()[:2])
        candidate = ingest.SourceCandidate("GROUP", segments)
        cases = (
            (lambda path: ("aac", path.name), lambda _l, out: out.write_bytes(b"x"), sum(x.duration_seconds for x in segments), "incompatible"),
            (lambda path: ("aac",), lambda _l, out: out.write_bytes(b""), sum(x.duration_seconds for x in segments), "empty"),
            (lambda path: ("aac",), lambda _l, out: out.write_bytes(b"x"), 1.0, "duration"),
        )
        for stream_probe, runner, duration, label in cases:
            target = self.destination / label
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                ingest._normalize_group(
                    candidate, target, self.clock_value,
                    duration_probe=lambda _path, value=duration: value,
                    stream_probe=stream_probe, concat_runner=runner,
                )
            self.assertFalse(target.exists())

    def test_nonfinite_or_nonpositive_canonical_duration_fails_closed(self):
        segments = tuple(self.real_shape()[:2])
        candidate = ingest.SourceCandidate("GROUP", segments)
        for index, duration in enumerate((float("nan"), float("inf"), -1.0, 0.0)):
            target = self.destination / f"invalid-duration-{index}"
            with self.subTest(duration=duration), self.assertRaisesRegex(
                RuntimeError, "invalid canonical duration"
            ):
                ingest._normalize_group(
                    candidate, target, self.clock_value,
                    duration_probe=lambda _path, value=duration: value,
                    stream_probe=lambda _path: ("aac",),
                    concat_runner=lambda _list, output: output.write_bytes(b"audio"),
                )
            self.assertFalse(target.exists())

    def test_concat_failure_and_nonregular_output_leave_no_visible_target(self):
        segments = tuple(self.real_shape()[:2])
        candidate = ingest.SourceCandidate("GROUP", segments)

        def fail_concat(_list, _output):
            raise subprocess.CalledProcessError(1, ["ffmpeg"])

        def directory_output(_list, output):
            output.mkdir()

        for label, runner in (("concat_failure", fail_concat), ("nonregular", directory_output)):
            target = self.destination / label
            with self.subTest(label=label), self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
                ingest._normalize_group(
                    candidate, target, self.clock_value,
                    duration_probe=lambda _path: 1800,
                    stream_probe=lambda _path: ("aac",), concat_runner=runner,
                )
            self.assertFalse(target.exists())

    def test_source_bytes_are_reverified_before_concat(self):
        segments = tuple(self.real_shape()[:2])
        candidate = ingest.SourceCandidate("GROUP", segments)
        segments[0].source_path.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed before concat"):
            ingest._normalize_group(
                candidate, self.destination / "target", self.clock_value,
                duration_probe=lambda _path: 1800, stream_probe=lambda _path: ("aac",),
                concat_runner=lambda _list, _out: self.fail("concat must not run"),
            )

    def test_real_shape_dry_run_reports_one_group_and_writes_nothing(self):
        segments = self.real_shape()
        durations = {item.source_path: item.duration_seconds for item in segments}
        output = io.StringIO()
        concat = mock.Mock()
        with redirect_stdout(output):
            summary = ingest.run_ingest(
                ingest.IngestConfig(self.root / "source", self.destination, True),
                duration_probe=lambda path: durations[path.resolve()],
                stream_probe=lambda _path: ("aac",), concat_runner=concat,
                clock=lambda: self.clock_value,
            )
        text = output.getvalue()
        self.assertEqual((3, 1, 0), (summary.discovered, summary.grouped, summary.ambiguous))
        self.assertIn("GROUP policy=asr_900s_segment_group_v1", text)
        self.assertIn("gaps=2.119478,1.124580", text)
        self.assertIn("classes=full,full,tail", text)
        self.assertIn("group_fingerprint=", text)
        self.assertIn("dry_run=true", text)
        self.assertFalse(self.destination.exists())
        concat.assert_not_called()

    def test_ambiguous_dry_run_fails_closed_without_writes(self):
        first = self.segment("2026_06_26_12_00_00", 900)
        tail = self.segment("2026_06_26_12_15_01", 20)
        durations = {first.source_path: 900, tail.source_path: 20}
        output = io.StringIO()
        with redirect_stdout(output):
            summary = ingest.run_ingest(
                ingest.IngestConfig(self.root / "source", self.destination, True),
                duration_probe=lambda path: durations[path.resolve()],
            )
        self.assertEqual((1, 1), (summary.ambiguous, summary.failed))
        self.assertIn("AMBIGUOUS", output.getvalue())
        self.assertIn("reason=one_full_chunk_plus_tail", output.getvalue())
        self.assertFalse(self.destination.exists())

    def test_group_ingest_creates_one_record_skips_and_recovers_interrupted_ledger(self):
        segments = self.real_shape()
        durations = {item.source_path: item.duration_seconds for item in segments}
        total = sum(durations.values())

        def probe(path):
            return total if path.name == ingest.CANONICAL_AUDIO_FILENAME else durations[path.resolve()]

        def concat(_list, output):
            output.write_bytes(b"joined")

        real_atomic = ingest.atomic_write_json
        failed_once = {"value": False}

        def interrupt_ledger(path, payload):
            if path.name == ingest.LEDGER_FILENAME and not failed_once["value"]:
                failed_once["value"] = True
                raise OSError("interrupted")
            return real_atomic(path, payload)

        with mock.patch.object(ingest, "atomic_write_json", side_effect=interrupt_ledger):
            first = ingest.run_ingest(
                ingest.IngestConfig(self.root / "source", self.destination),
                duration_probe=probe, stream_probe=lambda _path: ("aac",),
                concat_runner=concat, clock=lambda: self.clock_value,
            )
        self.assertEqual(1, first.failed)
        self.assertFalse((self.destination / ingest.LEDGER_FILENAME).exists())

        recovered = ingest.run_ingest(
            ingest.IngestConfig(self.root / "source", self.destination),
            duration_probe=probe, stream_probe=lambda _path: ("aac",),
            concat_runner=lambda _l, _o: self.fail("verified target should recover"),
            clock=lambda: self.clock_value,
        )
        self.assertEqual((1, 0), (recovered.ingested, recovered.failed))
        ledger = json.loads((self.destination / ingest.LEDGER_FILENAME).read_text())
        self.assertEqual(1, len(ledger["records"]))
        record = next(iter(ledger["records"].values()))
        self.assertEqual(ingest.GROUPING_POLICY, record["source_group_policy"])
        rerun = ingest.run_ingest(
            ingest.IngestConfig(self.root / "source", self.destination),
            duration_probe=probe, stream_probe=lambda _path: ("aac",), concat_runner=concat,
        )
        self.assertEqual(1, rerun.skipped)

    def test_partial_or_mismatched_group_target_is_rejected(self):
        segments = self.real_shape()
        durations = {item.source_path: item.duration_seconds for item in segments}
        total = sum(durations.values())

        def probe(path):
            return total if path.name == ingest.CANONICAL_AUDIO_FILENAME else durations[path.resolve()]

        candidate = ingest.SourceCandidate("GROUP", tuple(segments))
        target = self.destination / ingest.deterministic_group_folder_name(
            candidate.created_at, candidate.identity
        )
        target.mkdir(parents=True)
        partial = ingest.run_ingest(
            ingest.IngestConfig(self.root / "source", self.destination),
            duration_probe=probe, stream_probe=lambda _path: ("aac",),
            concat_runner=lambda _l, _o: self.fail("partial target must fail first"),
        )
        self.assertEqual(1, partial.failed)
        self.assertFalse((self.destination / ingest.LEDGER_FILENAME).exists())

        target.rmdir()
        ingest.run_ingest(
            ingest.IngestConfig(self.root / "source", self.destination),
            duration_probe=probe, stream_probe=lambda _path: ("aac",),
            concat_runner=lambda _l, output: output.write_bytes(b"joined"),
            clock=lambda: self.clock_value,
        )
        (self.destination / ingest.LEDGER_FILENAME).unlink()
        metadata_path = target / ingest.METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text())
        metadata["source_group"]["group_fingerprint_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
        mismatch = ingest.run_ingest(
            ingest.IngestConfig(self.root / "source", self.destination),
            duration_probe=probe, stream_probe=lambda _path: ("aac",),
            concat_runner=lambda _l, _o: self.fail("mismatched target must not concatenate"),
        )
        self.assertEqual(1, mismatch.failed)
        self.assertFalse((self.destination / ingest.LEDGER_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
