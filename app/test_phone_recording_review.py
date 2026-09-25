from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

import meetingintel_cli as cli
import phone_drive_fetch as fetch
import phone_recording_ingest as ingest
import phone_recording_review as review


class PhoneRecordingReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.date_root = self.staging / "2026" / "07" / "23"
        self.date_root.mkdir(parents=True)
        self.ingest_root = self.root / "ingest"
        self.fetch_ledger = self.root / "state" / "fetch.json"
        self.config = review.ReviewConfig(self.staging, self.fetch_ledger, self.ingest_root)
        self.now = datetime(2026, 7, 23, 18, 0, tzinfo=ingest.IST)
        self.durations: dict[str, float] = {}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_segment(self, timestamp: str, duration: float, body: bytes | None = None) -> Path:
        path = self.date_root / f"{timestamp}.m4a"
        path.write_bytes(body or timestamp.encode())
        self.durations[path.name] = duration
        return path

    def write_fetch_ledger(self, paths: list[Path], *, statuses: dict[str, str] | None = None) -> None:
        records = {}
        statuses = statuses or {}
        for index, path in enumerate(paths, 1):
            records[f"drive-{index}"] = {
                "status": statuses.get(path.name, "settled_ready"),
                "local_relative_path": str(path.relative_to(self.staging)),
                "size": path.stat().st_size,
                "sha256": ingest.sha256_file(path),
                "revision": f"rev-{index}",
            }
        self.fetch_ledger.parent.mkdir(parents=True, exist_ok=True)
        self.fetch_ledger.write_text(json.dumps({"schema_version": 1, "records": records}))

    def duration_probe(self, path: Path) -> float:
        return self.durations.get(path.name, sum(self.durations.values()))

    def runner(self, config: ingest.IngestConfig, *, duration_probe):
        return ingest.run_ingest(
            config,
            duration_probe=duration_probe,
            stream_probe=lambda _path: ("aac", "LC", "44100", 2),
            concat_runner=lambda _list, output: output.write_bytes(b"joined-audio"),
            clock=lambda: self.now,
        )

    def run_review(
        self,
        scope="today",
        *,
        inputs=(),
        dry_run=False,
        terminal=True,
        runner=None,
        process_runner=None,
    ):
        output = io.StringIO()
        answers = iter(inputs)

        def input_func(_prompt):
            return next(answers)

        with redirect_stdout(output):
            result = review.run_review(
                scope,
                config=self.config,
                clock=lambda: self.now,
                input_func=input_func,
                terminal_isatty=lambda: terminal,
                dry_run=dry_run,
                duration_probe=self.duration_probe,
                ingest_runner=runner or self.runner,
                process_runner=process_runner,
            )
        return result, output.getvalue()

    def ingest_records(self):
        path = self.ingest_root / ingest.LEDGER_FILENAME
        return json.loads(path.read_text())["records"] if path.exists() else {}

    def synthetic_review(self, group_indexes=((1,), (2,))) -> review.ReviewSet:
        segment_count = max(
            (index for group in group_indexes for index in group),
            default=1,
        )
        segments = tuple(
            review.ReviewSegment(
                self.date_root / f"chunk-{index}.m4a",
                self.now.replace(hour=12, minute=index),
                30.0,
                str(index) * 64,
                10,
            )
            for index in range(1, segment_count + 1)
        )
        identities = tuple(segment.fingerprint_sha256 for segment in segments)
        suggestion = ingest.OperatorResolution(
            ingest.OPERATOR_PLAN_SCHEMA_VERSION,
            "plan",
            identities,
            tuple(tuple(identities[index - 1] for index in group) for group in group_indexes),
            (),
        )
        return review.ReviewSet(
            self.date_root,
            self.now.date(),
            segments,
            (),
            suggestion,
            "synthetic continuity policy",
            "raw:2026/07/23:2026-07-23",
        )

    def plain_plan(self, selected: review.ReviewSet, inputs: tuple[object, ...]):
        answers = iter(inputs)

        def input_func(_prompt):
            answer = next(answers)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        output = io.StringIO()
        with redirect_stdout(output):
            result = review._interactive_plain_plan(selected, input_func)
        return result, output.getvalue()

    def test_inbox_edit_sequence_cannot_restore_original_suggestion(self):
        selected = self.synthetic_review(((1,), (2,)))
        result, output = self.plain_plan(selected, ("4", "1,2", "a", "y"))
        self.assertIsNotNone(result)
        self.assertEqual(1, len(result.groups))
        self.assertIn("plan is unchanged", output)

    def test_plain_evidence_renders_before_one_and_multi_group_decisions(self):
        one_group = self.synthetic_review(((1, 2),))
        multi_group = self.synthetic_review(((1,), (2,)))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                0,
                review._run_review_sets(
                    (one_group, multi_group),
                    config=self.config,
                    input_func=lambda _prompt: "q",
                    terminal_isatty=lambda: True,
                    dry_run=True,
                    duration_probe=self.duration_probe,
                    ingest_runner=self.runner,
                    process_runner=None,
                    clock=lambda: self.now,
                    plain_language=True,
                ),
            )
        rendered = output.getvalue()
        self.assertLess(rendered.index("Filename"), rendered.index("Suggested plan:"))
        self.assertGreaterEqual(rendered.count("Suggested plan:"), 2)

    def test_plain_confirmation_controls_are_fail_closed(self):
        selected = self.synthetic_review(((1, 2),))
        cases = {
            "y": (("2", "y"), True),
            "b_then_q": (("2", "b", "q"), False),
            "q": (("2", "q"), False),
            "invalid_then_y": (("2", "invalid", "y"), True),
            "blank": (("2", ""), False),
            "eof": (("2", EOFError()), False),
            "ctrl_c": (("2", KeyboardInterrupt()), False),
            "exhaustion": (("2", "x", "x", "x"), False),
        }
        for label, (answers, succeeds) in cases.items():
            with self.subTest(label=label):
                result, _output = self.plain_plan(selected, answers)
                self.assertEqual(succeeds, result is not None)

    def test_plain_suggested_join_separate_custom_and_discard_paths(self):
        cases = (
            (("1", "y"), ((1,), (2,))),
            (("2", "y"), ((1, 2),)),
            (("3", "y"), ((1,), (2,))),
            (("4", "1,2", "y"), ((1, 2),)),
        )
        for answers, expected_groups in cases:
            with self.subTest(answers=answers):
                selected = self.synthetic_review(((1,), (2,)))
                result, _output = self.plain_plan(selected, answers)
                actual = tuple(
                    tuple(
                        selected.segments.index(
                            next(
                                segment
                                for segment in selected.segments
                                if segment.fingerprint_sha256 == identity
                            )
                        )
                        + 1
                        for identity in group
                    )
                    for group in result.groups
                )
                self.assertEqual(expected_groups, actual)

        selected = self.synthetic_review(((1,), (2,), (3,)))
        result, _output = self.plain_plan(selected, ("5", "2", "DISCARD"))
        self.assertEqual(
            (selected.segments[1].fingerprint_sha256,),
            result.discard_source_fingerprints_sha256,
        )

    def test_review_numbers_chronologically_and_formats_calculated_end_and_signed_gap(self):
        first = self.add_segment("2026_07_23_12_54_59", 900.3)
        second = self.add_segment("2026_07_23_13_10_01", 900.1)
        self.write_fetch_ledger([second, first])
        sets = review.collect_review_sets(self.config, duration_probe=self.duration_probe)
        output = review.format_review(sets[0])
        self.assertIn(" 1  2026_07_23_12_54_59.m4a", output)
        self.assertIn("12:54:59", output)
        self.assertIn("13:09:59.300", output)
        self.assertIn("+1.700s", output)
        self.assertLess(output.index("12_54_59"), output.index("13_10_01"))

    def test_existing_policy_produces_complete_suggestion(self):
        paths = [
            self.add_segment("2026_07_23_12_00_00", 900),
            self.add_segment("2026_07_23_12_15_01", 900),
            self.add_segment("2026_07_23_12_30_02", 30),
        ]
        self.write_fetch_ledger(paths)
        sets = review.collect_review_sets(self.config, duration_probe=self.duration_probe)
        self.assertIsNotNone(sets[0].suggestion)
        self.assertEqual(1, len(sets[0].suggestion.groups))
        self.assertIn(ingest.GROUPING_POLICY, sets[0].suggestion_reason)

    def test_ambiguous_candidate_has_no_default_acceptance(self):
        paths = [
            self.add_segment("2026_07_23_12_00_00", 900),
            self.add_segment("2026_07_23_12_15_01", 20),
        ]
        self.write_fetch_ledger(paths)
        result, output = self.run_review(dry_run=True)
        self.assertEqual(0, result)
        self.assertIn("No safe automatic suggestion.", output)
        self.assertIn("one_full_chunk_plus_tail", output)
        self.assertFalse(self.ingest_root.exists())

    def test_custom_partitions_are_validated_and_supported(self):
        self.assertEqual(((1, 2), (3,)), review._parse_partition("1,2;3", 3))
        self.assertEqual(((1,), (2, 3)), review._parse_partition("1;2,3", 3))
        self.assertEqual(((1,), (2,), (3,)), review._parse_partition("1;2;3", 3))
        for value in ("1,1;2,3", "1,3", "0,2,3", "1,2;", "2,1;3"):
            with self.subTest(value=value), self.assertRaises(review.ReviewError):
                review._parse_partition(value, 3)

    def test_accept_suggested_plan_normalizes_without_processing(self):
        paths = [self.add_segment("2026_07_23_12_00_00", 42)]
        before = paths[0].read_bytes()
        self.write_fetch_ledger(paths)
        result, output = self.run_review(inputs=("a", "c"))
        self.assertEqual(0, result)
        self.assertIn("Normalized: 1 phone meetings", output)
        self.assertIn("Production processing: not started", output)
        records = self.ingest_records()
        record = next(iter(records.values()))
        self.assertEqual(3, record["operator_resolution"]["schema_version"])
        self.assertEqual(before, paths[0].read_bytes())

    def test_process_now_y_receives_only_exact_new_normalized_source(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        process = mock.Mock(return_value=0)
        result, output = self.run_review(
            inputs=("a", "c", "y"), process_runner=process
        )
        self.assertEqual(0, result)
        process.assert_called_once()
        sources = process.call_args.args[0]
        self.assertEqual(1, len(sources))
        self.assertEqual(path.name, sources[0].source_filename)
        self.assertIn("Process exactly this meeting now?", output)
        self.assertIn("Processing completed for the exact normalized set.", output)

    def test_production_runner_is_not_reached_until_explicit_y(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        preflight = mock.Mock(return_value=0)
        process = lambda _sources: preflight()
        result, _output = self.run_review(
            inputs=("a", "c", "n"), process_runner=process
        )
        self.assertEqual(0, result)
        preflight.assert_not_called()

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.date_root = self.staging / "2026" / "07" / "23"
        self.date_root.mkdir(parents=True)
        self.ingest_root = self.root / "ingest"
        self.fetch_ledger = self.root / "state" / "fetch.json"
        self.config = review.ReviewConfig(self.staging, self.fetch_ledger, self.ingest_root)
        self.durations = {}
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        result, _output = self.run_review(
            inputs=("a", "c", "y"), process_runner=process
        )
        self.assertEqual(0, result)
        preflight.assert_called_once_with()

    def test_process_now_multiple_outputs_excludes_unrelated_backlog(self):
        old_date_root = self.staging / "2026" / "07" / "22"
        old_date_root.mkdir(parents=True)
        old_path = old_date_root / "2026_07_22_12_00_00.m4a"
        old_path.write_bytes(b"older backlog")
        self.durations[old_path.name] = 20
        ingest.run_ingest(
            ingest.IngestConfig(old_date_root, self.ingest_root),
            duration_probe=self.duration_probe,
            clock=lambda: self.now,
        )
        paths = [
            self.add_segment("2026_07_23_12_00_00", 20),
            self.add_segment("2026_07_23_12_00_30", 20),
            self.add_segment("2026_07_23_12_01_00", 20),
        ]
        self.write_fetch_ledger([old_path, *paths])

        def grouped_runner(config, *, duration_probe):
            return ingest.run_ingest(
                config,
                duration_probe=lambda path: 40.0 if path.name == ingest.CANONICAL_AUDIO_FILENAME else duration_probe(path),
                stream_probe=lambda _path: ("aac", "LC", "44100", 2),
                concat_runner=lambda _list, output: output.write_bytes(b"joined-audio"),
                clock=lambda: self.now,
            )

        process = mock.Mock(return_value=0)
        result, output = self.run_review(
            inputs=("e", "1,2;3", "c", "y"),
            runner=grouped_runner,
            process_runner=process,
        )
        self.assertEqual(0, result)
        sources = process.call_args.args[0]
        self.assertEqual(2, len(sources))
        self.assertEqual(
            {"2026_07_23_12_00_00.m4a", "2026_07_23_12_01_00.m4a"},
            {source.source_filename for source in sources},
        )
        self.assertIn("Process exactly these 2 meetings now?", output)
        self.assertNotIn(old_path.name, {source.source_filename for source in sources})

    def test_process_now_list_displays_safe_exact_set_then_reprompts(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        process = mock.Mock(return_value=0)
        result, output = self.run_review(
            inputs=("a", "c", "l", "y"), process_runner=process
        )
        self.assertEqual(0, result)
        process.assert_called_once()
        self.assertIn("2026-07-23 12:00:00 IST | phone | 2026_07_23_12_00_00.m4a", output)
        self.assertNotIn(str(self.root), output)
        self.assertNotIn("sha256", output.lower())

    def test_process_now_declines_and_invalid_inputs_never_start_processing(self):
        for answers in (("a", "c", "n"), ("a", "c", ""), ("a", "c", "bad", "bad", "bad"), ("a", "c")):
            with self.subTest(answers=answers):
                path = self.add_segment("2026_07_23_12_00_00", 42, body=str(answers).encode())
                self.write_fetch_ledger([path])
                process = mock.Mock(return_value=0)
                result, _output = self.run_review(inputs=answers, process_runner=process)
                self.assertEqual(0, result)
                process.assert_not_called()
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.root = Path(self.temporary.name)
                self.staging = self.root / "staging"
                self.date_root = self.staging / "2026" / "07" / "23"
                self.date_root.mkdir(parents=True)
                self.ingest_root = self.root / "ingest"
                self.fetch_ledger = self.root / "state" / "fetch.json"
                self.config = review.ReviewConfig(self.staging, self.fetch_ledger, self.ingest_root)
                self.durations = {}

    def test_process_now_dry_run_non_tty_and_discard_only_never_start(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        process = mock.Mock(return_value=0)
        result, _output = self.run_review(inputs=("a", "c", "y"), dry_run=True, process_runner=process)
        self.assertEqual(0, result)
        process.assert_not_called()

    def test_process_now_ctrl_c_never_starts_processing(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        answers = iter(("a", "c"))
        process = mock.Mock(return_value=0)

        def input_func(prompt):
            if prompt == "Choice: ":
                raise KeyboardInterrupt
            return next(answers)

        output = io.StringIO()
        with redirect_stdout(output):
            result = review.run_review(
                config=self.config,
                clock=lambda: self.now,
                input_func=input_func,
                terminal_isatty=lambda: True,
                duration_probe=self.duration_probe,
                ingest_runner=self.runner,
                process_runner=process,
            )
        self.assertEqual(0, result)
        process.assert_not_called()

        process.reset_mock()
        result, _output = self.run_review(inputs=("a", "c", "y"), terminal=False, process_runner=process)
        self.assertEqual(0, result)
        process.assert_not_called()

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.date_root = self.staging / "2026" / "07" / "23"
        self.date_root.mkdir(parents=True)
        self.ingest_root = self.root / "ingest"
        self.fetch_ledger = self.root / "state" / "fetch.json"
        self.config = review.ReviewConfig(self.staging, self.fetch_ledger, self.ingest_root)
        self.durations = {}
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        result, _output = self.run_review(inputs=("d", "1", "DISCARD", "y"), process_runner=process)
        self.assertEqual(0, result)
        process.assert_not_called()

    def test_normalization_failure_never_asks_or_processes(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        process = mock.Mock(return_value=0)
        failed = lambda _config, *, duration_probe: ingest.IngestSummary(failed=1)
        result, output = self.run_review(
            inputs=("a", "c", "y"), runner=failed, process_runner=process
        )
        self.assertEqual(1, result)
        process.assert_not_called()
        self.assertNotIn("Process exactly", output)

    def test_processing_failure_preserves_normalized_sources_and_is_nonzero(self):
        path = self.add_segment("2026_07_23_12_00_00", 42)
        self.write_fetch_ledger([path])
        process = mock.Mock(return_value=1)
        result, output = self.run_review(
            inputs=("a", "c", "y"), process_runner=process
        )
        self.assertEqual(1, result)
        process.assert_called_once()
        self.assertTrue(self.ingest_root.exists())
        self.assertIn("Normalized source remains available.", output)
        self.assertIn("mi --audio-first new list", output)

    def test_edit_partition_and_provenance_support_join_plus_singleton(self):
        paths = [
            self.add_segment("2026_07_23_12_00_00", 20),
            self.add_segment("2026_07_23_12_00_30", 20),
            self.add_segment("2026_07_23_12_01_00", 20),
        ]
        self.write_fetch_ledger(paths)
        def grouped_runner(config, *, duration_probe):
            return ingest.run_ingest(
                config,
                duration_probe=lambda path: 40.0 if path.name == ingest.CANONICAL_AUDIO_FILENAME else duration_probe(path),
                stream_probe=lambda _path: ("aac", "LC", "44100", 2),
                concat_runner=lambda _list, output: output.write_bytes(b"joined-audio"),
                clock=lambda: self.now,
            )

        result, output = self.run_review(inputs=("e", "1,2;3", "c"), runner=grouped_runner)
        self.assertEqual(0, result)
        self.assertIn("Normalized: 2 phone meetings", output)
        records = self.ingest_records()
        self.assertEqual(2, len(records))
        for record in records.values():
            self.assertEqual(3, record["operator_resolution"]["schema_version"])
            self.assertEqual(3, len(record["operator_resolution"]["source_fingerprints_sha256"]))

    def test_discard_requires_exact_confirmation_and_retains_original_audio(self):
        paths = [
            self.add_segment("2026_07_23_12_00_00", 20),
            self.add_segment("2026_07_23_12_01_00", 20),
        ]
        before = [path.read_bytes() for path in paths]
        self.write_fetch_ledger(paths)
        result, output = self.run_review(inputs=("d", "2", "DISCARD"))
        self.assertEqual(0, result)
        self.assertIn("Discarded: 1 phone recordings", output)
        self.assertEqual(before, [path.read_bytes() for path in paths])
        self.assertEqual(2, len(self.ingest_records()))
        self.assertIn("discarded", {record["status"] for record in self.ingest_records().values()})

    def test_cancel_eof_and_non_tty_make_no_changes(self):
        path = self.add_segment("2026_07_23_12_00_00", 20)
        self.write_fetch_ledger([path])
        for inputs, terminal in (((), True), (("q",), True), (("", "", ""), True), (("a",), False)):
            with self.subTest(inputs=inputs, terminal=terminal):
                result, _ = self.run_review(inputs=inputs, terminal=terminal)
                self.assertEqual(0, result)
                self.assertFalse(self.ingest_root.exists())

    def test_dry_run_measures_but_writes_nothing(self):
        path = self.add_segment("2026_07_23_12_00_00", 20)
        self.write_fetch_ledger([path])
        result, output = self.run_review(dry_run=True)
        self.assertEqual(0, result)
        self.assertIn("Duration", output)
        self.assertFalse(self.ingest_root.exists())

    def test_pending_normalized_and_discarded_sources_are_not_reviewed(self):
        pending = self.add_segment("2026_07_23_12_00_00", 20)
        normalized = self.add_segment("2026_07_23_12_01_00", 20)
        discarded = self.add_segment("2026_07_23_12_02_00", 20)
        self.write_fetch_ledger([pending, normalized, discarded], statuses={pending.name: "pending_settlement"})
        self.ingest_root.mkdir()
        records = {
            "norm": {
                "status": "normalized",
                "source_fingerprint_sha256": ingest.sha256_file(normalized),
                "source_path": str(normalized.resolve()),
            },
            "drop": {
                "status": "discarded",
                "source_fingerprint_sha256": ingest.sha256_file(discarded),
                "source_path": str(discarded.resolve()),
            },
        }
        (self.ingest_root / ingest.LEDGER_FILENAME).write_text(json.dumps({"schema_version": 1, "records": records}))
        sets = review.collect_review_sets(self.config, duration_probe=self.duration_probe)
        self.assertEqual((), sets)

    def test_revalidation_ignores_unrelated_completed_recording_in_same_date_folder(self):
        completed = self.add_segment("2026_07_23_11_00_00", 20)
        selected_paths = [
            self.add_segment("2026_07_23_12_00_00", 20),
            self.add_segment("2026_07_23_12_00_30", 20),
        ]
        self.write_fetch_ledger([completed, *selected_paths])
        self.ingest_root.mkdir()
        (self.ingest_root / ingest.LEDGER_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": {
                        "completed": {
                            "status": "normalized",
                            "source_fingerprint_sha256": ingest.sha256_file(completed),
                            "source_path": str(completed.resolve()),
                        }
                    },
                }
            )
        )
        selected = review.collect_review_sets(
            self.config, duration_probe=self.duration_probe
        )[0]
        self.assertEqual(
            [path.name for path in selected_paths],
            [segment.source_path.name for segment in selected.segments],
        )
        review._revalidate(selected, self.config)

    def test_revalidation_still_rejects_new_unresolved_recording_in_same_set(self):
        selected_path = self.add_segment("2026_07_23_12_00_00", 20)
        self.write_fetch_ledger([selected_path])
        selected = review.collect_review_sets(
            self.config, duration_probe=self.duration_probe
        )[0]
        added = self.add_segment("2026_07_23_12_01_00", 20)
        self.write_fetch_ledger([selected_path, added])
        with self.assertRaisesRegex(review.ReviewError, "source set changed"):
            review._revalidate(selected, self.config)

    def test_mutation_between_review_and_confirmation_fails_closed(self):
        path = self.add_segment("2026_07_23_12_00_00", 20)
        self.write_fetch_ledger([path])
        mutated = False

        def input_func(prompt):
            nonlocal mutated
            if prompt.startswith("\n[a]") and not mutated:
                mutated = True
                path.write_bytes(b"changed")
                return "s"
            return "c"

        output = io.StringIO()
        with redirect_stdout(output):
            result = review.run_review(
                config=self.config,
                clock=lambda: self.now,
                input_func=input_func,
                terminal_isatty=lambda: True,
                duration_probe=self.duration_probe,
                ingest_runner=self.runner,
            )
        self.assertEqual(1, result)
        self.assertIn("size changed", output.getvalue())
        self.assertFalse(self.ingest_root.exists())

    def test_schema_three_rejects_incomplete_duplicate_and_reordered_plans(self):
        identities = ["a" * 64, "b" * 64, "c" * 64]
        valid = {
            "schema_version": 3,
            "source_fingerprints_sha256": identities,
            "groups": [{"action": "normalize", "source_fingerprints_sha256": identities[:2]}, {"action": "normalize", "source_fingerprints_sha256": identities[2:]}],
            "discard_source_fingerprints_sha256": [],
        }
        path = self.root / "plan.json"
        path.write_text(json.dumps(valid))
        plan = ingest.load_operator_resolution(path)
        self.assertEqual("plan", plan.resolution)
        for name, replacement in (
            ("missing", {**valid, "discard_source_fingerprints_sha256": [identities[0]]}),
            ("duplicate", {**valid, "groups": [{"action": "normalize", "source_fingerprints_sha256": identities[:2]}, {"action": "normalize", "source_fingerprints_sha256": [identities[1]]}]}),
            ("unlisted", {**valid, "groups": [{"action": "normalize", "source_fingerprints_sha256": ["d" * 64]}, {"action": "normalize", "source_fingerprints_sha256": identities[1:]}]}),
        ):
            with self.subTest(name=name):
                path.write_text(json.dumps(replacement))
                with self.assertRaises(ValueError):
                    ingest.load_operator_resolution(path)

    def test_schema_three_rejects_reordered_group_at_application(self):
        paths = [
            self.add_segment("2026_07_23_12_00_00", 20),
            self.add_segment("2026_07_23_12_01_00", 20),
        ]
        self.write_fetch_ledger(paths)
        review_set = review.collect_review_sets(self.config, duration_probe=self.duration_probe)[0]
        identities = tuple(segment.fingerprint_sha256 for segment in review_set.segments)
        plan = review._make_plan(review_set.segments, groups=[identities[::-1]])
        with self.assertRaisesRegex(ValueError, "chronological"):
            ingest._apply_operator_resolution(
                list(review_set.candidates),
                [review._as_ingest_segment(segment) for segment in review_set.segments],
                plan,
            )

    def test_dry_run_has_no_remote_model_or_normalizer_execution(self):
        path = self.add_segment("2026_07_23_12_00_00", 20)
        self.write_fetch_ledger([path])
        with mock.patch.object(fetch, "list_remote_recordings", side_effect=AssertionError("Drive called")), \
             mock.patch.object(ingest, "run_ingest", side_effect=AssertionError("normalizer called")):
            result, _ = self.run_review(dry_run=True)
        self.assertEqual(0, result)

    def test_cli_routes_review_without_discovery_or_model_paths(self):
        with mock.patch.object(cli.phone_review, "run_review", return_value=0) as run_review, \
             mock.patch.object(cli, "discover_candidates") as discover, \
             mock.patch.object(cli, "load_secret_environment") as secrets:
            result = cli.run_cli(["phone", "review", "today", "--dry-run"], terminal_isatty=lambda: False)
        self.assertEqual(0, result)
        run_review.assert_called_once()
        discover.assert_not_called()
        secrets.assert_not_called()


if __name__ == "__main__":
    unittest.main()
