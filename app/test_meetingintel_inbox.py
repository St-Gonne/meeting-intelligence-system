from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import meetingintel_cli as cli
import meetingintel_inbox as inbox
import meetingintel_inventory as inventory
import phone_recording_review as review


IST = timezone(timedelta(hours=5, minutes=30))


class InboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 5, 20, 0, tzinfo=IST)
        self.root = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.path = Path(self.root.name)
        self.config = inbox.InboxConfig(
            inventory=inventory.InventoryConfig(
                meetings_root=self.path / "meetings",
                phone_staging_root=self.path / "staging",
                phone_fetch_ledger=self.path / "state" / "fetch.json",
                phone_ingest_root=self.path / "ingest" / "phone",
                processing_ledger=self.path / "state" / "processed.json",
            ),
            laptop_ingest_root=self.path / "ingest" / "laptop",
            laptop_capture_root=self.path / "captures",
        )

    def tearDown(self) -> None:
        self.root.cleanup()

    def report(self, *items: inventory.InventoryItem) -> inventory.InventoryReport:
        counts: dict[str, int] = {}
        for item in items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return inventory.InventoryReport("all", False, items, counts)

    def test_mixed_lanes_are_truthful_and_newest_first(self) -> None:
        items = self.report(
            inventory.InventoryItem(self.now - timedelta(hours=4), "meetily", "private title", "meetily ready"),
            inventory.InventoryItem(self.now - timedelta(hours=2), "phone_recording", "x (6 staged segments)", "phone pending normalization", group_key="raw:2026/08/05"),
            inventory.InventoryItem(self.now - timedelta(hours=1), "phone_recording", "x.m4a", "phone pending settlement"),
        )
        with patch.object(inventory, "inventory", return_value=items):
            result = inbox.build_inbox(config=self.config, clock=lambda: self.now)
        self.assertEqual(["phone", "phone", "meetily"], [item.source_kind for item in result.items])
        self.assertEqual("waiting", result.items[0].category)
        self.assertEqual("Phone recording", result.items[0].display_name)
        self.assertNotIn("private title", inbox.format_inbox(result))
        self.assertIn("Waiting for the second unchanged Drive check", inbox.format_inbox(result))

    def test_laptop_validated_and_interrupted_states_are_local_and_read_only(self) -> None:
        self.config.laptop_capture_root.mkdir()
        source = SimpleNamespace(
            capture_id="capture-normalized",
            source_identity="a" * 64,
            source_folder=self.path / "ingest" / "laptop" / "source",
            created_at=self.now - timedelta(hours=1),
            source_segment_count=2,
        )
        normalized = SimpleNamespace(source=source)
        complete = self.config.laptop_capture_root / "complete"
        interrupted = self.config.laptop_capture_root / "interrupted"
        for folder, status, capture_id in ((complete, "complete", "complete-id"), (interrupted, "interrupted", "interrupted-id")):
            folder.mkdir()
            (folder / "capture_manifest.json").write_text(
                __import__("json").dumps({"status": status, "capture_id": capture_id, "started_at": self.now.isoformat()}),
                encoding="utf-8",
            )
        audit = SimpleNamespace(accepted=True, merge_ready=True, recovery_ready=True, finalized_segments=(Path("a"), Path("b")))
        with patch.object(inventory, "inventory", return_value=self.report()), patch.object(
            inbox, "audit_capture_session", return_value=audit
        ):
            result = inbox.build_inbox(
                config=self.config,
                clock=lambda: self.now,
                laptop_discover=lambda *args, **kwargs: (normalized,),
                laptop_audit=lambda *args, **kwargs: audit,
            )
        statuses = [item.status for item in result.items]
        self.assertIn("laptop normalized and ready", statuses)
        self.assertIn("laptop validated; waiting for normalization", statuses)
        self.assertIn("laptop interrupted; recovery eligible", statuses)
        self.assertFalse((self.config.inventory.processing_ledger).exists())

    def test_old_recording_manifest_is_attention_not_waiting(self) -> None:
        self.config.laptop_capture_root.mkdir()
        stale = self.config.laptop_capture_root / "stale"
        stale.mkdir()
        (stale / "capture_manifest.json").write_text(
            __import__("json").dumps(
                {
                    "status": "recording",
                    "capture_id": "stale-id",
                    "started_at": (self.now - timedelta(days=2)).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        audit = SimpleNamespace(
            accepted=False,
            merge_ready=False,
            recovery_ready=False,
            finalized_segments=(Path("a"),),
        )
        with patch.object(inventory, "inventory", return_value=self.report()):
            result = inbox.build_inbox(
                config=self.config,
                clock=lambda: self.now,
                laptop_discover=lambda *args, **kwargs: (),
                laptop_audit=lambda *args, **kwargs: audit,
            )

        self.assertEqual(1, len(result.items))
        self.assertEqual("needs_decision", result.items[0].category)
        self.assertEqual("laptop capture appears abandoned", result.items[0].status)
        self.assertEqual(0, result.counts["waiting"])

    def test_plain_language_six_chunk_path_preserves_exact_confirmation_and_context(self) -> None:
        segments = tuple(
            review.ReviewSegment(Path(f"/private/tmp/s{i}.m4a"), self.now + timedelta(minutes=i), 900.0, str(i) * 64, 10)
            for i in range(6)
        )
        plan = review.ingest.OperatorResolution(3, "plan", tuple(item.fingerprint_sha256 for item in segments), (tuple(item.fingerprint_sha256 for item in segments),), ())
        selected = review.ReviewSet(self.path, self.now.date(), segments, (), plan, "continuity")
        answers = iter(("1", "y"))
        with patch.object(review, "_preview_plan", return_value="plan"):
            result = review._interactive_plan(selected, lambda _prompt: next(answers), plain_language=True)
        self.assertEqual(plan, result)

    def test_latest_phone_item_is_selected_not_historical_backlog(self) -> None:
        older = inventory.InventoryItem(self.now - timedelta(days=3), "phone_recording", "old.m4a", "operator decision required")
        latest = inventory.InventoryItem(self.now - timedelta(hours=1), "phone_recording", "new (6 staged segments)", "phone pending normalization", group_key="raw:latest")
        report = inbox.InboxReport(
            (
                inbox.InboxItem(latest.created_at, "phone", "Phone recording", latest.status, "review", "needs_decision", identity=latest.group_key, segment_count=6),
                inbox.InboxItem(older.created_at, "phone", "Phone recording", older.status, "review", "needs_decision"),
            ),
            {"needs_decision": 2, "ready": 0, "waiting": 0, "completed": 0},
        )
        latest_review = Mock(group_key=latest.group_key, segments=(SimpleNamespace(created_at=latest.created_at),) * 6)
        old_review = Mock(group_key="raw:old", segments=(SimpleNamespace(created_at=older.created_at),))
        runner = Mock(return_value=0)
        with patch.object(review, "collect_review_sets", return_value=(old_review, latest_review)):
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "r",
                inbox_builder=lambda **_kwargs: report,
                review_runner=runner,
            )
        self.assertEqual(0, result)
        self.assertIs(runner.call_args.args[0], latest_review)

    def test_two_chunk_current_item_matches_by_group_identity(self) -> None:
        created = self.now - timedelta(hours=1)
        item = inbox.InboxItem(
            created,
            "phone",
            "Phone recording",
            "operator decision required",
            "review",
            "needs_decision",
            identity="raw:2026/08/06:2026-08-06",
            segment_count=2,
        )
        matching = Mock(
            group_key=item.identity,
            segments=(SimpleNamespace(created_at=created), SimpleNamespace(created_at=created + timedelta(seconds=1))),
        )
        self.assertIs(matching, cli._matching_phone_review(item, (matching,)))

    def test_missing_or_colliding_group_identity_fails_closed(self) -> None:
        created = self.now - timedelta(hours=1)
        item = inbox.InboxItem(
            created,
            "phone",
            "Phone recording",
            "operator decision required",
            "review",
            "needs_decision",
            identity="raw:current:2026-08-06",
            segment_count=2,
        )
        matching = Mock(
            group_key=item.identity,
            segments=(SimpleNamespace(created_at=created), SimpleNamespace(created_at=created + timedelta(seconds=1))),
        )
        self.assertIsNone(cli._matching_phone_review(item, (matching, matching)))
        self.assertIsNone(cli._matching_phone_review(item, (Mock(group_key="", segments=matching.segments),)))

    def test_failed_identity_revalidation_does_not_invoke_review_or_write_path(self) -> None:
        created = self.now - timedelta(hours=1)
        item = inbox.InboxItem(created, "phone", "Phone recording", "operator decision required", "review", "needs_decision", identity="raw:current:2026-08-06", segment_count=2)
        report = inbox.InboxReport((item,), {"needs_decision": 1, "ready": 0, "waiting": 0, "completed": 0})
        review_runner = Mock()
        with patch.object(review, "collect_review_sets", return_value=()), patch.object(
            cli, "_run_exact_phone_sources"
        ) as process:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "r",
                inbox_builder=lambda **_kwargs: report,
                review_runner=review_runner,
            )
        self.assertEqual(1, result)
        review_runner.assert_not_called()
        process.assert_not_called()

    def test_numbered_items_put_today_before_older_action_items(self) -> None:
        today = inbox.InboxItem(
            self.now - timedelta(hours=1),
            "phone",
            "Phone recording",
            "phone normalized and ready",
            "ready",
            "ready",
            identity="today-phone",
        )
        older = inbox.InboxItem(
            self.now - timedelta(days=2),
            "meetily",
            "Meetily recording",
            "meetily ready",
            "ready",
            "ready",
            identity=str(self.path / "older-meetily"),
        )
        report = inbox.InboxReport(
            (older, today),
            {"needs_decision": 0, "ready": 2, "waiting": 0, "completed": 0},
            today_counts={"needs_decision": 0, "ready": 1, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        rendered = inbox.format_inbox(report)
        detailed = inbox.format_inbox(report, details=True)
        self.assertIn("[1] Phone recording", rendered)
        self.assertNotIn("[2] Meetily recording", rendered)
        self.assertIn("Hidden from the daily view", rendered)
        self.assertLess(detailed.index("[1] Phone recording"), detailed.index("[2] Meetily recording"))
        self.assertIn("today actionable=1; today waiting=0; completed/closed=0", rendered)

    def test_numbered_phone_review_routes_only_the_selected_identity(self) -> None:
        created = self.now - timedelta(hours=1)
        item = inbox.InboxItem(
            created,
            "phone",
            "Phone recording",
            "operator decision required",
            "review",
            "needs_decision",
            identity="raw:selected",
            segment_count=2,
        )
        report = inbox.InboxReport(
            (item,),
            {"needs_decision": 1, "ready": 0, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        selected = Mock(
            group_key=item.identity,
            segments=(SimpleNamespace(created_at=created), SimpleNamespace(created_at=created + timedelta(seconds=1))),
        )
        review_runner = Mock(return_value=0)
        with patch.object(review, "collect_review_sets", return_value=(selected,)), patch.object(
            cli, "discover_candidates"
        ) as discover:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "1",
                inbox_builder=lambda **_kwargs: report,
                review_runner=review_runner,
            )
        self.assertEqual(0, result)
        review_runner.assert_called_once()
        self.assertIs(review_runner.call_args.args[0], selected)
        discover.assert_not_called()

    def test_numbered_ready_phone_item_uses_exact_process_gate_after_lowercase_y(self) -> None:
        item = inbox.InboxItem(
            self.now - timedelta(hours=1),
            "phone",
            "Phone recording",
            "phone normalized and ready",
            "ready",
            "ready",
            identity="phone-ready",
        )
        report = inbox.InboxReport(
            (item,),
            {"needs_decision": 0, "ready": 1, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        source = object()
        process = Mock(return_value=0)
        answers = iter(("1", "y"))
        with patch.object(cli, "_load_exact_phone_source", return_value=source), patch.object(
            cli, "discover_audio_first_sources"
        ) as discover:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: next(answers),
                inbox_builder=lambda **_kwargs: report,
                process_runner=process,
            )
        self.assertEqual(0, result)
        process.assert_called_once_with((source,))
        discover.assert_not_called()

    def test_inbox_phone_route_propagates_selected_qwen_backend(self) -> None:
        item = inbox.InboxItem(
            self.now - timedelta(hours=1), "phone", "Phone recording",
            "phone normalized and ready", "ready", "ready", identity="phone-ready",
        )
        report = inbox.InboxReport((item,), {
            "needs_decision": 0, "ready": 1, "waiting": 0, "completed": 0,
        }, today_date=self.now.date())
        answers = iter(("1", "y"))
        with patch.object(cli, "_load_exact_phone_source", return_value=object()), patch.object(
            cli, "_run_exact_phone_sources", return_value=0
        ) as process:
            result = cli.run_inbox(
                clock=lambda: self.now, terminal_isatty=lambda: True,
                input_func=lambda _prompt: next(answers), inbox_builder=lambda **_kwargs: report,
                phone_transcription_backend="qwen",
            )
        self.assertEqual(0, result)
        self.assertEqual("qwen", process.call_args.kwargs["phone_transcription_backend"])

    def test_numbered_laptop_routes_reuse_exact_processing_and_capture_gates(self) -> None:
        normalized_item = inbox.InboxItem(
            self.now - timedelta(hours=1),
            "laptop",
            "Laptop recording",
            "laptop normalized and ready",
            "ready",
            "ready",
            identity="laptop-ready",
        )
        report = inbox.InboxReport(
            (normalized_item,),
            {"needs_decision": 0, "ready": 1, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        source = object()
        answers = iter(("1", "y"))
        with patch.object(cli, "_load_exact_laptop_source", return_value=source), patch.object(
            cli, "_run_exact_laptop_source", return_value=0
        ) as process:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: next(answers),
                inbox_builder=lambda **_kwargs: report,
            )
        self.assertEqual(0, result)
        process.assert_called_once()
        self.assertIs(process.call_args.args[0], source)
        self.assertTrue(process.call_args.kwargs["suppress_private_paths"])

        capture_item = inbox.InboxItem(
            self.now - timedelta(hours=2),
            "laptop",
            "Laptop recording",
            "laptop validated; waiting for normalization",
            "normalize",
            "needs_decision",
            identity="capture-id",
        )
        capture_report = inbox.InboxReport(
            (capture_item,),
            {"needs_decision": 1, "ready": 0, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        with patch.object(cli, "_normalize_exact_laptop_item", return_value=0) as normalize:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "1",
                inbox_builder=lambda **_kwargs: capture_report,
            )
        self.assertEqual(0, result)
        normalize.assert_called_once()
        self.assertIs(normalize.call_args.args[0], capture_item)

    def test_numbered_meetily_route_is_exact_and_does_not_use_broad_discovery(self) -> None:
        item = inbox.InboxItem(
            self.now - timedelta(hours=1),
            "meetily",
            "Meetily recording",
            "meetily ready",
            "ready",
            "ready",
            identity=str(self.path / "meetily"),
        )
        report = inbox.InboxReport(
            (item,),
            {"needs_decision": 0, "ready": 1, "waiting": 0, "completed": 0},
            today_date=self.now.date(),
        )
        folder = self.path / "meetily"
        folder.mkdir()
        (folder / "metadata.json").write_text("{}", encoding="utf-8")
        (folder / "transcripts.json").write_text("{}", encoding="utf-8")
        created = self.now - timedelta(hours=1)
        answers = iter(("1", "y"))
        with patch.object(cli, "MEETINGS_ROOT", self.path), patch.object(
            cli.pipeline,
            "parse_meeting_folder",
            return_value=({"status": "completed", "created_at": created.isoformat()}, {}, "hash", "synthetic text", created),
        ), patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}), patch.object(
            cli, "_run_production", return_value=0
        ) as production, patch.object(cli, "discover_candidates") as discover:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: next(answers),
                inbox_builder=lambda **_kwargs: report,
            )
        self.assertEqual(0, result)
        discover.assert_not_called()
        production.assert_called_once()
        self.assertEqual(folder.resolve(), production.call_args.kwargs["explicit_meeting_folder"])

    def test_waiting_and_completed_items_have_no_interactive_write_route(self) -> None:
        report = inbox.InboxReport(
            (
                inbox.InboxItem(self.now, "phone", "Phone recording", "phone pending settlement", "wait", "waiting"),
                inbox.InboxItem(self.now - timedelta(days=1), "phone", "Phone recording", "phone already processed", "done", "completed"),
            ),
            {"needs_decision": 0, "ready": 0, "waiting": 1, "completed": 1},
            today_date=self.now.date(),
        )
        input_func = Mock(return_value="1")
        with patch.object(cli, "_run_production") as production, patch.object(
            cli, "normalize_capture"
        ) as normalize:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=input_func,
                inbox_builder=lambda **_kwargs: report,
            )
        self.assertEqual(0, result)
        input_func.assert_not_called()
        production.assert_not_called()
        normalize.assert_not_called()

    def test_prepare_and_process_runner_are_exact_and_explicit(self) -> None:
        item_time = self.now - timedelta(hours=1)
        item = inbox.InboxItem(item_time, "phone", "Phone recording", "phone pending normalization", "review", "needs_decision", identity="raw:current:2026-08-05", segment_count=1)
        report = inbox.InboxReport((item,), {"needs_decision": 1, "ready": 0, "waiting": 0, "completed": 0})
        selected = Mock(group_key=item.identity, segments=(SimpleNamespace(created_at=item_time),))
        exact_sources = (object(),)
        runner = Mock(return_value=0)
        answers = iter(("p", "y"))
        with patch.object(review, "collect_review_sets", return_value=(selected,)), patch.object(
            cli, "_run_exact_phone_sources", return_value=0
        ) as handoff:
            result = cli.run_inbox(
                clock=lambda: self.now,
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: next(answers),
                inbox_builder=lambda **_kwargs: report,
                review_runner=runner,
            )
        self.assertEqual(0, result)
        process_runner = runner.call_args.kwargs["process_runner"]
        with patch.object(cli, "_run_exact_phone_sources", return_value=0) as handoff:
            process_runner(exact_sources)
        self.assertEqual(1, runner.call_count)
        self.assertEqual(exact_sources, handoff.call_args.args[0])

    def test_noninteractive_cancel_eof_invalid_make_no_action(self) -> None:
        report = inbox.InboxReport(
            (inbox.InboxItem(self.now, "phone", "Phone recording", "phone pending settlement", "wait", "waiting"),),
            {"needs_decision": 0, "ready": 0, "waiting": 1, "completed": 0},
        )
        for terminal, answer in ((False, "p"), (True, ""), (True, None), (True, "x")):
            with self.subTest(terminal=terminal, answer=answer):
                runner = Mock()
                with patch.object(cli.inbox, "build_inbox", return_value=report):
                    result = cli.run_inbox(
                        terminal_isatty=lambda value=terminal: value,
                        input_func=lambda _prompt, value=answer: value,
                        review_runner=runner,
                    )
                self.assertEqual(0, result)
                runner.assert_not_called()

    def test_read_only_route_does_not_call_processing_or_external_surfaces(self) -> None:
        report = inbox.InboxReport((), {"needs_decision": 0, "ready": 0, "waiting": 0, "completed": 0})
        with patch.object(cli.inbox, "build_inbox", return_value=report), patch.object(cli, "discover_candidates") as discover, patch.object(cli, "load_secret_environment") as secrets:
            output = io.StringIO()
            with redirect_stdout(output):
                result = cli.run_cli(["inbox"], terminal_isatty=lambda: False)
        self.assertEqual(0, result)
        discover.assert_not_called()
        secrets.assert_not_called()
        self.assertIn("No matching meetings or phone recordings.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
