from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import meetingintel_cli as cli
import meetingintel_pipeline as pipeline
from audio_first_meeting_source import AudioFirstMeetingSource
from ollama_endpoint import OllamaModelDigestMismatch, OllamaUnreachable
from ollama_service import ServiceConflict


IST = timezone(timedelta(hours=5, minutes=30))


def candidate(name: str, created_at: datetime, kind: str = "meetily") -> cli.SourceCandidate:
    return cli.SourceCandidate(f"fp-{name}", created_at, kind, name)


class ScopeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 8, 18, 0, tzinfo=IST)
        self.today_meetily = candidate("today-meetily", datetime(2026, 7, 8, 9, 0, tzinfo=IST))
        self.today_phone = candidate("today-phone", datetime(2026, 7, 8, 10, 0, tzinfo=IST), "phone_recording")
        self.yesterday = candidate("yesterday", datetime(2026, 7, 7, 23, 59, tzinfo=IST))

    def test_no_argument_and_explicit_today(self):
        self.assertEqual("today", cli.parse_scope([]))
        self.assertEqual("today", cli.parse_scope(["today"]))

    def test_sync_is_routed_before_production_discovery(self):
        with patch.object(cli, "run_sync", return_value=0) as sync:
            self.assertEqual(0, cli.run_cli(["sync", "--dry-run"]))
        sync.assert_called_once_with(["--dry-run"])

    def test_record_is_routed_before_production_discovery(self):
        with patch.object(cli, "run_record", return_value=0) as record, patch.object(
            cli, "discover_candidates"
        ) as discover:
            self.assertEqual(0, cli.run_cli(["record"]))
        record.assert_called_once()
        self.assertEqual([], record.call_args.args[0])
        self.assertNotIn("phone_transcription_backend", record.call_args.kwargs)
        discover.assert_not_called()

    def test_help_routes_to_operator_cheat_sheet_without_discovery(self):
        for token in ("help", "--help", "-h"):
            with self.subTest(token=token), patch.object(
                cli, "discover_candidates"
            ) as discover, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(0, cli.run_cli([token]))
            self.assertIn("CHECK PHONE DELIVERY", output.getvalue())
            self.assertIn("mi recordings diagnose last", output.getvalue())
            self.assertIn("mi ollama status", output.getvalue())
            discover.assert_not_called()

    def test_topic_help_and_invalid_topic_do_not_discover_or_process(self):
        for topic, expected in (("phone", 0), ("recordings", 0), ("recovery", 0),
                                ("setup", 0), ("advanced", 0), ("all", 0), ("typo", 2)):
            with self.subTest(topic=topic), patch.object(cli, "discover_candidates") as discover, \
                    patch.object(cli, "_run_production") as production, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(expected, cli.run_cli(["help", topic]))
                discover.assert_not_called()
                production.assert_not_called()

    def test_empty_legacy_lane_points_to_inbox_without_processing(self):
        output = io.StringIO()
        with patch.object(cli, "discover_candidates", return_value=cli.Discovery((), ())), patch.object(
            cli, "_run_production", return_value=0
        ) as production, redirect_stdout(output):
            result = cli.run_cli(["today"], clock=lambda: self.now)
        self.assertEqual(0, result)
        self.assertIn("No Meetily meetings selected for this legacy processing lane.", output.getvalue())
        self.assertIn("Next: mi inbox", output.getvalue())
        production.assert_not_called()

    def test_record_creates_private_durable_session_and_runs_guard(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            recordings_root = Path(temporary) / "recordings"
            now = datetime(2026, 7, 28, 15, 0, tzinfo=IST)
            with patch.object(
                cli.laptop_capture_guard,
                "run_guard",
                return_value=0,
            ) as run_guard, patch.object(
                cli,
                "normalize_capture",
                return_value=Mock(
                    target_root=recordings_root / "normalized",
                    segment_count=1,
                ),
            ) as normalize, patch.object(
                cli,
                "load_laptop_meeting_source",
                return_value=Mock(source_identity="a" * 64),
            ):
                self.assertEqual(
                    0,
                    cli.run_record(
                        [],
                        clock=lambda: now,
                        recordings_root=recordings_root,
                        ingest_root=Path(temporary) / "ingest",
                        terminal_isatty=lambda: False,
                    ),
                )
            config = run_guard.call_args.args[0]
            self.assertEqual(recordings_root, config.allowed_root)
            self.assertEqual(
                recordings_root / "Meeting 2026-07-28_15-00-00",
                config.session_root,
            )
            self.assertEqual(0o700, recordings_root.stat().st_mode & 0o777)
            normalize.assert_called_once()

    def test_record_processes_only_exact_normalized_source_after_lowercase_y(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            recordings_root = Path(temporary) / "recordings"
            ingest_root = Path(temporary) / "ingest"
            normalized_root = ingest_root / "laptop-source"
            source = Mock(source_identity="a" * 64)
            with patch.object(
                cli.laptop_capture_guard, "run_guard", return_value=0
            ), patch.object(
                cli,
                "normalize_capture",
                return_value=Mock(target_root=normalized_root, segment_count=2),
            ), patch.object(
                cli, "load_laptop_meeting_source", return_value=source
            ), patch.object(
                cli, "_run_exact_laptop_source", return_value=0
            ) as process:
                result = cli.run_record(
                    [],
                    recordings_root=recordings_root,
                    ingest_root=ingest_root,
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: "y",
                )

            self.assertEqual(result, 0)
            process.assert_called_once()
            self.assertIs(process.call_args.args[0], source)

    def test_record_decline_or_noninteractive_never_processes(self):
        for interactive, decision in ((True, "n"), (True, "Y"), (False, "y")):
            with self.subTest(interactive=interactive, decision=decision), tempfile.TemporaryDirectory(
                dir="/private/tmp"
            ) as temporary:
                recordings_root = Path(temporary) / "recordings"
                with patch.object(
                    cli.laptop_capture_guard, "run_guard", return_value=0
                ), patch.object(
                    cli,
                    "normalize_capture",
                    return_value=Mock(
                        target_root=Path(temporary) / "normalized",
                        segment_count=1,
                    ),
                ), patch.object(
                    cli,
                    "load_laptop_meeting_source",
                    return_value=Mock(source_identity="a" * 64),
                ), patch.object(
                    cli, "_run_exact_laptop_source"
                ) as process:
                    result = cli.run_record(
                        [],
                        recordings_root=recordings_root,
                        ingest_root=Path(temporary) / "ingest",
                        terminal_isatty=lambda: interactive,
                        input_func=lambda _prompt: decision,
                    )
                self.assertEqual(result, 0)
                process.assert_not_called()

    def test_interrupted_recording_never_normalizes_or_processes(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            with patch.object(
                cli.laptop_capture_guard, "run_guard", return_value=1
            ), patch.object(cli, "normalize_capture") as normalize, patch.object(
                cli, "_run_exact_laptop_source"
            ) as process:
                result = cli.run_record(
                    [],
                    recordings_root=Path(temporary) / "recordings",
                    ingest_root=Path(temporary) / "ingest",
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: "y",
                )
            self.assertEqual(result, 1)
            normalize.assert_not_called()
            process.assert_not_called()

    def test_failed_battery_start_prints_no_audio_and_restart_instruction(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            def fail_start(config):
                config.session_root.mkdir(mode=0o700)
                (config.session_root / "capture_manifest.json").write_text(json.dumps({
                    "status": "interrupted", "stop_reason": "preflight failed: battery_critical",
                    "segments": [],
                }))
                return 2

            stderr = io.StringIO()
            with patch.object(cli.laptop_capture_guard, "run_guard", side_effect=fail_start), \
                 patch.object(cli, "normalize_capture") as normalize, \
                 redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                result = cli.run_record([], recordings_root=Path(temporary) / "recordings")
            self.assertEqual(2, result)
            normalize.assert_not_called()
            self.assertIn("RECORDING DID NOT START", stderr.getvalue())
            self.assertIn("NO AUDIO WAS SAVED", stderr.getvalue())
            self.assertIn("Connect power", stderr.getvalue())

    def test_sync_schedule_is_installed_without_production_discovery(self):
        with patch.object(cli, "install_daily_schedule", return_value=Path("/tmp/agent.plist")) as install:
            self.assertEqual(0, cli.run_cli(["sync", "--install-daily", "--daily-hour", "8"]))
        install.assert_called_once_with(8, 0)

    def test_sync_status_is_local_and_does_not_enter_discovery(self):
        with patch.object(cli, "print_fetch_status") as status, patch.object(cli, "discover_candidates") as discover:
            self.assertEqual(0, cli.run_cli(["sync", "status"]))
        status.assert_called_once()
        discover.assert_not_called()

    def test_sync_scheduler_status_and_diagnose_are_read_only_routes(self):
        healthy = {"installed": True, "loaded": True, "healthy": True, "run_count": 1, "last_exit_code": 0, "state": "running", "runtime_outcome": "confirmed_success", "receipt_at": "2026-07-23T00:00:00+05:30", "checks": ()}
        unhealthy = dict(healthy, healthy=False)
        with patch.object(cli, "scheduler_status", return_value=healthy) as status, patch.object(cli, "discover_candidates") as discover:
            self.assertEqual(0, cli.run_cli(["sync", "scheduler", "status"]))
        status.assert_called_once()
        discover.assert_not_called()
        with patch.object(cli, "scheduler_diagnose", return_value=unhealthy) as diagnose:
            self.assertEqual(1, cli.run_cli(["sync", "scheduler", "diagnose"]))
        diagnose.assert_called_once()

    def test_sync_scheduler_probe_is_explicit_and_does_not_discover(self):
        with patch.object(cli, "run_scheduler_probe", return_value=0) as probe, patch.object(
            cli, "discover_candidates"
        ) as discover, patch.object(cli, "fetch_recordings") as fetch_recordings:
            self.assertEqual(0, cli.run_cli(["sync", "scheduler", "probe"]))
        probe.assert_called_once_with()
        discover.assert_not_called()
        fetch_recordings.assert_not_called()

    def test_scheduled_sync_records_only_safe_receipt_facts(self):
        summary = argparse.Namespace(already_running=False, failed=0, requires_operator_decision=0, exit_code=0)
        with patch.dict(cli.os.environ, {"MEETINGINTEL_PHONE_FETCH_SCHEDULED": "1"}, clear=False), patch.object(
            cli, "fetch_recordings", return_value=summary
        ), patch.object(cli, "write_scheduler_receipt") as write_receipt:
            self.assertEqual(0, cli.run_sync(["--dry-run"]))
        write_receipt.assert_called_once_with(success=True, category="success", exit_code=0)

    def test_scheduled_sync_logs_a_safe_failure_category_to_stderr(self):
        summary = argparse.Namespace(already_running=False, failed=1, requires_operator_decision=0, exit_code=1)
        error = io.StringIO()
        with (
            patch.dict(cli.os.environ, {"MEETINGINTEL_PHONE_FETCH_SCHEDULED": "1"}, clear=False),
            patch.object(cli, "fetch_recordings", return_value=summary),
            patch.object(cli, "classify_fetch_failure", return_value="drive_auth_or_permission"),
            patch.object(cli, "write_scheduler_receipt") as write_receipt,
            redirect_stderr(error),
        ):
            self.assertEqual(1, cli.run_sync(["--dry-run"]))
        self.assertIn("PHONE_FETCH_FAILURE category=drive_auth_or_permission", error.getvalue())
        self.assertNotIn("unauthorized", error.getvalue())
        write_receipt.assert_called_once_with(
            success=False,
            category="fetch_failure_drive_auth_or_permission",
            exit_code=1,
        )

    def test_ollama_family_is_routed_before_production_discovery(self):
        with patch.object(cli, "run_ollama", return_value=0) as ollama, patch.object(
            cli, "discover_candidates"
        ) as discover:
            self.assertEqual(0, cli.run_cli(["ollama", "status"]))
        ollama.assert_called_once_with(["status"], manager=None)
        discover.assert_not_called()

    def test_audio_first_processing_requires_explicit_flag(self):
        with patch.object(cli, "discover_candidates", return_value=cli.Discovery((), ())) as discover:
            with patch.object(cli, "load_secret_environment", return_value=True):
                self.assertEqual(0, cli.run_cli(["today"]))
                self.assertEqual(0, cli.run_cli(["--audio-first", "today"]))
        self.assertEqual([None, cli.AUDIO_FIRST_ROOT], [call.kwargs["audio_first_root"] for call in discover.call_args_list])

    def test_pipeline_audio_first_only_requires_a_phone_root(self):
        with self.assertRaises(ValueError):
            pipeline.process_meetings(
                argparse.Namespace(audio_first_only=True, audio_first_root=None)
            )

    def test_today_uses_asia_kolkata_boundary_and_both_source_kinds(self):
        utc_same_day = candidate(
            "utc-boundary", datetime(2026, 7, 7, 19, 0, tzinfo=timezone.utc), "phone_recording"
        )
        selected = cli.select_candidates(
            "today",
            [self.yesterday, self.today_phone, self.today_meetily, utc_same_day],
            set(),
            self.now,
        )
        self.assertEqual(
            ["utc-boundary", "today-meetily", "today-phone"],
            [item.display_name for item in selected],
        )
        self.assertNotIn(self.yesterday, selected)

    def test_last_chooses_latest_by_created_at_across_sources_not_mtime(self):
        selected = cli.select_candidates(
            "last", [self.today_phone, self.today_meetily], set(), self.now
        )
        self.assertEqual((self.today_phone,), selected)

    def test_last_already_processed_does_not_choose_older_unprocessed(self):
        selected = cli.select_candidates(
            "last",
            [self.today_meetily, self.today_phone],
            {self.today_phone.fingerprint},
            self.now,
        )
        self.assertEqual((self.today_phone,), selected)

    def test_new_mixed_source_order_is_deterministic_and_excludes_processed(self):
        selected = cli.select_candidates(
            "new",
            [self.today_phone, self.yesterday, self.today_meetily],
            {self.today_meetily.fingerprint},
            self.now,
        )
        self.assertEqual(
            [self.yesterday, self.today_phone], list(selected)
        )


class CommandExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "state" / "processed_ledger.json"
        self.output = self.root / "output"
        self.secret = self.root / "meetingintel.env"
        self.now = datetime(2026, 7, 8, 18, 0, tzinfo=IST)
        self.item = candidate("meeting", datetime(2026, 7, 8, 17, 0, tzinfo=IST))
        self.discovery = cli.Discovery((self.item,), ())

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, scope, *, records=None, token=True, runner=None, answer="ALL", audio_first=False, backend=None):
        output = io.StringIO()
        errors = io.StringIO()
        ledger = {"schema_version": 2, "records": records or {}}
        runner = runner or Mock(return_value=(1, self.output / "brief.md"))
        patches = (
            patch.object(cli, "discover_candidates", return_value=self.discovery),
            patch.object(cli.pipeline, "load_ledger", return_value=ledger),
            patch.object(cli, "load_secret_environment", return_value=token),
            patch.object(cli, "resolve_ollama_endpoint", return_value=cli.pipeline.DEFAULT_OLLAMA_URL),
            patch.object(cli, "preflight_ollama", return_value={}),
            patch.object(cli, "LEDGER_PATH", self.ledger),
            patch.object(cli, "OUTPUT_DIR", self.output),
            patch.object(cli, "create_all_checkpoint", return_value=self.root / "checkpoint"),
        )
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], redirect_stdout(output), redirect_stderr(errors)
        ):
            command = (["--audio-first"] if audio_first else []) + ([scope] if scope else [])
            if backend is not None:
                command = ["--phone-transcription-backend", backend, *command]
            code = cli.run_cli(
                command, clock=lambda: self.now,
                input_func=lambda _prompt: answer, pipeline_runner=runner,
            )
        return code, output.getvalue(), errors.getvalue(), runner

    def test_today_selection_reaches_pipeline_before_staging_with_exact_filter(self):
        code, output, _errors, runner = self.run_cli("today")
        self.assertEqual(0, code)
        args = runner.call_args.args[0]
        self.assertEqual([self.item.fingerprint], args.selected_fingerprint)
        self.assertFalse(args.refresh_existing)
        self.assertIn("Scope: today", output)
        self.assertIn("Selected: 1 meetings", output)
        self.assertIn("Processed: 1", output)

    def test_audio_first_is_phone_only_and_passes_only_phone_to_pipeline(self):
        phone = candidate("phone", datetime(2026, 7, 8, 18, 0, tzinfo=IST), "phone_recording")
        self.discovery = cli.Discovery((self.item, phone), ())
        runner = Mock(return_value=(1, self.output / "brief.md"))
        code, _output, _errors, runner = self.run_cli(
            "today", runner=runner, audio_first=True
        )
        self.assertEqual(0, code)
        args = runner.call_args.args[0]
        self.assertEqual([phone.fingerprint], args.selected_fingerprint)
        self.assertTrue(args.audio_first_only)
        self.assertEqual(cli.AUDIO_FIRST_ROOT, args.audio_first_root)

    def test_phone_qwen_default_and_legacy_override_reach_pipeline(self):
        phone = candidate("phone", datetime(2026, 7, 8, 18, 0, tzinfo=IST), "phone_recording")
        self.discovery = cli.Discovery((phone,), ())
        for requested, expected in ((None, "qwen"), ("qwen", "qwen"), ("legacy", "legacy")):
            with self.subTest(requested=requested):
                runner = Mock(return_value=(1, self.output / "brief.md"))
                code, output, _errors, runner = self.run_cli(
                    "today", runner=runner, audio_first=True, backend=requested
                )
                self.assertEqual(0, code)
                self.assertEqual(expected, runner.call_args.args[0].phone_transcription_backend)
                self.assertIn(f"Phone transcription backend: {expected}", output)

    def test_backend_selector_does_not_make_read_only_list_start_pipeline(self):
        with patch.object(cli, "build_inventory", return_value={}) as inventory, patch.object(
            cli, "format_inventory", return_value="read only"
        ), patch.object(cli, "_run_production") as production:
            self.assertEqual(0, cli.run_cli([
                "--phone-transcription-backend", "qwen", "--audio-first", "today", "list"
            ], clock=lambda: self.now))
        inventory.assert_called_once()
        production.assert_not_called()

    def test_laptop_run_labels_actual_backend_and_does_not_claim_failed_meeting_in_brief(self):
        laptop = candidate("laptop", self.now, "laptop_capture")
        runner = Mock(return_value=(0, self.output / "brief.md"))
        output = io.StringIO()
        with patch.object(cli, "load_secret_environment", return_value=True), patch.object(
            cli, "resolve_ollama_endpoint", return_value=cli.pipeline.DEFAULT_OLLAMA_URL
        ), patch.object(cli, "preflight_ollama", return_value={}), redirect_stdout(output):
            code = cli._run_production(
                scope="exact laptop recording", selected=(laptop,), processed=set(),
                clock=lambda: self.now, pipeline_runner=runner, ollama_manager=None,
                audio_first_only=True, suppress_private_paths=True,
            )
        self.assertEqual(1, code)
        self.assertIn("Laptop transcription backend: legacy (Whisper)", output.getvalue())
        self.assertNotIn("Phone transcription backend", output.getvalue())
        self.assertNotIn("Brief written", output.getvalue())
        self.assertIn("no result from this attempt", output.getvalue())

    def test_exact_phone_handoff_passes_only_validated_sources_and_skips_meetily_discovery(self):
        source = AudioFirstMeetingSource(
            source_folder=self.root / "phone-source",
            source_kind="phone_recording",
            source_filename="2026_07_08_18_00_00.m4a",
            original_source_path=self.root / "staged" / "2026_07_08_18_00_00.m4a",
            source_fingerprint="a" * 64,
            created_at=self.now,
            duration_seconds=42.0,
            canonical_audio_path=self.root / "phone-source" / "audio.m4a",
            normalization_status="normalized",
            transcription_status="not_started",
            audio_handling="copy",
            ingested_at=self.now,
        )
        runner = Mock(return_value=(1, self.output / "brief.md"))
        with (
            patch.object(cli, "discover_candidates") as discover,
            patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}),
            patch.object(cli, "load_secret_environment", return_value=True),
            patch.object(cli, "resolve_ollama_endpoint", return_value=cli.pipeline.DEFAULT_OLLAMA_URL),
            patch.object(cli, "preflight_ollama", return_value={}),
            patch.object(cli, "LEDGER_PATH", self.ledger),
            patch.object(cli, "OUTPUT_DIR", self.output),
        ):
            result = cli._run_exact_phone_sources(
                (source,),
                clock=lambda: self.now,
                pipeline_runner=runner,
                ollama_manager=None,
                phone_transcription_backend="qwen",
            )
        self.assertEqual(0, result)
        discover.assert_not_called()
        args = runner.call_args.args[0]
        self.assertTrue(args.audio_first_only)
        self.assertEqual("qwen", args.phone_transcription_backend)
        self.assertEqual(
            [pipeline.build_audio_first_fingerprint(source.source_identity)],
            args.selected_fingerprint,
        )

    def test_last_already_done_exits_without_pipeline_or_older_fallback(self):
        runner = Mock()
        code, output, _errors, runner = self.run_cli(
            "last", records={self.item.fingerprint: {}}, runner=runner
        )
        self.assertEqual(0, code)
        runner.assert_not_called()
        self.assertIn("Already done: 1", output)

    def test_new_selects_unprocessed_and_excludes_processed(self):
        second = candidate("second", datetime(2026, 7, 8, 16, 0, tzinfo=IST), "phone_recording")
        self.discovery = cli.Discovery((second, self.item), ())
        code, _output, _errors, runner = self.run_cli(
            "new", records={second.fingerprint: {}}
        )
        self.assertEqual(0, code)
        self.assertEqual([self.item.fingerprint], runner.call_args.args[0].selected_fingerprint)

    def test_unreachable_managed_endpoint_gets_one_bounded_recovery_before_pipeline(self):
        runner = Mock(return_value=(1, self.output / "brief.md"))
        manager = Mock()
        manager.recover.return_value = True
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch.object(cli, "discover_candidates", return_value=self.discovery),
            patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}),
            patch.object(cli, "load_secret_environment", return_value=True),
            patch.object(cli, "resolve_ollama_endpoint", return_value="http://127.0.0.1:11435/api/generate"),
            patch.object(cli, "preflight_ollama", side_effect=OllamaUnreachable("connection refused")),
            patch.object(cli, "LEDGER_PATH", self.ledger),
            patch.object(cli, "OUTPUT_DIR", self.output),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            code = cli.run_cli(
                ["today"], clock=lambda: self.now,
                pipeline_runner=runner, ollama_manager=manager,
            )
        self.assertEqual(0, code)
        manager.recover.assert_called_once_with("http://127.0.0.1:11435/api/generate")
        runner.assert_called_once()
        self.assertIn("Recovered the MeetingIntel-managed local Ollama service", output.getvalue())

    def test_digest_mismatch_and_conflict_never_recover_or_run_pipeline(self):
        for failure in (
            OllamaModelDigestMismatch("wrong digest"),
            OllamaUnreachable("connection refused"),
        ):
            with self.subTest(failure=type(failure).__name__):
                runner = Mock(return_value=(1, self.output / "brief.md"))
                manager = Mock()
                if isinstance(failure, OllamaUnreachable):
                    manager.recover.side_effect = ServiceConflict("port conflict")
                output = io.StringIO()
                errors = io.StringIO()
                with (
                    patch.object(cli, "discover_candidates", return_value=self.discovery),
                    patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}),
                    patch.object(cli, "load_secret_environment", return_value=True),
                    patch.object(cli, "resolve_ollama_endpoint", return_value="http://127.0.0.1:11435/api/generate"),
                    patch.object(cli, "preflight_ollama", side_effect=failure),
                    patch.object(cli, "LEDGER_PATH", self.ledger),
                    patch.object(cli, "OUTPUT_DIR", self.output),
                    redirect_stdout(output),
                    redirect_stderr(errors),
                ):
                    code = cli.run_cli(
                        ["today"], clock=lambda: self.now,
                        pipeline_runner=runner, ollama_manager=manager,
                    )
                self.assertEqual(1, code)
                if isinstance(failure, OllamaUnreachable):
                    manager.recover.assert_called_once()
                else:
                    manager.recover.assert_not_called()
                runner.assert_not_called()
    def test_all_requires_exact_confirmation_and_refreshes_only_after_checkpoint(self):
        for answer in ("all", "ALL ", "no", ""):
            with self.subTest(answer=answer):
                code, output, _errors, runner = self.run_cli("all", answer=answer)
                self.assertEqual(0, code)
                runner.assert_not_called()
                self.assertIn("Aborted safely", output)
        code, output, _errors, runner = self.run_cli("all", answer="ALL")
        self.assertEqual(0, code)
        self.assertTrue(runner.call_args.args[0].refresh_existing)
        self.assertIn("WARNING", output)
        self.assertIn("Source candidates: 1", output)
        self.assertIn("Checkpoint:", output)

    def test_missing_secret_fails_before_pipeline_and_never_prints_token(self):
        code, output, errors, runner = self.run_cli("today", token=False)
        self.assertEqual(1, code)
        runner.assert_not_called()
        self.assertIn("HF_TOKEN is unavailable", errors)
        self.assertNotIn("secret-token", output + errors)

    def test_pipeline_persistence_failure_is_not_reported_as_skip(self):
        runner = Mock(side_effect=pipeline.ProductionPersistenceError("ledger failed"))
        code, output, errors, _runner = self.run_cli("today", runner=runner)
        self.assertEqual(1, code)
        self.assertIn("Pipeline persistence error", errors)
        self.assertNotIn("Already done: 1", output)


class SecretCheckpointAndArtifactTests(unittest.TestCase):
    def test_secret_loader_supports_export_and_quotes_without_printing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meetingintel.env"
            path.write_text("# local\nexport HF_TOKEN='secret-token'\nOTHER=value\n")
            environ = {}
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertTrue(cli.load_secret_environment(path, environ))
            self.assertEqual("secret-token", environ["HF_TOKEN"])
            self.assertEqual("", output.getvalue())

    def test_all_checkpoint_is_complete_deterministic_and_cleans_temporary_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "state" / "ledger.json"
            ledger.parent.mkdir()
            ledger.write_text('{"schema_version": 2, "records": {}}\n')
            output = root / "output"
            output.mkdir()
            (output / "brief.md").write_text("brief")
            checkpoints = root / "checkpoints"
            item = candidate("one", datetime(2026, 7, 8, 10, tzinfo=IST))
            clock = lambda: datetime(2026, 7, 8, 18, 30, tzinfo=IST)
            result = cli.create_all_checkpoint(ledger, output, [item], clock, checkpoints)
            self.assertTrue((result / "processed_ledger.json").is_file())
            self.assertTrue((result / "output" / "brief.md").is_file())
            manifest = json.loads((result / "checkpoint_manifest.json").read_text())
            self.assertEqual([item.fingerprint], manifest["selected_fingerprints"])
            self.assertEqual([], list(checkpoints.glob(".*.tmp")))

    def test_source_selection_and_plan_do_not_mutate_sources_or_make_selector_views(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.m4a"
            source.write_bytes(b"source")
            before = (source.read_bytes(), source.stat().st_mtime_ns)
            cli.select_candidates(
                "today", [candidate("x", datetime(2026, 7, 8, tzinfo=IST))], set(),
                datetime(2026, 7, 8, tzinfo=IST),
            )
            self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns))
            self.assertEqual([], list(Path(directory).glob("*selector*")))

    def test_desktop_cheat_sheet_exact_content(self):
        self.assertIn("start with mi inbox", cli.CHEAT_SHEET_TEXT)
        self.assertIn("LEGACY MEETILY PROCESSING LANE", cli.help_text("advanced"))
        self.assertIn("mi record", cli.CHEAT_SHEET_TEXT)
        self.assertIn("every two hours", cli.help_text("phone"))
        self.assertNotIn("EVERY DAY AT 07:00", cli.CHEAT_SHEET_TEXT)
        self.assertIn("mi sync status", cli.CHEAT_SHEET_TEXT)
        self.assertIn("mi today list", cli.CHEAT_SHEET_TEXT)


if __name__ == "__main__":
    unittest.main()
