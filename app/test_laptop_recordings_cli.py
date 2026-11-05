from __future__ import annotations
from mi_paths import configured_timezone

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import meetingintel_cli as cli
import meetingintel_pipeline as pipeline


IST = configured_timezone()


def laptop_source(identity: str, minute: int = 0) -> Mock:
    return Mock(
        source_identity=identity,
        created_at=datetime(2026, 7, 30, 10, minute, tzinfo=IST),
        source_segment_count=2,
        source_kind="laptop_capture",
        display_name=f"Meeting 2026-07-30_10-{minute:02d}-00",
    )


class LaptopRecordingsCliTests(unittest.TestCase):
    def test_known_speaker_count_is_exact_scoped_and_still_requires_confirmation(self):
        source = laptop_source('a' * 64)
        with patch.object(cli, '_laptop_sources', return_value=((source,), ())), \
             patch.object(cli, '_run_exact_laptop_source', return_value=0) as process, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, cli.run_recordings(['process', 'aaaaaaaaaaaa', '--speakers', '2'],
                terminal_isatty=lambda: True, input_func=lambda _: 'n'))
            process.assert_not_called()
            self.assertIn('mi recordings process aaaaaaaaaaaa --speakers 2', output.getvalue())
            self.assertEqual(0, cli.run_recordings(['process', 'aaaaaaaaaaaa', '--speakers', '2'],
                terminal_isatty=lambda: True, input_func=lambda _: 'y'))
            self.assertEqual(2, process.call_args.kwargs['speaker_count'])
            self.assertIs(source, process.call_args.args[0])

    def test_known_speaker_count_reaches_only_the_exact_pipeline_source(self):
        from types import SimpleNamespace
        source = laptop_source('a' * 64)
        selected = Mock(fingerprint='b'*64, created_at=source.created_at,
                        source_kind='laptop_capture', display_name='Synthetic')
        args = SimpleNamespace()
        def production(*pos, **kw):
            kw['pipeline_runner'](args)
            return 0
        with patch.object(cli.pipeline, 'production_source_from_laptop_capture', return_value=selected), \
             patch.object(cli.pipeline, 'load_ledger', return_value={'records': {}}), \
             patch.object(cli, '_run_production', side_effect=production), \
             patch.object(cli.pipeline, 'process_explicit_laptop_sources', return_value=(1, Path('/brief'))) as process, \
             redirect_stdout(io.StringIO()):
            cli._run_exact_laptop_source(source, clock=cli.pipeline.now_ist,
                                        ollama_manager=None, speaker_count=2)
        self.assertEqual(2, args.diarization_speakers)
        self.assertEqual((selected,), process.call_args.args[1])

    def test_route_precedes_normal_production_discovery(self) -> None:
        with patch.object(cli, "run_recordings", return_value=0) as recordings, patch.object(
            cli, "discover_candidates"
        ) as discovery:
            result = cli.run_cli(["recordings", "list"])
        self.assertEqual(result, 0)
        self.assertEqual(recordings.call_args.args[0], ["list"])
        discovery.assert_not_called()

    def test_list_is_read_only_and_reports_ready_or_processed(self) -> None:
        first = laptop_source("a" * 64)
        second = laptop_source("b" * 64, 5)
        processed_fingerprint = pipeline.build_laptop_capture_fingerprint(
            second.source_identity
        )
        output = io.StringIO()
        with patch.object(
            cli, "_laptop_sources", return_value=((first, second), ())
        ), patch.object(
            cli.pipeline,
            "load_ledger",
            return_value={"records": {processed_fingerprint: {}}},
        ), patch.object(cli, "_run_exact_laptop_source") as process, redirect_stdout(output):
            result = cli.run_recordings(["list"])

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("id=aaaaaaaaaaaa", rendered)
        self.assertIn("id=bbbbbbbbbbbb", rendered)
        self.assertIn("ready", rendered)
        self.assertIn("processed", rendered)
        process.assert_not_called()

    def test_process_requires_unique_selector_and_explicit_lowercase_y(self) -> None:
        first = laptop_source("abcdef12" + "a" * 56)
        second = laptop_source("abcdef12" + "b" * 56, 5)
        stderr = io.StringIO()
        with patch.object(
            cli, "_laptop_sources", return_value=((first, second), ())
        ), patch.object(cli, "_run_exact_laptop_source") as process, redirect_stderr(stderr):
            ambiguous = cli.run_recordings(
                ["process", "abcdef12"],
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "y",
            )
        self.assertEqual(ambiguous, 1)
        self.assertIn("ambiguous", stderr.getvalue())
        process.assert_not_called()

        with patch.object(
            cli, "_laptop_sources", return_value=((first, second), ())
        ), patch.object(
            cli, "_run_exact_laptop_source", return_value=0
        ) as process:
            declined = cli.run_recordings(
                ["process", first.source_identity[:12]],
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "Y",
            )
            accepted = cli.run_recordings(
                ["process", first.source_identity[:12]],
                terminal_isatty=lambda: True,
                input_func=lambda _prompt: "y",
            )
        self.assertEqual(declined, 0)
        self.assertEqual(accepted, 0)
        process.assert_called_once()
        self.assertIs(process.call_args.args[0], first)

    def test_normalize_last_uses_manifest_selected_capture_and_exact_offer(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            recordings_root = Path(temporary) / "recordings"
            ingest_root = Path(temporary) / "ingest"
            latest = recordings_root / "latest"
            source = laptop_source("c" * 64)
            with patch.object(
                cli, "_latest_capture_session", return_value=latest
            ), patch.object(
                cli,
                "normalize_capture",
                return_value=Mock(target_root=ingest_root / "canonical"),
            ) as normalize, patch.object(
                cli, "load_laptop_meeting_source", return_value=source
            ), patch.object(
                cli, "_run_exact_laptop_source", return_value=0
            ) as process:
                result = cli.run_recordings(
                    ["normalize-last"],
                    recordings_root=recordings_root,
                    ingest_root=ingest_root,
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: "y",
                )

        self.assertEqual(result, 0)
        config = normalize.call_args.args[0]
        self.assertEqual(config.session_root, latest)
        self.assertEqual(config.capture_root, recordings_root)
        self.assertEqual(config.ingest_root, ingest_root)
        process.assert_called_once()

    def test_noninteractive_process_never_runs_model_path(self) -> None:
        source = laptop_source("d" * 64)
        output = io.StringIO()
        with patch.object(
            cli, "_laptop_sources", return_value=((source,), ())
        ), patch.object(cli, "_run_exact_laptop_source") as process, redirect_stdout(output):
            result = cli.run_recordings(
                ["process", "last"],
                terminal_isatty=lambda: False,
            )
        self.assertEqual(result, 0)
        self.assertIn("interactive confirmation required", output.getvalue())
        process.assert_not_called()

    def test_unsupported_recovery_explains_no_command_available(self) -> None:
        from types import SimpleNamespace
        audit = SimpleNamespace(accepted=True, merge_ready=False, recovery_ready=True,
                                finalized_segments=(Path("chunk"),), findings=())
        manifest = {"status": "interrupted", "stop_reason": "control_lost", "segments": []}
        output = io.StringIO()
        with patch.object(cli, "load_manifest", return_value=manifest), patch.object(
            cli, "audit_capture_session", return_value=audit
        ), redirect_stdout(output):
            self.assertFalse(cli._print_capture_diagnostic(Path("synthetic-capture")))
        self.assertIn("No automatic recovery command is available", output.getvalue())
        self.assertIn("Keep the original chunks", output.getvalue())
        self.assertNotIn("next=operator review required", output.getvalue())

    def test_diagnose_is_read_only(self) -> None:
        session = Path("/private/tmp/synthetic-session")
        with patch.object(
            cli, "_latest_capture_session", return_value=session
        ), patch.object(
            cli, "_print_capture_diagnostic", return_value=True
        ) as diagnostic, patch.object(
            cli, "normalize_capture"
        ) as normalize, patch.object(
            cli, "_run_exact_laptop_source"
        ) as process:
            result = cli.run_recordings(["diagnose", "last"])
        self.assertEqual(result, 0)
        diagnostic.assert_called_once_with(session)
        normalize.assert_not_called()
        process.assert_not_called()

    def test_recover_last_requires_exact_attestation_and_exact_processing_gate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            recordings_root = Path(temporary) / "recordings"
            ingest_root = Path(temporary) / "ingest"
            session = recordings_root / "latest"
            source = laptop_source("e" * 64)
            with patch.object(
                cli, "_latest_capture_session", return_value=session
            ), patch.object(
                cli, "_print_capture_diagnostic", return_value=True
            ), patch.object(
                cli,
                "load_manifest",
                return_value={
                    "segments": [
                        {"duration_seconds": 900.0},
                        {"duration_seconds": 120.0},
                    ]
                },
            ), patch.object(
                cli,
                "normalize_capture",
                return_value=Mock(target_root=ingest_root / "canonical"),
            ) as normalize, patch.object(
                cli, "load_laptop_meeting_source", return_value=source
            ), patch.object(
                cli, "_run_exact_laptop_source", return_value=0
            ) as process:
                cancelled = cli.run_recordings(
                    ["recover-last"],
                    recordings_root=recordings_root,
                    ingest_root=ingest_root,
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: "recover",
                )
                answers = iter(("RECOVER", "y"))
                recovered = cli.run_recordings(
                    ["recover-last"],
                    recordings_root=recordings_root,
                    ingest_root=ingest_root,
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: next(answers),
                )

        self.assertEqual(cancelled, 0)
        self.assertEqual(recovered, 0)
        normalize.assert_called_once()
        config = normalize.call_args.args[0]
        self.assertTrue(config.allow_terminal_source_loss_recovery)
        self.assertTrue(
            config.operator_attested_meeting_ended_before_source_loss
        )
        self.assertIsNone(config.keep_finalized_segment_count)
        process.assert_called_once()

    def test_control_loss_recovery_flag_requires_exact_confirmation(self):
        from contextlib import ExitStack
        session = Path("/private/tmp/synthetic-control-loss")
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch.object(cli, "_latest_capture_session", return_value=session))
            stack.enter_context(patch.object(cli, "_print_capture_diagnostic", return_value=False))
            stack.enter_context(patch.object(cli, "load_manifest", return_value={"segments": [{"duration_seconds": 10}]}))
            stack.enter_context(patch.object(cli, "audit_capture_session"))
            stack.enter_context(patch.object(cli, "operator_stop_control_loss_recovery_eligible", return_value=True))
            normalize = stack.enter_context(patch.object(cli, "normalize_capture", return_value=Mock(target_root=session)))
            stack.enter_context(patch.object(cli, "load_laptop_meeting_source", return_value=laptop_source("c" * 64)))
            process = stack.enter_context(patch.object(cli, "_run_exact_laptop_source", return_value=0))
            command = ["recover-last", "--operator-stop-control-loss"]
            self.assertEqual(0, cli.run_recordings(command, terminal_isatty=lambda: True,
                                                input_func=lambda _: "recover"))
            normalize.assert_not_called()
            answers = iter(("RECOVER", "y"))
            self.assertEqual(0, cli.run_recordings(command, terminal_isatty=lambda: True,
                                                input_func=lambda _: next(answers)))
            config = normalize.call_args.args[0]
            self.assertTrue(config.allow_operator_stop_control_loss_recovery)
            self.assertTrue(config.operator_attested_meeting_ended_before_operator_stop)
            self.assertFalse(config.allow_terminal_source_loss_recovery)
            process.assert_called_once()
        self.assertIn("meeting finished before you stopped recording", output.getvalue())

    def test_recover_last_can_keep_exact_prefix_and_preserve_processing_gate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            recordings_root = Path(temporary) / "recordings"
            ingest_root = Path(temporary) / "ingest"
            session = recordings_root / "latest"
            source = laptop_source("f" * 64)
            output = io.StringIO()
            with patch.object(
                cli, "_latest_capture_session", return_value=session
            ), patch.object(
                cli, "_print_capture_diagnostic", return_value=True
            ), patch.object(
                cli,
                "load_manifest",
                return_value={
                    "segments": [
                        {"duration_seconds": 900.0},
                        {"duration_seconds": 900.0},
                        {"duration_seconds": 120.0},
                    ]
                },
            ), patch.object(
                cli,
                "normalize_capture",
                return_value=Mock(target_root=ingest_root / "canonical"),
            ) as normalize, patch.object(
                cli, "load_laptop_meeting_source", return_value=source
            ), patch.object(
                cli, "_run_exact_laptop_source", return_value=0
            ) as process, redirect_stdout(output):
                answers = iter(("RECOVER", "y"))
                result = cli.run_recordings(
                    ["recover-last", "--keep-segments", "2"],
                    recordings_root=recordings_root,
                    ingest_root=ingest_root,
                    terminal_isatty=lambda: True,
                    input_func=lambda _prompt: next(answers),
                )

        self.assertEqual(result, 0)
        config = normalize.call_args.args[0]
        self.assertEqual(config.keep_finalized_segment_count, 2)
        self.assertIn("keep chunks 1-2", output.getvalue())
        self.assertIn("exclude 1 tail chunks", output.getvalue())
        process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
