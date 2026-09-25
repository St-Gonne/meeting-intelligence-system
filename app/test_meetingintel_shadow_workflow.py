#!/usr/bin/env python3
"""Deterministic tests for the MeetingIntel shadow workflow wrapper."""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import meetingintel_shadow_workflow as workflow
import whispermlx_diarization_helper as diarization_helper


class CliTests(unittest.TestCase):
    def test_inputs_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            workflow.parse_args(
                [
                    "--meeting-folder",
                    "/meeting",
                    "--diarization-folder",
                    "/diarization",
                ]
            )

    def test_defaults(self) -> None:
        args = workflow.parse_args(["--diarization-folder", "/diarization"])
        self.assertFalse(args.with_layer3)
        self.assertFalse(args.dry_run)
        self.assertEqual(2, args.min_speakers)
        self.assertEqual(2, args.max_speakers)
        self.assertIsNone(args.trim_after_minutes)
        self.assertEqual([], args.speaker_map)

    def test_repeatable_speaker_map(self) -> None:
        args = workflow.parse_args(
            [
                "--diarization-folder",
                "/diarization",
                "--speaker-map",
                "SPEAKER_00=Alex",
                "--speaker-map",
                "SPEAKER_01=unknown",
            ]
        )
        self.assertEqual(
            ["SPEAKER_00=Alex", "SPEAKER_01=unknown"], args.speaker_map
        )

    def test_integer_trim_parsing(self) -> None:
        args = workflow.parse_args(
            ["--meeting-folder", "/meeting", "--trim-after-minutes", "49"]
        )
        self.assertEqual(49.0, args.trim_after_minutes)

    def test_decimal_trim_parsing(self) -> None:
        args = workflow.parse_args(
            ["--meeting-folder", "/meeting", "--trim-after-minutes", "48.5"]
        )
        self.assertEqual(48.5, args.trim_after_minutes)

    def test_invalid_trim_values_are_rejected(self) -> None:
        for value in ("0", "-1", "nan", "invalid"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                workflow.parse_args(
                    [
                        "--meeting-folder",
                        "/meeting",
                        "--trim-after-minutes",
                        value,
                    ]
                )

    def test_trim_is_rejected_for_existing_diarization(self) -> None:
        with self.assertRaises(SystemExit):
            workflow.parse_args(
                [
                    "--diarization-folder",
                    "/diarization",
                    "--trim-after-minutes",
                    "49",
                ]
            )


class DiscoveryTests(unittest.TestCase):
    def test_discovers_exactly_one_new_manifest(self) -> None:
        old = Path("/tests/old/run_manifest.json")
        new = Path("/tests/new/run_manifest.json")
        with patch.object(workflow, "snapshot_manifests", return_value={old, new}):
            self.assertEqual(
                new,
                workflow.discover_new_manifest(Path("/tests"), {old}, "Test"),
            )

    def test_rejects_ambiguous_discovery(self) -> None:
        manifests = {
            Path("/tests/one/run_manifest.json"),
            Path("/tests/two/run_manifest.json"),
        }
        with patch.object(workflow, "snapshot_manifests", return_value=manifests):
            with self.assertRaisesRegex(workflow.WorkflowError, "ambiguous"):
                workflow.discover_new_manifest(Path("/tests"), set(), "Test")


class TrimTests(unittest.TestCase):
    def test_mocked_ffprobe_duration_validation(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout="3344.47\n",
            stderr="",
        )
        with patch.object(workflow, "FFPROBE_BIN", Path("/bin/sh")), patch.object(
            workflow.subprocess,
            "run",
            return_value=completed,
        ) as run:
            duration = workflow.probe_source_duration(Path("/meeting/audio.mp4"))

        self.assertEqual(3344.47, duration)
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_prepare_trim_uses_exact_duration_and_preserves_original_path(self) -> None:
        meeting = Path("/meetings/exact")
        trim_folder = Path("/tests/trimmed/run")
        with patch.object(
            workflow,
            "probe_source_duration",
            return_value=3600.0,
        ), patch.object(
            workflow,
            "FFMPEG_BIN",
            Path("/bin/sh"),
        ), patch.object(
            workflow,
            "create_trim_folder",
            return_value=trim_folder,
        ), patch.object(workflow, "run_child") as run_child, patch.object(
            workflow,
            "validate_trimmed_audio",
        ), patch.object(workflow, "write_json") as write_json:
            result = workflow.prepare_trim(meeting, 48.5)

        command = run_child.call_args.args[0]
        self.assertEqual("2910", command[command.index("-t") + 1])
        self.assertEqual(str(meeting / "audio.mp4"), command[command.index("-i") + 1])
        self.assertEqual(str(trim_folder / "audio_trimmed.mp4"), command[-1])
        self.assertNotEqual(meeting / "audio.mp4", result.audio_path)
        self.assertEqual(48.5, result.details["trim_after_minutes"])
        write_json.assert_called_once()

    def test_out_of_range_trim_stops_before_ffmpeg_or_folder_creation(self) -> None:
        with patch.object(
            workflow,
            "probe_source_duration",
            return_value=120.0,
        ), patch.object(workflow, "create_trim_folder") as create_folder, patch.object(
            workflow,
            "run_child",
        ) as run_child:
            with self.assertRaisesRegex(workflow.WorkflowError, "must be shorter"):
                workflow.prepare_trim(Path("/meetings/exact"), 2.0)

        create_folder.assert_not_called()
        run_child.assert_not_called()

    def test_diarization_command_receives_trimmed_audio_override(self) -> None:
        trimmed = Path("/tests/trimmed/audio_trimmed.mp4")
        command = workflow.build_diarization_command(
            Path("/meetings/exact"),
            2,
            2,
            trimmed,
        )
        self.assertEqual(
            ["--audio-override", str(trimmed)],
            command[-2:],
        )

    def test_final_summary_includes_applied_trim_status(self) -> None:
        result = workflow.WorkflowResult(
            diarization_folder=Path("/tests/diarization"),
            layer2_path="/tests/layer2.md",
            layer3_path="not requested",
            comparison_path="/tests/comparison.md",
            manifest_path=Path("/tests/run_manifest.json"),
            warnings=[],
            trim_status="applied after 49 minutes — /tests/audio_trimmed.mp4",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            workflow.print_final_outputs(result)
        self.assertIn(
            "Trim: applied after 49 minutes — /tests/audio_trimmed.mp4",
            output.getvalue(),
        )

    def test_final_summary_uses_honest_check_language_and_timings(self) -> None:
        result = workflow.WorkflowResult(
            diarization_folder=Path("/tests/diarization"),
            layer2_path="/tests/layer2.md",
            layer3_path="/tests/layer3.md",
            comparison_path="/tests/comparison.md",
            manifest_path=Path("/tests/run_manifest.json"),
            warnings=[],
            trim_status="not applied",
            stage_timings={
                "trim": workflow.stage_entry("not_applied"),
                "diarization": workflow.stage_entry("success", 10.0),
                "speaker_confirmation": workflow.stage_entry("success", 4.0),
                "layer2": workflow.stage_entry("success", 3.0),
                "layer3": workflow.stage_entry("success", 2.0),
                "compute_stage_total": workflow.stage_entry("measured", 15.0),
                "workflow_elapsed": workflow.stage_entry("measured", 19.0),
            },
        )
        output = io.StringIO()
        with redirect_stdout(output):
            workflow.print_final_outputs(result)
        rendered = output.getvalue()
        self.assertIn("SPEAKER STRUCTURE / FORMAT CHECKS", rendered)
        self.assertIn(
            "Lint only — does not verify factual speaker attribution.", rendered
        )
        self.assertIn("Speaker confirmation: 4.0s", rendered)
        self.assertIn("Compute/stage total: 15.0s", rendered)
        self.assertIn("Workflow elapsed: 19.0s", rendered)

    def test_compute_total_excludes_speaker_confirmation(self) -> None:
        manifest = {
            "stage_timings": {
                "speaker_confirmation": workflow.stage_entry("success", 50.0),
                "layer2": workflow.stage_entry("success", 3.0),
                "layer3": workflow.stage_entry("not_requested"),
            }
        }
        timings = workflow.build_workflow_stage_timings(
            workflow.stage_entry("success", 2.0),
            workflow.stage_entry("success", 10.0),
            manifest,
            65.0,
        )
        self.assertEqual(15.0, timings["compute_stage_total"]["duration_seconds"])
        self.assertEqual(50.0, timings["speaker_confirmation"]["duration_seconds"])
        self.assertEqual(65.0, timings["workflow_elapsed"]["duration_seconds"])

    def test_shadow_command_passes_speaker_maps(self) -> None:
        command = workflow.build_shadow_command(
            Path("/tests/diarization"),
            True,
            ["SPEAKER_00=Alex", "SPEAKER_01=unknown"],
        )
        self.assertEqual(
            [
                "--speaker-map",
                "SPEAKER_00=Alex",
                "--speaker-map",
                "SPEAKER_01=unknown",
            ],
            command[-4:],
        )


class DiarizationOutputNormalizationTests(unittest.TestCase):
    def test_trimmed_stem_outputs_are_copied_to_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            for extension in ("srt", "txt", "tsv", "json", "vtt"):
                (output_folder / f"audio_trimmed.{extension}").write_text(
                    f"trimmed-{extension}",
                    encoding="utf-8",
                )

            copies = diarization_helper.normalize_output_files(
                output_folder,
                Path("/tests/trimmed/audio_trimmed.mp4"),
            )

            self.assertEqual(5, len(copies))
            for extension in ("srt", "txt", "tsv", "json", "vtt"):
                canonical = output_folder / f"audio.{extension}"
                self.assertEqual(
                    f"trimmed-{extension}",
                    canonical.read_text(encoding="utf-8"),
                )
                self.assertTrue(
                    any(copy["canonical"] == str(canonical) for copy in copies)
                )

    def test_existing_canonical_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            canonical = output_folder / "audio.srt"
            canonical.write_text("canonical-content", encoding="utf-8")
            (output_folder / "audio_trimmed.srt").write_text(
                "trimmed-content",
                encoding="utf-8",
            )

            copies = diarization_helper.normalize_output_files(
                output_folder,
                Path("/tests/trimmed/audio_trimmed.mp4"),
            )

            self.assertEqual("canonical-content", canonical.read_text(encoding="utf-8"))
            self.assertFalse(
                any(copy["canonical"] == str(canonical) for copy in copies)
            )

    def test_success_manifest_records_output_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            copied = {
                "source": str(output_folder / "audio_trimmed.srt"),
                "canonical": str(output_folder / "audio.srt"),
            }
            with patch.object(
                diarization_helper,
                "create_output_folder",
                return_value=output_folder,
            ), patch.object(
                diarization_helper.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ), patch.object(
                diarization_helper,
                "normalize_output_files",
                return_value=[copied],
            ):
                success = diarization_helper.run_diarization(
                    Path("/meetings/exact"),
                    2,
                    2,
                    "test-token",
                    Path("/tests/trimmed/audio_trimmed.mp4"),
                )

            manifest = json.loads(
                (output_folder / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(success)
            self.assertEqual("success", manifest["status"])
            self.assertEqual(
                {
                    "applied": True,
                    "preferred_source_stem": "audio_trimmed",
                    "copies": [copied],
                },
                manifest["output_normalization"],
            )


class ExplicitAudioHelperTests(unittest.TestCase):
    def test_exact_production_meetily_argv_reaches_meetily_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "meeting"
            source.mkdir()
            executable = root / "executable"
            executable.write_text("synthetic", encoding="utf-8")
            child = textwrap.dedent(
                """
                import os
                from pathlib import Path
                from unittest.mock import patch
                import whispermlx_diarization_helper as helper

                source = Path(os.environ["TEST_SOURCE_FOLDER"])
                executable = Path(os.environ["TEST_EXECUTABLE"])

                def select_meetily(args):
                    assert args.meeting_folder == source
                    assert args.audio_path is None
                    assert args.source_folder is None
                    assert args.source_id is None
                    print("MEETILY_BRANCH_REACHED")
                    return [source]

                with patch.object(helper, "WHISPERMLX_BIN", executable), \
                     patch.object(helper, "FFMPEG_BIN", executable), \
                     patch.object(helper, "select_folders", side_effect=select_meetily), \
                     patch.object(helper, "run_diarization", return_value=True) as inference:
                    result = helper.main()
                    assert inference.call_count == 1
                    assert inference.call_args.args[:4] == (source, 1, 8, "synthetic-token")
                raise SystemExit(result)
                """
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_TOKEN": "synthetic-token",
                    "TEST_SOURCE_FOLDER": str(source),
                    "TEST_EXECUTABLE": str(executable),
                }
            )
            command = [
                sys.executable,
                "-c",
                child,
                "--meeting-folder",
                str(source),
                "--min-speakers",
                "1",
                "--max-speakers",
                "8",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, env=environment
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("MEETILY_BRANCH_REACHED", result.stdout)
        self.assertNotIn("--audio-path", command)
        self.assertNotIn("--source-folder", command)
        self.assertNotIn("--source-id", command)

    def test_exact_production_meetily_argv_without_token_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "meeting"
            source.mkdir()
            executable = root / "executable"
            executable.write_text("synthetic", encoding="utf-8")
            child = textwrap.dedent(
                """
                import os
                from pathlib import Path
                from unittest.mock import patch
                import whispermlx_diarization_helper as helper

                executable = Path(os.environ["TEST_EXECUTABLE"])
                with patch.object(helper, "WHISPERMLX_BIN", executable), \
                     patch.object(helper, "FFMPEG_BIN", executable), \
                     patch.object(helper, "run_diarization") as inference:
                    try:
                        helper.main()
                    finally:
                        assert inference.call_count == 0
                """
            )
            environment = os.environ.copy()
            environment.pop("HF_TOKEN", None)
            environment["TEST_EXECUTABLE"] = str(executable)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child,
                    "--meeting-folder",
                    str(source),
                    "--min-speakers",
                    "1",
                    "--max-speakers",
                    "8",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
        self.assertEqual(2, result.returncode)
        self.assertEqual(
            "Error: HF_TOKEN is not set. Set it in your environment before running "
            "this helper; do not place the token in the command itself.\n",
            result.stderr,
        )

    def test_explicit_audio_validation_accepts_file_and_rejects_directory_or_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.m4a"
            audio.write_bytes(b"audio")
            before = audio.read_bytes()
            self.assertEqual(audio.resolve(), diarization_helper.validate_explicit_audio(audio))
            self.assertEqual(before, audio.read_bytes())
            with self.assertRaises(SystemExit):
                diarization_helper.validate_explicit_audio(root)
            with self.assertRaises(SystemExit):
                diarization_helper.validate_explicit_audio(root / "missing.m4a")

    def test_explicit_manifest_and_command_preserve_identity_bounds_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            audio = source / "audio.m4a"
            audio.write_bytes(b"unchanged")
            output = root / "output"
            output.mkdir()
            identity = "a" * 64
            with patch.object(
                diarization_helper, "create_output_folder", return_value=output
            ) as create, patch.object(
                diarization_helper.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run, patch.object(
                diarization_helper, "normalize_output_files", return_value=[]
            ):
                success = diarization_helper.run_diarization(
                    source, 1, 8, "token", audio,
                    source_mode="audio_first", source_identity=identity,
                )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertTrue(success)
            self.assertEqual("audio_first", manifest["source_mode"])
            self.assertEqual(identity, manifest["source_identity"])
            self.assertEqual(str(audio), manifest["source_audio"])
            command = run.call_args.args[0]
            self.assertEqual("1", command[command.index("--min_speakers") + 1])
            self.assertEqual("8", command[command.index("--max_speakers") + 1])
            self.assertEqual(b"unchanged", audio.read_bytes())
            create.assert_called_once_with(source, f"audio_first_{identity}")

    def test_existing_meetily_run_keeps_original_staging_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                diarization_helper, "create_output_folder", return_value=output
            ) as create, patch.object(
                diarization_helper.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1),
            ):
                diarization_helper.run_diarization(
                    Path("/meetings/exact"), 2, 2, "token"
                )
            create.assert_called_once_with(Path("/meetings/exact"), None)


class WorkflowTests(unittest.TestCase):
    def make_args(self, **overrides: object) -> argparse.Namespace:
        values = {
            "meeting_folder": None,
            "diarization_folder": Path("/tests/diarization/exact"),
            "with_layer3": True,
            "min_speakers": 2,
            "max_speakers": 2,
            "trim_after_minutes": None,
            "dry_run": False,
            "speaker_map": [],
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def shadow_manifest(self, diarization_folder: Path) -> dict[str, object]:
        return {
            "status": "success",
            "diarization_folder": str(diarization_folder),
            "layer2_output": "/tests/shadow/run/layer2_diarized.md",
            "layer3_status": "success",
            "layer3_path": "/tests/shadow/run/layer3_brief.md",
            "comparison_output": "/tests/shadow/run/comparison.md",
            "speaker_attribution_warnings": ["Layer 2 warning"],
            "layer3_attribution_warnings": ["Layer 3 warning"],
            "speaker_check_semantics": "structural_format_lint_only",
            "stage_timings": {
                "speaker_confirmation": workflow.stage_entry("success", 4.0),
                "layer2": workflow.stage_entry("success", 3.0),
                "layer3": workflow.stage_entry("success", 2.0),
            },
        }

    @patch.object(workflow, "print_final_outputs")
    @patch.object(workflow, "load_manifest")
    @patch.object(workflow, "discover_new_manifest")
    @patch.object(workflow, "snapshot_manifests", return_value=set())
    @patch.object(workflow, "run_child")
    @patch.object(workflow, "workflow_lock", return_value=nullcontext())
    @patch.object(workflow, "validate_diarization_folder")
    @patch.object(workflow, "record_trim_details")
    @patch.object(workflow, "inherited_trim_details")
    def test_existing_diarization_skips_diarization_stage(
        self,
        inherited_trim: Mock,
        _record_trim: Mock,
        validate_diarization: Mock,
        _lock: Mock,
        run_child: Mock,
        _snapshot: Mock,
        discover: Mock,
        load_manifest: Mock,
        print_outputs: Mock,
    ) -> None:
        diarization = Path("/tests/diarization/exact")
        shadow_manifest_path = Path("/tests/shadow/run/run_manifest.json")
        validate_diarization.return_value = diarization
        inherited_trim.return_value = {"applied": False, "mode": "none"}
        discover.return_value = shadow_manifest_path
        load_manifest.return_value = self.shadow_manifest(diarization)

        result = workflow.run_workflow(self.make_args())

        run_child.assert_called_once()
        command, stage = run_child.call_args.args
        self.assertEqual("Layer 2/3 shadow", stage)
        self.assertIn("--with-layer3", command)
        self.assertEqual(diarization, result.diarization_folder)
        self.assertEqual(["Layer 2 warning", "Layer 3 warning"], result.warnings)
        self.assertEqual("not applied", result.trim_status)
        print_outputs.assert_called_once_with(result)

    @patch.object(workflow, "print_final_outputs")
    @patch.object(workflow, "load_manifest")
    @patch.object(workflow, "discover_new_manifest")
    @patch.object(workflow, "snapshot_manifests", return_value=set())
    @patch.object(workflow, "run_child")
    @patch.object(workflow, "workflow_lock", return_value=nullcontext())
    @patch.object(workflow, "validate_meeting_folder")
    @patch.object(workflow, "record_trim_details")
    def test_meeting_input_runs_both_stages_with_exact_discovered_folder(
        self,
        _record_trim: Mock,
        validate_meeting: Mock,
        _lock: Mock,
        run_child: Mock,
        _snapshot: Mock,
        discover: Mock,
        load_manifest: Mock,
        _print_outputs: Mock,
    ) -> None:
        meeting = Path("/meetings/exact")
        diarization_manifest_path = Path(
            "/tests/diarization/new/run_manifest.json"
        )
        shadow_manifest_path = Path("/tests/shadow/new/run_manifest.json")
        diarization = diarization_manifest_path.parent
        validate_meeting.return_value = meeting
        discover.side_effect = [diarization_manifest_path, shadow_manifest_path]
        load_manifest.side_effect = [
            {"source_folder": str(meeting)},
            self.shadow_manifest(diarization),
        ]

        result = workflow.run_workflow(
            self.make_args(
                meeting_folder=meeting,
                diarization_folder=None,
                min_speakers=3,
                max_speakers=6,
            )
        )

        self.assertEqual(2, run_child.call_count)
        diarization_command = run_child.call_args_list[0].args[0]
        shadow_command = run_child.call_args_list[1].args[0]
        self.assertEqual(
            ["--min-speakers", "3", "--max-speakers", "6"],
            diarization_command[-4:],
        )
        self.assertEqual(str(diarization), shadow_command[-2])
        self.assertEqual("--with-layer3", shadow_command[-1])
        self.assertEqual(diarization, result.diarization_folder)
        self.assertEqual("not applied", result.trim_status)

    @patch.object(workflow, "print_dry_run")
    @patch.object(workflow, "workflow_lock")
    @patch.object(workflow, "subprocess")
    @patch.object(workflow, "validate_diarization_folder")
    @patch.object(workflow, "record_trim_details")
    @patch.object(workflow, "prepare_trim")
    def test_dry_run_uses_no_subprocess_and_no_lock_write(
        self,
        prepare_trim: Mock,
        record_trim: Mock,
        validate_diarization: Mock,
        subprocess_module: Mock,
        lock: Mock,
        print_dry_run: Mock,
    ) -> None:
        diarization = Path("/tests/diarization/exact")
        validate_diarization.return_value = diarization

        result = workflow.run_workflow(self.make_args(dry_run=True))

        self.assertIsNone(result)
        subprocess_module.run.assert_not_called()
        lock.assert_not_called()
        prepare_trim.assert_not_called()
        record_trim.assert_not_called()
        print_dry_run.assert_called_once()

    def test_child_failure_is_clear(self) -> None:
        with patch.object(
            workflow.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 7),
        ):
            with self.assertRaisesRegex(workflow.WorkflowError, "exit code 7"):
                workflow.run_child(["child"], "Test stage")

    @patch.object(workflow, "print_dry_run")
    @patch.object(workflow, "write_json")
    @patch.object(workflow, "create_trim_folder")
    @patch.object(workflow, "workflow_lock")
    @patch.object(workflow, "subprocess")
    @patch.object(workflow, "validate_meeting_folder")
    def test_trimmed_dry_run_performs_no_subprocesses_or_writes(
        self,
        validate_meeting: Mock,
        subprocess_module: Mock,
        lock: Mock,
        create_trim_folder: Mock,
        write_json: Mock,
        print_dry_run: Mock,
    ) -> None:
        meeting = Path("/meetings/exact")
        validate_meeting.return_value = meeting

        result = workflow.run_workflow(
            self.make_args(
                meeting_folder=meeting,
                diarization_folder=None,
                trim_after_minutes=49.0,
                dry_run=True,
            )
        )

        self.assertIsNone(result)
        subprocess_module.run.assert_not_called()
        lock.assert_not_called()
        create_trim_folder.assert_not_called()
        write_json.assert_not_called()
        print_dry_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
