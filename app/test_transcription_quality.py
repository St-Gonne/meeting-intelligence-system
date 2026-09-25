import unittest
import json
import subprocess
import tempfile
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

from transcription_quality import assess_segments, repetition_loop
from meetingintel_whisper_runtime import guarded_backend


class RepetitionTests(unittest.TestCase):
    def test_english_and_indic_loops(self):
        for unit in ("now", "अब", "अगर", "अब मतलब", "this is the story"):
            with self.subTest(unit=unit):
                self.assertTrue(repetition_loop((unit + " ") * 24))

    def test_natural_code_switching_and_emphasis_pass(self):
        for text in (
            "हाँ ठीक है, but the numbers need checking. फिर बात करते हैं।",
            "No, no, no, that is not what I meant. Yes, yes, I understand.",
            "The first report was late. The second report was late. The third arrived.",
            " ".join(f"item {i} approved" for i in range(100)),
            "now " * 23,
            "",
        ):
            self.assertFalse(repetition_loop(text), text)

    def test_no_transcript_content_in_report(self):
        report = assess_segments([{"text": "private example " * 24}])
        self.assertFalse(report["accepted"])
        self.assertEqual(report["flagged_segments"], 1)
        self.assertNotIn("private", str(report))


class BackendTests(unittest.TestCase):
    def test_preserves_options_result_and_records_actual_language(self):
        result = {"language": "hi", "segments": [{"text": "ठीक है, let us proceed."}]}
        backend = Mock(return_value=result)
        memory = Mock()
        memory.get_peak_memory.return_value = 123
        observations = []
        wrapped = guarded_backend(backend, memory, observations)
        self.assertIs(result, wrapped("audio", language=None, task="transcribe"))
        backend.assert_called_once_with("audio", language=None, task="transcribe")
        memory.clear_cache.assert_called_once_with()
        self.assertEqual(observations[0]["decoded_language"], "hi")
        self.assertNotIn("ठीक", str(observations))

    def test_rejects_loop_before_returning_to_alignment_and_always_clears_cache(self):
        backend = Mock(return_value={"segments": [{"text": "अब " * 40}], "language": "hi"})
        memory = Mock()
        memory.get_peak_memory.return_value = 456
        observations = []
        with self.assertRaisesRegex(RuntimeError, "transcription_quality:repetition_loop"):
            guarded_backend(backend, memory, observations)("audio", language="hi")
        self.assertFalse(observations[0]["quality"]["accepted"])
        memory.clear_cache.assert_called_once_with()

    def test_backend_failure_still_clears_cache_without_fabricating_observation(self):
        memory = Mock()
        observations = []
        backend = Mock(side_effect=RuntimeError("backend failed"))
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            guarded_backend(backend, memory, observations)("audio")
        memory.clear_cache.assert_called_once_with()
        self.assertEqual(observations, [])

    def test_rejects_loop_split_across_backend_timestamp_segments(self):
        for include_text in (False, True):
            result = {"language": "hi", "segments": [{"text": "अब " * 3} for _ in range(8)]}
            if include_text:
                result["text"] = "अब " * 24
            observations = []
            memory = Mock()
            memory.get_peak_memory.return_value = 1
            with self.subTest(include_text=include_text), self.assertRaisesRegex(RuntimeError, "transcription_quality:repetition_loop"):
                guarded_backend(Mock(return_value=result), memory, observations)("audio")
            self.assertEqual(observations[0]["quality"]["flagged_segments"], 0)
            self.assertTrue(observations[0]["quality"]["assembled_text_flagged"])
            memory.clear_cache.assert_called_once_with()


class HelperContractTests(unittest.TestCase):
    def test_qwen_candidate_failure_cannot_leave_success_or_transcript(self):
        import multilingual_transcription_candidate as candidate
        import whispermlx_diarization_helper as helper
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; source.mkdir()
            audio = source / "audio.wav"; audio.write_bytes(b"synthetic")
            output = root / "private" / "failed"; output.parent.mkdir()
            with patch.object(candidate, "run", side_effect=ValueError("candidate:qwen_asr_failed")):
                self.assertFalse(helper.run_diarization(
                    source, 1, 2, "", audio_override=audio, source_mode="audio_first",
                    source_identity="a" * 64, transcription_mode="qwen3-asr-1.7b-contiguous",
                    private_output=output))
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failure")
            self.assertNotEqual(manifest.get("exit_code"), 0)
            self.assertFalse(any(output.glob("audio.*")))

    def test_repaired_helper_rejects_sampled_fallback_for_new_asr(self):
        import whispermlx_diarization_helper as helper
        with self.assertRaisesRegex(ValueError, "not admitted"):
            helper.run_diarization(
                Path("/synthetic"), 1, 2, "", audio_override=Path("/synthetic/audio.m4a"),
                transcription_mode="repaired-standard-auto", private_output=Path("/private/tmp/new"),
                reference_python=Path("/reference/python"), reference_model_cache=Path("/reference/cache"),
            )

    def test_repaired_cached_mode_does_not_require_token_but_legacy_does(self):
        import whispermlx_diarization_helper as helper
        with patch.object(helper, "WHISPERMLX_BIN", Mock(is_file=Mock(return_value=True))), \
             patch.object(helper, "FFMPEG_BIN", Mock(is_file=Mock(return_value=True))), \
             patch.dict(helper.os.environ, {}, clear=True):
            self.assertEqual(helper.validate_environment("repaired-standard-auto"), "")
            with self.assertRaises(SystemExit):
                helper.validate_environment("legacy")

    def test_explicit_repaired_mode_uses_shared_runner_and_speaker_limits(self):
        import multilingual_transcription_candidate as candidate
        import whispermlx_diarization_helper as helper
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_folder = root / "source"
            source_folder.mkdir()
            audio = source_folder / "audio.m4a"
            audio.write_bytes(b"synthetic")
            output = root / "private" / "run"
            output.parent.mkdir()
            resume = root / "checkpoint"
            resume.mkdir()

            def complete(_audio, destination, **_kwargs):
                (destination / "audio.txt").write_text("synthetic transcript")
                return {"status": "ready_for_review"}

            with patch.object(candidate, "run", side_effect=complete) as run:
                self.assertTrue(helper.run_diarization(
                    source_folder,
                    1,
                    3,
                    "synthetic",
                    audio_override=audio,
                    source_mode="audio_first",
                    source_identity="a" * 64,
                    transcription_mode="repaired-standard-auto",
                    private_output=output,
                    resume_asr_dir=resume,
                ))
            self.assertEqual(run.call_args.kwargs["chunk_seconds"], 30)
            self.assertEqual(run.call_args.kwargs["language_policy"], "standard_auto")
            self.assertEqual(run.call_args.kwargs["min_speakers"], 1)
            self.assertEqual(run.call_args.kwargs["max_speakers"], 3)
            self.assertTrue(run.call_args.kwargs["allow_precreated_output"])
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["transcription_mode"], "repaired-standard-auto")
            self.assertEqual(manifest["runtime_adapter"], "multilingual_transcription_candidate_v1")
            self.assertEqual(manifest["resume_asr_dir"], str(resume))
            self.assertIn("--resume-asr-dir", manifest["command_used"])
            self.assertEqual(manifest["status"], "success")

    def test_runtime_receipt_is_never_mistaken_for_transcript_json(self):
        import whispermlx_diarization_helper as helper
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "audio.txt").write_text("synthetic transcript")
            (output / "transcription_runtime.json").write_text('{"completed": true}')
            self.assertEqual(helper.normalize_output_files(output, Path("/synthetic/audio.wav")), [])
            self.assertFalse((output / "audio.json").exists())

    def test_quality_failure_has_safe_operator_diagnostic(self):
        from meetingintel_pipeline import sanitized_helper_failure_reason
        reason = sanitized_helper_failure_reason(1, "RuntimeError: transcription_quality:repetition_loop; review source audio")
        self.assertEqual(reason, "diarization_helper:exit_1:transcription_repetition_rejected")

    def test_launch_uses_runtime_transcription_only_and_explicit_language(self):
        import whispermlx_diarization_helper as helper
        for language in ("en", "hi", "auto"):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(helper, "create_output_folder", return_value=output), \
                     patch.object(helper, "normalize_output_files", return_value=[]), \
                     patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                    self.assertTrue(helper.run_diarization(Path("/synthetic"), 1, 2, "synthetic", language=language))
                command = run.call_args.args[0]
                self.assertEqual(command[1], str(helper.WHISPERMLX_RUNTIME))
                self.assertEqual(command[command.index("--task") + 1], "transcribe")
                if language == "auto":
                    self.assertNotIn("--language", command)
                else:
                    self.assertEqual(command[command.index("--language") + 1], language)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["transcription_language"], language)

    def test_invalid_language_fails_before_creating_output(self):
        import whispermlx_diarization_helper as helper
        with patch.object(helper, "create_output_folder") as create:
            with self.assertRaises(ValueError):
                helper.run_diarization(Path("/synthetic"), 1, 2, "synthetic", language="translate")
            create.assert_not_called()


class RuntimeEntrypointTests(unittest.TestCase):
    def test_private_receipt_written_on_success_and_failure_and_backend_restored(self):
        from meetingintel_whisper_runtime import main, CACHE_LIMIT_BYTES
        for reject in (False, True):
            with self.subTest(reject=reject), tempfile.TemporaryDirectory() as directory:
                memory = types.ModuleType("mlx.core")
                memory.set_cache_limit = Mock()
                memory.clear_cache = Mock()
                memory.get_peak_memory = Mock(return_value=100)
                package = types.ModuleType("mlx")
                package.core = memory
                backend = types.ModuleType("mlx_whisper")
                original = Mock(return_value={"language": "hi", "segments": [{"text": "अब " * 30 if reject else "हाँ, proceed."}]})
                backend.transcribe = original
                cli_module = types.ModuleType("whispermlx.__main__")
                cli_module.cli = lambda: backend.transcribe("synthetic", language=None)
                modules = {"mlx": package, "mlx.core": memory, "mlx_whisper": backend, "whispermlx.__main__": cli_module}
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["runtime", "--output_dir", directory]):
                    if reject:
                        with self.assertRaisesRegex(RuntimeError, "transcription_quality"):
                            main()
                    else:
                        self.assertEqual(main(), 0)
                self.assertIs(backend.transcribe, original)
                memory.set_cache_limit.assert_called_once_with(CACHE_LIMIT_BYTES)
                path = Path(directory) / "transcription_runtime.json"
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                receipt = json.loads(path.read_text())
                self.assertEqual(receipt["completed"], not reject)
                self.assertEqual(receipt["chunks"][0]["decoded_language"], "hi")
                self.assertNotIn("proceed", path.read_text())


if __name__ == "__main__":
    unittest.main()
