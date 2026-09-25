import inspect
import hashlib
import json
import array
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from multilingual_transcription_candidate import ASR_CHECKPOINT_SCHEMA_VERSION, ASR_OPTIONS, admission_assessment, alignment_languages_for, alignment_preserves_text, attach_languages, callable_sha256, choose_bilingual_language, merge_aligned, prepare_output, requested_backend_language, run, transcribe_vad_chunks, validate_asr_checkpoint
from transcription_quality import assess_backend_result
class CandidateTests(unittest.TestCase):
    def test_missing_or_unbound_assessment_never_admits_quality(self):
        report = {"source_sha256": "source", "asr_backend": "backend", "model_revision": "revision"}
        self.assertEqual(admission_assessment(report)["status"], "not_assessed")
        self.assertFalse(admission_assessment(report)["eligible"])
        report["asr_content_review"] = {"status": "accepted", "unresolved_indices": []}
        self.assertFalse(admission_assessment(report)["eligible"])

    def test_only_explicit_bound_clear_assessment_admits_quality(self):
        report = {"source_sha256": "source", "asr_backend": "backend", "model_revision": "revision",
            "asr_content_review": {"status": "accepted", "source_sha256": "source",
                "asr_backend": "backend", "model_revision": "revision", "unresolved_indices": []}}
        self.assertTrue(admission_assessment(report)["eligible"])
        report["asr_content_review"]["unresolved_indices"] = [1]
        self.assertFalse(admission_assessment(report)["eligible"])

    def test_qwen_backend_requires_stable_runtime_and_forbids_reference_fallback(self):
        with self.assertRaisesRegex(ValueError, "qwen_runtime_required"):
            run(Path("/unused/source"), Path("/unused/output"), asr_backend="qwen3-asr-1.7b-contiguous")
        with self.assertRaisesRegex(ValueError, "qwen_reference_fallback_unsupported"):
            run(Path("/unused/source"), Path("/unused/output"), asr_backend="qwen3-asr-1.7b-contiguous",
                qwen_python=Path("/runtime/python"), qwen_model_dir=Path("/runtime/model"),
                resume_asr_dir=Path("/checkpoint"), reference_python=Path("/ref/python"),
                reference_model_cache=Path("/ref/model"))

    def test_versioned_checkpoint_accepts_downstream_only_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"; source.write_bytes(b"source")
            raw = json.dumps({"segments":[{"start":0.0,"end":1.0,"text":"x","language":"en"}]}).encode()
            backend = json.dumps([{"start":0.0,"end":1.0}]).encode()
            class FloatWave(array.array):
                @property
                def dtype(self):
                    return "float32"
            waveform = FloatWave("f", [0.0] * 16000)
            model={"identifier":"model","revision":"rev","assets":[]}; fallback={"enabled":False}
            bounds=[{"start":0.0,"end":1.0,"first_sample":0,"last_sample":16000}]
            checkpoint={"schema_version":ASR_CHECKPOINT_SCHEMA_VERSION,"source_audio":str(source),
                "source_sha256":hashlib.sha256(b"source").hexdigest(),"model":model,"chunk_seconds":30,
                "language_policy":"standard_auto","runtime_versions":{"runtime":"fixed"},"decoder_options":ASR_OPTIONS,
                "transcribe_vad_chunks_sha256":callable_sha256(transcribe_vad_chunks),
                "quality_guard_sha256":callable_sha256(assess_backend_result),"fallback":fallback,
                "vad_runtime_sha256":"vad-code","vad_options":{"chunk_size":30},
                "raw_sha256":hashlib.sha256(raw).hexdigest(),"backend_chunks_sha256":hashlib.sha256(backend).hexdigest(),
                "raw_segment_count":1,"vad":{"count":1,"bounds":bounds,"waveform":{"sample_rate":16000,"dtype":"float32","sample_count":16000,"sha256":hashlib.sha256(waveform.tobytes()).hexdigest()}}}
            with patch("multilingual_transcription_candidate.package_versions",return_value={"runtime":"fixed"}):
                self.assertTrue(validate_asr_checkpoint(checkpoint,checkpoint["source_sha256"],raw,backend,waveform,model,30,"standard_auto",fallback,"vad-code"))
                for key,value in (("model",{"identifier":"changed"}),("chunk_seconds",20),("fallback",{"enabled":True})):
                    changed=dict(checkpoint); changed[key]=value
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError,"provenance_mismatch"):
                        validate_asr_checkpoint(changed,checkpoint["source_sha256"],raw,backend,waveform,model,30,"standard_auto",fallback,"vad-code")
                with self.assertRaisesRegex(ValueError,"hash_mismatch"):
                    validate_asr_checkpoint(checkpoint,checkpoint["source_sha256"],raw+b"corrupt",backend,waveform,model,30,"standard_auto",fallback,"vad-code")

    def test_empty_chunk_language_does_not_require_alignment_model(self):
        segments = [
            {"language": "en", "text": "spoken"},
            {"language": "jw", "text": "  "},
            {"language": "hi", "text": "बोला"},
        ]
        self.assertEqual(alignment_languages_for(segments), {"en", "hi"})

    def test_repaired_defaults_use_standard_auto_language_and_30_second_chunks(self):
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(run).parameters.items()
        }
        self.assertEqual(defaults["language_policy"], "standard_auto")
        self.assertEqual(defaults["chunk_seconds"], 30)

    def test_sampled_reference_fallback_is_not_admitted_for_new_asr(self):
        with self.assertRaisesRegex(ValueError, "sampled_reference_fallback_not_admitted"):
            run(Path("/unused/source"), Path("/unused/output"),
                reference_python=Path("/unused/python"),
                reference_model_cache=Path("/unused/cache"))

    def test_bilingual_selection_can_switch_and_does_not_select_urdu_script(self):
        self.assertEqual(choose_bilingual_language({"en": .8, "hi": .1, "ur": .1}), "en")
        self.assertEqual(choose_bilingual_language({"en": .1, "hi": .3, "ur": .6}), "hi")
        with self.assertRaises(ValueError):
            choose_bilingual_language({"en": .9})

    def test_standard_auto_never_silently_forces_english_or_hindi(self):
        self.assertIsNone(requested_backend_language("standard_auto"))
        self.assertIsNone(requested_backend_language("unrestricted"))
        self.assertEqual(
            requested_backend_language("bilingual", {"en": .2, "hi": .8}),
            "hi",
        )
        with self.assertRaisesRegex(ValueError, "missing_language_probabilities"):
            requested_backend_language("bilingual")

    def test_real_backend_boundary_restarts_auto_after_hindi_and_english(self):
        backend = Mock(side_effect=[
            {"language": "hi", "text": "first", "segments": [{"text": "first"}]},
            {"language": "en", "text": "second", "segments": [{"text": "second"}]},
            {"language": "hi", "text": "third", "segments": [{"text": "third"}]},
        ])
        clear_cache = Mock()
        raw, observations = transcribe_vad_chunks(
            list(range(48000)),
            [
                {"start": 0, "end": 1},
                {"start": 1, "end": 2},
                {"start": 2, "end": 3},
            ],
            backend,
            "/models/large-v3",
            "standard_auto",
            clear_cache,
        )
        self.assertEqual([x["language"] for x in raw["segments"]], ["hi", "en", "hi"])
        self.assertEqual([x["decoded_language"] for x in observations], ["hi", "en", "hi"])
        self.assertEqual([call.kwargs["language"] for call in backend.call_args_list], [None, None, None])
        self.assertTrue(all(call.kwargs["task"] == "transcribe" for call in backend.call_args_list))
        self.assertTrue(all(call.kwargs["initial_prompt"] is None for call in backend.call_args_list))
        self.assertEqual(clear_cache.call_count, 3)

    def test_backend_boundary_clears_cache_when_decode_or_quality_fails(self):
        for result in (
            RuntimeError("decode failed"),
            {"language": "hi", "text": "अब " * 30, "segments": [{"text": "अब " * 30}]},
        ):
            backend = Mock(side_effect=result if isinstance(result, Exception) else None)
            if not isinstance(result, Exception):
                backend.return_value = result
            clear_cache = Mock()
            with self.subTest(result=type(result).__name__), self.assertRaises(Exception):
                transcribe_vad_chunks(
                    [0] * 16000,
                    [{"start": 0, "end": 1}],
                    backend,
                    "/models/large-v3",
                    "standard_auto",
                    clear_cache,
                )
            clear_cache.assert_called_once_with()

    def test_repetition_only_uses_reference_on_identical_array(self):
        failing = {"language": "hi", "text": "अब " * 30, "segments": [{"text": "अब " * 30}]}
        recovered = {"language": "hi", "text": "recovered", "segments": [{"text": "recovered"}]}
        backend = Mock(return_value=failing)
        fallback = Mock(return_value=recovered)
        samples = [0] * 16000
        raw, observations = transcribe_vad_chunks(
            samples,
            [{"start": 0, "end": 1}],
            backend,
            "/models/large-v3",
            "standard_auto",
            Mock(),
            fallback_backend=fallback,
        )
        self.assertEqual(raw["segments"][0]["text"], "recovered")
        self.assertTrue(observations[0]["fallback_used"])
        self.assertFalse(observations[0]["primary_quality"]["accepted"])
        self.assertTrue(observations[0]["quality"]["accepted"])
        self.assertEqual(fallback.call_args.args[0], samples)

    def test_passing_primary_never_invokes_reference(self):
        backend = Mock(return_value={"language": "en", "text": "clean", "segments": [{"text": "clean"}]})
        fallback = Mock()
        transcribe_vad_chunks(
            [0] * 16000,
            [{"start": 0, "end": 1}],
            backend,
            "/models/large-v3",
            "standard_auto",
            Mock(),
            fallback_backend=fallback,
        )
        fallback.assert_not_called()

    def test_alignment_must_not_drop_or_reorder_content(self):
        raw = {"segments": [{"text": "Hello नमस्ते"}, {"text": "world"}]}
        self.assertTrue(alignment_preserves_text(raw, {"segments": [{"text": "Hello"}, {"text": "नमस्ते world"}]}))
        self.assertFalse(alignment_preserves_text(raw, {"segments": [{"text": "Hello world"}]}))
        self.assertFalse(alignment_preserves_text(raw, {"segments": [{"text": "world Hello नमस्ते"}]}))

    def test_languages_follow_each_chunk_and_do_not_mutate_source(self):
        segments = [{"text": "Hello", "start": 0, "end": 1}, {"text": "नमस्ते", "start": 1, "end": 2}]
        result = attach_languages(segments, [{"decoded_language": "en"}, {"decoded_language": "hi"}])
        self.assertEqual([x["language"] for x in result], ["en", "hi"])
        self.assertNotIn("language", segments[0])

    def test_mismatch_fails_instead_of_shifting_languages(self):
        with self.assertRaisesRegex(ValueError, "count_mismatch"):
            attach_languages([{"text": "one"}], [])

    def test_merge_restores_timeline_without_dropping_unaligned_words(self):
        en = {"segments": [{"start": 10, "end": 12, "language": "en", "text": "World", "words": [{"word": "World"}]}]}
        hi = {"segments": [{"start": 2, "end": 4, "language": "hi", "text": "नमस्ते", "words": [{"word": "नमस्ते", "start": 2, "end": 4}]}]}
        result = merge_aligned([en, hi])
        self.assertEqual([s["start"] for s in result["segments"]], [2, 10])
        self.assertEqual(result["language"], "mixed")
        self.assertEqual(result["languages"], ["en", "hi"])
        self.assertEqual(len(result["word_segments"]), 2)

    def test_output_cannot_replace_source_or_reuse_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            audio = source_dir / "audio.wav"
            audio.write_bytes(b"synthetic")
            with self.assertRaisesRegex(ValueError, "protected_output"):
                prepare_output(audio, source_dir / "candidate")
            _, output = prepare_output(audio, root / "candidate")
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                prepare_output(audio, root / "candidate")

    def test_helper_may_use_only_an_empty_precreated_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "audio.wav"
            source.write_bytes(b"synthetic")
            output = root / "output"
            output.mkdir()
            _, accepted = prepare_output(source, output, allow_precreated=True)
            self.assertEqual(accepted, output.resolve())
            (output / "existing").write_text("no")
            with self.assertRaisesRegex(ValueError, "precreated_output_not_empty"):
                prepare_output(source, output, allow_precreated=True)


if __name__ == "__main__":
    unittest.main()
