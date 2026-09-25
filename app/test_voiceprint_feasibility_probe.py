from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import voiceprint_feasibility_probe as probe


class FakeEmbedding:
    shape = (1, 192)
    dtype = "float32"

    def __call__(self, audio_path: str):
        self.audio_path = audio_path
        return self


class VoiceprintFeasibilityProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "synthetic-sample.wav"
        self.audio.write_bytes(b"synthetic test placeholder")
        self.output = self.root / "disposable-output"

    def tearDown(self):
        self.temporary.cleanup()

    def test_cli_requires_explicit_audio_and_output(self):
        with self.assertRaises(SystemExit):
            probe.parse_args([])

    def test_explicit_mock_probe_reports_shape_without_embedding_values(self):
        calls = []
        embedding = FakeEmbedding()

        def factory(model, token, cache_dir):
            calls.append((model, token, cache_dir))
            return embedding

        result, report_path = probe.run_probe(
            self.audio,
            self.output,
            embedding_factory=factory,
            runtime_version="test-runtime",
        )

        self.assertEqual("success", result.status)
        self.assertEqual((1, 192), result.embedding_shape)
        self.assertEqual([("pyannote/embedding", None, None)], calls)
        self.assertEqual(str(self.audio.resolve()), embedding.audio_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual([1, 192], report["embedding_shape"])
        self.assertNotIn("embedding", report)
        self.assertNotIn("audio_path", report)
        self.assertNotIn("source_path", report)

    def test_runtime_failure_reports_type_only(self):
        def factory(_model, _token, _cache_dir):
            raise RuntimeError("token or meeting content must not be persisted")

        result, report_path = probe.run_probe(
            self.audio,
            self.output,
            embedding_factory=factory,
            runtime_version="test-runtime",
        )

        self.assertEqual("failure", result.status)
        self.assertEqual("RuntimeError", result.error_type)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual("RuntimeError", report["error_type"])
        self.assertNotIn("token", report)
        self.assertNotIn("meeting content", report)

    def test_repository_audio_and_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            probe.validate_audio_path(Path(__file__))
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            probe.validate_output_dir(Path(__file__).parent / "voiceprint-test-output")

    def test_symlink_audio_is_rejected(self):
        link = self.root / "sample-link.wav"
        link.symlink_to(self.audio)
        with self.assertRaisesRegex(ValueError, "symlink"):
            probe.validate_audio_path(link)

    def test_repository_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            probe.validate_cache_dir(Path(__file__).parent / "voiceprint-cache")

    def test_invalid_embedding_shape_fails_without_vector_persistence(self):
        class BadEmbedding:
            shape = (192,)

            def __call__(self, _audio_path):
                return self

        result, report_path = probe.run_probe(
            self.audio,
            self.output,
            embedding_factory=lambda *_args: BadEmbedding(),
            runtime_version="test-runtime",
        )
        self.assertEqual("failure", result.status)
        self.assertEqual("ValueError", result.error_type)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertNotIn("embedding_values", report)


if __name__ == "__main__":
    unittest.main()
