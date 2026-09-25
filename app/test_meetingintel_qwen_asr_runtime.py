import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import meetingintel_qwen_asr_runtime as runtime


class Tensor:
    def __init__(self, values): self.values, self.shape = list(values), (1, len(values))
    def __getitem__(self, key):
        if isinstance(key, tuple): return Tensor(self.values[key[1]])
        if key == 0: return self
        return self.values[key]
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.values)


class Inputs(dict):
    def __init__(self): super().__init__(input_ids=Tensor([99]))
    def to(self, *args): return self


class Processor:
    def __init__(self, parsed): self.parsed = list(parsed)
    def apply_transcription_request(self, audio): return Inputs()
    def decode(self, ids, return_format): return [self.parsed.pop(0)] if self.parsed else []


class Model:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.generation_config = types.SimpleNamespace(eos_token_id=2)
    def generate(self, **kwargs): return types.SimpleNamespace(sequences=Tensor(self.outputs.pop(0)))


TORCH = types.SimpleNamespace(float16="float16", inference_mode=contextlib.nullcontext)


class QwenRuntimeTests(unittest.TestCase):
    def waveform(self, seconds): return runtime.np.zeros(seconds * runtime.SAMPLE_RATE, dtype=runtime.np.float32)

    def test_missing_language_tag_preserves_valid_text_with_script_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chunks = runtime.transcribe_waveform(
                self.waveform(1), output,
                Processor([{"language": None, "transcription": "usable text"}]),
                Model([[99, 2]]), TORCH,
            )
            self.assertEqual("en", chunks[0]["language"])
            self.assertEqual("script_fallback", chunks[0]["language_resolution"])

    def test_resume_prefix_is_hash_validated_and_continues_from_its_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); waveform = self.waveform(60)
            samples = waveform[:30 * runtime.SAMPLE_RATE]
            prefix = [{"index": 0, "sample_start": 0, "sample_end": len(samples),
                "sample_count": len(samples), "sample_sha256": runtime.hashlib.sha256(samples.tobytes()).hexdigest(),
                "start": 0.0, "end": 30.0, "language": "en", "text": "first",
                "ended_with_eos": True}]
            (root / "backend_chunks.json").write_text(json.dumps(prefix))
            resumed = runtime.load_resume_chunks(waveform, root)
            chunks = runtime.transcribe_waveform(waveform, root,
                Processor([{"language": "English", "transcription": "second"}]),
                Model([[99, 2]]), TORCH, resumed)
            self.assertEqual([(0, 30), (30, 60)], [
                (int(x["start"]), int(x["end"])) for x in chunks
            ])

    def test_cap_eos_split_minimum_failure_and_sample_continuity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); waveform = self.waveform(60); cap = runtime.MAX_NEW_TOKENS
            chunks = runtime.transcribe_waveform(waveform, output,
                Processor([{"language": "English", "transcription": "left"},
                           {"language": "Hindi", "transcription": "right"}]),
                Model([[99] + [7] * cap, [99, 7, 2], [99, 8, 2]]), TORCH)
            self.assertEqual([(x["sample_start"], x["sample_end"]) for x in chunks],
                [(0, 30 * runtime.SAMPLE_RATE), (30 * runtime.SAMPLE_RATE, 60 * runtime.SAMPLE_RATE)])
            admitted = runtime.transcribe_waveform(self.waveform(30), output,
                Processor([{"language": "English", "transcription": "complete"}]),
                Model([[99] + [7] * (cap - 1) + [2]]), TORCH)
            self.assertEqual(admitted[0]["output_token_count"], cap)
            with self.assertRaisesRegex(ValueError, "minimum_window"):
                runtime.transcribe_waveform(self.waveform(30), output, Processor([]),
                    Model([[99] + [7] * cap]), TORCH)
            self.assertEqual(json.loads((output / "qwen_asr_failure.json").read_text())["failure_category"],
                             "generation_not_terminated_at_minimum_window")

    def test_empty_and_invalid_parsed_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ValueError):
                runtime.transcribe_waveform(self.waveform(1), output, Processor([]), Model([[99, 7, 2]]), TORCH)
            self.assertEqual(json.loads((output / "qwen_asr_failure.json").read_text())["failure_category"],
                             "empty_or_invalid_parsed_output")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chunks = runtime.transcribe_waveform(
                self.waveform(1), output,
                Processor([{"language": "English", "transcription": "  "}]),
                Model([[99, 7, 2]]), TORCH,
            )
            self.assertEqual("", chunks[0]["text"])
            self.assertEqual("no_transcription_at_minimum_window", chunks[0]["empty_resolution"])

    def test_dependency_report_success_and_missing_prerequisites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "revision"; model.mkdir()
            for name in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json"):
                (model / name).write_text(name)
            ffmpeg = root / "ffmpeg"; ffmpeg.write_text("#!/bin/sh\n"); ffmpeg.chmod(0o700)
            mps = types.SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
            torch = types.SimpleNamespace(__version__="test", backends=types.SimpleNamespace(mps=mps))
            transformers = types.SimpleNamespace(__version__="test")
            with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), patch.object(runtime, "FFMPEG", ffmpeg):
                self.assertEqual(runtime.dependency_report(model)["status"], "ready")
            mps.is_available = lambda: False
            with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), patch.object(runtime, "FFMPEG", root / "missing"):
                report = runtime.dependency_report(model)
            self.assertEqual(report["status"], "failed")
            self.assertEqual({x["prerequisite"] for x in report["failed_prerequisites"]}, {"ffmpeg", "mps"})


if __name__ == "__main__": unittest.main()
