from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import voiceprint_provider_pyannote as provider


class PyannoteProviderTests(unittest.TestCase):
    def test_decoder_requires_valid_float_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "sample.wav"
            audio.write_bytes(b"fixture")
            good = subprocess.CompletedProcess([], 0, b"\x00\x00\x00\x00" * 8000, b"")
            values = provider.decode_mono_16k(audio, runner=lambda *args, **kwargs: good)
            self.assertEqual(8000, len(values))

            bad = subprocess.CompletedProcess([], 1, b"", b"private path")
            with self.assertRaisesRegex(provider.ProviderError, "decode failed"):
                provider.decode_mono_16k(audio, runner=lambda *args, **kwargs: bad)

    def test_ordinary_files_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(provider.ProviderError, "unsafe"):
                provider._ordinary_file(link, "fixture")

    def test_cli_outputs_only_embedding_json(self):
        output = io.StringIO()
        with patch.object(provider, "embed", return_value=(0.1, 0.2)), redirect_stdout(output):
            self.assertEqual(0, provider.main(["--model-asset", "/tmp/model", "--audio", "/tmp/audio"]))
        self.assertEqual({"embedding": [0.1, 0.2]}, json.loads(output.getvalue()))

    def test_cli_failure_is_sanitized(self):
        output, error = io.StringIO(), io.StringIO()
        with (
            patch.object(provider, "embed", side_effect=provider.ProviderError("/private/secret")),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            self.assertEqual(2, provider.main(["--model-asset", "/tmp/model", "--audio", "/tmp/audio"]))
        self.assertEqual("", output.getvalue())
        self.assertNotIn("secret", error.getvalue())


if __name__ == "__main__":
    unittest.main()
