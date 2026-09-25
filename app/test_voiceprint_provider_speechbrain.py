from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import voiceprint_provider_speechbrain as provider


class SpeechBrainProviderTests(unittest.TestCase):
    def test_model_asset_must_be_an_ordinary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            link = root / "link"
            link.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(provider.ProviderError, "unsafe"):
                provider._ordinary_directory(link, "model")

            (model / "nested-link").symlink_to(root / "outside")
            with self.assertRaisesRegex(provider.ProviderError, "contains a symlink"):
                provider._ordinary_directory(model, "model")

    def test_cli_outputs_only_embedding_json(self):
        output = io.StringIO()
        with patch.object(provider, "embed", return_value=(0.3, 0.4)), redirect_stdout(output):
            self.assertEqual(0, provider.main(["--model-asset", "/tmp/model", "--audio", "/tmp/audio"]))
        self.assertEqual({"embedding": [0.3, 0.4]}, json.loads(output.getvalue()))

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
