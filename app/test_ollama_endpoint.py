from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import diarized_layer2_shadow as shadow
import meetingintel_pipeline as pipeline
from ollama_endpoint import (
    DEFAULT_OLLAMA_URL,
    OLLAMA_APPROVED_DIGEST,
    OLLAMA_MODEL,
    InvalidOllamaEndpoint,
    OllamaModelMissing,
    OllamaModelDigestMismatch,
    OllamaUnreachable,
    preflight_ollama,
    resolve_ollama_endpoint,
    validate_ollama_endpoint,
)


class FakeResponse:
    def __init__(self, payload: object, status: int = 200):
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class OllamaEndpointTests(unittest.TestCase):
    def test_resolution_priority_is_explicit_then_private_file_then_default(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "meetingintel.env"
            env_file.write_text(
                "export MEETINGINTEL_OLLAMA_URL='http://127.0.0.1:11435/api/generate'\n",
                encoding="utf-8",
            )
            self.assertEqual(
                "http://localhost:11435/api/generate",
                resolve_ollama_endpoint(
                    "http://localhost:11435/api/generate", env_file=env_file
                ),
            )
            self.assertEqual(
                "http://127.0.0.1:11435/api/generate",
                resolve_ollama_endpoint(env_file=env_file),
            )
            self.assertEqual(
                DEFAULT_OLLAMA_URL,
                resolve_ollama_endpoint(env_file=Path(directory) / "missing.env"),
            )

    def test_validator_accepts_only_loopback_exact_generate_endpoint(self):
        accepted = (
            "http://localhost:11435/api/generate",
            "http://127.0.0.1:11435/api/generate",
        )
        for endpoint in accepted:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(endpoint, validate_ollama_endpoint(endpoint))

        rejected = (
            "http://localhost:11434/api/generate",
            "http://[::1]:11436/api/generate",
            "https://localhost:11434/api/generate",
            "http://192.168.1.10:11434/api/generate",
            "http://127.0.0.1:11435/api/tags",
            "http://localhost:11434/api/generate/",
            "http://localhost:11434/api/generate?x=1",
            "http://user:pass@localhost:11434/api/generate",
        )
        for endpoint in rejected:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(InvalidOllamaEndpoint):
                    validate_ollama_endpoint(endpoint)

    def test_preflight_accepts_required_aliases_and_only_gets_tags(self):
        for name in (OLLAMA_MODEL, f"{OLLAMA_MODEL}:latest"):
            opener = Mock(return_value=FakeResponse({"models": [{"name": name, "digest": OLLAMA_APPROVED_DIGEST}]}))
            result = preflight_ollama(DEFAULT_OLLAMA_URL, opener=opener)
            self.assertTrue(result["model_available"])
            request = opener.call_args.args[0]
            self.assertEqual("GET", request.method)
            self.assertEqual(
                "http://127.0.0.1:11435/api/tags",
                request.full_url,
            )

    def test_preflight_rejects_alias_with_wrong_digest(self):
        with self.assertRaisesRegex(OllamaModelDigestMismatch, "digest mismatch"):
            preflight_ollama(
                DEFAULT_OLLAMA_URL,
                opener=Mock(return_value=FakeResponse({"models": [{"name": OLLAMA_MODEL, "digest": "wrong"}]})),
            )

    def test_preflight_distinguishes_missing_alias_and_safe_http_failure(self):
        with self.assertRaises(OllamaModelMissing):
            preflight_ollama(
                DEFAULT_OLLAMA_URL,
                opener=Mock(return_value=FakeResponse({"models": [{"name": "other"}]})),
            )

        error = urllib.error.HTTPError(
            "http://127.0.0.1:11435/api/tags", 503, "Service Unavailable", {}, None
        )
        with self.assertRaisesRegex(OllamaUnreachable, r"503 Service Unavailable"):
            preflight_ollama(DEFAULT_OLLAMA_URL, opener=Mock(side_effect=error))

        with self.assertRaisesRegex(OllamaUnreachable, r"HTTP 502"):
            preflight_ollama(
                DEFAULT_OLLAMA_URL,
                opener=Mock(return_value=FakeResponse({}, status=502)),
            )

        with self.assertRaisesRegex(OllamaUnreachable, "connection refused"):
            preflight_ollama(
                DEFAULT_OLLAMA_URL,
                opener=Mock(side_effect=urllib.error.URLError("connection refused")),
            )

    def test_pipeline_and_shadow_skip_network_only_in_safe_modes(self):
        pipeline_args = Namespace(
            ollama_url=DEFAULT_OLLAMA_URL,
            model="not-used",
            dry_run=True,
        )
        with patch.object(pipeline, "preflight_ollama") as pipeline_preflight:
            pipeline.prepare_ollama(pipeline_args)
        pipeline_preflight.assert_not_called()
        self.assertEqual(OLLAMA_MODEL, pipeline_args.model)

        for mode in ("dry_run", "eligibility_only"):
            args = Namespace(
                ollama_url=DEFAULT_OLLAMA_URL,
                dry_run=mode == "dry_run",
                eligibility_only=mode == "eligibility_only",
            )
            with patch.object(shadow, "preflight_ollama") as shadow_preflight:
                shadow.prepare_ollama(args)
            shadow_preflight.assert_not_called()

    def test_pipeline_preflight_runs_for_real_mode(self):
        args = Namespace(
            ollama_url="http://127.0.0.1:11435/api/generate",
            model="not-used",
            dry_run=False,
        )
        with patch.object(pipeline, "preflight_ollama") as preflight:
            self.assertEqual(args.ollama_url, pipeline.prepare_ollama(args))
        preflight.assert_called_once_with(args.ollama_url)
