import json
import unittest
from unittest.mock import patch

import diarized_layer2_shadow as shadow
import meetingintel_pipeline as pipeline

from meetingintel_model_policy import (
    GEMMA_CONTEXT_TOKENS,
    GEMMA_OPTIONS,
    GEMMA_OUTPUT_TOKENS,
    GEMMA_SAFE_INPUT_TOKENS,
    ollama_payload,
    require_complete_response,
)


class ModelPolicyTests(unittest.TestCase):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def test_payload_makes_sampling_thinking_context_and_output_cap_explicit(self):
        payload = ollama_payload("meetingintel", "fixture")
        self.assertIs(False, payload["think"])
        self.assertIs(False, payload["truncate"])
        self.assertIs(False, payload["shift"])
        self.assertEqual(GEMMA_OPTIONS, payload["options"])
        self.assertEqual(65_536, GEMMA_CONTEXT_TOKENS)
        self.assertEqual(8_192, GEMMA_OUTPUT_TOKENS)
        self.assertEqual(GEMMA_CONTEXT_TOKENS - GEMMA_OUTPUT_TOKENS, GEMMA_SAFE_INPUT_TOKENS)
        self.assertEqual(
            {"temperature", "top_p", "top_k", "repeat_penalty", "seed", "num_ctx", "num_predict"},
            set(payload["options"]),
        )

    def test_incomplete_or_length_limited_response_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            require_complete_response({"done": True, "done_reason": "length"})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            require_complete_response({"done_reason": "stop"})
        with self.assertRaisesRegex(RuntimeError, "prompt token"):
            require_complete_response({"done": True, "done_reason": "stop"})
        require_complete_response({
            "done": True, "done_reason": "stop", "prompt_eval_count": 10,
            "eval_count": 2,
        })

    def test_multilingual_prompt_and_output_budgets_are_enforced_from_server_counts(self):
        with self.assertRaisesRegex(RuntimeError, "reserved input budget"):
            require_complete_response({
                "done": True, "done_reason": "stop",
                "prompt_eval_count": GEMMA_SAFE_INPUT_TOKENS + 1, "eval_count": 1,
            })

    def test_flat_and_diarized_callers_send_no_truncation_controls_and_check_counts(self):
        response = self.Response({
            "response": "valid", "done": True, "done_reason": "stop",
            "prompt_eval_count": 12, "eval_count": 2,
        })
        with patch.object(pipeline.urllib.request, "urlopen", return_value=response) as flat:
            text, metrics = pipeline.summarize_with_ollama(
                "fixture", "http://127.0.0.1:11435/api/generate", pipeline.DEFAULT_MODEL
            )
        flat_payload = json.loads(flat.call_args.args[0].data)
        self.assertEqual("valid", text)
        self.assertEqual(12, metrics["prompt_eval_count"])
        self.assertIs(False, flat_payload["truncate"])
        self.assertIs(False, flat_payload["shift"])

        with patch.object(shadow.urllib.request, "urlopen", return_value=response) as diarized:
            text, raw = shadow.call_ollama("fixture")
        diarized_payload = json.loads(diarized.call_args.args[0].data)
        self.assertEqual("valid", text)
        self.assertEqual(12, raw["prompt_eval_count"])
        self.assertIs(False, diarized_payload["truncate"])
        self.assertIs(False, diarized_payload["shift"])

    def test_flat_caller_rejects_missing_input_accounting_before_returning_text(self):
        response = self.Response({
            "response": "must not persist", "done": True, "done_reason": "stop",
            "eval_count": 2,
        })
        with patch.object(pipeline.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "prompt token"):
                pipeline.summarize_with_ollama(
                    "fixture", "http://127.0.0.1:11435/api/generate", pipeline.DEFAULT_MODEL
                )
        with self.assertRaisesRegex(RuntimeError, "output exceeded"):
            require_complete_response({
                "done": True, "done_reason": "stop", "prompt_eval_count": 20,
                "eval_count": GEMMA_OUTPUT_TOKENS + 1,
            })





# Existing policy/routing tests mock model/subprocess work. Explicitly replace
# only admission here; test_meetingintel_gpu.py exercises the real integration.
class _SyntheticSession:
    def __init__(self, *_args, **_kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def run(self, command, **kwargs):
        import subprocess
        return subprocess.run(command, check=False, **kwargs)


def setUpModule():
    global _gpu_patch
    _gpu_patch = patch('meetingintel_gpu.Session', _SyntheticSession)
    _gpu_patch.start()


def tearDownModule():
    _gpu_patch.stop()

if __name__ == "__main__":
    unittest.main()
