"""Shared, explicit Ollama generation policy for MeetingIntel analysis."""

from __future__ import annotations

from typing import Any

GEMMA_CONTEXT_TOKENS = 65_536
GEMMA_OUTPUT_TOKENS = 8_192
GEMMA_SAFE_INPUT_TOKENS = GEMMA_CONTEXT_TOKENS - GEMMA_OUTPUT_TOKENS

# The bounded synthetic comparison did not establish that the upstream sampling
# recommendation improves grounded meeting extraction, so retain the established
# low-temperature profile while making every inherited value explicit.
GEMMA_SAMPLING_PROFILE = "meetingintel-grounded-v1"
GEMMA_OPTIONS: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "seed": -1,
    "num_ctx": GEMMA_CONTEXT_TOKENS,
    "num_predict": GEMMA_OUTPUT_TOKENS,
}
GEMMA_THINK = False


def ollama_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "keep_alive": "2m",
        "think": GEMMA_THINK,
        # Ollama otherwise defaults both controls on. The server must reject an
        # oversized prompt instead of dropping or shifting supplied meeting text.
        "truncate": False,
        "shift": False,
        "prompt": prompt,
        "options": dict(GEMMA_OPTIONS),
    }


def require_complete_response(payload: dict[str, Any]) -> None:
    if payload.get("done") is not True or payload.get("done_reason") != "stop":
        reason = payload.get("done_reason", "unknown")
        raise RuntimeError(f"Ollama response was incomplete (done_reason: {reason})")
    prompt_tokens = payload.get("prompt_eval_count")
    output_tokens = payload.get("eval_count")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise RuntimeError("Ollama response omitted valid prompt token accounting")
    if prompt_tokens > GEMMA_SAFE_INPUT_TOKENS:
        raise RuntimeError(
            f"Ollama prompt used {prompt_tokens} tokens, above the reserved input budget"
        )
    if type(output_tokens) is not int or output_tokens < 0:
        raise RuntimeError("Ollama response omitted valid output token accounting")
    if output_tokens > GEMMA_OUTPUT_TOKENS:
        raise RuntimeError("Ollama output exceeded the configured token budget")
