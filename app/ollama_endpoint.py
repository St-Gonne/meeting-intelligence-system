"""Shared local Ollama endpoint resolution and model preflight."""

from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435/api/generate"
OLLAMA_MODEL = "meetingintel"
OLLAMA_APPROVED_DIGEST = "577c70d8e8ba905b49daffcf4a2c6c908d7be73415eaace75ba4004d3cb270e0"
OLLAMA_ENV_NAME = "MEETINGINTEL_OLLAMA_URL"
PRIVATE_ENV_FILE = Path(__file__).resolve().parent / ".secrets" / "meetingintel.env"


class OllamaEndpointError(RuntimeError):
    """Base class for safe endpoint/preflight failures."""


class InvalidOllamaEndpoint(OllamaEndpointError):
    """The configured endpoint is not an allowed local generate URL."""


class OllamaUnreachable(OllamaEndpointError):
    """The local Ollama tags endpoint could not be queried safely."""


class OllamaModelMissing(OllamaEndpointError):
    """The required fixed model alias was not advertised by Ollama."""


class OllamaModelDigestMismatch(OllamaEndpointError):
    """The fixed alias exists but is not the approved model instance."""


def _read_private_env_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_ollama_endpoint(raw_endpoint: str) -> str:
    if not isinstance(raw_endpoint, str) or not raw_endpoint.strip():
        raise InvalidOllamaEndpoint(
            "Ollama endpoint must be a local URL ending exactly in /api/generate"
        )
    endpoint = raw_endpoint.strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not _is_loopback_host(parsed.hostname):
        raise InvalidOllamaEndpoint(
            "Ollama endpoint must use http and a loopback host"
        )
    if parsed.username is not None or parsed.password is not None:
        raise InvalidOllamaEndpoint("Ollama endpoint must not contain credentials")
    if parsed.path != "/api/generate" or parsed.query or parsed.fragment:
        raise InvalidOllamaEndpoint(
            "Ollama endpoint path must be exactly /api/generate"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise InvalidOllamaEndpoint("Ollama endpoint port is invalid") from exc
    if port != 11435 or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise InvalidOllamaEndpoint("MeetingIntel requires its coordinated server on 127.0.0.1:11435")
    return endpoint


def resolve_ollama_endpoint(
    explicit: str | None = None,
    *,
    env_file: Path = PRIVATE_ENV_FILE,
) -> str:
    """Resolve explicit URL, private env URL, then the localhost default."""

    if explicit is not None:
        candidate = explicit
    else:
        candidate = _read_private_env_value(env_file, OLLAMA_ENV_NAME)
        if candidate is None:
            candidate = DEFAULT_OLLAMA_URL
    return validate_ollama_endpoint(candidate)


def _tags_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    tags = SplitResult(parsed.scheme, parsed.netloc, "/api/tags", "", "")
    return urlunsplit(tags)


def _safe_http_detail(error: urllib.error.HTTPError) -> str:
    reason = str(error.reason).strip()
    return f"HTTP {error.code}{f' {reason}' if reason else ''}"


def preflight_ollama(
    endpoint: str,
    *,
    opener: Any = urllib.request.urlopen,
    timeout: float = 5.0,
    required_model: str = OLLAMA_MODEL,
    required_digest: str = OLLAMA_APPROVED_DIGEST,
) -> dict[str, Any]:
    """Require /api/tags and the fixed model before any model call."""

    validated = validate_ollama_endpoint(endpoint)
    request = urllib.request.Request(_tags_endpoint(validated), method="GET")
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise OllamaUnreachable(
            f"Ollama /api/tags {_safe_http_detail(exc)}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason).strip()
        raise OllamaUnreachable(
            f"Ollama /api/tags unreachable{f': {reason}' if reason else ''}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaUnreachable("Ollama /api/tags request timed out") from exc
    except OSError as exc:
        raise OllamaUnreachable(
            f"Ollama /api/tags unreachable: {type(exc).__name__}"
        ) from exc

    if status is not None and not 200 <= status < 300:
        raise OllamaUnreachable(f"Ollama /api/tags HTTP {status}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        detail = f"HTTP {status}" if status is not None else "HTTP response"
        raise OllamaUnreachable(
            f"Ollama /api/tags returned invalid JSON ({detail})"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise OllamaUnreachable("Ollama /api/tags returned an invalid model list")

    models = [item for item in payload["models"] if isinstance(item, dict)]
    names = {
        item.get("name")
        for item in models
        if isinstance(item.get("name"), str)
    }
    accepted_aliases = {required_model, f"{required_model}:latest"}
    if not names.intersection(accepted_aliases):
        raise OllamaModelMissing(
            f"Ollama model alias missing: expected {required_model} or {required_model}:latest"
        )
    matching_models = [item for item in models if item.get("name") in accepted_aliases]
    matching_digests = {
        item.get("digest")
        for item in matching_models
        if isinstance(item.get("digest"), str)
    }
    if required_digest not in matching_digests:
        observed = ", ".join(sorted(digest for digest in matching_digests if digest)) or "missing"
        raise OllamaModelDigestMismatch(
            f"Ollama model alias digest mismatch: expected approved digest; observed {observed}"
        )
    return {
        "endpoint": validated,
        "tags_endpoint": _tags_endpoint(validated),
        "required_model": required_model,
        "model_available": True,
        "approved_digest": required_digest,
        "model_digest_matches": True,
    }
