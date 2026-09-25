"""Explicit local audio-to-candidate bridge for the private Gate 5 inbox.

Exactly one caller-supplied private clip and one caller-supplied pinned local
model asset are accepted.  Audio and model hashes are recorded as provenance;
audio bytes, source paths, and vectors are never emitted to logs or reports.
The provider runs with offline model-cache controls and no production imports.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from typing import Callable

from voiceprint_gate5_inbox import Gate5Error, Gate5Inbox, InboxCandidate, _private_file
from voiceprint_provider_pyannote import ProviderError, embed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _offline_environment():
    keys = {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _admit(inbox: Gate5Inbox, model_asset: Path, audio: Path) -> tuple[Path, Path, str, str]:
    try:
        model_path = _private_file(model_asset, "model asset")
        audio_path = _private_file(audio, "audio")
    except Gate5Error as exc:
        raise ProviderError(str(exc)) from exc
    model_hash, audio_hash = _sha256(model_path), _sha256(audio_path)
    expected = inbox.provenance.model_asset_sha256
    if expected is not None and expected != model_hash:
        raise Gate5Error("model asset does not match the pinned Gate 4 model")
    # Pin the first explicitly admitted model and make subsequent calls match
    # it. This is an in-memory configuration update; the model path is never
    # persisted.
    inbox.pin_model_digest(model_hash)
    return model_path, audio_path, model_hash, audio_hash


def enroll_audio(
    inbox: Gate5Inbox,
    identity_id: str,
    *,
    model_asset: Path,
    audio: Path,
    consent: str | None = None,
    embedder: Callable[[Path, Path], tuple[float, ...]] = embed,
):
    """Create a pending enrollment from one explicitly consented clip."""
    if consent != "yes":
        raise Gate5Error("explicit audio evaluation consent is required")
    model_path, audio_path, _model_hash, audio_hash = _admit(inbox, model_asset, audio)
    try:
        with _offline_environment():
            vector = embedder(model_path, audio_path)
    except ProviderError:
        raise
    return inbox.enroll_pending(identity_id, vector, audio_sha256=audio_hash)


def review_audio(
    inbox: Gate5Inbox,
    observed_speaker: str,
    *,
    meeting_id: str,
    model_asset: Path,
    audio: Path,
    consent: str | None = None,
    embedder: Callable[[Path, Path], tuple[float, ...]] = embed,
) -> InboxCandidate:
    """Score one explicitly consented clip in one explicit meeting scope."""
    if consent != "yes":
        raise Gate5Error("explicit audio evaluation consent is required")
    model_path, audio_path, _model_hash, _audio_hash = _admit(inbox, model_asset, audio)
    with _offline_environment():
        vector = embedder(model_path, audio_path)
    return inbox.review({observed_speaker: vector}, meeting_id=meeting_id, audio_sha256=_audio_hash)[0]


__all__ = ["enroll_audio", "review_audio", "ProviderError"]


def _cli() -> int:
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Explicit local-audio Gate 5 candidate bridge")
    parser.add_argument("command", choices=("enroll", "review"))
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--identity-id", required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--meeting-id", required=False)
    parser.add_argument("--speaker", required=False)
    parser.add_argument("--consent", choices=("yes", "no"), required=True)
    parser.add_argument("--gate4-report", type=Path)
    args = parser.parse_args()
    try:
        inbox = Gate5Inbox(args.state_root, clock=lambda: datetime.now(timezone.utc), gate4_report=args.gate4_report, model_asset=args.model_asset)
        if args.command == "enroll":
            record = enroll_audio(inbox, args.identity_id, model_asset=args.model_asset, audio=args.audio, consent=args.consent)
            print(json.dumps({"state": record.state, "identity": record.identity_id}, sort_keys=True))
        else:
            if not args.meeting_id or not args.speaker:
                raise Gate5Error("review requires --meeting-id and --speaker")
            candidate = review_audio(inbox, args.speaker, meeting_id=args.meeting_id, model_asset=args.model_asset, audio=args.audio, consent=args.consent)
            print(json.dumps({"candidate_id": candidate.candidate_id, "meeting_id": candidate.meeting_id, "speaker": candidate.observed_speaker, "identity": candidate.identity_id, "score": candidate.score, "status": candidate.status}, sort_keys=True))
        return 0
    except (Gate5Error, ProviderError, OSError, ValueError) as exc:
        print(f"GATE5_REJECTED={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
