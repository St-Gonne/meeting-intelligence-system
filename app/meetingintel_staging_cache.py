"""Private reuse of a fully completed, validated transcription stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from datetime import datetime
from pathlib import Path

from transcription_quality import assess_segments

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CACHE = _ROOT / "state" / "staging-cache"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_SPEAKER = re.compile(r"^SPEAKER_\d+$")
_CONTRACT_KEYS = {
    "source_identity", "source_audio_sha256", "source_folder", "source_audio",
    "backend", "model_identifier", "revision", "config_sha256",
}


def _canonical_contract(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict) or set(value) != _CONTRACT_KEYS:
        return None
    result = dict(value)
    for key in ("source_identity", "source_audio_sha256", "config_sha256"):
        if not isinstance(result[key], str) or not _HEX.fullmatch(result[key]):
            return None
    for key in ("source_folder", "source_audio"):
        raw = result[key]
        if not isinstance(raw, str) or not Path(raw).is_absolute() or ".." in Path(raw).parts:
            return None
    if result["backend"] not in {"legacy", "qwen"}:
        return None
    if not isinstance(result["model_identifier"], str) or not result["model_identifier"]:
        return None
    if result["revision"] is not None and not isinstance(result["revision"], str):
        return None
    return result


def _no_symlink_path(path: Path) -> bool:
    if not path.is_absolute() or ".." in path.parts:
        return False
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return False
        except FileNotFoundError:
            continue
    return True


def _owned(path: Path, kind: str) -> bool:
    if not _no_symlink_path(path):
        return False
    try:
        info = path.lstat()
        return info.st_uid == os.getuid() and (
            stat.S_ISDIR(info.st_mode) if kind == "dir" else stat.S_ISREG(info.st_mode)
        )
    except OSError:
        return False


def _read(path: Path, limit: int) -> tuple[bytes, str] | None:
    if not _owned(path, "file"):
        return None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                return None
            data = stream.read(limit + 1)
        if len(data) > limit:
            return None
        return data, hashlib.sha256(data).hexdigest()
    except OSError:
        return None


def _digest(path: Path) -> str | None:
    if not _owned(path, "file"):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
                return None
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _json_file(path: Path, limit: int = 16 * 1024 * 1024) -> tuple[object, str] | None:
    read = _read(path, limit)
    if read is None or not read[0]:
        return None
    try:
        return json.loads(read[0]), read[1]
    except (ValueError, UnicodeDecodeError):
        return None


def _source_ok(contract: dict[str, object]) -> bool:
    folder, audio = Path(str(contract["source_folder"])), Path(str(contract["source_audio"]))
    if not _owned(folder, "dir") or not _owned(audio, "file"):
        return False
    return _digest(audio) == contract["source_audio_sha256"]


def _validate_stage(stage_value: object, contract: dict[str, object]) -> tuple[Path, dict[str, str]] | None:
    try:
        stage = Path(stage_value)
        if not stage.is_absolute() or stage != stage.resolve(strict=True):
            return None
    except (OSError, TypeError, ValueError):
        return None
    if (not _owned(stage, "dir") or stage.is_relative_to(_ROOT)
            or stage.is_relative_to(Path(str(contract["source_folder"])) )):
        return None
    if not _source_ok(contract):
        return None

    manifest_pair = _json_file(stage / "run_manifest.json")
    if manifest_pair is None or not isinstance(manifest_pair[0], dict):
        return None
    manifest = manifest_pair[0]
    expected_mode = "legacy" if contract["backend"] == "legacy" else "qwen3-asr-1.7b-contiguous"
    completed = manifest.get("completed_at")
    try:
        timestamp = datetime.fromisoformat(completed) if isinstance(completed, str) else None
        aware = timestamp is not None and timestamp.utcoffset() is not None
    except ValueError:
        aware = False
    if (manifest.get("status") != "success" or type(manifest.get("exit_code")) is not int
            or manifest.get("exit_code") != 0 or not aware
            or manifest.get("source_mode") != "audio_first"
            or manifest.get("source_identity") != contract["source_identity"]
            or manifest.get("source_folder") != contract["source_folder"]
            or manifest.get("source_audio") != contract["source_audio"]
            or manifest.get("output_folder") != str(stage)
            or manifest.get("transcription_mode") != expected_mode
            or manifest.get("model") != contract["model_identifier"]):
        return None

    audio_pair = _json_file(stage / "audio.json")
    if audio_pair is None or not isinstance(audio_pair[0], dict):
        return None
    segments = audio_pair[0].get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    for segment in segments:
        if (not isinstance(segment, dict) or
                not (segment.get("speaker") is None or segment.get("speaker") == "UNKNOWN" or
                     _SPEAKER.fullmatch(str(segment.get("speaker"))) )):
            return None
        start, end, text = segment.get("start"), segment.get("end"), segment.get("text")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float))
                or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or not isinstance(text, str)):
            return None
    if assess_segments(segments).get("accepted") is not True:
        return None

    hashes = {"audio.json": audio_pair[1], "run_manifest.json": manifest_pair[1]}
    for name in ("audio.srt", "audio.txt"):
        item = _read(stage / name, 64 * 1024 * 1024)
        if item is None or not item[0].strip():
            return None
        hashes[name] = item[1]

    if contract["backend"] == "legacy":
        runtime_pair = _json_file(stage / "transcription_runtime.json")
        if runtime_pair is None or not isinstance(runtime_pair[0], dict):
            return None
        runtime = runtime_pair[0]
        chunks = runtime.get("chunks")
        if runtime.get("completed") is not True or not isinstance(chunks, list) or not chunks:
            return None
        if any(not isinstance(chunk, dict) or not isinstance(chunk.get("quality"), dict)
               or chunk["quality"].get("accepted") is not True for chunk in chunks):
            return None
        hashes["transcription_runtime.json"] = runtime_pair[1]
    else:
        provenance = manifest.get("candidate_stage_provenance")
        if not isinstance(provenance, dict):
            return None
        qwen = provenance.get("qwen_asr")
        if not isinstance(qwen, dict):
            return None
        model = qwen.get("model")
        count = qwen.get("chunk_count")
        if (qwen.get("status") != "success" or qwen.get("all_generations_ended_with_eos") is not True
                or isinstance(count, bool) or not isinstance(count, int) or count <= 0
                or qwen.get("source_sha256") != contract["source_audio_sha256"]
                or not isinstance(model, dict) or model.get("identifier") != contract["model_identifier"]
                or model.get("revision") != contract["revision"]):
            return None
        qwen_path = stage / "qwen_asr_manifest.json"
        try:
            qwen_path.lstat()
        except FileNotFoundError:
            pass
        else:
            qwen_pair = _json_file(qwen_path)
            if qwen_pair is None or qwen_pair[0] != qwen:
                return None
            hashes["qwen_asr_manifest.json"] = qwen_pair[1]
    return stage, hashes


def _cache_root(root: str | os.PathLike[str] | None, create: bool) -> Path | None:
    try:
        path = Path(root) if root is not None else _DEFAULT_CACHE
        if not path.is_absolute() or ".." in path.parts or not _no_symlink_path(path):
            return None
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (not _owned(path, "dir") or path.stat().st_mode & 0o077):
            return None
        return path
    except (OSError, TypeError, ValueError):
        return None


def _fingerprint(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def save(fingerprint: str, contract: dict, stage, *, root=None) -> bool:
    """Cache a stage only after every provenance, content and quality check passes."""
    try:
        if not _fingerprint(fingerprint):
            return False
        clean = _canonical_contract(contract)
        cache = _cache_root(root, True)
        checked = _validate_stage(stage, clean) if clean and cache else None
        if checked is None:
            return False
        payload = json.dumps({"schema": 1, "fingerprint": fingerprint, "contract": clean,
                              "stage": str(checked[0]), "hashes": checked[1]},
                             sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > 16 * 1024:
            return False
        target = cache / f"{fingerprint}.json"
        try:
            target_info = target.lstat()
            if not stat.S_ISREG(target_info.st_mode) or target_info.st_uid != os.getuid():
                return False
        except FileNotFoundError:
            pass
        temporary = cache / f".{fingerprint}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        return True
    except Exception:
        return False


def load(fingerprint: str, contract: dict, *, root=None) -> Path | None:
    """Return a completed stage only when its exact contract and all hashes still match."""
    try:
        if not _fingerprint(fingerprint):
            return None
        clean = _canonical_contract(contract)
        cache = _cache_root(root, False)
        if clean is None or cache is None:
            return None
        entry = _read(cache / f"{fingerprint}.json", 16 * 1024)
        if entry is None:
            return None
        if ((cache / f"{fingerprint}.json").stat().st_mode & 0o777) != 0o600:
            return None
        payload = json.loads(entry[0])
        if (not isinstance(payload, dict) or payload.get("schema") != 1
                or payload.get("fingerprint") != fingerprint or payload.get("contract") != clean
                or not isinstance(payload.get("hashes"), dict)):
            return None
        checked = _validate_stage(payload.get("stage"), clean)
        if checked is None or checked[1] != payload["hashes"]:
            return None
        return checked[0]
    except Exception:
        return None
