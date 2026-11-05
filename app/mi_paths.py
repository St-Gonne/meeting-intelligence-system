"""Portable paths; importing this module creates no files."""
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('MI_DATA_ROOT', str(PROJECT / '.local'))).expanduser().absolute()

def configured_timezone():
    return ZoneInfo(os.environ.get('MI_TIMEZONE', 'Asia/Kolkata'))


def transcription_backend() -> str:
    value = os.environ.get('MI_TRANSCRIPTION_BACKEND', 'qwen')
    if value not in {'qwen', 'whisper', 'legacy'}:
        raise ValueError('MI_TRANSCRIPTION_BACKEND must be qwen or whisper')
    return 'legacy' if value in {'whisper', 'legacy'} else 'qwen'


def public_path(value: str) -> Path:
    root, _, tail = value.partition('/')
    if root == 'project':
        if tail.split('/')[0] in {'state', 'output', 'ingest', 'staging', 'artifacts', 'checkpoints', '.secrets'}:
            return DATA / tail
        return PROJECT / tail
    if root == 'phone-source':
        return Path(os.environ.get('MI_PHONE_SOURCE', str(DATA / 'phone-source'))).expanduser()
    if root == 'work':
        if tail == 'transcription-runtime/qwen-env/bin/python':
            return Path(os.environ.get('MI_QWEN_PYTHON', str(DATA / tail)))
        if tail.startswith('transcription-runtime/cache/'):
            return Path(os.environ.get('MI_QWEN_MODEL_DIR', str(DATA / tail)))
        return DATA / 'work' / tail
    if root == 'home':
        if tail.startswith('Library/Application Support/MeetingIntel/voiceprint-'):
            private = Path(os.environ.get('MI_VOICE_ROOT', str(Path.home() / 'Library/Application Support/MeetingIntel-public')))
            return private / tail.removeprefix('Library/Application Support/MeetingIntel/')
        if tail == 'homebrew' or tail.startswith('homebrew/'):
            prefix = Path(os.environ.get('HOMEBREW_PREFIX', '/opt/homebrew'))
            suffix = tail.removeprefix('homebrew').lstrip('/')
            if suffix and suffix.rsplit('/', 1)[-1] not in {'bin', 'rclone'}:
                binary = suffix.rsplit('/', 1)[-1]
                found = shutil.which(binary)
                if found: return Path(found)
            return prefix / suffix
        if tail.startswith('whispermlx-env/'):
            key = 'MI_WHISPER_PYTHON' if tail.endswith('/python') else 'MI_WHISPERMLX_BIN'
            return Path(os.environ.get(key, str(DATA / tail)))
        if tail == 'Applications/OBS.app':
            return Path(os.environ.get('MI_OBS_APP', '/Applications/OBS.app'))
        if tail.startswith('Movies/meetily-recordings'):
            return Path(os.environ.get('MI_MEETILY_ROOT', str(DATA / 'meetily-recordings')))
        return DATA / 'home' / tail
    raise ValueError('Unknown public path namespace')
