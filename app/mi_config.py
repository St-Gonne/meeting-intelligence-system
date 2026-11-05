"""User-local setup settings; no shell evaluation, downloads or file creation."""
import json
import os
from pathlib import Path

KEYS = {'MI_DATA_ROOT', 'MI_PHONE_SOURCE', 'MI_PHONE_REMOTE', 'MI_MEETILY_ROOT',
        'MI_QWEN_PYTHON', 'MI_QWEN_MODEL_DIR', 'MI_WHISPER_PYTHON', 'MI_WHISPERMLX_BIN',
        'MI_OBS_APP', 'MI_VOICE_ROOT', 'MI_GPU_MODE', 'MI_TIMEZONE', 'MI_TRANSCRIPTION_BACKEND'}


def load_settings(environ=None):
    env = os.environ if environ is None else environ
    default = Path(__file__).resolve().parent / '.local' / 'settings.json'
    path = Path(env.get('MI_CONFIG_FILE', str(default))).expanduser()
    if path.is_symlink():
        raise ValueError('settings file must not be a symlink')
    if not path.exists():
        return
    if not path.is_file() or path.stat().st_size > 16384:
        raise ValueError('settings file is missing or too large')
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict) or set(settings) - KEYS:
        raise ValueError('settings contains unsupported keys; see docs/SETUP.md')
    if any(not isinstance(v, str) or not v or '\x00' in v for v in settings.values()):
        raise ValueError('settings values must be nonempty strings')
    for key, value in settings.items():
        env.setdefault(key, value)
