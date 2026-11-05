#!/usr/bin/env python3
"""Read-only setup inventory. No installs, model loads, recording or authorization."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
from mi_config import load_settings


def check(*, mode='demo', phone=False, capture=False):
    load_settings()
    from mi_paths import public_path, transcription_backend, configured_timezone
    import gpu_lock
    checks = []
    def add(name, ok, next_step, required=True):
        checks.append(dict(name=name, ok=bool(ok), required=required, next_step='' if ok else next_step))
    add('Python 3.11+', sys.version_info >= (3, 11), 'Install Python 3.11; see docs/SETUP.md.')
    try:
        configured_timezone(); timezone_ok=True
    except (ValueError, KeyError): timezone_ok=False
    add('Timezone', timezone_ok, 'Set MI_TIMEZONE to your IANA timezone in app/.local/settings.json.')
    backend = transcription_backend()
    if mode == 'runtime':
        add('Apple Silicon Mac', platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Full processing targets Apple Silicon macOS; use demos on other systems.')
        for command in ['ffmpeg', 'ffprobe']:
            add(command, shutil.which(command), 'Install FFmpeg using Homebrew, then check your PATH.')
        def environment(name, binary, modules):
            available=Path(binary).is_file() and os.access(binary, os.X_OK)
            if available:
                code='import importlib.util,sys;sys.exit(not all(importlib.util.find_spec(m) is not None for m in '+repr(modules)+'))'
                try:
                    result=subprocess.run([str(binary), '-B', '-c', code],capture_output=True,timeout=15)
                    available=result.returncode == 0
                except (OSError, subprocess.TimeoutExpired):available=False
            add(name, available, 'Build the separate environment and set its interpreter path; docs/SETUP.md.')
        environment('Whisper/diarization environment', public_path('home/whispermlx-env/bin/python'), ['torch','whispermlx','pyannote'])
        if backend == 'qwen':
            environment('Qwen environment',public_path('work/transcription-runtime/qwen-env/bin/python'),['torch','transformers','librosa'])
            model=public_path('work/transcription-runtime/cache/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3')
            add('Qwen local assets', (model/'config.json').is_file() and any(model.glob('*.safetensors')), 'Download the pinned Qwen model and set MI_QWEN_MODEL_DIR; docs/SETUP.md. Run the documented model preflight afterwards.')
        secret=public_path('project/.secrets/meetingintel.env')
        has_token=bool(os.environ.get('HF_TOKEN','').strip())
        if not has_token and secret.is_file() and not secret.is_symlink():
            has_token=any(line.partition('=')[0].strip() == 'HF_TOKEN' and bool(line.partition('=')[2].strip()) for line in secret.read_text().splitlines())
        add('Local model authorization',has_token,'Create your own private .secrets/meetingintel.env with HF_TOKEN after accepting model terms; never paste the token into logs or chat.')
        add('Ollama CLI',shutil.which('ollama'),'Install Ollama from its official source.')
        try:
            from ollama_endpoint import preflight_ollama, DEFAULT_OLLAMA_URL
            preflight_ollama(DEFAULT_OLLAMA_URL); alias=True
        except Exception:alias=False
        add('Pinned analysis model at 127.0.0.1:11435',alias,'Set up the managed Ollama service and create the model from the included recipe; mi ollama status.')
        limits=gpu_lock.server_limits(gpu_lock.runtime_endpoints())
        add('Ollama service limits ('+os.environ.get('MI_GPU_MODE','standalone')+')', bool(limits) and all(r['max_loaded_models_one'] for r in limits.values()), 'Use the managed service with OLLAMA_MAX_LOADED_MODELS=1. Standalone needs 11435 only; shared needs both services.')
        if phone:
            add('rclone',shutil.which('rclone'),'Install rclone and complete your own read-only Drive authorization; docs/PHONE_SETUP.md.')
            add('Phone folder selected',bool(os.environ.get('MI_PHONE_REMOTE')),'Set MI_PHONE_REMOTE to your recorder-only remote folder. No Drive access is performed by this checker.')
        if capture:
            add('Node 22+',False,'Run node --version and install Node 22+ if needed.',False)
            node=shutil.which('node')
            if node:
                try:
                    result=subprocess.run([node,'-p','Number(process.versions.node.split(".")[0]) >= 22 && typeof WebSocket !== "undefined"'],capture_output=True,text=True,timeout=5)
                    checks[-1]['ok']=result.returncode==0 and result.stdout.strip()=='true'
                    if checks[-1]['ok']:checks[-1]['next_step']=''
                except (OSError,subprocess.TimeoutExpired):pass
            checks[-1]['required']=True
            add('Signed OBS app',public_path('home/Applications/OBS.app').is_dir(),'Install OBS; grant microphone and screen/system-audio permissions yourself. Run a deliberate short capture to verify audio.')
    return {'mode':mode,'backend':backend,'checks':checks,
            'ready':all(r['ok'] for r in checks if r['required']),
            'limits':'Dependency inventory only. Model cache integrity, Metal readiness, permissions, audio continuity and end-to-end output require the documented checks and a short consented trial.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['demo','runtime'],default='demo')
    parser.add_argument('--phone',action='store_true')
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    try:result=check(mode=args.mode,phone=args.phone,capture=args.capture)
    except (ValueError,OSError,KeyError) as error:
        print('Setup settings invalid: '+str(error),file=sys.stderr);return 2
    if args.json:print(json.dumps(result,indent=2))
    else:
        for row in result['checks']:
            print(('OK   ' if row['ok'] else 'NEED ')+row['name']+((': '+row['next_step']) if row['next_step'] else ''))
        print(result['limits'])
    return 0 if result['ready'] else 1


if __name__=='__main__':raise SystemExit(main())
