"""The two owner-selected references, weekly at night, via the existing agent.

No discovery, retries, production output, enrollment copies, service recovery,
or policy changes. One zero-wait GPU admission supervises one bounded worker.
"""
from mi_paths import configured_timezone
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import gpu_lock
import meetingintel_learning as learning
import meetingintel_run_health as health
import meetingintel_runtime as runtime
from meetingintel_audio_tools import audio_tool_environment

IST = configured_timezone()


def _root(create=False):
    path = learning._root() / 'canary'
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe canary root')
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and (path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077):
        raise ValueError('unsafe canary owner')
    return path


def configure(references, enabled=True):
    config = {'schema_version': 1, 'enabled': enabled, 'references': references}
    _validate(config)
    runtime._write_json(_root(True), 'references.json', config)


def _validate(config):
    if not isinstance(config, dict) or set(config) != {'schema_version', 'enabled', 'references'} or config['schema_version'] != 1 or type(config['enabled']) is not bool:
        raise ValueError('invalid canary config')
    refs = config['references']
    if not isinstance(refs, list) or len(refs) != 2 or {r['lane'] for r in refs} != {'phone', 'laptop'}:
        raise ValueError('exactly two reference lanes required')
    for ref in refs:
        if set(ref) != {'lane', 'source_folder', 'stage', 'source_audio_sha256', 'stage_manifest_sha256', 'stage_artifact_sha256', 'expected_label'}:
            raise ValueError('invalid reference')
        import re
        for key in ('source_audio_sha256', 'stage_manifest_sha256'):
            if re.fullmatch('[a-f0-9]{64}', ref[key]) is None:
                raise ValueError('invalid reference hash')
        if set(ref['stage_artifact_sha256']) != {'audio.json', 'audio.srt', 'audio.txt'} or any(re.fullmatch('[a-f0-9]{64}', digest) is None for digest in ref['stage_artifact_sha256'].values()):
            raise ValueError('invalid reference artifacts')
        if re.fullmatch('SPEAKER_[0-9]+', ref['expected_label']) is None:
            raise ValueError('invalid reference label')
        for key in ('source_folder', 'stage'):
            path = Path(ref[key])
            if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('unsafe reference path')
    return config


def due():
    """No GPU work in daytime, unverified builds, busy state or uncertain peers."""
    current = datetime.now(IST)
    if not 0 <= current.hour < 6:
        return {'status': 'not_night'}
    try:
        root = _root()
        config = runtime._read_json(root, 'references.json')
        if config is None:
            return {'status': 'not_configured'}
        _validate(config)
        if not config['enabled']:
            return {'status': 'disabled'}
        if not health.verified_build():
            return {'status': 'unverified_build'}
        if runtime.status()['active']:
            return {'status': 'deferred_busy'}
        if any(gpu_lock.loaded(endpoint) != [] for endpoint in gpu_lock.runtime_endpoints()):
            return {'status': 'deferred_gpu'}
        fd = runtime._open_file(root, 'schedule.lock', create=True)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {'status': 'deferred_busy'}
            schedule = runtime._read_json(root, 'schedule.json') or {'last_full_at': None, 'next_lane': 'laptop'}
            if schedule['last_full_at'] and current - datetime.fromisoformat(schedule['last_full_at']) < timedelta(days=7):
                return {'status': 'not_due'}
            with gpu_lock.Session('meetingintel', wait_seconds=0) as session:
                run = root / ('run-' + uuid.uuid4().hex)
                run.mkdir(mode=0o700)
                previous = dict(schedule)
                schedule['last_full_at'] = current.isoformat()
                schedule['next_lane'] = 'phone' if previous['next_lane'] == 'laptop' else 'laptop'
                runtime._write_json(root, 'schedule.json', schedule)
                result = session.run([sys.executable, str(Path(__file__).resolve()), 'worker', str(run), previous['next_lane']],
                                     capture_output=True, env=dict(os.environ, MI_CANARY='1', MI_ORIGIN='scheduled', MI_ACTOR='scheduler'))
                if result.returncode == 75 and not (run / 'full-started.json').exists():
                    runtime._write_json(root, 'schedule.json', previous)
                category = 'passed' if result.returncode == 0 else 'deferred_busy' if result.returncode == 75 else 'failed'
                runtime._write_json(root, 'receipt.json', {'at': current.isoformat(), 'status': category, 'lane': previous['next_lane']})
                return {'status': category}
        finally:
            os.close(fd)
    except gpu_lock.Deferred:
        return {'status': 'deferred_gpu'}
    except Exception:
        return {'status': 'failed', 'failure_code': 'validation_failure'}


def _source(ref):
    import meetingintel_pipeline as pipeline
    import meetingintel_identity as identity
    source = identity._source(Path(ref['source_folder']))
    production = (pipeline.production_source_from_laptop_capture(source) if ref['lane'] == 'laptop'
                  else pipeline.production_source_from_audio_first(source))
    if pipeline.sha256_file(production.source_audio_path) != ref['source_audio_sha256'] or pipeline.sha256_file(Path(ref['stage']) / 'run_manifest.json') != ref['stage_manifest_sha256']:
        raise ValueError('reference changed')
    import meetingintel_staging_cache as cache
    pair = cache._json_file(Path(ref['stage']) / 'run_manifest.json')
    if pair is None or not isinstance(pair[0], dict):
        raise ValueError('reference manifest invalid')
    manifest = pair[0]
    mode = manifest.get('transcription_mode', 'legacy')
    if mode not in {'legacy', 'qwen3-asr-1.7b-contiguous'}:
        raise ValueError('reference backend invalid')
    # These are frozen, accepted identity references, not ASR cache entries.
    # The human-confirmed anchor predates runtime sidecars. Never loosen the
    # production cache contract to admit it for a fresh full processing run.
    from voiceprint_gate5_preview import _load_diarization
    _load_diarization(Path(ref['stage']), production.source_identity, include_unassigned=True)
    if (manifest.get('exit_code') != 0 or Path(manifest.get('source_folder', '')).resolve() != production.source_folder
            or Path(manifest.get('source_audio', '')).resolve() != production.source_audio_path
            or Path(manifest.get('output_folder', '')).resolve() != Path(ref['stage'])
            or datetime.fromisoformat(manifest['completed_at']).tzinfo is None):
        raise ValueError('reference stage invalid')
    if any(pipeline.sha256_file(Path(ref['stage']) / name) != digest for name, digest in ref['stage_artifact_sha256'].items()):
        raise ValueError('reference artifact changed')
    return production


def _timed_out(*_):
    raise TimeoutError('canary_time_limit')


@health.batch
def _checks(config, run, lane):
    import meetingintel_pipeline as pipeline
    import meetingintel_cli as cli
    import meetingintel_identity as identity
    for ref in config['references']:
        os.environ['MI_CANARY_PHASE'] = 'cheap'
        source = _source(ref)
        health.begin(source.fingerprint, source.source_kind, select=True)
        stage = Path(ref['stage'])
        started = time.monotonic()
        result = identity.create_presentation(source, stage, run / ('cheap-' + ref['lane']))
        health.identity_result(result)
        health.reuse(True)
        health.gpu_timing(0, int((time.monotonic() - started) * 1000))
        projection = identity._json(run / ('cheap-' + ref['lane']) / 'transcripts/identity' / (source.fingerprint + '.json'))
        receipt = identity._json(Path(projection.get('receipt_path', '/nonexistent')))
        passed = result.get('status') == 'matched' and receipt.get('named_label') == ref['expected_label']
        health.finish(health._active.get(), 'complete' if passed else 'failed', None if passed else 'validation_failure')
        if not passed:
            return 1
    ref = next(r for r in config['references'] if r['lane'] == lane)
    source = _source(ref)
    # Cheap and full are distinct measurement runs for the same reference.
    health._runs.get().pop(source.fingerprint, None)
    runtime._write_json(run, 'full-started.json', {'started_at': health.now().isoformat()})
    cli.LEDGER_PATH, cli.OUTPUT_DIR = run / 'ledger.json', run / 'output'
    pipeline.QWEN_STAGING_ROOT = run / 'asr'
    pipeline.qwen_resume_candidate = lambda _: None
    candidate = cli.SourceCandidate(source.fingerprint, source.created_at, source.source_kind, 'Reference recording')
    def execute(args):
        if lane == 'laptop':
            return pipeline.process_explicit_laptop_sources(args, (source,))
        return pipeline.process_meetings(args)
    os.environ['MI_CANARY_PHASE'] = 'full'
    cli.load_secret_environment()
    from ollama_endpoint import resolve_ollama_endpoint, preflight_ollama
    preflight_ollama(resolve_ollama_endpoint())
    code = cli._run_production((candidate,), scope='exact canary reference', processed=set(),
        clock=pipeline.now_ist, pipeline_runner=execute, ollama_manager=None,
        audio_first_only=True, explicit_audio_first_source_folders=(source.source_folder,),
        suppress_private_paths=True)
    record = health._runs.get().get(source.fingerprint)
    if code != 0 or not record or record['outcome'] != 'complete' or not record['first_try']:
        return 1
    # This independent process has no ContextVar from the pipeline. It must see
    # the completed real-run sidecar by its recorded operation ID.
    reader = subprocess.run([sys.executable, '-c',
        'import json,meetingintel_progress as p; print(json.dumps(p.read()))'],
        env=dict(os.environ, MI_OPERATION_ID=record['progress_operation_id']),
        capture_output=True, text=True, timeout=10)
    measured = json.loads(reader.stdout) if reader.returncode == 0 else None
    if not measured or measured['stage'] != 'complete':
        record.update(outcome='failed', first_try=False, failure_code='validation_failure')
        record['failure_codes'].append('validation_failure')
        health._save(record)
        return 1
    return 0


def worker(run, lane):
    if os.environ.get('MI_CANARY') != '1':
        return 1
    gpu_lock.assert_child()
    root = _root()
    if run.parent != root or run.is_symlink() or not run.is_dir() or lane not in {'phone', 'laptop'}:
        return 1
    config = _validate(runtime._read_json(root, 'references.json'))
    os.environ.update(MI_DISABLE_STAGE_REUSE='1', MI_STAGING_CACHE_ROOT=str(run / 'stage-cache'),
        MI_PROGRESS_ROOT=str(run), MI_IDENTITY_RECEIPT_ROOT=str(run / 'identity-receipts'),
        MI_CANARY_ASR_ROOT=str(run / 'asr'), MI_CANARY_SHADOW_ROOT=str(run / 'analysis'))
    os.environ.pop('MI_OPERATION_ID', None)
    signal.signal(signal.SIGALRM, _timed_out)
    signal.alarm(1800)
    try:
        with runtime.operation_lock('processing'), audio_tool_environment():
            return _checks(config, run, lane)
    except runtime.BusyError:
        return 75
    except Exception:
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'worker':
        raise SystemExit(worker(Path(sys.argv[2]), sys.argv[3]))
    raise SystemExit('Use the existing learning scheduler due-check.')
