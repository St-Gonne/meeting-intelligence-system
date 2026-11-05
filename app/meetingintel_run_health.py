"""Content-free receipts for ordinary processing and the bounded canary.

Receipts are measurement evidence, never an alternate production ledger. A
missing receipt, incomplete stage or untested build cannot pass the stability
gate. Storage failures leave a coverage gap and never interrupt processing.
"""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from functools import wraps
import json
import os
from pathlib import Path
import re
import stat
import uuid

import meetingintel_learning as learning
import meetingintel_progress as progress
import meetingintel_runtime as runtime

_runs = ContextVar('mi_health_runs', default=None)
_active = ContextVar('mi_health_active', default=None)
STAGES = {'preflight', 'processing', 'transcription', 'eligibility', 'analysis',
          'identity', 'transcript', 'report', 'ledger', 'brief'}
STATUSES = {'running', 'complete', 'failed', 'deferred', 'interrupted'}
FIELDS = {'schema_version', 'run_id', 'source_id', 'source_kind', 'build_id',
          'origin', 'actor', 'canary', 'canary_phase', 'started_at', 'finished_at', 'outcome',
          'first_try', 'failure_code', 'failure_codes', 'stages', 'identity_status',
          'identity_reason', 'stage_reuse', 'progress_operation_id',
          'progress_terminal', 'gpu_wait_ms', 'gpu_duration_ms'}


def now():
    return datetime.now(timezone.utc)


def _root(create=False):
    path = Path(os.environ.get('MI_RUN_HEALTH_ROOT', str(learning._root() / 'run-health')))
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe health root')
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('unsafe health root')
    return path


def _save(record):
    if os.environ.get('MI_RUN_HEALTH_OFF') == '1':
        return
    try:
        runtime._write_json(_root(True), record['run_id'] + '.json', record)
    except Exception:
        learning._gap()


def _valid(record):
    if not isinstance(record, dict) or set(record) != FIELDS or record['schema_version'] != 1:
        raise ValueError('invalid health record')
    for key in ('source_id', 'build_id'):
        if not isinstance(record[key], str) or re.fullmatch('[a-f0-9]{64}', record[key]) is None:
            raise ValueError('invalid health identity')
    if re.fullmatch('[a-f0-9]{32}', record['run_id']) is None:
        raise ValueError('invalid run id')
    if record['source_kind'] not in learning.SOURCE_KINDS or record['origin'] not in learning.ORIGINS or record['actor'] not in learning.ACTORS:
        raise ValueError('invalid health category')
    if record['outcome'] not in STATUSES or type(record['first_try']) is not bool or type(record['canary']) is not bool:
        raise ValueError('invalid health outcome')
    if record['canary_phase'] not in {'cheap', 'full', 'none'}:
        raise ValueError('invalid canary phase')
    from meetingintel_identity import REASONS
    if record['identity_status'] not in {'matched', 'anonymous', 'revoked', 'disabled', 'unavailable', 'not_applicable', 'not_run'} or record['identity_reason'] not in REASONS | {'not_applicable', 'not_run'}:
        raise ValueError('invalid identity status')
    if record['stage_reuse'] not in {'hit', 'miss', 'not_applicable', 'unknown'} or record['progress_terminal'] not in STATUSES | {'unknown'}:
        raise ValueError('invalid health progress')
    if record['progress_operation_id'] is not None:
        progress._identifier(record['progress_operation_id'])
    for key in ('gpu_wait_ms', 'gpu_duration_ms'):
        if record[key] is not None and (type(record[key]) is not int or record[key] < 0):
            raise ValueError('invalid timing')
    if record['failure_code'] is not None and record['failure_code'] not in learning.SAFE_CODES:
        raise ValueError('invalid failure')
    if not isinstance(record['failure_codes'], list) or any(code not in learning.SAFE_CODES for code in record['failure_codes']):
        raise ValueError('invalid failures')
    if not isinstance(record['stages'], dict):
        raise ValueError('invalid stages')
    for name, stage in record['stages'].items():
        if name not in STAGES or not isinstance(stage, dict) or set(stage) != {'result', 'attempts', 'first_try', 'failure_code'}:
            raise ValueError('invalid stage')
        if stage['result'] not in {'started', 'passed', 'failed'} or type(stage['attempts']) is not int or stage['attempts'] < 1 or type(stage['first_try']) is not bool:
            raise ValueError('invalid stage outcome')
        if stage['failure_code'] is not None and stage['failure_code'] not in learning.SAFE_CODES:
            raise ValueError('invalid stage failure')
    for key in ('started_at', 'finished_at'):
        if record[key] is not None and datetime.fromisoformat(record[key]).tzinfo is None:
            raise ValueError('invalid time')
    if (record['outcome'] == 'running') != (record['finished_at'] is None):
        raise ValueError('invalid terminal time')
    return record


def records():
    """Read only private, schema-checked receipts. Damage is visible as a gap."""
    result, gaps = [], 0
    try:
        root = _root()
        for path in root.glob('*.json'):
            if re.fullmatch('[a-f0-9]{32}\\.json', path.name) is None:
                continue
            try:
                record = _valid(runtime._read_json(root, path.name))
                if path.stem != record['run_id']:
                    raise ValueError('wrong run id')
                result.append(record)
            except Exception:
                gaps += 1
    except Exception:
        gaps += 1
    return sorted(result, key=lambda r: (r['started_at'], r['run_id'])), gaps


def _prior_staging_attempt(source_identity, source_folder):
    """Read historical attempt evidence before processing creates new staging."""
    import meetingintel_pipeline as pipeline
    import whispermlx_diarization_helper as helper
    if source_identity and re.fullmatch('[a-f0-9]{64}', source_identity) is None:
        raise ValueError('invalid source identity')
    for root in {pipeline.QWEN_STAGING_ROOT, helper.OUTPUT_ROOT}:
        for folder in root.iterdir() if root.is_dir() else ():
            if not folder.is_dir() or folder.is_symlink():
                continue
            label = 'audio_first_' + source_identity if source_identity else Path(source_folder).name
            if folder.name == label or folder.name.startswith(label + '__'):
                return True
            short_label = bool(source_identity and folder.name.startswith('audio_first_' + source_identity[:16] + '__'))
            manifest_path = folder / 'run_manifest.json'
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if not manifest_path.is_symlink() else {}
            except (OSError, ValueError):
                if short_label:
                    return True
                continue
            if not isinstance(manifest, dict):
                continue
            same_source = (manifest.get('source_identity') == source_identity if source_identity
                           else manifest.get('source_folder') == str(source_folder))
            if same_source and (short_label or manifest.get('status') == 'failure'):
                return True
    return False


def begin(source_id, source_kind, *, select=False, source_identity=None, source_folder=None):
    group = _runs.get()
    if group is None or os.environ.get('MI_RUN_HEALTH_OFF') == '1':
        return
    if re.fullmatch('[a-f0-9]{64}', source_id or '') is None:
        return
    record = group.get(source_id)
    prior_staging = False
    if (record is None or select) and (source_identity or source_folder) and os.environ.get('MI_CANARY') != '1':
        try:
            prior_staging = _prior_staging_attempt(source_identity, source_folder)
        except Exception:
            learning._gap()
            prior_staging = True
    if record is None:
        previous, gaps = records()
        repeated = any(r['source_id'] == source_id and not r['canary'] and r['outcome'] != 'running' for r in previous)
        record = dict(schema_version=1, run_id=uuid.uuid4().hex, source_id=source_id,
            source_kind=source_kind, build_id=learning.build_id(),
            origin=os.environ.get('MI_ORIGIN', 'direct_script'),
            actor=os.environ.get('MI_ACTOR', 'unknown'), canary=os.environ.get('MI_CANARY') == '1',
            canary_phase=os.environ.get('MI_CANARY_PHASE', 'cheap') if os.environ.get('MI_CANARY') == '1' else 'none',
            started_at=now().isoformat(), finished_at=None, outcome='running', first_try=not repeated and not gaps and not prior_staging,
            failure_code=None, failure_codes=[], stages={}, identity_status='not_run', identity_reason='not_run',
            stage_reuse='unknown', progress_operation_id=None, progress_terminal='unknown',
            gpu_wait_ms=None, gpu_duration_ms=None)
        if record['origin'] not in learning.ORIGINS:
            record['origin'] = 'unknown'
        if record['actor'] not in learning.ACTORS:
            record['actor'] = 'unknown'
        group[source_id] = record
    if prior_staging:
        record['first_try'] = False
    _active.set(record)
    if select:
        event('stage.succeeded', stage='preflight')
        try:
            snapshot = progress.start(fingerprint64hex=source_id)
            record['progress_operation_id'] = snapshot['operation_id']
        except Exception:
            learning._gap()
    _save(record)


def event(event_type, *, stage=None, source_id=None, error_code=None, **_ignored):
    record = (_runs.get() or {}).get(source_id) if source_id else None
    if record is None and event_type.startswith('stage.'):
        record = _active.get()
    if record is None and source_id is None:
        record = _active.get()
    if not record or record['outcome'] != 'running' or stage not in STAGES:
        return
    previous = record['stages'].get(stage)
    code = error_code if error_code in learning.SAFE_CODES else 'operation_failed'
    failed = event_type.endswith('.failed')
    started = event_type.endswith('.started')
    attempts = (previous or {}).get('attempts', 0) + int(started)
    first = (previous or {}).get('first_try', True) and not failed and attempts <= 1
    record['stages'][stage] = dict(result='failed' if failed else 'started' if started else 'passed',
        attempts=max(1, attempts), first_try=first,
        failure_code=code if failed else (previous or {}).get('failure_code'))
    if failed:
        record['first_try'] = False
        if code not in record['failure_codes']:
            record['failure_codes'].append(code)
    _save(record)
    if failed and stage == 'processing':
        finish(record, 'deferred' if code == 'deferred' else 'failed', code)
    if stage == 'analysis' and not started and not failed:
        event('artifact.saved', stage='report', source_id=record['source_id'])


def identity_result(result):
    record = _active.get()
    if record:
        from meetingintel_identity import REASONS
        status, reason = result.get('status'), result.get('reason')
        record['identity_status'] = status if status in {'matched', 'anonymous', 'revoked', 'disabled', 'unavailable'} else 'unavailable'
        record['identity_reason'] = reason if reason in REASONS else 'unavailable'
        event('stage.succeeded' if status != 'unavailable' else 'stage.failed', stage='identity', error_code='missing_artifact')


def reuse(hit):
    record = _active.get()
    if record:
        record['stage_reuse'] = 'hit' if hit else 'miss'
        _save(record)


def gpu_timing(wait_ms, duration_ms):
    record = _active.get()
    if record:
        record['gpu_wait_ms'], record['gpu_duration_ms'] = wait_ms, duration_ms
        _save(record)


def finish(record, outcome, code=None):
    if record['outcome'] != 'running':
        return
    operation_id = record['progress_operation_id']
    if operation_id:
        try:
            progress.finish('complete' if outcome == 'complete' else 'deferred' if outcome == 'deferred' else 'failed', operation_id=operation_id)
        except Exception:
            pass
        snapshot = progress.read(operation_id)
        record['progress_terminal'] = snapshot['stage'] if snapshot else 'unknown'
    record.update(outcome=outcome, finished_at=now().isoformat(), failure_code=code)
    record['first_try'] = record['first_try'] and outcome == 'complete' and record['progress_terminal'] == 'complete'
    if code and code not in record['failure_codes']:
        record['failure_codes'].append(code)
    _save(record)


def delivered(fingerprints):
    for fingerprint in fingerprints:
        record = (_runs.get() or {}).get(fingerprint)
        if record and record['outcome'] == 'running':
            event('artifact.saved', stage='brief', source_id=fingerprint)
            required = {'processing', 'transcript', 'ledger', 'brief'}
            complete = all(record['stages'].get(stage, {}).get('result') == 'passed' for stage in required)
            finish(record, 'complete' if complete else 'failed', None if complete else 'missing_artifact')


def _scope(function, args, kwargs, candidates=None):
    existing = _runs.get()
    token = _runs.set({}) if existing is None else None
    active_token = _active.set(None) if token is not None else None
    try:
        if candidates:
            for candidate in candidates:
                begin(candidate.fingerprint, candidate.source_kind)
        return function(*args, **kwargs)
    except BaseException as exc:
        from meetingintel_observe import safe_error
        code = safe_error(exc)
        for record in (_runs.get() or {}).values():
            finish(record, 'deferred' if code == 'deferred' else 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', code)
        raise
    finally:
        if token is not None:
            for record in _runs.get().values():
                finish(record, 'failed', 'outcome_unknown' if record['progress_operation_id'] else 'preflight_failed')
            _runs.reset(token)
            _active.reset(active_token)


def batch(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return _scope(function, args, kwargs)
    return wrapped


def attempt(function):
    @wraps(function)
    def wrapped(selected, *args, **kwargs):
        candidates = [c for c in selected if kwargs.get('refresh_existing') or c.fingerprint not in kwargs.get('processed', set())]
        return _scope(function, (selected, *args), kwargs, candidates)
    return wrapped


def verified_build():
    try:
        payload = runtime._read_json(_root(), 'verified-build.json')
        if (set(payload) != {'build_id', 'verified_at', 'tests_passed'} or payload['build_id'] != learning.build_id()
                or type(payload['tests_passed']) is not int or payload['tests_passed'] < 1
                or datetime.fromisoformat(payload['verified_at']).tzinfo is None):
            return None
        return payload
    except Exception:
        return None


def record_suite_pass(tests_passed):
    if type(tests_passed) is not int or tests_passed < 1:
        raise ValueError('invalid suite count')
    payload = dict(build_id=learning.build_id(), verified_at=now().isoformat(), tests_passed=tests_passed)
    runtime._write_json(_root(True), 'verified-build.json', payload)
    return payload


def _reference_check_overdue(current, rows, root, receipt):
    config = runtime._read_json(root, 'references.json')
    if config is None:
        return False
    schedule = runtime._read_json(root, 'schedule.json') or {}
    attempts = [datetime.fromisoformat(r['started_at']) for r in rows if r['canary']]
    if schedule.get('last_full_at'):
        attempts.append(datetime.fromisoformat(schedule['last_full_at']))
    if receipt and receipt.get('status') in {'passed', 'failed'}:
        attempts.append(datetime.fromisoformat(receipt['at']))
    last_attempt = max(attempts) if attempts else datetime.fromtimestamp((root / 'references.json').stat().st_mtime, timezone.utc)
    return current - last_attempt > timedelta(days=8)


def summary():
    rows, gaps = records()
    current = now()
    for record in rows:
        if record['outcome'] == 'running':
            try:
                active = runtime.status()['active']
            except Exception:
                active = True
                gaps += 1
            if not active or current - datetime.fromisoformat(record['started_at']) > timedelta(hours=24):
                record.update(outcome='interrupted', first_try=False, failure_code='interrupted')
                record['failure_codes'] = [*record['failure_codes'], 'interrupted']
    ordinary = [r for r in rows if not r['canary']]
    recent = [r for r in ordinary if current - timedelta(hours=24) <= datetime.fromisoformat(r['started_at']) <= current]
    canaries = [r for r in rows if r['canary'] and current - timedelta(hours=24) <= datetime.fromisoformat(r['started_at']) <= current]
    verification = verified_build()
    streak, seen = 0, set()
    for record in ordinary:
        if not verification or record['build_id'] != verification['build_id'] or record['started_at'] < verification['verified_at']:
            continue
        eligible = record['origin'] in {'cli', 'gui'} and record['actor'] == 'human'
        if not eligible:
            continue
        if record['outcome'] == 'running':
            continue
        passed = record['outcome'] == 'complete' and record['first_try'] and record['progress_terminal'] == 'complete'
        if not passed:
            streak = 0
        elif record['source_id'] not in seen:
            streak += 1
        seen.add(record['source_id'])
    if gaps or os.environ.get('MI_RUN_HEALTH_OFF') == '1':
        streak = 0
    coverage = learning._root() / 'coverage-gap.json'
    if coverage.exists():
        gaps += 1
        streak = 0
    last = ordinary[-5:]
    ok = sum(r['outcome'] == 'complete' and r['first_try'] for r in last)
    header = f'Last {len(last)} runs: {ok} OK · stability {min(streak, 5)}/5' if last else 'No run-health evidence yet · stability 0/5'
    if not verification:
        header += ' · build verification missing'
    if gaps:
        header += ' · measurement gap'
    canary_failure = next((r for r in reversed(rows) if r['canary'] and r['outcome'] in {'failed', 'interrupted'}), None)
    if canary_failure:
        header += ' · reference check failed: ' + (canary_failure['failure_code'] or 'outcome_unknown')
        from meetingintel_identity import PREVIEW_FAILURE_REASONS, WORKER_FAILURE_REASONS
        reason = canary_failure['identity_reason']
        if reason in set(PREVIEW_FAILURE_REASONS.values()) | WORKER_FAILURE_REASONS:
            header += ' (' + reason + ')'
    reference_check_overdue = False
    try:
        canary_root = learning._root() / 'canary'
        receipt = runtime._read_json(canary_root, 'receipt.json') if canary_root.exists() else None
        if receipt and receipt.get('status') == 'failed' and not canary_failure:
            header += ' · reference check failed'
        reference_check_overdue = _reference_check_overdue(current, rows, canary_root, receipt)
        if reference_check_overdue:
            header += ' · reference check overdue'
    except Exception:
        header += ' · reference check unavailable'
    failures = Counter(code for r in recent for code in r['failure_codes'])
    return dict(runs=len(recent), first_try_passes=sum(r['outcome'] == 'complete' and r['first_try'] for r in recent),
        failures=dict(failures), stability_streak=min(streak, 5), stable=streak >= 5 and not gaps,
        build_verified=verification is not None, coverage_gaps=gaps, header=header,
        reference_check_overdue=reference_check_overdue,
        canary_runs=len(canaries), canary_failures=sum(r['outcome'] in {'failed', 'interrupted'} for r in canaries))


if __name__ == '__main__':
    import json
    print(json.dumps(summary(), sort_keys=True))
