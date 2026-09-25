#!/usr/bin/env python3
"""Small, content-free local observations and deterministic usage reviews.

This is not the production ledger. Collection is best effort; source snapshots
must be supplied by the authoritative metadata adapter, never discovered here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from statistics import median
import sys
import tempfile
import uuid

SCHEMA_VERSION = 1
WINDOW_DAYS = 14
EVENT_TYPES = frozenset(
    f'{family}.{action}'
    for family, actions in {
        'action': 'offered selected confirmed cancelled terminal_handoff',
        'capture': 'started stopped interrupted',
        'notification': 'attempted delivered acknowledged failed',
        'stage': 'started succeeded failed',
        'operation': 'started succeeded failed busy interrupted cancelled',
        'artifact': 'saved failed opened exported',
        'phone': 'delivered reviewed normalized failed',
        'source': 'observed',
        'measurement': 'gap',
    }.items() for action in actions.split()
)
ORIGINS = {'gui', 'cli', 'scheduled', 'direct_script', 'unknown'}
ACTORS = {'human', 'agent', 'scheduler', 'unknown'}
SOURCE_KINDS = {'phone_recording', 'laptop_capture', 'meetily', 'unknown'}
STAGES = {'capture', 'delivery', 'settlement', 'review', 'normalization', 'preflight',
          'transcription', 'diarization', 'eligibility', 'analysis', 'layer2', 'layer3',
          'transcript', 'report', 'ledger', 'brief', 'processing', 'notification'}
OUTCOMES = {'started', 'success', 'failure', 'unknown', 'interrupted', 'busy',
            'cancelled', 'ready', 'waiting', 'unresolved', 'completed', 'saved',
            'missing', 'deferred', 'skipped', 'available', 'unavailable'}
ACTIONS = {'process', 'retry', 'record', 'phone_review', 'recover', 'normalize',
           'open_transcript', 'open_report', 'repair_brief', 'refresh',
           'learning_review', 'sync', 'details', 'stop', 'open_brief'}
SAFE_CODES = {'operation_failed', 'transcription_failed', 'diarization_failed', 'brief_write_failed', 'ledger_write_failed', 'capture_incomplete', 'capture_interrupted', 'unknown', 'busy', 'interrupted', 'source_lost', 'source_not_recovered',
              'capture_permission_failed', 'invalid_owner_field', 'storage_failure',
              'validation_failure', 'preflight_failed', 'process_failed', 'worker_failed',
              'process_interrupted', 'persistence_failure', 'ledger_persistence_failed',
              'brief_persistence_failed', 'transcript_persistence_failed',
              'flat_layer2:invalid_owner_field', 'flat_layer3:invalid_owner_field',
              'missing_artifact', 'collector_unavailable', 'telemetry_rejected',
              'outcome_unknown', 'stale_operation', 'source_incomplete', 'command_failed'}
DETAIL_ENUMS = {
    'backend': {'qwen', 'legacy', 'whisper', 'gemma', 'none', 'unknown'},
    'action': ACTIONS,
    'artifact_kind': {'transcript', 'report', 'brief', 'ledger'},
    'processing_mode': {'diarized_1to1', 'flat', 'flat_fallback',
                        'flat_from_diarized_transcript', 'flat_fallback_from_diarized_transcript'},
    'capture_status': {'recording', 'completed', 'interrupted', 'unknown'},
}
BOOL_DETAILS = {'reconstructed', 'empty_action_list', 'enabled', 'acknowledged',
                'complete', 'partial', 'known_incomplete', 'persisted'}
COUNT_DETAILS = {'count', 'part_count', 'segment_count', 'selected_count',
                 'missing_owner_count', 'duplicate_owner_count', 'duration_ms',
                 'source_count', 'dropped_count', 'attempt_count'}
IDENTITY = re.compile(r'^(?:[a-fA-F0-9]{16,64}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})$')
_GAPS = 0


def _now():
    return datetime.now(timezone.utc)


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat()


def _root():
    return Path(os.environ.get('MI_LEARNING_ROOT', str(Path.home() / 'Library/Application Support/MeetingIntel/learning'))).expanduser()


def _safe_root(create=False):
    root = _root()
    if not root.is_absolute() or '..' in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise OSError('unsafe learning storage')
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.exists():
        if root.stat().st_uid != os.getuid() or not root.is_dir():
            raise OSError('unsafe learning owner')
        if create:
            root.chmod(0o700)
    return root


def _atomic_json(path, payload):
    if path.is_symlink():
        raise OSError('unsafe output path')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(payload, output, sort_keys=True, separators=(',', ':'))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _gap():
    global _GAPS
    _GAPS += 1
    try:
        root = _safe_root(create=True)
        # A gap marker is deliberately outside SQLite so DB failures remain visible.
        _atomic_json(root / 'coverage-gap.json', {'status': 'incomplete', 'last_gap_at': _stamp(_now()), 'process_gap_count': _GAPS})
    except Exception:
        pass


@lru_cache(maxsize=1)
def build_id():
    """Hash actual source bytes, including dirty edits; cached per process."""
    root = Path(__file__).resolve().parent
    paths = [p for p in root.glob('*.py') if not p.name.startswith('test_')]
    for folder in ('ui', 'prompts'):
        if (root / folder).is_dir():
            paths.extend(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.html', '.css', '.js', '.txt', '.md', '.json'})
    alert_source = root / 'scripts' / 'capture_alert.swift'
    if alert_source.is_file():
        paths.append(alert_source)
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_symlink():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _identity(value):
    if value is None:
        return None
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError('invalid identity')
    return value


def _choice(value, allowed):
    if value is not None and value not in allowed:
        raise ValueError('invalid category')
    return value


def _details(details):
    if details is None:
        return {}
    if not isinstance(details, dict) or len(details) > 20:
        raise ValueError('invalid details')
    result = {}
    for key, value in details.items():
        if key == 'source_refs' and isinstance(value, list) and len(value) <= 100:
            if any(item is None for item in value):
                raise ValueError('invalid source reference')
            result[key] = list(dict.fromkeys(_identity(item) for item in value))
        elif key in DETAIL_ENUMS:
            result[key] = _choice(value, DETAIL_ENUMS[key])
        elif key in BOOL_DETAILS and isinstance(value, bool):
            result[key] = value
        elif key in COUNT_DETAILS and type(value) is int and 0 <= value <= 10**12:
            result[key] = value
        else:
            raise ValueError('unapproved detail')
    return result


@contextmanager
def _db(create=True):
    root = _safe_root(create=create)
    path = root / 'events.sqlite'
    for candidate in (path, root / 'events.sqlite-wal', root / 'events.sqlite-shm'):
        if candidate.is_symlink():
            raise OSError('unsafe database path')
    if create and not path.exists():
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
    connection = sqlite3.connect(str(path) if create else path.as_uri() + '?mode=ro', uri=not create, timeout=0.03)
    connection.row_factory = sqlite3.Row
    try:
        if not create and connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
            raise ValueError('unsupported observation schema')
        if create:
            path.chmod(0o600)
            connection.execute('PRAGMA journal_mode=WAL')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError('unsupported observation schema')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS events (
                  event_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL,
                  event_type TEXT NOT NULL, source_id TEXT, source_kind TEXT,
                  stage TEXT, outcome TEXT, error_code TEXT, elapsed_ms INTEGER,
                  operation_id TEXT, parent_operation_id TEXT, origin TEXT NOT NULL,
                  actor TEXT NOT NULL, build_id TEXT NOT NULL, details TEXT NOT NULL,
                  schema_version INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS events_time ON events(occurred_at);
                CREATE INDEX IF NOT EXISTS events_source ON events(source_id, occurred_at);
                CREATE TABLE IF NOT EXISTS review_runs (
                  review_id TEXT PRIMARY KEY, window_start TEXT NOT NULL,
                  window_end TEXT NOT NULL, created_at TEXT NOT NULL,
                  status TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS changes (
                  change_id TEXT PRIMARY KEY, review_id TEXT NOT NULL,
                  finding_type TEXT NOT NULL, state TEXT NOT NULL,
                  created_at TEXT NOT NULL, evaluation TEXT NOT NULL);
            ''')
            connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
            now = _stamp(_now())
            connection.execute('INSERT OR IGNORE INTO review_runs VALUES(?,?,?,?,?,?)', ('anchor', now, now, now, 'anchor', '{}'))
            connection.commit()
            for suffix in ('', '-wal', '-shm'):
                sidecar = root / ('events.sqlite' + suffix)
                if sidecar.exists():
                    sidecar.chmod(0o600)
        yield connection
    finally:
        connection.close()


def emit(event_type, *, source_id=None, source_kind=None, stage=None, outcome=None,
         error_code=None, elapsed_ms=None, operation_id=None, parent_operation_id=None,
         origin=None, actor=None, details=None):
    """Persist one bounded observation; never propagate telemetry failure."""
    try:
        return _emit(event_type, source_id=source_id, source_kind=source_kind, stage=stage,
                     outcome=outcome, error_code=error_code, elapsed_ms=elapsed_ms,
                     operation_id=operation_id, parent_operation_id=parent_operation_id,
                     origin=origin, actor=actor, details=details)
    except Exception:
        _gap()
        return None


def _emit(event_type, *, source_id=None, source_kind=None, stage=None, outcome=None,
          error_code=None, elapsed_ms=None, operation_id=None, parent_operation_id=None,
          origin=None, actor=None, details=None, event_id=None):
    _choice(event_type, EVENT_TYPES)
    if event_type is None:
        raise ValueError('missing event type')
    if outcome is None:
        outcome = {'started': 'started', 'succeeded': 'success', 'failed': 'failure', 'saved': 'saved', 'busy': 'busy', 'interrupted': 'interrupted', 'cancelled': 'cancelled'}.get(event_type.rsplit('.', 1)[-1])
    outcome = {'succeeded': 'success', 'failed': 'failure', 'confirmed': 'completed'}.get(outcome, outcome)
    payload = _details(details)
    if elapsed_ms is not None and (type(elapsed_ms) is not int or not 0 <= elapsed_ms <= 10**12):
        raise ValueError('invalid elapsed duration')
    selected_origin = origin or os.environ.get('MI_ORIGIN', 'direct_script')
    selected_actor = actor or os.environ.get('MI_ACTOR', 'unknown')
    # Unknown safe error identifiers are deliberately coarsened, never retained.
    safe_error = error_code if error_code in SAFE_CODES else ('unknown' if error_code is not None else None)
    event_id = event_id or uuid.uuid4().hex
    values = (event_id, _stamp(_now()), event_type, _identity(source_id),
              _choice(source_kind, SOURCE_KINDS), _choice(stage, STAGES),
              _choice(outcome, OUTCOMES), safe_error, elapsed_ms,
              _identity(operation_id or os.environ.get('MI_OPERATION_ID')),
              _identity(parent_operation_id), _choice(selected_origin, ORIGINS),
              _choice(selected_actor, ACTORS), build_id(), json.dumps(payload, sort_keys=True), SCHEMA_VERSION)
    with _db() as connection:
        connection.execute('INSERT OR IGNORE INTO events VALUES(' + ','.join('?' for _ in values) + ')', values)
        connection.commit()
    return event_id


def reconcile(snapshots):
    """Import validated metadata snapshots idempotently; no filesystem scan."""
    imported = 0
    failed = 0
    for snapshot in snapshots:
        try:
            allowed = {'source_id', 'source_kind', 'stage', 'outcome', 'error_code', 'details'}
            if not isinstance(snapshot, dict) or set(snapshot) - allowed:
                raise ValueError('invalid snapshot')
            data = dict(snapshot)
            data['details'] = {**(data.get('details') or {}), 'reconstructed': True}
            _identity(data.get('source_id'))
            _choice(data.get('source_kind'), SOURCE_KINDS)
            _choice(data.get('stage'), STAGES)
            _choice(data.get('outcome'), OUTCOMES)
            data['details'] = _details(data['details'])
            if data.get('error_code') not in SAFE_CODES and data.get('error_code') is not None:
                data['error_code'] = 'unknown'
            canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
            with _db() as connection:
                prior = connection.execute(
                    "SELECT * FROM events WHERE source_id IS ? AND stage IS ? AND event_type='source.observed' "
                    "ORDER BY occurred_at DESC,rowid DESC LIMIT 1",
                    (data.get('source_id'), data.get('stage'))).fetchone()
            identical = prior and all(prior[key] == data.get(key) for key in ('source_kind', 'stage', 'outcome', 'error_code')) and json.loads(prior['details']) == data['details']
            if not identical:
                event_id = hashlib.sha256(('reconciled:' + (prior['event_id'] if prior else '') + canonical).encode()).hexdigest()
                _emit('source.observed', event_id=event_id, origin='unknown', actor='unknown', **data)
                imported += 1
        except Exception:
            failed += 1
            _gap()
    return {'imported': imported, 'failed': failed}


def _coverage(root):
    return {'status': 'incomplete' if _GAPS or (root / 'coverage-gap.json').exists() else 'observed',
            'process_gap_count': _GAPS, 'external_app_usage': 'unobserved',
            'direct_uninstrumented_scripts': 'unobserved'}


def status():
    """Read-only: opening the GUI must not initialize observation storage."""
    try:
        root = _safe_root()
        coverage = _coverage(root)
        if not (root / 'events.sqlite').exists():
            return {'status': 'degraded' if coverage['status'] == 'incomplete' else 'not_initialized',
                    'event_count': 0, 'coverage': coverage, 'last_review': None, 'changes': []}
        with _db(create=False) as connection:
            row = connection.execute('SELECT count(*) AS count,min(occurred_at) AS first,max(occurred_at) AS last FROM events').fetchone()
            anchor = connection.execute("SELECT window_end FROM review_runs WHERE review_id='anchor'").fetchone()
            latest = connection.execute("SELECT * FROM review_runs WHERE status='completed' ORDER BY window_end DESC LIMIT 1").fetchone()
            changes = [dict(r) for r in connection.execute('SELECT change_id,review_id,finding_type,state,created_at,evaluation FROM changes ORDER BY created_at DESC LIMIT 50')]
            for change in changes:
                change['evaluation'] = json.loads(change['evaluation'])
            due = datetime.fromisoformat(latest['window_end'] if latest else anchor['window_end']) + timedelta(days=WINDOW_DAYS)
        return {'status': 'healthy' if coverage['status'] == 'observed' else 'degraded',
                'event_count': row['count'], 'first_event_at': row['first'], 'last_event_at': row['last'],
                'next_review_due': _stamp(due), 'last_review': json.loads(latest['result']) if latest else None,
                'changes': changes, 'coverage': coverage}
    except Exception:
        return {'status': 'degraded', 'event_count': None, 'coverage': {'status': 'unavailable'}, 'changes': [], 'last_review': None}


def source_state(source_id):
    """Latest observed stage/artifact outcome, never a production authority."""
    try:
        source_id = _identity(source_id)
        if source_id is None or not (_safe_root() / 'events.sqlite').exists():
            return {}
        with _db(create=False) as connection:
            row = connection.execute(
                "SELECT stage,outcome,error_code,occurred_at,event_type FROM events "
                "WHERE source_id=? AND (event_type LIKE 'stage.%' OR event_type IN ('artifact.saved','artifact.failed') "
                "OR event_type LIKE 'operation.%') "
                "ORDER BY occurred_at DESC,rowid DESC LIMIT 1", (source_id,)).fetchone()
        if not row:
            return {}
        return {'stage': row['stage'], 'outcome': row['outcome'], 'error_code': row['error_code'],
                'updated_at': row['occurred_at'], 'event_type': row['event_type']}
    except Exception:
        return {}


def _summarize(rows, start, end, root):
    events = [dict(row) for row in rows]
    for event in events:
        event['details'] = json.loads(event['details'])
    observed = [e for e in events if not e['details'].get('reconstructed')]
    findings = []

    def finding(kind, selected, denominator, candidate):
        if selected:
            findings.append({'finding_type': kind, 'observed_count': len(selected),
                             'eligible_count': denominator, 'evidence_ids': [e['event_id'] for e in selected[:100]],
                             'proposed_action': candidate, 'evaluation_status': 'not_evaluated'})

    captures = [e for e in observed if e['event_type'] == 'capture.started']
    interrupted = [e for e in observed if e['event_type'] == 'capture.interrupted']
    finding('capture_interruptions', interrupted, len(captures), 'inspect_capture_failure_categories')
    stages = Counter(e['stage'] or 'unknown' for e in observed if e['event_type'] == 'stage.started')
    for stage in sorted({e['stage'] for e in observed if e['event_type'] == 'stage.failed'}, key=str):
        failed = [e for e in observed if e['event_type'] == 'stage.failed' and e['stage'] == stage]
        finding('stage_failure:' + (stage or 'unknown'), failed, stages[stage or 'unknown'], 'inspect_stage_failure_categories')
    switches = []
    failed_gui = {}
    for event in observed:
        source = event['source_id']
        if not source:
            continue
        if event['origin'] == 'gui' and event['event_type'] in {'operation.failed', 'stage.failed'}:
            failed_gui[source] = event
        elif event['origin'] == 'cli' and (event['event_type'] == 'operation.started' or (event['event_type'] == 'stage.started' and event['stage'] == 'processing')) and source in failed_gui:
            prior = failed_gui.pop(source)
            if datetime.fromisoformat(event['occurred_at']) - datetime.fromisoformat(prior['occurred_at']) <= timedelta(days=1):
                switches.append(event)
    finding('gui_failure_then_cli_attempt', switches, len({e['source_id'] for e in observed if e['origin'] == 'gui' and e['event_type'] in {'operation.failed', 'stage.failed'}}), 'inspect_exact_source_retry_flow')
    action_counts = {}
    confirmed_actions = {}
    completed_actions = set()
    completion_evidence = {}
    for event in observed:
        action = event['details'].get('action')
        if action and event['event_type'].startswith('action.'):
            counts = action_counts.setdefault(action, {'completed': 0})
            counts[event['event_type'].split('.')[1]] = counts.get(event['event_type'].split('.')[1], 0) + 1
        operation_source = (event['operation_id'], event['source_id'])
        if event['event_type'] == 'action.confirmed' and action and all(operation_source):
            confirmed_actions.setdefault(operation_source, {})[action] = event['event_id']
        elif event['event_type'] == 'operation.succeeded' and all(operation_source):
            confirmations = confirmed_actions.get(operation_source, {})
            if len(confirmations) != 1:
                continue
            for confirmed_action, confirmation_id in confirmations.items():
                completion_key = (*operation_source, confirmed_action)
                if completion_key not in completed_actions:
                    completed_actions.add(completion_key)
                    action_counts[confirmed_action]['completed'] += 1
                    evidence = completion_evidence.setdefault(confirmed_action, [])
                    if len(evidence) < 100:
                        evidence.append({'confirmation_event_id': confirmation_id, 'success_event_id': event['event_id']})
    started = {e['operation_id']: e for e in observed if e['event_type'] == 'operation.started' and e['operation_id']}
    finished = {e['operation_id'] for e in observed if e['event_type'] in {'operation.succeeded', 'operation.failed', 'operation.cancelled', 'operation.interrupted'}}
    unresolved = [e for key, e in started.items() if key not in finished]
    finding('unknown_operation_outcome', unresolved, len(started), 'reconcile_authoritative_artifacts')
    latest_sources = {}
    for event in events:
        if event['event_type'] == 'source.observed' and event['source_id']:
            latest_sources[(event['source_id'], event['stage'])] = event
    pending = [e for e in latest_sources.values() if e['outcome'] in {'unresolved', 'unknown', 'interrupted', 'missing'}]
    pending_by_source = {e['source_id']: e for e in pending if e['source_id']}
    finding('unresolved_source_state', list(pending_by_source.values()), len({e['source_id'] for e in latest_sources.values() if e['source_id']}), 'inspect_source_next_action')
    if pending_by_source:
        findings[-1]['unresolved_stage_counts'] = dict(Counter(e['stage'] or 'unknown' for e in pending))
        findings[-1]['state_evidence_ids'] = [e['event_id'] for e in pending[:100]]
    cohorts = {}
    for event in observed:
        key = (event['source_kind'] or 'unknown', event['build_id'], event['details'].get('backend', 'unknown'), event['stage'] or 'unknown')
        cohort = cohorts.setdefault(key, {'source_kind': key[0], 'build_id': key[1], 'backend': key[2], 'stage': key[3], 'event_count': 0, 'stage_failures': 0, 'elapsed_ms': []})
        cohort['event_count'] += 1
        cohort['stage_failures'] += event['event_type'] == 'stage.failed'
        if event['elapsed_ms'] is not None:
            cohort['elapsed_ms'].append(event['elapsed_ms'])
    for cohort in cohorts.values():
        durations = sorted(cohort.pop('elapsed_ms'))
        cohort['duration_sample_count'] = len(durations)
        cohort['median_elapsed_ms'] = median(durations) if durations else None
    return {'schema_version': SCHEMA_VERSION, 'window_start': _stamp(start), 'window_end': _stamp(end),
            'coverage': _coverage(root), 'sample_counts': {'events': len(events), 'observed_events': len(observed),
            'reconstructed_events': len(events)-len(observed), 'sources': len({e['source_id'] for e in events if e['source_id']}),
            'capture_starts': len(captures), 'operation_starts': len(started)},
            'sample_status': 'sparse' if len(started) < 5 else 'descriptive_only',
            'findings': findings, 'actions': action_counts, 'action_completion_evidence': completion_evidence, 'cohorts': list(cohorts.values()),
            'limits': ['no_accuracy_measurement', 'no_causal_attribution', 'unobserved_external_app_usage',
                       'sources_are_not_unique_real_world_meetings', 'window_boundary_outcomes_may_be_unknown']}


def _review_locked(force=False):
    root = _safe_root(create=True)
    now = _now()
    results = []
    with _db() as connection:
        connection.execute('BEGIN IMMEDIATE')
        anchor = connection.execute("SELECT window_end FROM review_runs WHERE review_id='anchor'").fetchone()
        latest = connection.execute("SELECT window_end FROM review_runs WHERE status='completed' ORDER BY window_end DESC LIMIT 1").fetchone()
        start = datetime.fromisoformat(latest[0] if latest else anchor[0])
        # Reconciliation happens now, possibly after the reporting window ended.
        # Keep its current inventory separate from historical activity counts.
        inventory_rows = connection.execute("SELECT * FROM events WHERE event_type='source.observed' ORDER BY occurred_at,rowid").fetchall()
        current_sources = {}
        for row in inventory_rows:
            if row['source_id']:
                current_sources[(row['source_id'], row['stage'])] = row
        current_pending = [row for row in current_sources.values() if row['outcome'] in {'unresolved','unknown','interrupted','missing'}]
        current_inventory = {'as_of': _stamp(now), 'source_count': len({row['source_id'] for row in current_sources.values()}),
                             'unresolved_source_count': len({row['source_id'] for row in current_pending if row['source_id']}),
                             'unresolved_state_count': len(current_pending),
                             'unresolved_stage_counts': dict(Counter(row['stage'] or 'unknown' for row in current_pending)), 'evidence_ids': [row['event_id'] for row in current_pending[:100]],
                             'evidence_class': 'reconstructed_current_state'}
        while start + timedelta(days=WINDOW_DAYS) <= now:
            end = start + timedelta(days=WINDOW_DAYS)
            review_id = hashlib.sha256((_stamp(start) + '/' + _stamp(end)).encode()).hexdigest()
            rows = connection.execute('SELECT * FROM events WHERE occurred_at>=? AND occurred_at<? ORDER BY occurred_at,rowid', (_stamp(start), _stamp(end))).fetchall()
            result = _summarize(rows, start, end, root)
            result['review_id'] = review_id
            result['current_inventory'] = current_inventory
            connection.execute('INSERT OR IGNORE INTO review_runs VALUES(?,?,?,?,?,?)', (review_id, _stamp(start), _stamp(end), _stamp(now), 'completed', json.dumps(result, sort_keys=True)))
            for item in result['findings']:
                change_id = hashlib.sha256((review_id + item['finding_type']).encode()).hexdigest()
                connection.execute('INSERT OR IGNORE INTO changes VALUES(?,?,?,?,?,?)', (change_id, review_id, item['finding_type'], 'candidate', _stamp(now), json.dumps(item, sort_keys=True)))
            results.append(result)
            start = end
        if force and not results:
            rows = connection.execute('SELECT * FROM events WHERE occurred_at>=? AND occurred_at<? ORDER BY occurred_at,rowid', (_stamp(start), _stamp(now))).fetchall()
            result = _summarize(rows, start, now, root)
            result['review_id'] = 'preview'
            result['current_inventory'] = current_inventory
            result['partial_window'] = True
            results.append(result)
        # Compact lifecycle facts remain; only granular interactions expire.
        cutoff = _stamp(now - timedelta(days=90))
        connection.execute("DELETE FROM events WHERE occurred_at<? AND (event_type LIKE 'action.%' OR event_type IN ('artifact.opened','artifact.exported'))", (cutoff,))
        annual = _stamp(now - timedelta(days=365))
        connection.execute("DELETE FROM review_runs WHERE created_at<? AND status!='anchor'", (annual,))
        connection.execute('DELETE FROM changes WHERE created_at<?', (annual,))
        if results:
            review_root = root / 'reviews'
            if review_root.is_symlink():
                raise OSError('unsafe review output')
            review_root.mkdir(exist_ok=True, mode=0o700)
            review_root.chmod(0o700)
            # Materialize before committing the window. An output failure rolls
            # back its DB completion so a later due check can retry safely.
            for result in results:
                _atomic_json(review_root / (result['review_id'] + '.json'), result)
            for path in review_root.glob('*.json'):
                if not path.is_symlink() and path.stat().st_mtime < (now - timedelta(days=365)).timestamp():
                    path.unlink()
        connection.commit()
    return {'status': 'reviewed' if results else 'not_due', 'reviews': results}


def review(force=False):
    try:
        from meetingintel_runtime import BusyError, operation_lock
        try:
            with operation_lock(kind='review'):
                import meetingintel_actions
                snapshot = meetingintel_actions.reconcile()
                # Operator notices (including another visible operation) are
                # distinct from missing/unreadable measurement evidence.
                if snapshot.get('coverage_warnings', snapshot.get('warnings')):
                    _gap()
                return _review_locked(force=force)
        except BusyError:
            return {'status': 'deferred_busy', 'reviews': []}
    except Exception:
        _gap()
        return {'status': 'failed', 'error_code': 'review_unavailable', 'reviews': []}


def update_change(change_id, state, *, build=None, tests_passed=None, outcome=None, comparison_review_id=None):
    """Record a reviewed change and its later evaluation, never apply software."""
    try:
        _identity(change_id)
        _choice(state, {'candidate', 'deferred', 'rejected', 'applied', 'evaluated'})
        with _db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT * FROM changes WHERE change_id=?', (change_id,)).fetchone()
            if not row:
                raise ValueError('unknown change')
            evaluation = json.loads(row['evaluation'])
            if state == 'applied':
                if row['state'] not in {'candidate', 'deferred'} or not isinstance(build, str) or not re.fullmatch(r'[a-f0-9]{64}', build) or type(tests_passed) is not int or tests_passed < 1:
                    raise ValueError('applied change requires build and passing tests')
                evaluation.update({'applied_build': build, 'tests_passed': tests_passed, 'applied_at': _stamp(_now()), 'evaluation_status': 'pending'})
            elif state == 'evaluated':
                _choice(outcome, {'improved', 'unchanged', 'worse', 'insufficient_evidence'})
                if row['state'] != 'applied' or outcome is None:
                    raise ValueError('only applied changes can be evaluated')
                _identity(comparison_review_id)
                compared = connection.execute("SELECT * FROM review_runs WHERE review_id=? AND status='completed'", (comparison_review_id,)).fetchone()
                if not compared or compared['window_start'] < evaluation['applied_at']:
                    raise ValueError('evaluation requires a subsequent complete window')
                evaluation.update({'evaluation_status': outcome, 'comparison_review_id': comparison_review_id, 'evaluated_at': _stamp(_now())})
            elif row['state'] in {'applied', 'evaluated'}:
                raise ValueError('cannot discard applied change evidence')
            connection.execute('UPDATE changes SET state=?,evaluation=? WHERE change_id=?', (state, json.dumps(evaluation, sort_keys=True), change_id))
            connection.commit()
        return {'status': 'updated', 'change_id': change_id, 'state': state}
    except Exception:
        return {'status': 'failed', 'error_code': 'invalid_change_update'}


def run_cli(argv):
    parser = argparse.ArgumentParser(prog='mi learning')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    review_parser = sub.add_parser('review')
    review_parser.add_argument('--force', action='store_true', help='Preview an incomplete window without advancing the scheduled window')
    change_parser = sub.add_parser('change')
    change_parser.add_argument('change_id')
    change_parser.add_argument('--state', required=True, choices=['candidate','deferred','rejected','applied','evaluated'])
    change_parser.add_argument('--build')
    change_parser.add_argument('--tests-passed', type=int)
    change_parser.add_argument('--outcome', choices=['improved','unchanged','worse','insufficient_evidence'])
    change_parser.add_argument('--comparison-review-id')
    scheduler = sub.add_parser('scheduler')
    scheduler.add_argument('action', choices=['install', 'status', 'uninstall', 'due'])
    args = parser.parse_args(argv)
    if args.command == 'status':
        result = status()
    elif args.command == 'review':
        result = review(force=args.force)
    elif args.command == 'change':
        result = update_change(args.change_id, args.state, build=args.build, tests_passed=args.tests_passed, outcome=args.outcome, comparison_review_id=args.comparison_review_id)
    else:
        import meetingintel_learning_scheduler as scheduler_module
        result = getattr(scheduler_module, args.action)()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if result.get('status') in {'failed', 'degraded'} else 0


if __name__ == '__main__':
    raise SystemExit(run_cli(sys.argv[1:]))
