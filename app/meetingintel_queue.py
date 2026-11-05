"""Private foreground queue of owner-approved exact sources; no scheduler.

The consumer lease is separate from the existing per-job operation/GPU locks.
Use the runtime's private, atomic JSON boundary rather than another data store.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import re
import signal
import time

import meetingintel_inbox as inbox
import meetingintel_pipeline as pipeline
from meetingintel_runtime import (BusyError, RuntimeSafetyError, operation_lock,
    _root, _open_file, _read_json, _write_json)

STATES = {'pending', 'running', 'succeeded', 'failed', 'deferred', 'skipped'}
CODES = {'none', 'busy', 'gpu_deferred', 'already_completed', 'source_changed',
         'validation_failed', 'processing_failed', 'interrupted_unknown'}
FIELDS = {'identity', 'lane', 'created_at', 'fingerprint', 'seal', 'state', 'code', 'backend'}
SHA = re.compile(r'[a-f0-9]{64}\Z')


@contextmanager
def consumer_lock():
    root = _root(create=True)
    fd = _open_file(root, 'queue.lock', create=True)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BusyError('Another queue consumer or approval is active; work is retained.') from error
        yield root
    finally:
        os.close(fd)


def _validate(state):
    if not isinstance(state, dict) or set(state) != {'schema_version', 'jobs'} or type(state['schema_version']) is not int or state['schema_version'] != 1 or not isinstance(state['jobs'], list):
        raise RuntimeSafetyError('Invalid queue state')
    seen = set()
    for job in state['jobs']:
        if not isinstance(job, dict) or set(job) != FIELDS:
            raise RuntimeSafetyError('Invalid queue job')
        if any(not isinstance(value, str) for value in job.values()):
            raise RuntimeSafetyError('Invalid queue field')
        if (job['lane'] not in {'phone', 'laptop', 'meetily'} or job['state'] not in STATES
                or job['code'] not in CODES or job['backend'] not in {'qwen', 'legacy'}
                or not SHA.fullmatch(job['fingerprint']) or not SHA.fullmatch(job['seal'])
                or not job['identity'] or len(job['identity']) > 4096):
            raise RuntimeSafetyError('Invalid queue admission')
        if job['lane'] != 'meetily' and not SHA.fullmatch(job['identity']):
            raise RuntimeSafetyError('Invalid queue source identity')
        try:
            stamp = datetime.fromisoformat(job['created_at'])
            if stamp.tzinfo is None:
                raise ValueError()
        except ValueError as error:
            raise RuntimeSafetyError('Invalid queue recording timestamp') from error
        if job['fingerprint'] in seen:
            raise RuntimeSafetyError('Duplicate queue job')
        seen.add(job['fingerprint'])
    return state


def read_state(root):
    payload = _read_json(root, 'queue.json')
    return _validate({'schema_version': 1, 'jobs': []} if payload is None else payload)


def write_state(root, state):
    _write_json(root, 'queue.json', _validate(state))


def _item(job):
    return inbox.InboxItem(datetime.fromisoformat(job['created_at']), job['lane'],
        'Recording', '', '', 'ready', job['identity'])


def _hash_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Source evidence unavailable')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


def resolve(item):
    """Validate through the existing exact admission gates and seal input bytes."""
    import meetingintel_cli as cli
    if item.source_kind == 'phone':
        source = cli._load_exact_phone_source(item)
        production = pipeline.production_source_from_audio_first(source)
    elif item.source_kind == 'laptop':
        source = cli._load_exact_laptop_source(item)
        production = pipeline.production_source_from_laptop_capture(source)
    elif item.source_kind == 'meetily':
        folder = cli._exact_direct_child(Path(item.identity), cli.MEETINGS_ROOT, 'Meetily source')
        paths = [folder / name for name in ('metadata.json', 'transcripts.json')]
        seal = _hash_files(paths)
        metadata, _, transcript_hash, text, created = pipeline.parse_meeting_folder(folder)
        if metadata.get('status') != 'completed' or not text or created != item.created_at:
            raise ValueError('Meetily source is no longer ready')
        fp = pipeline.build_fingerprint(folder.name, metadata['created_at'], transcript_hash)
        return {'fingerprint': fp, 'seal': seal}, folder
    else:
        raise ValueError('Unsupported queue lane')
    if production.created_at != item.created_at:
        raise ValueError('Recording timestamp changed')
    paths = [source.source_folder / 'meeting_source.json', source.canonical_audio_path]
    if item.source_kind == 'laptop':
        paths.append(source.capture_manifest_path)
    return {'fingerprint': production.fingerprint, 'seal': _hash_files(paths)}, source


def preview(report, target_date, *, resolver=None):
    resolver = resolver or resolve
    jobs, rows, seen = [], [], set()
    for item in sorted(report.items, key=lambda i: (i.created_at, i.source_kind, i.identity)):
        if item.created_at.astimezone(inbox.IST).date() != target_date:
            continue
        label = item.created_at.astimezone(inbox.IST).strftime('%Y-%m-%d %H:%M %Z') + ' | ' + item.source_kind
        if item.category == 'completed':
            rows.append(label + ' | completed/skipped')
            continue
        if item.category != 'ready':
            next_step = ('mi sync status; wait for settlement, then mi inbox for grouping/normalization'
                         if item.source_kind == 'phone' else 'mi inbox; review normalization/recovery evidence separately')
            rows.append(label + ' | blocked: ' + item.status + ' | Next: ' + next_step)
            continue
        try:
            facts, _ = resolver(item)
        except Exception:
            rows.append(label + ' | blocked: exact validation failed | Next: mi inbox')
            continue
        if facts['fingerprint'] in seen:
            continue
        seen.add(facts['fingerprint'])
        jobs.append(dict(identity=item.identity, lane=item.source_kind, created_at=item.created_at.isoformat(),
                         **facts, state='pending', code='none', backend='qwen'))
        rows.append(f"{len(jobs)}. {label} | ready | {facts['fingerprint'][:12]}")
    return jobs, rows


def admit(root, state, jobs, backend):
    existing = {j['fingerprint'] for j in state['jobs']}
    for frozen in jobs:
        if frozen['fingerprint'] not in existing:
            state['jobs'].append(dict(frozen, backend=(pipeline.LAPTOP_TRANSCRIPTION_BACKEND_DEFAULT if frozen['lane'] == 'laptop' else ('legacy' if frozen['lane'] == 'meetily' else backend))))
            existing.add(frozen['fingerprint'])
    write_state(root, state)


def completed(job, *, delivery=False):
    """Ledger is the dedup boundary; success also requires durable delivery."""
    import meetingintel_cli as cli
    ledger = pipeline.load_ledger(cli.LEDGER_PATH)
    record = ledger['records'].get(job['fingerprint'])
    if record is None and job['lane'] == 'meetily':
        record = next((r for r in ledger['records'].values() if r.get('folder_path') == job['identity']), None)
    if record is None:
        return False
    if not delivery:
        return True
    import meetingintel_actions as actions
    transcript = actions._safe_file(record.get('transcript_path'), cli.OUTPUT_DIR)
    report = actions._safe_file(record.get('layer2_report_path'), cli.OUTPUT_DIR)
    if not transcript or not report or not transcript.stat().st_size or not report.stat().st_size:
        return False
    if hashlib.sha256(transcript.read_bytes()).hexdigest() != record.get('transcript_sha256'):
        return False
    from datetime import date
    target = date.fromisoformat(record['meeting_date_ist'])
    brief = actions._safe_file(cli.OUTPUT_DIR / ('brief_input_' + target.isoformat() + '.md'), cli.OUTPUT_DIR)
    return bool(brief and brief.read_text() == pipeline.render_brief(target, pipeline.collect_today_records(ledger, target)))


def execute(job, source):
    import meetingintel_cli as cli
    clock = pipeline.now_ist
    if job['lane'] == 'phone':
        code = cli._run_exact_phone_sources((source,), clock=clock, pipeline_runner=pipeline.process_meetings,
            ollama_manager=None, suppress_private_paths=True, phone_transcription_backend=job['backend'])
    elif job['lane'] == 'laptop':
        with pipeline.selected_laptop_backend(job['backend']):
            code = cli._run_exact_laptop_source(source, clock=clock, ollama_manager=None, suppress_private_paths=True)
    else:
        candidate = cli.SourceCandidate(job['fingerprint'], datetime.fromisoformat(job['created_at']), 'meetily', 'Meetily recording')
        code = cli._run_production((candidate,), scope='exact queued recording', processed=set(), clock=clock,
            pipeline_runner=pipeline.process_meetings, ollama_manager=None, audio_first_only=False,
            explicit_meeting_folder=source, suppress_private_paths=True)
    return code == 0 and completed(job, delivery=True)


def retry_failed(root, state):
    for job in state['jobs']:
        if job['state'] in {'failed', 'running'}:
            job.update(state='pending', code='none')
    write_state(root, state)


def consume(root, *, resolver=None, executor=None, completed=None, wait_seconds=600, sleeper=time.sleep, only=None):
    resolver, executor = resolver or resolve, executor or execute
    is_completed = completed or globals()['completed']
    state = read_state(root)
    # A released consumer lease never proves a previous running attempt finished.
    # It may have persisted a record before dying; never rerun it automatically.
    for job in state['jobs']:
        if job['state'] == 'running':
            job.update(state='failed', code='interrupted_unknown')
    write_state(root, state)
    for job in state['jobs']:
        if job['state'] not in {'pending', 'deferred'} or (only is not None and job['fingerprint'] not in only):
            continue
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                with operation_lock('processing'):
                    if is_completed(job):
                        job.update(state='skipped', code='already_completed')
                        break
                    try:
                        facts, source = resolver(_item(job))
                    except Exception:
                        job.update(state='failed', code='validation_failed')
                        break
                    if facts != {key: job[key] for key in ('fingerprint', 'seal')}:
                        job.update(state='failed', code='source_changed')
                        break
                    job.update(state='running', code='none')
                    write_state(root, state)  # Must be durable before any model work.
                    print(f"Queue: processing {job['lane']} {job['fingerprint'][:12]}", flush=True)
                    try:
                        success = executor(job, source)
                    except pipeline.gpu.Deferred:
                        job.update(state='deferred', code='gpu_deferred')
                        write_state(root, state)
                        return 75
                    except Exception:
                        job.update(state='failed', code='processing_failed')
                    else:
                        job.update(state='succeeded' if success else 'failed', code='none' if success else 'processing_failed')
                    break
            except BusyError:
                if job['code'] != 'busy':
                    print('Queue waiting for active capture/processing; work retained (up to ten minutes).', flush=True)
                job.update(state='pending', code='busy')
                write_state(root, state)
                if time.monotonic() >= deadline:
                    job.update(state='deferred', code='busy')
                    write_state(root, state)
                    return 75
                sleeper(min(5, max(0, deadline-time.monotonic())))
        write_state(root, state)
        print(f"Queue: {job['state']} {job['fingerprint'][:12]} ({job['code']})", flush=True)
    return 1 if any(j['state'] == 'failed' for j in state['jobs']) else 0


def status_text():
    root = _root(create=False)
    state = read_state(root)
    if not state['jobs']:
        return 'Queue is empty.'
    active = False
    try:
        fd = _open_file(root, 'queue.lock')
    except FileNotFoundError:
        pass
    else:
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
        finally:
            os.close(fd)
    lines = ['Queue consumer: ' + ('active (or approval open)' if active else 'stopped'),
             'Order | recording time (' + str(inbox.IST) + ') | lane | source | state | diagnostic']
    for index, job in enumerate(state['jobs'], 1):
        stamp = datetime.fromisoformat(job['created_at']).astimezone(inbox.IST).strftime('%Y-%m-%d %H:%M %Z')
        code = 'interrupted_unknown' if job['state'] == 'running' and not active else job['code']
        lines.append(f"{index} | {stamp} | {job['lane']} | {job['fingerprint'][:12]} | {job['state']} | {code}")
    lines.append('Resume pending/deferred: mi queue resume. Explicit failed/ambiguous retry: mi queue resume --retry-failed.')
    return '\n'.join(lines)


@contextmanager
def _interruptions():
    previous = {}
    def stop(_signum, _frame):
        raise KeyboardInterrupt()
    try:
        for sig in (signal.SIGHUP, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_cli(argv, *, clock=pipeline.now_ist, input_func=input, terminal_isatty=lambda: False,
            backend=pipeline.PHONE_TRANSCRIPTION_BACKEND_DEFAULT):
    command = list(argv)
    if command == ['status']:
        print(status_text())
        return 0
    if command not in (['today'], ['resume'], ['resume', '--retry-failed']):
        print('Usage: mi queue today | status | resume [--retry-failed]')
        return 2
    try:
        with consumer_lock() as root:
            state = read_state(root)
            only = None
            if command == ['today']:
                selected_now = clock()
                report = inbox.build_inbox(clock=lambda: selected_now)
                jobs, rows = preview(report, selected_now.astimezone(inbox.IST).date())
                existing = {j['fingerprint'] for j in state['jobs']}
                jobs = [j for j in jobs if j['fingerprint'] not in existing]
                print('TODAY QUEUE PREVIEW — Asia/Kolkata; oldest first')
                print('\n'.join(rows) or 'No recordings for today.')
                for warning in report.warnings:
                    print('Inventory warning: source readiness is incomplete; review mi inbox.')
                if any('ledger' in warning for warning in report.warnings):
                    print('Queue admission stopped until ledger warnings are resolved.')
                    return 1
                print(f'New exact sources selected: {len(jobs)}. Phone backend: {backend}.')
                for index, job in enumerate(jobs, 1):
                    print(f"  Approve {index}: {job['created_at']} | {job['lane']} | {job['fingerprint'][:12]}")
                if state['jobs']:
                    print('Existing queue entries are retained without duplication. Use mi queue status/resume for them.')
                if not jobs:
                    return 0
                print('Keep this Terminal open and the Mac awake. Closure/crash stops advancement; restart needs mi queue resume.')
                if not terminal_isatty() or input_func('Approve this exact set together? Type y: ') != 'y':
                    print('Queue not changed.')
                    return 0
                admit(root, state, jobs, backend)
                # Today's approval cannot expand to an older approved backlog.
                only = {j['fingerprint'] for j in jobs}
            elif '--retry-failed' in command:
                print(status_text())
                print('Retry all failed/ambiguous entries shown above using their original frozen sources. Accepted stages may be reused.')
                if not terminal_isatty() or input_func('Approve these retries? Type y: ') != 'y':
                    print('Queue not changed.')
                    return 0
                retry_failed(root, state)
            with _interruptions():
                result = consume(root, only=only)
            print('Queue pass finished. Use mi queue status for all entries and safe diagnostics.')
            return result
    except (KeyboardInterrupt, EOFError):
        print('\nQueue stopped; pending work is retained. Running work has an ambiguous outcome and requires explicit retry review.')
        return 130
    except BusyError:
        print('Another queue consumer or approval is active. Work is retained; mi queue status is read-only.')
        return 75
    except (RuntimeSafetyError, OSError, ValueError):
        print('Queue stopped safely: local state or source validation unavailable. Use mi queue status and mi inbox.')
        return 1
