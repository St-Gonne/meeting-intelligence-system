"""Synthetic queue admission, persistence and interruption regressions; no models."""
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import meetingintel_inbox as inbox
import meetingintel_queue as queue
from meetingintel_runtime import BusyError

NOW = datetime(2026, 10, 23, 12, tzinfo=inbox.IST)


def item(key='a', kind='phone', category='ready', day=NOW):
    return inbox.InboxItem(day, kind, 'Private title must not appear',
        {'phone': 'phone normalized and ready', 'laptop': 'laptop normalized and ready',
         'meetily': 'meetily ready'}[kind], 'Prepare using mi inbox', category, key * 64)


def resolved(value):
    fp = value.identity if value.source_kind != 'laptop' else 'c' * 64
    return {'fingerprint': fp, 'seal': 'd' * 64}, object()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.env = patch.dict(os.environ, MI_RUNTIME_ROOT=str(Path(self.temp.name) / 'runtime'))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)

    def preview(self, values):
        report = inbox.InboxReport(tuple(values), {}, today_date=NOW.date())
        return queue.preview(report, NOW.date(), resolver=resolved)

    def save(self, values):
        jobs, _ = self.preview(values)
        with queue.consumer_lock() as root:
            state = queue.read_state(root)
            queue.admit(root, state, jobs, 'qwen')
        return state

    def consume(self, **kwargs):
        with queue.consumer_lock() as root:
            return queue.consume(root, resolver=resolved, completed=lambda *_: False,
                                 executor=lambda *_: True, **kwargs)

    def test_today_ist_order_blocked_and_completed_visible_no_old_backlog(self):
        late = item('b', 'laptop')
        early = item('a', day=NOW - timedelta(hours=1))
        blocked = item('e', category='waiting')
        done = item('f', category='completed')
        jobs, rows = self.preview([late, done, blocked, early, item('9', day=NOW-timedelta(days=1))])
        self.assertEqual([j['identity'] for j in jobs], ['a'*64, 'b'*64])
        self.assertEqual(len(rows), 4)
        self.assertIn('blocked', '\n'.join(rows))
        self.assertIn('completed/skipped', '\n'.join(rows))
        self.assertNotIn('Private title', '\n'.join(rows))
        # UTC previous day at 20:00 is today's 01:30 IST.
        utc = datetime(2026, 10, 22, 20, tzinfo=timezone.utc)
        self.assertEqual(len(self.preview([item(day=utc)])[0]), 1)

    def test_freeze_revalidation_failure_continues_independent_job(self):
        self.save([item(), item('b')])
        def changed(value):
            facts, source = resolved(value)
            if value.identity == 'a'*64:
                facts['seal'] = 'e'*64
            return facts, source
        ran = []
        with queue.consumer_lock() as root:
            queue.consume(root, resolver=changed, completed=lambda *_: False,
                          executor=lambda job, _: ran.append(job['identity']) or True)
            jobs = queue.read_state(root)['jobs']
        self.assertEqual(ran, ['b'*64])
        self.assertEqual([j['state'] for j in jobs], ['failed', 'succeeded'])
        self.assertEqual(jobs[0]['code'], 'source_changed')

    def test_dedup_and_completed_before_run(self):
        self.save([item(), item()])
        self.save([item()])
        with queue.consumer_lock() as root:
            self.assertEqual(len(queue.read_state(root)['jobs']), 1)
            queue.consume(root, resolver=resolved, completed=lambda *_: True,
                          executor=lambda *_: self.fail('completed source launched'))
            job = queue.read_state(root)['jobs'][0]
        self.assertEqual((job['state'], job['code']), ('skipped', 'already_completed'))

    def test_busy_retains_work_no_model_and_competing_consumer_rejected(self):
        self.save([item()])
        with queue.consumer_lock() as root:
            with self.assertRaises(BusyError):
                with queue.consumer_lock():
                    pass
            with patch.object(queue, 'operation_lock', side_effect=BusyError('private error')):
                queue.consume(root, resolver=resolved, completed=lambda *_: False,
                              executor=lambda *_: self.fail('busy launched'), wait_seconds=0)
            self.assertEqual(queue.read_state(root)['jobs'][0]['state'], 'deferred')
        self.assertEqual(self.consume(), 0)

    def test_crash_running_not_retried_and_pending_continues(self):
        self.save([item(), item('b')])
        with queue.consumer_lock() as root:
            state = queue.read_state(root)
            state['jobs'][0]['state'] = 'running'
            queue.write_state(root, state)
        calls = []
        with queue.consumer_lock() as root:
            queue.consume(root, resolver=resolved, completed=lambda *_: False,
                          executor=lambda job, _: calls.append(job['identity']) or True)
            jobs = queue.read_state(root)['jobs']
        self.assertEqual(calls, ['b'*64])
        self.assertEqual((jobs[0]['state'], jobs[0]['code']), ('failed', 'interrupted_unknown'))

    def test_failure_and_interrupt_never_false_success_or_raw_diagnostic(self):
        self.save([item(), item('b')])
        def fail(*_):
            raise RuntimeError('PRIVATE transcript secret')
        with queue.consumer_lock() as root:
            queue.consume(root, resolver=resolved, completed=lambda *_: False, executor=fail)
            state = queue.read_state(root)
            self.assertEqual([j['state'] for j in state['jobs']], ['failed', 'failed'])
            self.assertNotIn('PRIVATE', (root/'queue.json').read_text())
            queue.retry_failed(root, state)
            with self.assertRaises(KeyboardInterrupt):
                queue.consume(root, resolver=resolved, completed=lambda *_: False,
                              executor=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            state = queue.read_state(root)
            self.assertEqual(state['jobs'][0]['state'], 'running')
            self.assertEqual(state['jobs'][1]['state'], 'pending')

    def test_status_read_only_missing_and_unsafe_state_rejected(self):
        self.assertEqual(queue.status_text(), 'Queue is empty.')
        self.assertFalse((Path(self.temp.name)/'runtime').exists())
        self.save([item()])
        with queue.consumer_lock() as root:
            state = queue.read_state(root)
            state['jobs'][0]['state'] = 'invented'
            with self.assertRaises(queue.RuntimeSafetyError):
                queue.write_state(root, state)

    def test_cli_confirmation_decline_and_frozen_set_no_late_addition(self):
        report = inbox.InboxReport((item(),), {}, today_date=NOW.date())
        with patch.object(queue.inbox, 'build_inbox', return_value=report), patch.object(queue, 'resolve', side_effect=resolved):
            self.assertEqual(queue.run_cli(['today'], clock=lambda: NOW, input_func=lambda _: 'n', terminal_isatty=lambda: True), 0)
            self.assertEqual(queue.status_text(), 'Queue is empty.')
            with patch.object(queue, 'consume', return_value=0) as consumer:
                self.assertEqual(queue.run_cli(['today'], clock=lambda: NOW, input_func=lambda _: 'y', terminal_isatty=lambda: True), 0)
                self.assertEqual(consumer.call_count, 1)
        with queue.consumer_lock() as root:
            jobs = queue.read_state(root)['jobs']
        self.assertEqual([j['identity'] for j in jobs], ['a'*64])
        self.assertEqual(jobs[0]['backend'], 'qwen')

    def test_cli_dispatch_preserves_backend_without_legacy_discovery(self):
        import meetingintel_cli as cli
        with patch.object(queue, 'run_cli', return_value=0) as run, patch.object(cli, 'discover_candidates') as discover:
            self.assertEqual(cli.run_cli(['--phone-transcription-backend', 'legacy', 'queue', 'today'], clock=lambda: NOW), 0)
            self.assertEqual(run.call_args.args[0], ['today'])
            self.assertEqual(run.call_args.kwargs['backend'], 'legacy')
            discover.assert_not_called()

    def test_real_shared_operation_lock_blocks_queue_until_manual_work_finishes(self):
        self.save([item()])
        entered, release = threading.Event(), threading.Event()
        def manual():
            with queue.operation_lock('processing'):
                entered.set()
                release.wait(5)
        thread = threading.Thread(target=manual)
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            with queue.consumer_lock() as root:
                queue.consume(root, resolver=resolved, completed=lambda *_: False,
                    executor=lambda *_: self.fail('manual operation overlapped'), wait_seconds=0)
                self.assertEqual(queue.read_state(root)['jobs'][0]['state'], 'deferred')
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(self.consume(), 0)

    def test_gpu_deferred_is_durable_without_automatic_retry(self):
        self.save([item(), item('b')])
        calls = []
        def gpu_busy(job, _):
            calls.append(job['identity'])
            raise queue.pipeline.gpu.Deferred('PRIVATE internal reason')
        with queue.consumer_lock() as root:
            self.assertEqual(queue.consume(root, resolver=resolved, completed=lambda *_: False, executor=gpu_busy), 75)
            jobs = queue.read_state(root)['jobs']
            self.assertEqual([j['state'] for j in jobs], ['deferred', 'pending'])
            self.assertNotIn('PRIVATE', (root/'queue.json').read_text())
        self.assertEqual(calls, ['a'*64])

    def test_new_batch_does_not_process_old_pending_backlog(self):
        self.save([item('a', day=NOW-timedelta(days=1))])  # preview excludes it
        # Persist an older owner-approved job independently of today's selector.
        jobs, _ = self.preview([item('a')])
        with queue.consumer_lock() as root:
            state = queue.read_state(root)
            queue.admit(root, state, jobs, 'qwen')
        self.save([item('b')])
        calls = []
        with queue.consumer_lock() as root:
            queue.consume(root, resolver=resolved, completed=lambda *_: False,
                executor=lambda job, _: calls.append(job['identity']) or True, only={'b'*64})
            self.assertEqual(queue.read_state(root)['jobs'][0]['state'], 'pending')
        self.assertEqual(calls, ['b'*64])

    def test_status_does_not_rewrite_abandoned_running_and_retry_needs_confirmation(self):
        self.save([item()])
        with queue.consumer_lock() as root:
            state = queue.read_state(root)
            state['jobs'][0]['state'] = 'running'
            queue.write_state(root, state)
            before = (root/'queue.json').read_bytes()
        self.assertIn('interrupted_unknown', queue.status_text())
        self.assertEqual(queue.run_cli(['resume', '--retry-failed'], input_func=lambda _: 'n', terminal_isatty=lambda: True), 0)
        self.assertEqual((root/'queue.json').read_bytes(), before)

    def test_approved_grouped_phone_uses_exact_loader_and_source_bytes_are_frozen(self):
        import meetingintel_cli as cli
        folder = Path(self.temp.name)/'phone'
        folder.mkdir()
        (folder/'meeting_source.json').write_text('synthetic grouping provenance')
        (folder/'audio.m4a').write_bytes(b'not decoded: loader is mocked')
        source = SimpleNamespace(source_folder=folder, canonical_audio_path=folder/'audio.m4a', group_fingerprint='a'*64)
        production = SimpleNamespace(fingerprint='a'*64, created_at=NOW)
        with patch.object(cli, '_load_exact_phone_source', return_value=source) as loader, patch.object(queue.pipeline, 'production_source_from_audio_first', return_value=production):
            first, result = queue.resolve(item())
            self.assertIs(result, source)
            loader.assert_called_once()
            (folder/'audio.m4a').write_bytes(b'changed exact source')
            second, _ = queue.resolve(item())
        self.assertEqual(first['fingerprint'], second['fingerprint'])
        self.assertNotEqual(first['seal'], second['seal'])

    def test_execute_reuses_exact_lanes_no_prompts_and_delivery_required(self):
        import meetingintel_cli as cli
        for lane, method in [('phone', '_run_exact_phone_sources'), ('laptop', '_run_exact_laptop_source'), ('meetily', '_run_production')]:
            jobs, _ = self.preview([item(kind=lane)])
            job = dict(jobs[0], backend='legacy')
            source = object()
            with self.subTest(lane=lane), patch.object(cli, method, return_value=0) as runner, patch.object(queue, 'completed', return_value=False) as delivered:
                self.assertFalse(queue.execute(job, source))
                delivered.assert_called_once_with(job, delivery=True)
                runner.assert_called_once()
                if lane == 'phone':
                    self.assertEqual(runner.call_args.args[0], (source,))
                    self.assertEqual(runner.call_args.kwargs['phone_transcription_backend'], 'legacy')
                elif lane == 'meetily':
                    self.assertIs(runner.call_args.kwargs['explicit_meeting_folder'], source)

    def test_delivery_requires_checksum_report_and_exact_source_day_brief(self):
        import hashlib
        import meetingintel_cli as cli
        jobs, _ = self.preview([item()])
        root = Path(self.temp.name)/'output'
        root.mkdir()
        transcript, report = root/'transcript.txt', root/'report.md'
        transcript.write_text('synthetic anonymous transcript')
        report.write_text('synthetic report')
        brief = root/'brief_input_2026-10-23.md'
        record = dict(transcript_path=str(transcript), layer2_report_path=str(report), meeting_date_ist='2026-10-23',
                      transcript_sha256=hashlib.sha256(transcript.read_bytes()).hexdigest())
        ledger = {'records': {'a'*64: record}}
        with patch.object(cli, 'OUTPUT_DIR', root), patch.object(queue.pipeline, 'load_ledger', return_value=ledger), patch.object(queue.pipeline, 'render_brief', return_value='expected brief'), patch.object(queue.pipeline, 'collect_today_records', return_value=[]):
            self.assertTrue(queue.completed(jobs[0]))
            self.assertFalse(queue.completed(jobs[0], delivery=True))
            brief.write_text('expected brief')
            self.assertTrue(queue.completed(jobs[0], delivery=True))
            transcript.write_text('tampered')
            self.assertFalse(queue.completed(jobs[0], delivery=True))

    def test_process_death_releases_consumer_but_never_replays_running_job(self):
        self.save([item(), item('b')])
        script = """
import os
import meetingintel_queue as q
with q.consumer_lock() as root:
    def resolve(item):
        return {'fingerprint': item.identity, 'seal': 'd'*64}, None
    q.consume(root, resolver=resolve, completed=lambda job: False,
              executor=lambda job, source: os._exit(23))
"""
        process = subprocess.run([sys.executable, '-c', script], capture_output=True)
        self.assertEqual(process.returncode, 23)
        self.assertIn('interrupted_unknown', queue.status_text())
        calls = []
        with queue.consumer_lock() as root:
            queue.consume(root, resolver=resolved, completed=lambda *_: False,
                executor=lambda job, _: calls.append(job['identity']) or True)
            jobs = queue.read_state(root)['jobs']
        self.assertEqual(calls, ['b'*64])
        self.assertEqual(jobs[0]['code'], 'interrupted_unknown')

    def test_malformed_empty_state_and_symlink_rejected(self):
        with queue.consumer_lock() as root:
            queue._write_json(root, 'queue.json', {})
            with self.assertRaises(queue.RuntimeSafetyError):
                queue.read_state(root)
            (root/'queue.json').unlink()
            (root/'queue.json').symlink_to(Path(self.temp.name)/'outside')
            with self.assertRaises(queue.RuntimeSafetyError):
                queue.read_state(root)


if __name__ == '__main__':
    unittest.main()
