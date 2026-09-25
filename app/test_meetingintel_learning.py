import contextlib
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

import meetingintel_learning as learning


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name) / 'learning'
        self.env = patch.dict(os.environ, {'MI_LEARNING_ROOT': str(self.root)}, clear=False)
        self.env.start()
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.clock = patch.object(learning, '_now', side_effect=lambda: self.now)
        self.clock.start()
        learning._GAPS = 0
        self.runtime = types.SimpleNamespace(operation_lock=lambda **kw: contextlib.nullcontext(), BusyError=type('BusyError', (RuntimeError,), {}))
        self.modules = patch.dict('sys.modules', {'meetingintel_runtime': self.runtime, 'meetingintel_actions': types.SimpleNamespace(reconcile=lambda: {'warnings': []})})
        self.modules.start()
        self.source = 'a' * 64
        self.operation = 'b' * 32

    def tearDown(self):
        self.modules.stop()
        self.clock.stop()
        self.env.stop()
        self.tmp.cleanup()

    def emit(self, event, **kwargs):
        return learning.emit(event, source_id=self.source, source_kind='laptop_capture', **kwargs)

    def test_read_only_status_does_not_create_storage(self):
        self.assertEqual(learning.status()['status'], 'not_initialized')
        self.assertEqual(learning.source_state(self.source), {})
        self.assertFalse(self.root.exists())

    def test_event_content_origin_permissions_and_source_snapshot_build(self):
        with patch.dict(os.environ, {'MI_ORIGIN': 'gui', 'MI_ACTOR': 'agent', 'MI_OPERATION_ID': self.operation}):
            event_id = self.emit('stage.succeeded', stage='transcription', outcome='succeeded', details={'backend': 'legacy'})
        self.assertIsNotNone(event_id)
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            row = db.execute('SELECT origin,actor,operation_id,outcome,build_id FROM events').fetchone()
        self.assertEqual(row[:4], ('gui', 'agent', self.operation, 'success'))
        self.assertEqual(len(row[4]), 64)
        self.assertEqual((self.root / 'events.sqlite').stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(learning.source_state(self.source)['stage'], 'transcription')

    def test_sensitive_details_and_invalid_values_never_persist(self):
        for payload in ({'title': 'private meeting'}, {'action': 'private_name'}, {'count': True}):
            self.assertIsNone(self.emit('action.selected', details=payload))
        self.assertIsNone(learning.emit('capture.started', source_id='/private/secret/audio.m4a'))
        self.assertIsNone(self.emit('private event'))
        self.assertIsNone(self.emit('stage.started', elapsed_ms=-1))
        self.assertIsNone(self.emit('stage.started', actor='private name'))
        self.assertEqual(learning.status()['coverage']['status'], 'incomplete')
        self.emit('stage.failed', error_code='a private exception including a name')
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            rows = db.execute('SELECT error_code,details FROM events').fetchall()
        self.assertEqual(rows, [('unknown', '{}')])

    def test_database_unavailable_and_locked_do_not_raise(self):
        with patch.object(learning, '_db', side_effect=OSError('disk full')):
            self.assertIsNone(self.emit('capture.started'))
        self.assertTrue((self.root / 'coverage-gap.json').exists())
        self.emit('capture.started')
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            db.execute('BEGIN IMMEDIATE')
            self.assertIsNone(self.emit('capture.stopped'))
        self.assertEqual(learning.status()['status'], 'degraded')

    def test_symlink_root_fails_closed_without_writing_target(self):
        destination = Path(self.tmp.name) / 'private'
        destination.mkdir()
        self.root.symlink_to(destination, target_is_directory=True)
        self.assertIsNone(self.emit('capture.started'))
        self.assertEqual(list(destination.iterdir()), [])

    def test_reconciliation_deduplicates_but_preserves_state_reversion(self):
        snapshot = {'source_id': self.source, 'source_kind': 'laptop_capture', 'stage': 'report', 'outcome': 'available'}
        self.assertEqual(learning.reconcile([snapshot, snapshot]), {'imported': 1, 'failed': 0})
        self.now += timedelta(seconds=1)
        self.assertEqual(learning.reconcile([{**snapshot, 'outcome': 'missing'}])['imported'], 1)
        self.now += timedelta(seconds=1)
        self.assertEqual(learning.reconcile([snapshot])['imported'], 1)
        self.assertEqual(learning.status()['event_count'], 3)
        self.assertEqual(learning.source_state(self.source), {})
        self.assertEqual(learning.reconcile([{**snapshot, 'title': 'private'}])['failed'], 1)

    def test_completed_windows_catch_up_once_and_preview_does_not_advance(self):
        self.emit('capture.started')
        self.now += timedelta(days=2)
        preview = learning.review(force=True)
        self.assertTrue(preview['reviews'][0]['partial_window'])
        self.assertIsNone(learning.status()['last_review'])
        self.now += timedelta(days=28)
        result = learning.review()
        self.assertEqual(len(result['reviews']), 2)
        self.assertEqual(result['reviews'][0]['sample_counts']['capture_starts'], 1)
        self.assertEqual(result['reviews'][1]['sample_counts']['events'], 0)
        self.assertEqual(learning.review()['status'], 'not_due')
        self.assertEqual(len(list((self.root / 'reviews').glob('*.json'))), 3)

    def test_patterns_are_content_free_counted_and_sparse(self):
        self.emit('capture.started')
        self.emit('capture.interrupted', error_code='source_lost')
        self.emit('stage.started', stage='analysis')
        self.emit('stage.failed', stage='analysis', outcome='failed', origin='gui')
        self.now += timedelta(seconds=5)
        self.emit('operation.started', operation_id=self.operation, origin='cli')
        self.emit('action.terminal_handoff', origin='gui', details={'action': 'record'})
        self.now += timedelta(days=14)
        review = learning.review()['reviews'][0]
        findings = {f['finding_type']: f for f in review['findings']}
        self.assertEqual(findings['capture_interruptions']['observed_count'], 1)
        self.assertEqual(findings['stage_failure:analysis']['eligible_count'], 1)
        self.assertEqual(findings['gui_failure_then_cli_attempt']['observed_count'], 1)
        self.assertEqual(findings['unknown_operation_outcome']['observed_count'], 1)
        self.assertEqual(review['sample_status'], 'sparse')
        self.assertEqual(review['actions']['record']['terminal_handoff'], 1)
        self.assertEqual(len(learning.status()['changes']), 4)

    def test_intended_terminal_handoff_does_not_count_as_gui_failure(self):
        self.emit('action.terminal_handoff', origin='gui', details={'action': 'record'})
        self.emit('operation.started', operation_id=self.operation, origin='cli')
        self.emit('operation.succeeded', operation_id=self.operation, origin='cli')
        self.now += timedelta(days=14)
        findings = learning.review()['reviews'][0]['findings']
        self.assertFalse(any(f['finding_type'] == 'gui_failure_then_cli_attempt' for f in findings))

    def test_retention_purges_only_old_interactions_and_one_year_reviews(self):
        self.emit('capture.started')
        self.emit('action.selected', details={'action': 'record'})
        self.now += timedelta(days=100)
        learning.review()
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            self.assertEqual(db.execute('SELECT event_type FROM events').fetchall(), [('capture.started',)])
        self.now += timedelta(days=366)
        learning.review()
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM review_runs WHERE status='anchor'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM changes WHERE created_at<?", ((self.now - timedelta(days=365)).isoformat(),)).fetchone()[0], 0)

    def test_review_defers_while_core_operation_owns_lock(self):
        self.emit('capture.started')
        self.runtime.operation_lock = lambda **kw: (_ for _ in ()).throw(self.runtime.BusyError())
        self.assertEqual(learning.review()['status'], 'deferred_busy')
        self.assertIsNone(learning.status()['last_review'])

    def test_direct_cli_processing_start_counts_switch_once(self):
        self.emit('stage.failed', origin='gui', stage='analysis')
        self.now += timedelta(seconds=1)
        self.emit('stage.started', origin='cli', stage='processing', operation_id=self.operation)
        self.emit('operation.started', origin='cli', operation_id=self.operation)
        self.now += timedelta(days=14)
        findings = learning.review()['reviews'][0]['findings']
        self.assertEqual(next(f['observed_count'] for f in findings if f['finding_type'] == 'gui_failure_then_cli_attempt'), 1)

    def test_review_reconciles_only_after_acquiring_lock(self):
        actions = __import__('meetingintel_actions')
        with patch.object(actions, 'reconcile', return_value={'warnings': []}) as reconcile:
            learning.review()
            reconcile.assert_called_once()
            self.runtime.operation_lock = lambda **kw: (_ for _ in ()).throw(self.runtime.BusyError())
            learning.review()
            self.assertEqual(reconcile.call_count, 1)

    def test_change_evaluation_requires_build_tests_and_later_complete_window(self):
        self.emit('capture.started')
        self.emit('capture.interrupted')
        self.now += timedelta(days=14)
        first = learning.review()['reviews'][0]
        change = learning.status()['changes'][0]
        self.assertEqual(learning.update_change(change['change_id'], 'applied')['status'], 'failed')
        self.assertEqual(learning.update_change(change['change_id'], 'applied', build='c'*64, tests_passed=18)['status'], 'updated')
        self.assertEqual(learning.update_change(change['change_id'], 'evaluated', outcome='improved', comparison_review_id=first['review_id'])['status'], 'failed')
        self.now += timedelta(days=14)
        later = learning.review()['reviews'][0]
        self.assertEqual(learning.update_change(change['change_id'], 'evaluated', outcome='insufficient_evidence', comparison_review_id=later['review_id'])['status'], 'updated')
        self.assertEqual(learning.update_change(change['change_id'], 'candidate')['status'], 'failed')


    def test_review_export_failure_does_not_advance_completed_window(self):
        self.emit('capture.started')
        self.now += timedelta(days=14)
        with patch.object(learning, '_atomic_json', side_effect=OSError('disk full')):
            self.assertEqual(learning.review()['status'], 'failed')
        self.assertIsNone(learning.status()['last_review'])
        self.assertEqual(learning.review()['status'], 'reviewed')


    def test_source_lineage_references_allow_only_bounded_opaque_identities(self):
        self.assertIsNotNone(self.emit('source.observed', details={'source_refs': ['c'*64, '12345678-1234-1234-1234-123456789abc']}))
        self.assertIsNone(self.emit('source.observed', details={'source_refs': ['/private/meeting/audio.m4a']}))
        self.assertIsNone(self.emit('source.observed', details={'source_refs': ['d'*64]*101}))
        self.assertIsNone(self.emit('source.observed', details={'source_refs': [None]}))
        with contextlib.closing(sqlite3.connect(self.root / 'events.sqlite')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM events').fetchone()[0], 1)


    def test_action_completion_requires_exact_confirmed_operation_source_and_deduplicates(self):
        confirmation = self.emit('action.confirmed', operation_id=self.operation, origin='gui', details={'action': 'process'})
        success = self.emit('operation.succeeded', operation_id=self.operation, origin='gui')
        self.emit('operation.succeeded', operation_id=self.operation, origin='gui')
        self.emit('action.confirmed', operation_id='d'*32, details={'action': 'retry'})
        self.emit('operation.succeeded', operation_id='e'*32)
        learning.emit('operation.succeeded', source_id='f'*64, operation_id='d'*32)
        self.emit('action.confirmed', details={'action': 'repair_brief'})
        self.emit('operation.succeeded')
        self.now += timedelta(days=14)
        review = learning.review()['reviews'][0]
        self.assertEqual(review['actions']['process']['completed'], 1)
        self.assertEqual(review['actions']['retry']['completed'], 0)
        self.assertEqual(review['actions']['repair_brief']['completed'], 0)
        self.assertEqual(review['action_completion_evidence']['process'], [{'confirmation_event_id': confirmation, 'success_event_id': success}])

    def test_ambiguous_or_later_confirmation_cannot_invent_action_completion(self):
        self.emit('operation.succeeded', operation_id=self.operation)
        self.emit('action.confirmed', operation_id=self.operation, details={'action': 'retry'})
        self.emit('action.confirmed', operation_id='d'*32, details={'action': 'process'})
        self.emit('action.confirmed', operation_id='d'*32, details={'action': 'repair_brief'})
        self.emit('operation.succeeded', operation_id='d'*32)
        self.now += timedelta(days=14)
        review = learning.review()['reviews'][0]
        self.assertTrue(all(counts['completed'] == 0 for counts in review['actions'].values()))

    def test_duration_cohorts_are_stage_specific_and_use_actual_median(self):
        for stage, duration in [('transcription', 100), ('transcription', 300), ('analysis', 900), ('processing', 1500)]:
            self.emit('stage.succeeded', stage=stage, elapsed_ms=duration, details={'backend': 'legacy'})
        self.now += timedelta(days=14)
        cohorts = {c['stage']: c for c in learning.review()['reviews'][0]['cohorts']}
        self.assertEqual(cohorts['transcription']['duration_sample_count'], 2)
        self.assertEqual(cohorts['transcription']['median_elapsed_ms'], 200)
        self.assertEqual(cohorts['analysis']['median_elapsed_ms'], 900)
        self.assertEqual(cohorts['processing']['median_elapsed_ms'], 1500)

    def test_multistage_inventory_counts_unique_sources_and_separate_states(self):
        snapshots = [{'source_id': self.source, 'source_kind': 'laptop_capture', 'stage': stage, 'outcome': outcome}
                     for stage, outcome in [('capture','interrupted'), ('transcription','success'), ('report','missing'), ('brief','missing'), ('processing','unresolved')]]
        snapshots.append({'source_id': 'd'*64, 'source_kind': 'phone_recording', 'stage': 'processing', 'outcome': 'success'})
        learning.reconcile(snapshots)
        self.now += timedelta(days=14)
        review = learning.review()['reviews'][0]
        inventory = review['current_inventory']
        self.assertEqual(inventory['source_count'], 2)
        self.assertEqual(inventory['unresolved_source_count'], 1)
        self.assertEqual(inventory['unresolved_state_count'], 4)
        self.assertEqual(inventory['unresolved_stage_counts'], {'capture': 1, 'report': 1, 'brief': 1, 'processing': 1})
        finding = next(f for f in review['findings'] if f['finding_type'] == 'unresolved_source_state')
        self.assertEqual(finding['observed_count'], 1)
        self.assertEqual(finding['eligible_count'], 2)
        self.assertEqual(len(finding['state_evidence_ids']), 4)
        self.assertEqual(review['sample_status'], 'sparse')


    def test_operator_busy_notice_does_not_create_measurement_gap(self):
        self.emit('capture.started')
        actions = __import__('meetingintel_actions')
        with patch.object(actions, 'reconcile', return_value={'warnings': ['MeetingIntel review is active'], 'coverage_warnings': []}):
            self.assertEqual(learning.review()['status'], 'not_due')
        self.assertEqual(learning.status()['coverage']['status'], 'observed')
        self.assertFalse((self.root / 'coverage-gap.json').exists())

    def test_real_coverage_warning_still_marks_incomplete_and_legacy_warning_falls_back(self):
        self.emit('capture.started')
        actions = __import__('meetingintel_actions')
        with patch.object(actions, 'reconcile', return_value={'warnings': [], 'coverage_warnings': ['local processing ledger unreadable']}):
            learning.review()
        self.assertTrue((self.root / 'coverage-gap.json').exists())
        (self.root / 'coverage-gap.json').unlink()
        learning._GAPS = 0
        with patch.object(actions, 'reconcile', return_value={'warnings': ['local processing ledger unreadable']}):
            learning.review()
        self.assertEqual(learning.status()['coverage']['status'], 'incomplete')



if __name__ == '__main__':
    unittest.main()
