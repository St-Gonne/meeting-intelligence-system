import json
import os
from pathlib import Path
import plistlib
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import meetingintel_learning_scheduler as scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name) / 'learning'
        self.plist = Path(self.tmp.name) / 'LaunchAgents' / (scheduler.LABEL + '.plist')
        self.env = patch.dict(os.environ, {'MI_LEARNING_ROOT': str(self.root)})
        self.env.start()
        self.path_patch = patch.object(scheduler, '_plist_path', return_value=self.plist)
        self.path_patch.start()
        self.calls = []
        self.launch = patch.object(scheduler, '_launchctl', side_effect=self.launchctl)
        self.launch.start()

    def launchctl(self, args):
        self.calls.append(args)
        return subprocess.CompletedProcess(args, 0, '', '')

    def tearDown(self):
        self.launch.stop()
        self.path_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_status_does_not_install_or_create(self):
        self.assertEqual(scheduler.status()['status'], 'not_installed')
        self.assertFalse(self.root.exists())
        self.assertFalse(self.plist.exists())
        self.assertEqual(self.calls, [])

    def test_explicit_install_is_private_daily_and_idempotent(self):
        with patch.object(scheduler.sys, 'platform', 'darwin'):
            self.assertEqual(scheduler.install()['status'], 'installed')
            self.assertEqual(scheduler.install()['status'], 'installed')
        payload = plistlib.loads(self.plist.read_bytes())
        self.assertEqual(payload['StartInterval'], 86400)
        self.assertTrue(payload['RunAtLoad'])
        self.assertEqual(payload['EnvironmentVariables']['MI_ORIGIN'], 'scheduled')
        self.assertEqual(payload['ProgramArguments'][1], str(self.root / 'scheduler/runner.py'))
        self.assertEqual(sum(args[0] == 'bootstrap' for args in self.calls), 1)
        self.assertEqual(self.plist.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / 'scheduler/runner.py').stat().st_mode & 0o777, 0o600)

    def test_uninstall_keeps_observations_and_receipts(self):
        with patch.object(scheduler.sys, 'platform', 'darwin'):
            scheduler.install()
        private = self.root / 'events.sqlite'
        private.write_bytes(b'keep')
        self.assertEqual(scheduler.uninstall()['status'], 'not_installed')
        self.assertFalse(self.plist.exists())
        self.assertEqual(private.read_bytes(), b'keep')

    def test_unowned_plist_and_symlink_refused(self):
        self.plist.parent.mkdir()
        self.plist.write_bytes(plistlib.dumps({'Label': 'other', 'ProgramArguments': ['other']}))
        before = self.plist.read_bytes()
        self.assertEqual(scheduler.uninstall()['status'], 'failed')
        self.assertEqual(self.plist.read_bytes(), before)
        self.plist.unlink()
        target = Path(self.tmp.name) / 'target'
        target.write_text('keep')
        self.plist.symlink_to(target)
        with patch.object(scheduler.sys, 'platform', 'darwin'):
            self.assertEqual(scheduler.install()['status'], 'failed')
        self.assertEqual(target.read_text(), 'keep')

    def test_due_receipts_busy_and_success_have_distinct_categories(self):
        import meetingintel_learning
        with patch.object(meetingintel_learning, 'review', return_value={'status': 'deferred_busy', 'reviews': []}):
            self.assertEqual(scheduler.due()['status'], 'deferred_busy')
        receipt = json.loads((self.root / 'scheduler-receipt.json').read_text())
        self.assertEqual(receipt['category'], 'deferred_busy')
        self.assertEqual(receipt['stage'], 'completed')
        with patch.object(meetingintel_learning, 'review', return_value={'status': 'not_due', 'reviews': []}):
            self.assertEqual(scheduler.due()['status'], 'not_due')
        self.assertEqual(json.loads((self.root / 'scheduler-receipt.json').read_text())['category'], 'not_due')

    def test_due_import_or_review_failure_is_observable_without_exception_text(self):
        import meetingintel_learning
        with patch.object(meetingintel_learning, 'review', side_effect=OSError('private audio location')):
            self.assertEqual(scheduler.due()['status'], 'failed')
        text = (self.root / 'scheduler-receipt.json').read_text()
        self.assertNotIn('private audio', text)
        self.assertEqual(json.loads(text)['category'], 'review_unavailable')

    def test_failed_bootout_does_not_remove_live_owned_schedule(self):
        with patch.object(scheduler.sys, 'platform', 'darwin'):
            scheduler.install()
        with patch.object(scheduler, '_launchctl', side_effect=[subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]):
            self.assertEqual(scheduler.uninstall()['error_code'], 'scheduler_bootout_failed')
        self.assertTrue(self.plist.exists())

    def test_install_preserves_existing_launchagents_directory_mode(self):
        self.plist.parent.mkdir(mode=0o755)
        before = self.plist.parent.stat().st_mode & 0o777
        with patch.object(scheduler.sys, 'platform', 'darwin'):
            scheduler.install()
        self.assertEqual(self.plist.parent.stat().st_mode & 0o777, before)



if __name__ == '__main__':
    unittest.main()
