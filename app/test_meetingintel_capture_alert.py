import hashlib
import json
import os
import plistlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import meetingintel_capture_alert as alerts


class CaptureAlertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name)
        self.session = self.root / 'session'
        self.session.mkdir(mode=0o700)
        self.capture_id = '12345678-1234-1234-1234-123456789abc'
        self.manifest = self.session / 'capture_manifest.json'
        self.manifest.write_text(json.dumps({'capture_id': self.capture_id, 'status': 'interrupted'}))
        self.manifest.chmod(0o600)
        self.original_manifest = self.manifest.read_bytes()
        self.env = patch.dict(os.environ, {'MI_RUNTIME_ROOT': str(self.root / 'runtime'), 'MI_LEARNING_ROOT': str(self.root / 'learning')})
        self.env.start()
        self.events = []
        self.emitter = patch.object(alerts, '_emit', side_effect=lambda kind, capture_id, **fields: self.events.append((kind, fields)))
        self.emitter.start()

    def tearDown(self):
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)
        self.emitter.stop()
        self.env.stop()
        self.tmp.cleanup()

    def cached_binary(self, key, payload):
        bundle, helper, info_path, receipt_path = alerts._bundle_paths(alerts._cache(), key)
        helper.parent.mkdir(parents=True, mode=0o700)
        bundle.chmod(0o700)
        (bundle / 'Contents').chmod(0o700)
        helper.write_bytes(payload)
        helper.chmod(0o700)
        info_path.write_bytes(plistlib.dumps(alerts._bundle_info()))
        info_path.chmod(0o600)
        alerts._atomic_json(receipt_path, {'schema_version': 1, 'source_key': key, 'binary_sha256': alerts._hash(helper)})
        return helper

    def helper(self, body):
        key = hashlib.sha256(body.encode()).hexdigest()
        return self.cached_binary(key, ('#!' + sys.executable + '\n' + body).encode())

    def receipt(self):
        return json.loads((self.session / 'capture_alert.json').read_text())

    def test_preparation_compile_failure_and_timeout_fall_back(self):
        process = Mock()
        process.wait.return_value = 1
        process.poll.return_value = 1
        with patch.object(alerts.sys, 'platform', 'darwin'), patch.object(alerts.subprocess, 'Popen', return_value=process):
            self.assertIsNone(alerts.prepare())
        process.wait.side_effect = subprocess.TimeoutExpired('compiler', 30)
        with patch.object(alerts.sys, 'platform', 'darwin'), patch.object(alerts.subprocess, 'Popen', return_value=process), patch.object(alerts, '_terminate') as terminate:
            self.assertIsNone(alerts.prepare())
            terminate.assert_called_once_with(process)
        self.assertFalse((self.session / 'capture_alert.json').exists())

    def test_prepare_reuses_only_hash_verified_private_cache(self):
        key = hashlib.sha256(alerts._SOURCE.read_bytes() + alerts.platform.machine().encode() + alerts.platform.mac_ver()[0].encode()).hexdigest()
        target = self.cached_binary(key, b'verified test stub')
        with patch.object(alerts.sys, 'platform', 'darwin'), patch.object(alerts.subprocess, 'Popen') as popen:
            self.assertEqual(alerts.prepare(), target)
            popen.assert_not_called()
            target.write_bytes(b'mutated')
            self.assertIsNone(alerts.prepare())
            popen.assert_not_called()

    def test_notify_is_detached_quick_and_does_not_claim_delivery(self):
        helper = self.helper("print('presented', flush=True)\n")
        with patch.object(alerts.subprocess, 'Popen') as spawn, patch.dict(os.environ, {'HF_TOKEN': 'must-not-inherit'}):
            self.assertTrue(alerts.notify(self.session, self.capture_id, 'private exception text', helper))
        kwargs = spawn.call_args.kwargs
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], subprocess.DEVNULL)
        self.assertNotIn('HF_TOKEN', kwargs['env'])
        self.assertIn('capture_interrupted', spawn.call_args.args[0])
        self.assertEqual(self.events, [])
        self.assertFalse((self.session / 'capture_alert.json').exists())

    def test_notify_none_bad_identity_unverified_and_symlink_refuse(self):
        self.assertFalse(alerts.notify(self.session, self.capture_id, 'source_lost', None))
        helper = self.helper("print('presented', flush=True)\n")
        self.assertFalse(alerts.notify(self.session, 'c'*64, 'source_lost', helper))
        helper.write_bytes(b'tampered')
        self.assertFalse(alerts.notify(self.session, self.capture_id, 'source_lost', helper))
        link = self.root / 'linked'
        link.symlink_to(self.session, target_is_directory=True)
        self.assertFalse(alerts.notify(link, self.capture_id, 'source_lost', helper))

    def test_real_stub_stream_distinguishes_presented_then_acknowledged(self):
        helper = self.helper("print('presented', flush=True)\nprint('acknowledged', flush=True)\n")
        self.assertTrue(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'acknowledged')
        self.assertIn('presented_at', receipt)
        self.assertIn('acknowledged_at', receipt)
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'delivered', 'acknowledged'])
        self.assertEqual((self.session / 'capture_alert.json').stat().st_mode & 0o777, 0o600)
        self.assertNotIn(str(self.session), json.dumps(receipt))

    def test_startup_battery_alert_uses_distinct_fixed_message_mode(self):
        helper = self.helper("import sys\nassert sys.argv[1] == '--startup-battery-critical'\nprint('presented', flush=True)\nprint('acknowledged', flush=True)\n")
        self.assertTrue(alerts._worker(self.session, self.capture_id,
                                       'startup_battery_critical', helper, expiry=2))
        self.assertEqual('startup_battery_critical', self.receipt()['reason'])
        self.assertEqual('acknowledged', self.receipt()['status'])

    def test_zero_exit_without_visible_signal_is_not_delivery(self):
        helper = self.helper('pass\n')
        self.assertFalse(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        self.assertEqual(self.receipt()['failure_reason'], 'presentation_unconfirmed')
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'failed'])

    def test_unordered_acknowledgement_and_untrusted_stdout_fail(self):
        helper = self.helper("print('acknowledged', flush=True)\n")
        self.assertFalse(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        self.assertEqual(self.receipt()['status'], 'failed')
        self.assertNotIn('acknowledged_at', self.receipt())
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'failed'])

    def test_expiry_preserves_unacknowledged_truth_and_stops_worker_child(self):
        helper = self.helper("import time\nprint('presented', flush=True)\ntime.sleep(20)\n")
        started = time.monotonic()
        self.assertFalse(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=1.5))
        self.assertLess(time.monotonic()-started, 4)
        receipt = self.receipt()
        self.assertEqual(receipt['failure_reason'], 'expired_unacknowledged')
        self.assertIn('presented_at', receipt)
        self.assertNotIn('acknowledged_at', receipt)
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'delivered', 'failed'])

    def test_duplicate_worker_cannot_overwrite_acknowledgement(self):
        helper = self.helper("print('presented', flush=True)\nprint('acknowledged', flush=True)\n")
        self.assertTrue(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        before = (self.session / 'capture_alert.json').read_bytes()
        self.assertFalse(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        self.assertEqual((self.session / 'capture_alert.json').read_bytes(), before)
        self.assertFalse(alerts.notify(self.session, self.capture_id, 'source_lost', helper))

    def test_telemetry_failure_does_not_prevent_private_receipt(self):
        helper = self.helper("print('presented', flush=True)\nprint('acknowledged', flush=True)\n")
        import meetingintel_learning
        self.emitter.stop()
        with patch.object(meetingintel_learning, 'emit', side_effect=OSError('telemetry unavailable')):
            self.assertTrue(alerts._worker(self.session, self.capture_id, 'source_lost', helper, expiry=2))
        self.emitter.start()
        self.assertEqual(self.receipt()['status'], 'acknowledged')

    def test_failed_dispatch_records_attempt_without_false_delivery(self):
        self.assertFalse(alerts.notify(self.session, self.capture_id, 'source_lost', None))
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'failed'])
        self.events.clear()
        helper = self.helper("pass\n")
        with patch.object(alerts.subprocess, 'Popen', side_effect=OSError('private command details')):
            self.assertFalse(alerts.notify(self.session, self.capture_id, 'control_lost', helper))
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'failed'])
        self.assertFalse((self.session / 'capture_alert.json').exists())

    def test_hardlinked_helper_is_not_admitted(self):
        helper = self.helper("pass\n")
        os.link(helper, self.root / 'other-link')
        self.assertFalse(alerts.notify(self.session, self.capture_id, 'source_lost', helper))
        self.assertEqual([kind for kind, _ in self.events], ['attempted', 'failed'])


    def test_bundle_identity_and_containment_are_verified(self):
        helper = self.helper("pass\n")
        self.assertEqual(alerts._verified_binary(helper), helper)
        info_path = helper.parent.parent / 'Info.plist'
        info = plistlib.loads(info_path.read_bytes())
        self.assertEqual(info['CFBundleIdentifier'], 'com.meetingintel.capture-alert')
        self.assertFalse(info['LSUIElement'])
        info['CFBundleIdentifier'] = 'unowned.other'
        info_path.write_bytes(plistlib.dumps(info))
        with self.assertRaises(ValueError):
            alerts._verified_binary(helper)
        with self.assertRaises(ValueError):
            alerts._verified_binary(self.root / 'MeetingIntelCaptureAlert')



if __name__ == '__main__':
    unittest.main()
