"""Public calibration must be owner-local, finite and bound to its model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from voiceprint_gate5_inbox import Gate5Inbox, Gate5Error, GATE4_PROVIDER_ID

class PublicCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.model=self.root/'model.bin'
        self.model.write_bytes(b'invented test model');self.model.chmod(0o600)
        self.digest=hashlib.sha256(self.model.read_bytes()).hexdigest()
    def inbox(self, threshold, digest=None, supplied=None):
        report=self.root/'report.json'
        report.write_text(json.dumps({'schema_version':1,'report_type':'voiceprint_owner_bakeoff',
            'providers':[{'provider_id':GATE4_PROVIDER_ID,'passed':True,
                          'threshold':threshold,'model_asset_sha256':digest or self.digest}]}))
        report.chmod(0o600)
        return Gate5Inbox(self.root/'state',clock=lambda:datetime.now(timezone.utc),
                          gate4_report=report,model_asset=self.model,threshold=supplied)
    def test_new_owner_calibration_not_private_owner_constant(self):
        inbox=self.inbox(0.9)
        inbox.enroll_pending('demo-owner',(1.0,0.0))
        row=inbox.review({'SPEAKER_00':(0.8,0.6)},meeting_id='demo-meeting')[0]
        self.assertEqual(row.status,'unknown')
        self.assertEqual(inbox.provenance.threshold,0.9)
    def test_nonfinite_and_out_of_range_thresholds_rejected(self):
        for threshold in (float('nan'),float('inf'),-1.1,1.1):
            with self.subTest(threshold=threshold),self.assertRaises(Gate5Error):self.inbox(threshold)
    def test_local_model_and_report_must_match(self):
        with self.assertRaisesRegex(Gate5Error,'digest'):self.inbox(0.8,'c'*64)
    def test_runtime_threshold_cannot_override_evaluation(self):
        with self.assertRaisesRegex(Gate5Error,'threshold'):self.inbox(0.8,supplied=0.1)

if __name__=='__main__':unittest.main()
