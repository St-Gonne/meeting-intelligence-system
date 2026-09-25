#!/usr/bin/env python3
"""Exercise the real review API with invented vectors, not a voice recording."""
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from voiceprint_gate5_inbox import Gate5Inbox, GATE4_PROVIDER_ID

def main():
    with TemporaryDirectory(prefix='mi-voice-demo-') as temp:
        root = Path(temp)
        report = root / 'synthetic-calibration.json'
        report.write_text(json.dumps({'schema_version': 1, 'report_type': 'voiceprint_owner_bakeoff',
            'providers': [{'provider_id': GATE4_PROVIDER_ID, 'passed': True,
                           'threshold': 0.8, 'model_asset_sha256': 'b' * 64}]}))
        report.chmod(0o600)
        inbox = Gate5Inbox(root / 'private', clock=lambda: datetime.now(timezone.utc),
                          gate4_report=report, model_asset_sha256='b' * 64)
        print('Voice ID / synthetic lifecycle demo')
        print('No audio, model, network, or biometric enrollment is used.\n')
        inbox.enroll_pending('demo-owner', (1.0, 0.0))
        print('1. Enroll invented vector: pending retention')
        candidates = inbox.review({'SPEAKER_00': (1.0, 0.0), 'SPEAKER_01': (0.0, 1.0)}, meeting_id='demo-meeting')
        for c in candidates:
            print(f'2. {c.observed_speaker}: {c.status}, cosine={c.score:.2f}')
        candidate = next(c for c in candidates if c.status == 'candidate')
        print(f'3. Before confirmation: {len(inbox.trusted_map())} trusted mappings')
        inbox.confirm(candidate, operator_confirmed=True)
        print(f'4. Simulated confirmation, retention pending: {len(inbox.trusted_map())} trusted mappings')
        inbox.set_retention('demo-owner', consent=True)
        assert len(inbox.trusted_map()) == 1
        print('5. Simulated retention consent: 1 trusted mapping for demo-meeting')
        inbox.set_retention('demo-owner', consent=False)
        assert not inbox.trusted_map()
        print('6. Revoke retention: 0 trusted mappings; managed enrollment deleted')
        print('\nThis demonstrates consent and state transitions, not recognition accuracy.')

if __name__ == '__main__': main()
