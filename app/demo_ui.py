#!/usr/bin/env python3
"""Actual inbox UI with an invented read-only adapter and no model execution."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import webbrowser
import meetingintel_ui as ui

class SyntheticAdapter:
    def __init__(self):
        self.items = []
        for index, title, status, recording, transcript, report, brief, warning in [
            (1, 'Synthetic / product review', 'completed', 'retained', 'saved', 'saved', 'saved', None),
            (2, 'Synthetic / interrupted recording', 'incomplete', 'interrupted', 'saved', 'not saved', 'not saved', 'Capture ended early. Saved audio does not cover the whole meeting.'),
            (3, 'Synthetic / phone recording', 'ready', 'retained', 'not saved', 'not saved', 'not saved', None),
        ]:
            self.items.append(dict(id=f'synthetic-{index}', title=title, source_kind='phone' if index == 3 else 'laptop',
                created_at=datetime.now(timezone.utc).isoformat(), status=status, category=status,
                recording_state=recording, transcript_state=transcript, report_state=report, brief_state=brief,
                can_process=False, can_repair_brief=False, artifacts=[{'kind':'transcript'}] if transcript == 'saved' else [],
                duration_seconds=720, duration_is_final=True, warning=warning,
                next_action='Read-only synthetic demo. Use the setup guide before connecting your recordings.'))
    def snapshot(self):
        return {'items': copy.deepcopy(self.items), 'warnings': ['SYNTHETIC DEMO — invented examples; processing is disabled.']}
    def detail(self, source_id):
        return copy.deepcopy(next(x for x in self.items if x['id'] == source_id))
    def artifact(self, source_id, kind):
        self.detail(source_id)
        return 'SYNTHETIC EXAMPLE\nSPEAKER_00: Let us test the local inbox.\nSPEAKER_01: Keep incomplete recordings clearly marked.'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mi-ui-demo-') as temp:
        root=Path(temp).resolve()
        os.environ['MI_LEARNING_ROOT']=str(root/'learning')
        ui._emit=lambda *args, **kwargs: None
        state=ui.UIState(adapter=SyntheticAdapter(), runtime_root=root)
        server=ui.UIServer(('127.0.0.1', args.port), state)
        url=server.origin+'/#token='+state.token
        print('Read-only synthetic inbox. No recordings or model calls.', flush=True)
        print(url, flush=True)
        if not args.no_browser: webbrowser.open(url)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()

if __name__ == '__main__': main()
