#!/usr/bin/env python3
"""Render captured output as HTML for reproducible documentation screenshots."""
from pathlib import Path
import html
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
for name, args, title in [('voice-id',[sys.executable,'-B','app/demo_voice_id.py'],'Voice ID · synthetic lifecycle'),('terminal',[sys.executable,'-B','app/mi','help'],'MeetingIntel · terminal help')]:
    result=subprocess.run(args,cwd=root,capture_output=True,text=True,check=True)
    output=root/'docs/images'/f'{name}-output.txt'
    output.write_text(result.stdout.rstrip()+"\n")
    page='''<!doctype html><meta charset="utf-8"><title>'''+html.escape(title)+'''</title><style>
body{margin:0;padding:32px;background:#ecefe8;color:#eff4ed;font:16px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}main{max-width:1080px;margin:auto;background:#18342e;border-radius:16px;box-shadow:0 12px 35px #18342e22}header{padding:18px 26px;border-bottom:1px solid #ffffff22;color:#cce2d0}pre{white-space:pre-wrap;margin:0;padding:26px;font:inherit;font-size:14px}footer{padding:0 26px 24px;color:#b4cbbd;font:13px/1.5 system-ui}</style><main><header>'''+html.escape(title)+'''</header><pre>'''+html.escape(result.stdout)+'''</pre><footer>Captured from the published source. Synthetic demo output is not model-quality evidence.</footer></main>'''
    (root/'docs/images'/f'{name}.html').write_text(page)
