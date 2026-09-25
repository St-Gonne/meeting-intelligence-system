#!/usr/bin/env python3
"""Run each synthetic test module with fresh state and a bounded timeout.

Independent processes avoid module-level monkeypatch leakage between suites.
No private workspace, model weights, credentials, or account state is needed.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]/'app'
failed=[]
modules=sys.argv[1:] or [p.stem for p in sorted(root.glob('test_*.py'))]
for module in modules:
    with tempfile.TemporaryDirectory(prefix='mi-public-test-') as temp:
        temp=str(Path(temp).resolve())
        env={**os.environ, 'HOME':temp, 'PYTHONDONTWRITEBYTECODE':'1',
             'MI_DATA_ROOT':temp+'/data', 'MI_RUNTIME_ROOT':temp+'/runtime',
             'MI_UI_ROOT':temp+'/ui', 'MI_LEARNING_ROOT':temp+'/learning'}
        try:
            result=subprocess.run([sys.executable,'-B','-m','unittest',module,'-q'],cwd=root,env=env,
                                  capture_output=True,text=True,timeout=75)
            lines=[line for line in result.stderr.splitlines() if line.startswith(('Ran ', 'OK', 'FAILED'))]
            print(module+': '+ ('; '.join(lines) or str(result.returncode)),flush=True)
            if result.returncode:
                failed.append(module); print(result.stderr[-10000:],flush=True)
        except subprocess.TimeoutExpired:
            failed.append(module);print(module+': TIMEOUT (75 seconds)',flush=True)
print('Modules:',len(modules),'Failures:',failed,flush=True)
raise SystemExit(bool(failed))
