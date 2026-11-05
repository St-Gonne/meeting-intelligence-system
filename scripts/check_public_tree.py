#!/usr/bin/env python3
"""Reject runtime artifacts, credential signatures and stale source hashes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.secrets','state','output','ingest','staging','artifacts','checkpoints','source-backups','review-package','m3-review-package','.local'}
BINARY={'.m4a','.wav','.mp3','.mkv','.mp4','.gguf','.safetensors','.sqlite','.db','.pt'}
SIGNATURES=[r'(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}',r'hf_[A-Za-z0-9]{25,}',
            r'sk-(?:proj-)?[A-Za-z0-9_-]{35,}',r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----',
            r'GoogleDrive-[^\s/"\x27]*@']


def main():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    findings=[]
    for name in names:
        path=ROOT/name
        if not path.is_file():continue
        if path.is_symlink() or FORBIDDEN.intersection(Path(name).parts) or path.suffix in BINARY:
            findings.append(('runtime/unsafe file',name));continue
        if path.suffix in {'.jpg','.png','.svg'}:continue
        body=path.read_text(encoding='utf-8',errors='ignore')
        if any(re.search(pattern,body) for pattern in SIGNATURES):findings.append(('credential/account signature',name))
    manifest=json.loads((ROOT/'docs/SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        name=row['path'];path=ROOT/name
        if FORBIDDEN.intersection(Path(name).parts) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['release_sha256']:
            findings.append(('source manifest',name))
    for kind,name in findings:print(kind+': '+name)
    print('Public tree: '+str(len(names))+' files; '+str(len(findings))+' findings.')
    return bool(findings)


if __name__=='__main__':raise SystemExit(main())
