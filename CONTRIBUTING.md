# Contributing

Start with a small reproduction, measurement or setup fix. The demos do not need private recordings or a model download.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/test_public.py
```

FFmpeg/ffprobe must be on PATH for media-container tests. GPU lifecycle tests use fake HTTP servers and real local subprocesses; they need process inspection (`ps`), so a restrictive sandbox can reject them even though no model is run.

Choose an open issue, explain the narrow change, and include before/after evidence. For speed work, give hardware, model revision, stage timings, warm/cold runs, peak memory and quality checks. For transcription or Voice ID, synthetic tests alone do not establish accuracy: describe the consented evaluation protocol and publish only material you have permission to share.

Keep source selection, model preflight, anonymous normal processing, confirmation, retention and deletion boundaries intact. Do not upload meetings, portfolio files, credentials, private paths or biometric vectors. Prefer generated fixtures for bugs. No provider should be enabled or model downloaded simply by importing a module or opening a demo.

Code contributions use the code's PolyForm Noncommercial terms. The separate phone ingestion toolkit is Apache-2.0. Please check the license scope before reusing code in a commercial application.
