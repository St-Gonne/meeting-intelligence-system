# Contributing

Read [AGENTS.md](AGENTS.md) and [current status](docs/STATUS.md), then start with a small reproduction, measurement or setup fix. The demos do not need private recordings or a model download.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/test_public.py
```

FFmpeg/ffprobe must be on PATH for media-container tests. GPU lifecycle tests use fake HTTP servers and real local subprocesses; they need process inspection (`ps`), so a restrictive sandbox can reject them even though no model is run.

Choose an open issue, explain the narrow change, and include before/after evidence. For speed work, give hardware, model revision, stage timings, warm/cold runs, peak memory and quality checks. For transcription or Voice ID, synthetic tests alone do not establish accuracy: describe the consented evaluation protocol and publish only material you have permission to share.

Keep source selection, model preflight, anonymous normal processing, confirmation, retention and deletion boundaries intact. Do not upload meetings, portfolio files, credentials, private paths or biometric vectors. Prefer generated fixtures for bugs. No provider should be enabled or model downloaded simply by importing a module or opening a demo.

Code contributions use MIT, including contributions to Voice ID. Commercial reuse is permitted under the MIT terms. See [license scope](LICENSE.md) for earlier architecture documents and third-party materials. The separate phone ingestion toolkit is Apache-2.0.

On Linux, run `.venv/bin/python scripts/test_public.py --portable`. The full suite also checks Mac-only capture, AppKit and launchd integration fixtures. Linux core/demo success is not Linux recorder support.

Use the single-service public default unless testing shared coordination. Include the selected backend and timezone in reproducible setup reports. Runtime speaker integration tests are optional when dependencies are absent; report skips and run them in the separate installed runtime before claiming upstream export parity. Never attach private setup settings or model tokens.
