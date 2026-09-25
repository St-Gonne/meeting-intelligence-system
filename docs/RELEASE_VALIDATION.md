# Public alpha validation — 25 September 2026

The application and Voice ID module are extracted into a fresh public source tree. Private Git history, working app state, recordings, transcripts, reports, credentials, embeddings and weights are absent.

- 49 synthetic test modules passed on macOS with Python 3.11. The exact per-module counts are in [TEST_RECEIPT.json](TEST_RECEIPT.json). Rechecked failures are replaced by their passing final result, not counted twice.
- Demos: actual inbox UI with invented recording states; Voice ID candidate/confirmation/retention/revocation lifecycle with invented vectors; actual terminal help.
- Full suite requires NumPy for waveform tests and FFmpeg for generated-media tests. No model inference is part of this receipt.
- GPU tests use fake loopback servers and real OS process/lock behavior. They passed with process inspection enabled; the restricted sandbox initially blocked `ps`, causing truthful deferral and test timeouts. No live model server was changed.
- Portability fixes resolved original author paths, private-owner calibration constants, missing scheduler wrapper, a private-only quality-gate reference and temporary-directory symlink handling in the test runner. The app's path safety checks were preserved.
- Source files and extraction adaptations are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Publication scanning found only fictional `/Users/example` paths in negative tests; no original user paths, private email addresses or credential signatures in application source.

This is a contributor alpha. Real ASR quality, microphone continuity, model provisioning and a full clean-machine capture-to-report run remain unverified for adopters. Automated tests cannot establish those claims. The setup guide states the current pinned/two-server runtime assumptions explicitly.
