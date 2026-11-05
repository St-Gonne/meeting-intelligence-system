# Public refresh validation — 5 November 2026

Local validation completed a **1,049-test full macOS synthetic run in 61
modules**, with five optional upstream-runtime cases skipped in the lightweight
environment. A later private-asset layout regression is separately covered by
the eight-case public setup module; the current full suite contains 1,050 cases.
GitHub platform checks validate the final PR head. The full **13-case speaker assignment/export integration module**
was also run against the installed upstream runtime and passed without skips.
No real audio or model inference was used. Exact counts are in TEST_RECEIPT.json.

The refresh adds speaker/recovery/Qwen/queue changes, portable settings, explicit
backend choice, standalone/shared service configuration, and onboarding guidance.
CI pins Ubuntu 24.04 and macOS 15 Intel for reproducible portable/full synthetic
checks. These model-free checks do not establish Intel Mac inference support;
full processing still targets Apple Silicon. Single-endpoint GPU tests use fake
servers and real process/lock lifecycle checks;
existing coordinated-service and per-owner calibration checks remain covered.
The initial focused extraction checks caught and fixed a read-only lifecycle
adaptation and test fixture path assumptions before the full passing run.

Public-content review covers the complete candidate tree, including new files:
no known credential signatures, original machine/account paths, author calibration
constants or prohibited recording/model/state files were found. SOURCE_MANIFEST.json
records hashes of the final public source bytes. Original private history, actual
meeting data, enrollment/calibration assets and visual-speaker work are excluded.

Demos and setup inventory are checked separately from actual model/capture readiness.
A complete installation on a second physical machine, new-owner Voice ID quality,
phone background upload behavior and whole-call speaker/transcription accuracy
remain unproven. No model/service configuration on the existing private installation
was changed by this refresh. The public setup remains a contributor path.

## Earlier September release

### Public alpha validation — 25 September 2026

The application and Voice ID module are extracted into a fresh public source tree. Private Git history, working app state, recordings, transcripts, reports, credentials, embeddings and weights are absent.

- 49 synthetic test modules passed on macOS with Python 3.11. The exact per-module counts are in [TEST_RECEIPT.json](TEST_RECEIPT.json). Rechecked failures are replaced by their passing final result, not counted twice.
- Demos: actual inbox UI with invented recording states; Voice ID candidate/confirmation/retention/revocation lifecycle with invented vectors; actual terminal help.
- Full suite requires NumPy for waveform tests and FFmpeg for generated-media tests. No model inference is part of this receipt.
- GPU tests use fake loopback servers and real OS process/lock behavior. They passed with process inspection enabled; the restricted sandbox initially blocked `ps`, causing truthful deferral and test timeouts. No live model server was changed.
- Portability fixes resolved original author paths, private-owner calibration constants, missing scheduler wrapper, a private-only quality-gate reference and temporary-directory symlink handling in the test runner. The app's path safety checks were preserved.
- Source files and extraction adaptations are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Publication scanning found only fictional `/Users/example` paths in negative tests; no original user paths, private email addresses or credential signatures in application source.

This is a contributor alpha. Real ASR quality, microphone continuity, model provisioning and a full clean-machine capture-to-report run remain unverified for adopters. Automated tests cannot establish those claims. The setup guide states the current pinned/two-server runtime assumptions explicitly.

The first clean CI run exposed test-only assumptions about an installed Python path and rclone. These now use the running interpreter and an explicitly mocked executable for the fake download runner. CI runs the full suite on macOS and the portable core on Linux; Mac capture/launchd fixtures are not Linux-support evidence.
