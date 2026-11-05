# MeetingIntel: start here

This is the public application. Read README.md, docs/AI_SETUP.md and docs/SETUP.md
before installing or processing anything. docs/STATUS.md is current; the older
control/release-plan documents explicitly marked historical are background.

## If the user wants to get started

1. Inspect OS, CPU architecture, RAM, free disk, Python, installed tools and any
   existing Ollama or MeetingIntel installation. Use read-only checks first.
2. Run `python3.11 scripts/setup_check.py --mode demo`, terminal help, the inbox
   demo and the Voice ID lifecycle demo. These use invented data and no models.
3. Ask which lanes the user needs: laptop capture, phone recordings, or both;
   their timezone and meeting languages; and whether another app uses the local
   GPU/Ollama service.
4. Follow the exact dependency, asset and service commands in docs/SETUP.md.
   Explain downloads and disk use. Install into separate virtual environments;
   preserve existing apps, models and account configuration. Download only assets
   whose terms the user has accepted. The user handles login, secrets and OS grants.
5. Configure app/.local/settings.json (ignored by Git) using actual absolute
   machine paths. Secrets never belong in it. Qwen is the phone/laptop default;
   `mi --transcription-backend whisper …` is the explicit alternative. No hidden
   fallback. Meetily retains Whisper.
6. Default to one managed Ollama service on 127.0.0.1:11435. Do not create an idle
   second server. If the user needs concurrent local AI apps, follow shared mode;
   never bypass GPU admission, model digest or cleanup checks.
7. Follow docs/PHONE_SETUP.md for the chosen phone. Do not assume every recorder,
   iPhone export or cellular/VoIP recording matches the supported Android lane.
8. Run the read-only runtime setup check, the Qwen dependency check if selected,
   and an explicitly selected short trial. Verify actual audio, coverage, speakers,
   report and brief. Synthetic passes alone do not establish real-call quality.
9. Produce a short tutorial and command sheet in the user's private data folder.
   Adapt docs/TUTORIAL.md and docs/COMMANDS.md to their installed launcher, paths,
   phone, timezone and backend. Include start/stop, sync/settlement/grouping,
   processing, queue/resume, finding results and safe recovery. Mark unfinished
   setup clearly. Leave a private setup receipt with versions and remaining steps;
   exclude tokens, meeting content and biometric data.

## Code map and validation

- app/mi, mi_config.py, mi_paths.py: launcher, persisted settings, portable paths.
- meetingintel_cli.py, meetingintel_inbox.py, meetingintel_actions.py, ui/: operators.
- meetingintel_pipeline.py, whispermlx_diarization_helper.py: source admission,
  processing and persistence. Qwen/Whisper runtime modules handle transcription.
- meetingintel_speaker_turns.py: word-level speaker changes and overlap abstention.
- meetingintel_queue.py: one approved foreground queue, frozen source/backend seals.
- gpu_lock.py, meetingintel_gpu.py, ollama_service.py: exclusive work and cleanup.
- phone_drive_fetch.py, phone_recording_review.py, phone_recording_ingest.py:
  fetch, settle, review and canonicalization. These remain distinct from processing.
- meetingintel_identity.py and voiceprint_*: optional owner-local acoustic matching;
  no enrollment, calibration or reference recordings ship with this repository.

Use `python3.11 -m venv .venv`, install requirements-test.txt, then
`.venv/bin/python scripts/test_public.py` on macOS or add `--portable` on Linux.
Use focused modules while developing, then the appropriate full suite and
`git diff --check`, then `python3.11 scripts/check_public_tree.py` before publishing. Process-lifecycle tests need OS process inspection. Five
upstream speaker-writer integration cases additionally need the optional runtime;
report skipped tests rather than presenting them as executed checks.

## Boundaries

Keep raw recordings and earlier outputs. Processing selects exact sources and
requires the existing confirmation. No backlog sweeps, failed-job automatic retry,
forced-sleep capture, arbitrary model substitution or enrollment from normal
meetings. Recovery cannot recreate missing speech. Names never enter analysis or
action ownership; machine suggestions and human confirmations remain distinct.
Voice ID is optional and starts disabled. A new owner needs independent calibration
and retention consent, never the author's constants. No visual-speaker experiment
is included. Use synthetic fixtures and sanitized diagnostics in issues and PRs.
