# Changes

## 2026-11-05 — reliability and setup refresh

- Default phone and guarded laptop transcription to pinned Qwen3-ASR 1.7B. Add
  `--transcription-backend qwen|whisper` and persisted settings for explicit choice.
  Meetily keeps Whisper. Failure never silently switches engines.
- Preserve aligned word-speaker changes through TXT/SRT/JSON and general analysis.
  Accept finite NumPy timestamps, abstain on tied overlaps and keep uncertain
  owners/voices unknown. Exact laptop processing accepts a known speaker count.
- Retry unfinished Qwen windows down to 15 seconds, validate contiguous prefixes,
  fix completed-prefix checkpoint hashing, and reuse only complete accepted stages.
- Preserve speaker structure during general analysis and retain validated
  transcripts after downstream failures. Save source-day and processing-day briefs.
- Honor recorder stop intent around status requests and admit only explicit,
  integrity-checked recovery paths without rewriting interruption provenance.
- Add today's foreground queue with frozen source/configuration/backend choices,
  completed-source skipping, status/resume and separate failed-retry approval.
- Default public GPU admission to the one MeetingIntel service on 11435. Retain
  explicit shared mode and process/lock/cleanup checks.
- Add saved user settings, timezone configuration, a read-only setup checker,
  AI onboarding, model/dependency commands, phone setup, tutorial and command sheet.
- Include optional owner presentation after anonymous analysis, separate machine
  provenance and per-meeting revocation. New installs remain disabled; new owners
  supply independently evaluated calibration/enrollment. Private reference assets,
  original calibration and visual-speaker experiments remain excluded.

These are software and synthetic verification claims. They do not establish
universal accuracy, complete second-machine setup or an ordinary-run stability streak.
