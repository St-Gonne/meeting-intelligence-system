> Historical document, retained for context. The public application alpha and Voice ID source were released on 25 September 2026. See the [current README](../README.md), [setup](SETUP.md) and [validation](RELEASE_VALIDATION.md) for current public scope.

<!-- Copied from the working system on 2026-07-16, refreshing the 2026-07-11 copy. This is the live contract the AI builder (Codex) works under. Nothing edited except this note and two privacy redactions: the local machine path and the Drive recorder-app folder are replaced with placeholders. -->

# MeetingIntel — Project Control

Last reconciled: 2026-07-15
Purpose: the single current source of truth for milestone state and the next
build priority. It outranks historical handovers and plans.

## Current status

| Milestone | Status | Current meaning |
|---|---|---|
| M1 — Diarized path ready to judge | PASS — closed | Operator-confirmed diarized workflow was proven for quality review. |
| M2 — Production state safe | PASS — closed | Schema-v2 slim ledger, atomic replacement, and bounded backups are in place. |
| M3 — Five-meeting quality gate | PASS — closed | Diarized output won or tied the reviewed flat output and met the defined gate. |
| M4 — Promote diarized 1:1 path | PASS — closed | Supported 1:1 meetings use the diarized production path; flat remains controlled fallback. |
| M5 — Phone audio-first production admission | PASS — closed | Phone-source normalization, routing admission, and the isolated real-phone end-to-end proof are complete; production ledger/output were unchanged. |

M4's isolated production safety gate succeeded: it exited successfully, invoked
the diarization helper once, reused the staged folder, produced one schema-v2
record and one brief entry in isolation, and left source and real production
state/output unchanged. The detailed, privacy-sensitive evidence remains in the
isolated proof location and the M4 checkpoint.

## Active priority

**Single next build priority: make the supported on-demand and daily Drive → Mac
fetch reliable.**

The isolated real-phone proof is complete. The remaining work is operational
automation hardening around the narrow `mi sync` delivery step; it must preserve
the production and privacy boundaries and must not widen routing policy.

## Production boundary

Production is the local MeetingIntel pipeline's configured ledger and output
locations. For supported Meetily 1:1 sources, it performs one diarization staging
attempt, selects either `diarized_1to1`, `flat`, or `flat_fallback`, and persists
only after a selected result is ready. A persistence failure stops processing; it
is never a fallback signal.

Group calls remain outside the production claim. The older shadow-artifact
prototype remains archival and is not a third live pipeline.

The phone normalizer is deliberately outside production: it creates a canonical,
read-only source folder before production discovery. Production does not run the
normalizer or delete source audio.

## Recording sources and mobile ingestion

- **Meetily lane:** local Meetily export folders are the normal laptop source.
- **Phone lane:** the configured normalizer is for ASR Voice Recorder `.m4a`
  files from in-person recording. The current app is not a supported capture
  path for cellular or VoIP call audio. The expected operating sequence is
  in-person phone recording → Google Drive → Mac sync → manual normalization →
  explicit production discovery.
- **Delivery:** Google Drive for desktop is one delivery option. The separate
  `mi sync` fetcher is the preferred reliable option once its one-time,
  read-only local Google authorization is explicitly approved. It downloads only the configured phone
  folder to local staging, records immutable Drive metadata and failures in a
  private local ledger, and never invokes normalization or production. The
  remaining automation gap is approved credential setup and daily scheduler
  installation; no credentials or real Drive data are created or accessed by
  this project work.
- **Raw `.m4a` ingestion:** implemented as a standalone normalizer
  (`phone_recording_ingest.py`). It discovers `.m4a` files, accepts singleton or
  validated segmented recordings, and writes canonical `audio.m4a` plus
  `meeting_source.json` under the ingest root. It is not auto-run by production.
- **Configured paths:** the normalizer's default source is an account-scoped Mac
  Google Drive location ending in `My Drive/<recorder-app>/MeetingIntel/phone-recordings`;
  the account-specific prefix is intentionally not repeated in durable docs.
  Its verified default normalized root is
  `<local>/MeetingIntel/ingest/phone`. The production
  audio-first root has no default and must be passed explicitly with
  `--audio-first-root`.

## Operating rules

1. Keep one active priority. Only a blocker may interrupt it.
2. Treat current code, deterministic tests, this file, and current durable docs
   as current truth. Treat checkpoints, plans, handovers, review packages, and
   generated artifacts as dated evidence.
3. Never run capture, transcription, diarization, model calls, real meetings, or
   production-data mutations merely to validate documentation.
4. The phone lane is field-proven only for the isolated in-person recording
   flow; cellular and VoIP call capture remain unsupported.
5. Do not place meeting content, names, credentials, raw model output, or
   account-specific cloud paths in durable documentation.

## Important deferred capture gap

**Phone-call recording is not supported by the current ASR Voice Recorder lane.**
The lane may be used for in-person meetings only. This creates a potentially
material coverage gap for calls that do not enter the supported laptop lane;
the actual share is not yet measured.

The fast, bounded investigation is to test the phone's native dialer recording
feature, if its model, carrier, and region expose one, with explicit participant
consent. If that is unavailable or fails the isolated quality check, select a
separate phone-call capture lane later (for example, an approved dedicated
hardware recorder or an already-supported laptop meeting route). Do not rely on
unverified third-party call-recording apps, bypass OS protections, or widen
production routing as part of this decision. This deferred item does not
supersede the isolated in-person real-phone proof.

## Document status / index

### Current

- `PROJECT_CONTROL.md` — current priority and status.
- `docs/ARCHITECTURE.md` — current system shape and boundaries.
- `docs/OPERATIONS.md` — current safe operating procedure.
- `docs/DECISIONS.md` — current decisions and limits.

### Historical

- `docs/checkpoint_m1_*`, `docs/checkpoint_m2_*`, `docs/checkpoint_m3_*`,
  `docs/checkpoint_m4_*`, and `docs/checkpoint_phone_*` — dated closure evidence.
- `docs/m3_*`, `docs/handover_*`, `docs/milestone_*`,
  `docs/normalized_meeting_artifact_v1.md`, and
  `docs/shadow_artifact_pipeline_v0.2.md` — archived plans, protocols, or
  prototype documentation.

### Generated

- `state/`, `output/`, `artifacts/`, `ingest/`, `context/`, `checkpoints/`,
  `m3-review-package/`, and `review-package/` contain local/generated evidence
  and may include private data. They are not current documentation.

## Evidence references

- M4 production promotion and isolated safety proof:
  `docs/checkpoint_m4_promote_diarized_1to1_path_pass_2026-07-07.md`
- M5 phone admission, bounded to code and synthetic coverage:
  `docs/checkpoint_phone_audio_first_production_admission_pass_2026-07-07.md`
- Synthetic, non-production context-trust harness instructions:
  `docs/m4_context_trust_isolated_proof.md`
