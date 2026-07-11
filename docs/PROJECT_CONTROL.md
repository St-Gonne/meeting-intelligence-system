<!-- Copied verbatim from the working system on 2026-07-11. This is the live contract the AI builder (Codex) works under. Nothing edited except this note. -->

# Meeting Intelligence System — Project Control

Last updated: 2026-07-07
Owner: Sharan Tulsiani
Purpose: Living execution map. This file controls current project priority and milestone sequence.

## NORTH STAR

Build a local-first, privacy-preserving Meeting Intelligence System that turns recorded meetings into accurate, speaker-attributed intelligence with near-zero operator effort.

The useful product loop is:

Record → transcribe + diarize → identify speakers → detailed Layer 2 intelligence → short Layer 3 brief → one daily brief → manual EOD synthesis.

The product is successful when Sharan uses it on real meetings as part of his normal day, not when the shadow pipeline merely produces technically correct artifacts.

## CURRENT TARGET ARCHITECTURE

For 1:1 meetings:

Meetily capture
→ optional trim
→ WhisperMLX large-v3 + pyannote diarization
→ speaker identity resolution
→ diarized Layer 2
→ diarized Layer 3
→ production ledger
→ one daily brief
→ manual Claude EOD synthesis

Target operating rule:

* Diarized path = normal 1:1 path after quality gate.
* Existing flat Meetily-transcript path = fallback if diarized processing fails.
* Group meetings remain shadow-only until separately proven.
* Production integration is not approved until M3 passes.

## OPERATING RULES

1. WIP cap = 1 active milestone.
2. Work only on the ACTIVE milestone unless it is genuinely blocked.
3. Every new issue must be classified:

   * BLOCKER
   * QUALITY GATE FINDING
   * FOLLOW-UP
   * PARKED
4. Only a BLOCKER may interrupt the active milestone.
5. A side question may be investigated without changing the roadmap.
6. Do not create a new workstream merely because an adjacent problem is interesting.
7. Prove on a real meeting before declaring a milestone PASS.
8. Current code, current tests, this control file, and the latest milestone checkpoint outrank historical planning documents.
9. Do not modify production integration boundaries unless the active milestone explicitly approves it.
10. When in doubt: return to the current milestone PASS criteria.

## MILESTONE MAP

### M1 — DIARIZED PATH READY TO JUDGE

Status: PASS

Goal:
Make the proven diarized workflow produce identity-confirmed, internally coherent, honestly instrumented outputs that are ready for a human production-quality gate.

Required work:

M1.1 Operator-confirmed speaker map

* After diarization, show 2–3 representative utterances for each speaker label.
* Operator confirms a name or enters unknown.
* Confirmed map is passed into Layer 2.
* Confirmed identity status is preserved in outputs and manifest.

M1.2 First-class diarized Layer 3 prompt

* Create a clean diarized Layer 3 prompt variant.
* Remove contradictory assumptions that transcripts are untagged or that inferred speakers should become first-person "I".
* Do not modify the production flat-transcript prompt.

M1.3 Real stage timing

* Record and display duration for trim, diarization, Layer 2, Layer 3, and total workflow.
* Use real stage timing to judge throughput.
* Do not infer throughput from gaps between manually run commands.

M1.4 Honest speaker-check language

* Operator-facing attribution validator output must not imply factual attribution correctness.
* Rename/reframe as speaker structure or speaker format checks.
* Existing validators remain useful lints, not a production quality gate.

M1 PASS GATE:
One real 1:1 meeting completes end to end with:

* operator-confirmed names reaching Layer 2 and Layer 3 correctly;
* clean diarized Layer 3 instructions with no flat-transcript contradiction;
* stage timing visible;
* validator wording clearly described as structural/format checking;
* production output, ledger, and daily brief still untouched.

### M2 — PRODUCTION STATE SAFE

Status: PASS — CLOSED

Checkpoint:
`docs/checkpoint_m2_production_state_safe_pass_2026-07-05.md`

Goal:
Harden the production state before the diarized path is allowed to write to it.

Required work:

* atomic ledger write using temporary file + replace;
* save state after each successfully processed meeting;
* rolling ledger backup;
* slim ledger to required metadata, summary, and artifact paths rather than full transcript and full Layer 2 text;
* minimal deterministic tests for production pure/state functions;
* record Ollama prompt/context instrumentation such as prompt_eval_count where available;
* verify the reported HF_TOKEN process-argument exposure against the actual installed WhisperMLX behavior and fix safely if supported.

M2 PASS GATE:

* production tests pass;
* ledger survives simulated interrupted write behavior;
* successful earlier meetings remain recorded if a later meeting fails;
* current production records remain loadable or are migrated safely;
* ledger no longer serves as a permanent duplicate store of full transcript + full Layer 2 text;
* no diarized workflow writes production state yet.

### M3 — FIVE-MEETING QUALITY GATE

Status: PASS — CLOSED

Goal:
Make a human product-quality decision using real meetings rather than validator lints.

Run both flat and diarized paths on 5 consecutive real 1:1 meetings.

For each meeting evaluate:

1. Important entities missed.
2. Wrong speaker attribution.
3. Stage/deal read accuracy.
4. Action and action-owner correctness.
5. Which output Sharan would trust.

Additionally:

* spot-check 3 attributed evidence items per meeting against the diarized transcript/SRT;
* record actual stage and total processing duration.

M3 PASS / PROMOTION CRITERIA:

* zero unresolved wrong-action-owner errors in the diarized outputs;
* zero unresolved speaker swaps in the checked attributed evidence;
* diarized output wins or ties the flat output on important intelligence quality;
* processing duration is operationally acceptable for Sharan's real workflow.

If M3 fails:
Classify failure categories before changing architecture:

* missed entities;
* wrong stage reads;
* attribution swaps;
* action-owner errors;
* throughput/friction.

Do not default to a model upgrade or a new architecture without evidence.

### M4 — PROMOTE DIARIZED 1:1 PATH

Status: PASS — CLOSED

Checkpoint:
`docs/checkpoint_m4_promote_diarized_1to1_path_pass_2026-07-07.md`

Goal:
Collapse the dual-system tax and make the better path part of the useful daily loop.

Target:

* diarized workflow becomes the normal path for supported 1:1 meetings;
* diarized Layer 2/3 results write to the hardened production state;
* one daily brief uses diarized entries when available;
* existing flat path remains the fallback on diarized failure;
* fallback entries are clearly identifiable;
* group calls remain shadow-only.

M4 PASS GATE:
A real production day produces one daily brief containing a successfully diarized 1:1 meeting entry through the normal path, while the flat fallback is separately verified to remain functional.

Closure evidence:

* isolated real-meeting proof selected `diarized_1to1` with exactly one helper invocation and staged-folder reuse;
* selected diarized Layer 2 was promoted durably and one schema-v2 record produced one brief entry;
* flat and `flat_fallback` behavior remain separately verified;
* production persistence failure stops rather than changing processing mode;
* `dominant_two_plus_fragment_v1` is approved as a provisional monitored-field policy.

### M5 — PHONE AUDIO-FIRST PRODUCTION ADMISSION

Status: PASS / CLOSED

Goal:
Admit validated canonical audio-first phone sources into the production routing boundary without disguising them as Meetily transcripts or duplicating transcription/diarization.

Closure evidence:

* standalone phone `.m4a` normalizer and atomic slim ingest ledger built/tested;
* canonical `audio.m4a` plus `meeting_source.json` contract built;
* read-only audio-first discovery/validation adapter built/tested;
* explicit helper audio seam and audio-first diarized workflow/source-context seam built;
* production routing supports eligible `diarized_1to1`, ineligible `flat_from_diarized_transcript`, and late `flat_fallback_from_diarized_transcript`;
* malformed staged SRT now fails closed and valid timed blocks flatten chronologically;
* mixed Meetily + phone brief support remains intact;
* synthetic-test coverage passed for the phone workstream and closure seam.

Scope discipline:
Phone normalization remains a separate stage. Production still must not auto-run normalization, and the phone capture lane is not yet claimed as fully production-proven until the isolated real-phone proof completes.

## CURRENT ACTIVE MILESTONE

ISOLATED REAL-PHONE END-TO-END PROOF

Current status:
ACTIVE

Next action:
Run one isolated real-phone proof against the admitted phone lane and confirm end-to-end field behavior without broadening the scope of M4 or the router.

Codex must not implement FOLLOW-UP or PARKED work.

## PARKING LOT

### P-001 — Automatic meeting-end / dead-air detection

Classification: PARKED
Reason: Manual trim solves the observed accidental over-recording case.
Revisit trigger: Manual trim causes repeated real-world friction.

### P-002 — Sharan voice recognition / persistent voice bank

Classification: FOLLOW-UP
Reason: Strategically valuable Identity v1, but operator-confirmed speaker mapping is the faster Identity v0.
Measured M1 finding: Interactive speaker confirmation blocks unattended or overnight continuation when Sharan is AFK. Preserve this as identity/unattended-workflow follow-up context; do not treat the inflated wait as compute time.
Revisit trigger: M4 passes and manual speaker confirmation friction is measured in real use.

### P-003 — Retention and cleanup command

Classification: FOLLOW-UP
Reason: Real privacy/data-sprawl concern, but deletion logic should not be mixed into production-state hardening immediately before integration.
Revisit trigger: After M4 promotion, unless storage/privacy evidence elevates it earlier.

### P-004 — Long-meeting chunking

Classification: PARKED
Reason: Chunk/merge logic exists in the dormant prototype, but the current diarized path has not yet failed on a legitimate meeting because of the input cap.
Revisit trigger: First legitimate meeting exceeds the Layer 2 input limit.

### P-005 — Phone recording ingest

Classification: PROMOTED TO ACTIVE M5
Reason: M4 passed. The phone normalizer and read-only adapter are built/tested but remain outside production routing.
Revisit trigger: Active now; keep WIP limited to the production admission seam.

### P-006 — Shared llm_client / textutils refactor

Classification: FOLLOW-UP
Reason: Real duplicated logic and divergence risk, but not required to judge or promote the current product path.
Revisit trigger: After M4, or earlier if duplication directly blocks an active milestone.

### P-007 — Hardcoded shadow paths / config portability

Classification: FOLLOW-UP
Reason: Current machine and paths are known and operational.
Revisit trigger: Machine migration, path change, or post-M4 hardening.

### P-008 — Partial-success wrapper reporting

Classification: FOLLOW-UP
Reason: Valid operator UX issue; does not block M1 quality proof.
Revisit trigger: A real run loses hours or causes a mistaken rerun because Layer 2 succeeded and Layer 3 failed.

### P-009 — Artifact normalization prototype

Classification: FROZEN
Decision: Do not develop as a live third pipeline.
Preserve code and tests.
Harvest proven sanitization or chunking ideas only when an active milestone has a concrete need.

### P-010 — Group-call production integration

Classification: PARKED
Reason: Group diarization passed with caveats and has a lower confidence bar.
Revisit trigger: Diarized 1:1 path has been in real production use and group quality receives its own evidence gate.

## CURRENT DECISIONS

* Keep WhisperMLX large-v3 + pyannote.
* Keep two-pass Layer 2 → Layer 3 architecture.
* Keep Meetily for capture.
* Keep manual Claude EOD synthesis for now.
* Diarized path is the intended normal 1:1 production path after M3.
* Flat transcript path is the intended fallback after promotion.
* Group calls remain shadow-only.
* `dominant_two_plus_fragment_v1` is provisional for monitored field use, not an expert-final speaker-count classifier.
* Production persistence failure stops and never triggers flat fallback.
* Phone audio-first sources remain outside production until M5 admission is proven.
* Identity v0 is operator-confirmed speaker mapping.
* Persistent Sharan voice recognition remains a planned Identity v1, not the current milestone.
* Stop investing in additional attribution regex validators except evidence-backed truth checks needed by M3.
* Freeze the artifact-normalization prototype as a parallel/dormant prototype.
* Do not design around an assumed five-hour diarization runtime; measure real stage timing.

## LATEST VERIFIED CHECKPOINT

Implementation checkpoints:

* `4d7990e61cda994d5c32924ce66000f4f2341848` — M1 implementation ready for first gate.
* `c33321f8be118b9f4b64b68f2b2dfa6ac74104a5` — confirmed-identity propagation fixed in diarized Layer 3.
* `3e0fe6089cacdab5e86e98f2475a59a42ce49a94` — deterministic M2 production-state hardening.
* `4b5213877940a64409cd2efd9ba284fc71239930` — canonical durable source paths for selector-based production processing.
* `docs/checkpoint_m4_promote_diarized_1to1_path_pass_2026-07-07.md` — M4 diarized 1:1 production promotion PASS.

Known-good state:
July 7 M4 isolated production-routing proof and closure verification PASS.

Immutable checkpoint:
`docs/checkpoint_m4_promote_diarized_1to1_path_pass_2026-07-07.md`

Verified:

* M1, M2, M3, and M4 are PASS — CLOSED;
* supported validated 1:1 meetings select diarized production;
* ineligible meetings use flat and pre-persistence processing failures use identifiable `flat_fallback`;
* persistence failures stop in single-shot and watch modes;
* selected diarized Layer 2 artifacts are durable and lineage-validated;
* schema-v2 compatibility, slim records, atomic replacement, and rolling backups remain verified;
* exactly-once diarization and unattended anonymous speaker handling passed an isolated real-meeting proof;
* phone normalization and audio-first validation are built/tested but not production-integrated.

## BRANCH START RULE

At the start of a new ChatGPT or Codex milestone thread:

1. Read PROJECT_CONTROL.md.
2. State the ACTIVE milestone.
3. State its PASS gate.
4. State the single next action.
5. Do not propose work from NEXT, FOLLOW-UP, or PARKED unless a current blocker requires it.

The control file wins over conversational drift.
