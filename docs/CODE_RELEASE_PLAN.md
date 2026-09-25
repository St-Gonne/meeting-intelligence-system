> Historical document, retained for context. The public application alpha and Voice ID source were released on 25 September 2026. See the [current README](../README.md), [setup](SETUP.md) and [validation](RELEASE_VALIDATION.md) for current public scope.

# Can someone else use MeetingIntel yet?

Assessment: 25 September 2026.

**The phone-ingest toolkit can be used now. The full application is close enough
to prepare a developer alpha, but it is not yet a clean install for another person.**
The gap is packaging, portability and validation on a second setup. It is not a
lack of features to demonstrate.

## What exists in the private application

- A local inbox and explicit recording/processing actions.
- Laptop capture, phone intake, local transcription and local analysis.
- Notes, follow-ups and a daily brief.
- Capture interruption handling that preserves available audio without pretending
  it captured the whole meeting.
- An optional, explicit owner Voice ID review workflow.
- A source-only backup with an audited list of 146 files and restore instructions.

The backup is useful preparation, but it is a private recovery package. It includes
business prompts and machine-specific setup that have not been approved as a
public general-purpose distribution. Twenty-five of its allowlisted text files
still contain original machine home paths, including documentation and tests.

## What Voice ID actually does

Diarization separates voices into anonymous speaker labels. Voice ID compares a
speaker against a previously enrolled voice. They are different jobs.

The current accepted path is deliberately narrow:

1. An owner enrolls with explicit consent and a local evaluation.
2. After a supported phone meeting is diarized, the operator explicitly requests
   a voice review.
3. The system measures a candidate match locally.
4. The operator confirms the identity for that meeting and separately controls
   retention.

One owner match on a real 1:1 meeting was accepted. That establishes an example,
not recognition accuracy across users, accents, microphones or group calls.
Automatic naming and additional-person enrollment are not accepted capabilities.
The current CLI also expects the owner's local evaluation report and model asset;
a new user cannot just inherit that enrollment or threshold.

No voiceprints, enrollment clips, identity maps or private meeting data should
ship with the software. A new user needs their own enrollment and calibration.
Voice ID should remain optional and off by default in the first public alpha.

## Recommended alpha

Target technical users on Apple Silicon Macs first. Publish a clean source
package, not the private repository's history:

- Configurable runtime roots and model locations instead of author-specific paths.
- A synthetic example from audio through notes, with generic prompts.
- Separate, documented Whisper and Qwen environments.
- Explicit model download/terms steps; no model weights or tokens bundled.
- A read-only setup check for dependencies and configuration.
- A small local inbox and exact-source processing, with no automatic backlog run.
- Voice ID as an optional experimental module with its own evaluation guide.

A first release need not include every recorder, scheduler, business prompt or
cross-application GPU integration from the author's setup.

## Remaining release work

| Work | Why it matters |
| --- | --- |
| Review and sanitize a public file manifest | The private backup passes its own scan, which is not a public-content review. |
| Remove fixed machine paths and household/business assumptions | Restore-by-replacing-the-author's-home-directory is not a normal install experience. |
| Generic prompts and model configuration | Others need reusable behavior without the author's business context. |
| Independent clean install with generated audio | Existing local and restore evidence does not establish that another user can finish setup. |
| Model/dependency inventory and license choice for the full code | The architecture license and the toolkit's Apache-2.0 license do not automatically cover a new application release. |
| Voice enrollment/evaluation workflow | The current owner-specific report is not a universal threshold or setup fixture. |
| GPU activation closure or a bounded standalone alternative | Joint synthetic caller checks passed; remaining shared-runtime activation gates were still open in the latest receipt. |

No local models, recordings or capture processes were run or altered for this
readiness review. The full source has not been published by this update.

## Where contributions can start today

Use [meetingintel-phone-ingest](https://github.com/St-Gonne/meetingintel-phone-ingest)
for code changes now. For the full application, design feedback on installation,
configuration and evaluation is useful while the clean alpha is prepared. Do not
send real recordings, enrollment samples, identities, transcripts or private logs.
