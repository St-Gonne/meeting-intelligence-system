# Current public scope — 6 November 2026

The public application now includes the recent speaker export/overlap repairs,
Qwen window recovery, accepted-stage reuse, source-day delivery fixes, explicit
speaker count and the foreground processing queue. Qwen is the default for phone
and guarded laptop audio; Whisper is an explicit option. Meetily retains Whisper.
The public installer path uses one managed Ollama service by default.

This remains a contributor release. The existing local installation has practical
use and recovery evidence, but ordinary first-try stability, whole-call speaker
accuracy and a complete second-machine capture-to-report setup remain unproven.
A passing synthetic suite is not evidence of those outcomes. No universal minimum
RAM or Windows/full Linux processing support has been established. Demos and
portable core checks support macOS/Linux; full processing targets Apple Silicon.

Owner matching is optional, disabled on a new install, and requires the new user's
own calibration, retained enrollment and consent. Automatic suggestions remain
separate from human confirmations. No voiceprints, author calibration constants,
private reference media or visual-speaker experiments are distributed.

Current guidance: README.md, AGENTS.md, AI_SETUP.md, SETUP.md, PHONE_SETUP.md,
TUTORIAL.md, COMMANDS.md and VOICE_ID.md. Earlier PROJECT_CONTROL.md and
CODE_RELEASE_PLAN.md under docs/ are explicitly historical.

Recording metadata and phone filename timestamps currently require Asia/Kolkata.
General timezone portability remains outside this release; unsupported settings
fail before processing rather than creating invalid canonical sources.

Dependency maintenance on 6 November updates compatible non-ML security pins and
the matching OpenTelemetry family. Torch/Transformers advisories in the shared
Whisper speaker runtime remain open and block a security-cleared full setup, even
with Qwen transcription. See [dependency security status](DEPENDENCY_SECURITY.md)
and [issue #15](https://github.com/St-Gonne/meeting-intelligence-system/issues/15).
