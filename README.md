# Meeting Intelligence System

I built MeetingIntel to turn recordings into notes I can use: what was said,
what needs doing, and a short daily brief. The working system handles private
meetings, so this repository publishes its design rather than the full application.

**Looking for code to run?** Start with
[meetingintel-phone-ingest](https://github.com/St-Gonne/meetingintel-phone-ingest),
the Apache-2.0 toolkit for downloading, checking and grouping split phone recordings.
It can prepare audio for a separate transcription pipeline.

## Can I run the full application?

Not from this repository yet. The private app is now substantial enough to prepare
an Apple Silicon developer alpha. The main gaps are a clean install, portable
configuration, generic prompts and optional Voice ID setup. See the
[code-release assessment](docs/CODE_RELEASE_PLAN.md) for what exists and what still
needs to be done. The phone toolkit remains the runnable public part today.

## What changed by September 2026

The private system has moved beyond the July architecture snapshot:

- A local browser inbox shows recording, transcript, report and brief status
  separately. Opening the inbox does not process recordings; processing requires
  an explicit selection and confirmation.
- Capture handling distinguishes a recording that never started, an interrupted
  recording, and a completed recording. Saved audio is not proof that the whole
  meeting was captured.
- Phone transcription uses a local Qwen path; laptop and Meetily sources use
  Whisper. Local Gemma analysis produces the notes and brief. These private
  integrations are not included in the public phone toolkit.
- An explicit owner voice-review workflow exists locally. Automatic naming and
  enrollment of other people remain outside the accepted scope.
- The latest local work coordinates GPU use between applications. Joint synthetic
  caller checks passed; shared-runtime activation gates remain open in the latest
  receipt. This page does not claim live activation is complete.

These are summaries of local project records, not a public release or an
independently reproduced installation. The public toolkit has its own tests,
support limits and release history.

## The lesson that changed the interface

A transcript can be accurate for the audio that exists and still miss part of a
meeting because capture stopped early. Those are different failures.

The interface now keeps four questions separate:

1. Did recording start and continue for the intended meeting?
2. Is the saved audio intact?
3. Did transcription finish?
4. Did the report and brief finish?

Acknowledging an interruption alert answers none of the completeness questions.
Missing conversation cannot be reconstructed from a successful transcription.
The revised failed-start alert has synthetic checks; its revised native wording
still needs field verification.

## How the pieces fit

```text
consented laptop or phone recording
    -> capture / transfer checks
    -> explicit source and grouping review
    -> local transcription
    -> local analysis
    -> per-meeting note and daily brief
```

Phone delivery may pass through the operator's configured cloud storage.
Transcription and analysis run locally in the private system. The public toolkit
uses an existing rclone remote and does not manage cloud credentials.

## What you can reuse here

The main design ideas are separate evidence for every stage, preserved originals,
explicit decisions for ambiguous chunks, and small changes judged against a
written acceptance criterion. The [July control-file snapshot](docs/PROJECT_CONTROL.md)
shows how the project was run at that point. It is historical documentation,
not the current private task queue.

For executable code, setup instructions and bounded contribution opportunities,
use [the phone toolkit](https://github.com/St-Gonne/meetingintel-phone-ingest).
The full private pipeline, prompts, recordings, transcripts, identity records
and runtime state are not included here.

## Who did what

I chose the problems, workflow, privacy boundaries and acceptance criteria.
Codex wrote the implementation under those constraints. I use the system and
review the results; local tests and my acceptance do not establish general
transcription accuracy or reliability on somebody else's setup.

## License

This architecture record is available to read and share with credit under
[LICENSE.md](LICENSE.md). Commercial use and modified redistribution require
permission. The separate phone toolkit uses Apache-2.0.
