# Meeting Intelligence System

A meeting pipeline that runs entirely locally. It takes meeting recordings, made with participants' knowledge and consent (non-negotiable, and check your local law), and turns them into speaker-labelled intelligence notes + one short daily brief. Nothing leaves the machine. My meetings are live deal conversations, M&A, policy impact, negotiations, so privacy was the primary thing for me.

**Status: work in progress, in daily production use since early July 2026.** Five milestones passed and closed, each with a written pass gate and a checkpoint document. The bet the earlier version of this page was still waiting on... that speaker-attributed processing could beat the simple path on real meetings and be promoted to production... has since paid off. Details below, including the parts that didn't work.

**Working code:** the phone-recording ingestion boundary is now available as a
separate Apache-2.0 project:
[meetingintel-phone-ingest](https://github.com/sharantulsiani-ui/meetingintel-phone-ingest).
It retrieves segmented recorder files, waits for stable uploads, prevents
duplicates, asks a human how chunks should be grouped, and emits canonical
audio for any transcription pipeline.

## What it produces

The daily brief is the whole point: one short read over morning coffee instead of replaying hours of calls. Here's the shape of it (illustrative sample, fictional names and deals... the real ones stay private):

```
DAILY BRIEF — sample

MEETING: 1:1 — Sharan x "Arjun" (AcmeFund)   [2 speakers, confirmed]

SIGNALS
- AcmeFund closing a $2M bridge into "Nimbus Games" this month;
  Arjun wants co-investor intros.
- Warm on a Q4 co-hosted founder event. Low commitment, revisit
  after their close.

ACTIONS (owner)
- Sharan: send Nimbus deck to Arjun by Friday.
- Arjun: share bridge terms note ("early next week").

WATCH
- "Meridian Studio" mentioned twice as acquisition-curious.
  New name, nothing on file yet.
```

Behind that sits a detailed per-meeting note (people, companies, deal signals, who committed to what), and behind that the full transcript, which almost never gets read.

## What works today, what doesn't yet

Works: laptop-recorded 1:1 meetings run end to end every day... capture, transcript, who-said-what, intelligence note, daily brief. The diarized path is now the normal one, promoted after it won its quality gate on real meetings. The old flat path stays as a labelled fallback.

Phone recordings: the in-person lane now works end to end. Scheduled
phone-to-laptop delivery has been proven on the real Mac workflow, and the
operator can review, join, separate or discard recorder chunks before choosing
whether to process exactly the normalized meeting. The reusable ingestion
boundary has been extracted into the working-code repository linked above.

Not yet: group calls remain outside the production claim, and the complete
private MeetingIntel pipeline is not published. The public code is deliberately
limited to the reusable phone-ingestion boundary.

One real limitation, stated plainly because it's the kind of thing that usually gets buried: **the phone lane records in-person conversations only. It does not capture phone calls**, cellular or VoIP. That's a coverage gap for the conversations that never touch a laptop, and I haven't measured how big it is yet. The next test is bounded... try the phone's native dialer recording, with consent, and see if it clears the quality bar. If it doesn't, that's a separate capture lane and a separate decision. What I'm not going to do is reach for an unverified call-recording app or work around the OS to close it faster.

## Repository status

This documents a working personal system I use in my daily workflow. The full
system is not packaged as an installable app. Its first reusable working
component is published separately as
[meetingintel-phone-ingest](https://github.com/sharantulsiani-ui/meetingintel-phone-ingest).

The operational system contains private recordings, transcripts, identity
mappings, personal context and confidential business information. All of that
stays in a separate private repository and will not be published. This
repository remains the system design, operating rules, quality gates, current
limitations and roadmap. Reusable code is extracted into fresh-history
repositories only after a privacy and dependency pass; the private repository
and its history are never made public.

## Who wrote what

The code: OpenAI's Codex, working milestone by milestone.

Me: the product definition, the architecture boundaries, the control file, the quality gates, and the calls on what got parked, frozen or promoted. I don't write or read code, so the system had to be designed in a way where I never have to trust code I can't check. I gate on behaviour against real meetings instead.

## This is Interesting and took work: the control file

[docs/PROJECT_CONTROL.md](docs/PROJECT_CONTROL.md) is the live contract the AI builder works under, copied from the working system on 16 July 2026 (two redactions: my machine path and a cloud folder name). A few of its rules do most of the work:

- One active milestone at a time. Everything else waits, no matter how interesting. Otherwise my ADHD had me chasing rabbitholes, new things to learn and experiment and constant dopamine from trying new things.
- A milestone passes only on proof from a real meeting, not on tests looking green. The written success bar for the whole product is that I use it on real meetings as part of my normal day, not that the pipeline produces technically correct output.
- Every new issue gets classified before it's allowed to consume attention: blocker, quality finding, follow-up or parked. Only a blocker may interrupt the active milestone.
- Parked ideas carry a written trigger for when they come back, so nothing gets lost and nothing sneaks back in early either. There are hundreds of these.
- "The control file wins over conversational drift." The builder re-reads it at the start of every session and states the active milestone before doing anything.

There's also a rule in there I'd put in front of anyone building with AI: never run capture, transcription, diarization or model calls just to validate documentation. Docs describe the system, they don't get to summon it.

## The rule earning its keep: I deleted a working feature

The cleanest proof the WIP cap is real, rather than a nice paragraph on a README, is what happened on 11 July. A live Gmail sync engine got built... 622 lines, tests, a privacy model, its own setup docs. Genuinely useful, and nowhere near the active milestone.

It survived about nineteen hours. The revert took out 1,160 lines. Same day I was busy publishing this repo for a deadline, which is exactly when you're most tempted to keep a shiny thing around because it looks good.

Parking it wasn't an option, half-built features rot and then lie to you about what the system does. So it went, with a trigger written down for when it comes back. Losing a day of the builder's work costs almost nothing. Carrying an unproven lane inside a system I'm trying to trust costs a lot.

## Documents have a chain of command

Once there were five milestones of history, the docs started disagreeing with each other, which is its own failure mode... every stale plan sounds authoritative if you read it on its own. So the docs got reconciled down to four current ones (control, architecture, operations, decisions) and everything else got demoted rather than deleted, under a written authority order:

```
1. current code + deterministic tests    <- what is actually true
2. the control file + current docs       <- what we've decided
3. dated checkpoints + proof artifacts   <- historical evidence
4. old plans, handovers, generated output <- dated, not current
```

Nothing is thrown away, it just loses the right to be believed. The lower tiers are evidence of what was true on a date, not instructions for today. This solved a problem I kept hitting, where an old handover doc would quietly overrule a newer decision because it happened to be the file I opened first.

## How it works, plain version

```
Record meeting (local)
   |
   v
Transcribe + work out who said what   (local models, nothing uploaded)
   |
   +-- Layer 1: full transcript        (kept, almost never read)
   +-- Layer 2: detailed note          (people, companies, signals, actions)
   +-- Layer 3: short action brief
   |
   v
One daily brief
```

New processing paths run in "shadow" alongside the trusted path, and they only get promoted after passing a gate on five consecutive real meetings, judged by me on things I can actually verify: did it miss anything important, did it put words in the wrong person's mouth, did it assign actions to the right owner. The old path stays around as a clearly labelled fallback.

```
trusted path ────────────────────────────► production (daily brief)
                                                ▲
new path (shadow) ──► GATE: 5 consecutive ─────┘
                      real meetings,
                      judged by me
                          │
                      fails? stays in shadow.
                      production never notices.
```

## What I'd tell you it gets wrong

Speaker identification is genuinely unreliable at the source. The fix wasn't demanding perfect labelling (probably impossible), it was a standing rule downstream: never take a speaker label at face value, reconcile it against known context, flag rather than assert. A lot of what I do here comes down to pushing correctness to the layer where it's cheapest to get.

## Where this is heading

- **Phone recordings.** The in-person phone lane and scheduled delivery are now
  working in daily use. The reusable retrieval, settlement, chunk-review and
  normalization boundary is published as
  [meetingintel-phone-ingest](https://github.com/sharantulsiani-ui/meetingintel-phone-ingest).
  Cellular and VoIP call capture remain outside the supported lane.
- **Self notes.** Same pipeline, different input: voice memos to myself. After a meeting I couldn't record (no consent, no setup), or when a thought or learning pops up mid-day, I talk into the phone and it lands in the same intelligence flow as everything else. Planned, not built yet.
- **Group calls.** Working in shadow mode, but group speaker-separation has a lower confidence bar, so it stays out of production until it earns its own evidence gate.
- **Plenty more in the parking lot.** Every parked idea has a written trigger for when it comes back. The rule is the trigger promotes it, not my enthusiasm on a random Tuesday.

## License

This case-study repository remains under its existing read/share licence.
The working phone-ingestion code is separately available under Apache-2.0.
Details are in each repository's licence file.
