# Meeting Intelligence System

A meeting pipeline that runs entirely locally. It takes meeting recordings, made with participants' knowledge and consent (non-negotiable, and check your local law), and turns them into speaker-labelled intelligence notes + one short daily brief. Nothing leaves the machine. My meetings are live deal conversations, M&A, policy impact, negotiations, so privacy was the primary thing for me.

**Status: work in progress, in daily production use since early July 2026.** Five milestones passed so far, each one closed with a written pass gate and a checkpoint document.

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

Works: laptop-recorded 1:1 meetings run end to end every day... capture, transcript, who-said-what, intelligence note, daily brief. Five milestones passed on real meetings.

Not yet: phone recordings (lane built, final real-recording proof is the active milestone), group calls (shadow mode only), and the source code isn't published here yet.

## Repository status

This documents a working personal system I use in my daily workflow. Its not packaged as an installable app, and isn't trying to be one yet.

The operational system contains private recordings, transcripts, identity mappings, personal context and confidential business information. All of that stays in a separate private repository and will not be published. What I've made public is what I am focused on: the system design, the operating rules and quality gates, plus sanitized control documents, current limitations and the roadmap. The source code lands after a privacy pass (machine paths, personal specifics). Documents are the thing in my wheelhouse, so the docs first approach is as planned.

## Who wrote what

The code: OpenAI's Codex, working milestone by milestone.

Me: the product definition, the architecture boundaries, the control file, the quality gates, and the calls on what got parked, frozen or promoted. I don't write or read code, so the system had to be designed in a way where I never have to trust code I can't check. I gate on behaviour against real meetings instead.

## This is Interesting and took work: the control file

[docs/PROJECT_CONTROL.md](docs/PROJECT_CONTROL.md) is the live contract the AI builder works under, copied from the working system on 11 July 2026. A few of its rules do most of the work:

- One active milestone at a time. Everything else waits, no matter how interesting. Otherwise my ADHD had me chasing rabbitholes, new things to learn and experiment and constant dopamine from trying new things.
- A milestone passes only on proof from a real meeting, not on tests looking green. The written success bar for the whole product is that I use it on real meetings as part of my normal day, not that the pipeline produces technically correct output.
- Every new issue gets classified before it's allowed to consume attention: blocker, quality finding, follow-up or parked. Only a blocker may interrupt the active milestone.
- Parked ideas carry a written trigger for when they come back, so nothing gets lost and nothing sneaks back in early either. There are hundreds of these.
- "The control file wins over conversational drift." The builder re-reads it at the start of every session and states the active milestone before doing anything.

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

- **Phone recordings.** Most of my conversations don't happen at a laptop, they happen on calls and in rooms. The phone ingest lane is built and has passed its production-admission milestone; the final end-to-end proof on a real phone recording is the active milestone right now. Once that closes, laptop and phone feed the same daily brief.
- **Self notes.** Same pipeline, different input: voice memos to myself. After a meeting I couldn't record (no consent, no setup), or when a thought or learning pops up mid-day, I talk into the phone and it lands in the same intelligence flow as everything else. Planned, not built yet.
- **Group calls.** Working in shadow mode, but group speaker-separation has a lower confidence bar, so it stays out of production until it earns its own evidence gate.
- **Plenty more in the parking lot.** Every parked idea has a written trigger for when it comes back. The rule is the trigger promotes it, not my enthusiasm on a random Tuesday.

## License

Read it, learn from it, share it with credit. Not open source though: no commercial use and no modified redistribution without permission. Details in [LICENSE.md](LICENSE.md).
