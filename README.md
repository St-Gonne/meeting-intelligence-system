# Meeting Intelligence System

A meeting pipeline that runs entirely on my Mac. It takes meeting recordings, made with everyone's knowledge and consent (non-negotiable, and check your local law), and turns them into speaker-labelled intelligence notes and one short daily brief. Nothing leaves the machine. My meetings are live deal conversations so privacy was the starting constraint, not something I added later.

**Status: work in progress, in daily production use since early July 2026.** Five milestones passed so far, each one closed with a written pass gate and a checkpoint document.

## Repository status

This documents a working personal system I use in my daily workflow. Its not packaged as an installable app, and isn't trying to be one yet.

The operational system contains private recordings, transcripts, identity mappings, personal context and confidential business information. All of that stays in a separate private repository and will not be published. What's public here is the system design, the operating rules and quality gates, plus sanitized control documents, current limitations and the roadmap. The source code lands after a privacy pass (machine paths, personal specifics). Docs first is deliberate, the documents are the part I actually author.

## Who wrote what

The code: OpenAI's Codex, working milestone by milestone.

Me: the product definition, the architecture boundaries, the control file, the quality gates, and the calls on what got parked, frozen or promoted. I don't write or read code, so the system had to be designed in a way where I never have to trust code I can't check. I gate on behaviour against real meetings instead.

## The part worth reading: the control file

[docs/PROJECT_CONTROL.md](docs/PROJECT_CONTROL.md) is the live contract the AI builder works under, copied from the working system on 11 July 2026. A few of its rules do most of the work:

- One active milestone at a time. Everything else waits, no matter how interesting.
- A milestone passes only on proof from a real meeting, not on tests looking green. The written success bar for the whole product is that I use it on real meetings as part of my normal day, not that the pipeline produces technically correct output.
- Every new issue gets classified before it's allowed to consume attention: blocker, quality finding, follow-up, or parked. Only a blocker may interrupt the active milestone.
- Parked ideas carry a written trigger for when they come back, so nothing gets lost and nothing sneaks back in early either.
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

## What I'd tell you it gets wrong

Speaker identification is genuinely unreliable at the source. The fix wasn't demanding perfect labelling (probably impossible), it was a standing rule downstream: never take a speaker label at face value, reconcile it against known context, flag rather than assert. A lot of what I do here comes down to pushing correctness to the layer where it's cheapest to get.
