"""Operator help, kept separate from processing and service initialization."""
from __future__ import annotations

OVERVIEW = """MeetingIntel — start with mi inbox

EVERYDAY USE
  mi ui                       Open the local browser inbox
  mi inbox                    Find a recording in Terminal
  mi queue today              Approve today's ready mixed-source set once
  mi queue status             Read saved queue state
  mi queue resume             Continue approved pending/deferred work
  mi learning status          Check local usage collection and next review
  mi record                   Record a laptop meeting; Ctrl-C stops it
  mi sync                     Download phone recordings from Google Drive
  mi today list               List today's Meetily sources without processing

CHECK PHONE DELIVERY
  mi sync status              Check delivery and settlement
  mi sync scheduler status    Check the optional two-hour fetch schedule

WHEN SOMETHING STOPS
  mi recordings diagnose last Check the latest laptop capture
  mi ollama status            Check the local analysis service
  mi help recovery            Read the next steps before retrying

Phone and guarded laptop transcription default to Qwen; Gemma writes notes.
Use mi --transcription-backend whisper to select Whisper explicitly.
Meetily exports retain Whisper. Opening the inbox is read-only;
processing requires selecting and confirming an item. Keep the lid open while
recording. Do not use mi all for routine recovery: it reprocesses old meetings.

MORE HELP
  mi help queue               Batch approval, queue status and resume
  mi help phone               Phone delivery, processing and Whisper override
  mi help recordings          Laptop recording and recovery
  mi help advanced            Legacy commands, schedules and owner Voice ID
  mi help setup               Installation, backup and restoring another Mac
  mi help all                 Show every topic

Written guide: docs/COMMANDS.md
"""

TOPICS = {
    "queue": """One approval, sequential local processing

  mi queue today                    Preview today's admitted sources; y approves
  mi queue status                   Read-only order, state and safe diagnostics
  mi queue resume                   Continue approved pending/deferred jobs
  mi queue resume --retry-failed     Preview and confirm failed/ambiguous retries

Today uses MI_TIMEZONE (Asia/Kolkata by default), oldest first. Includes normalized phone, normalized
laptop and ready Meetily sources. Completed sources are skipped. Blocked phone
items show settlement/grouping preparation; use mi inbox, then run queue today
again. Grouping, normalization and interrupted-capture attestations stay separate.
Previously approved normalized phone groups work through the same exact loader.

Keep Terminal open and the Mac awake. This is a foreground consumer, not a
service: closure, restart or crash stops advancement; sleep can pause or disrupt
work. No automatic restart or failed-job retry. A stale running job is ambiguous
and needs --retry-failed confirmation. An ordinary failure stays visible while
independent jobs continue. Busy waits up to ten minutes, then retains deferred
work for resume; GPU deferral stops the pass with work retained.

Only today's newly approved set runs from queue today; it never expands to old
backlog. Existing approved pending work is resumed deliberately with queue resume.
Manual/UI actions still use their existing Busy behavior; only this CLI queue
retains approved work. Models, identity policy and delivery paths are unchanged.
""",
    "phone": """Phone recordings

1. mi sync                         Download to local staging
2. mi sync status                  Check readiness (settlement takes 10 minutes)
3. mi inbox                        Review grouping, then confirm exact processing

Fetching does not transcribe. The optional scheduler fetches at login and
every two hours; it does not process meetings. Run mi help advanced for setup.

Qwen is the phone default. To choose Whisper explicitly:
  mi --phone-transcription-backend legacy inbox
To explicitly select Qwen:
  mi --phone-transcription-backend qwen inbox

Phone-only inventory: mi --audio-first today list
Expert local grouping: mi phone review today --dry-run
This recorder lane supports in-person recordings, not cellular/VoIP capture.
""",
    "recordings": """Laptop recordings

  mi record                         Start; keep the terminal and lid open
  Ctrl-C                            Stop and finalize; then choose processing
  mi recordings list                List saved captures
  mi recordings process last        Confirm processing of the latest capture
  mi recordings process <short-id>  Select one earlier capture
  mi recordings process <short-id> --speakers 2  Use a known speaker count
  mi recordings diagnose last       Inspect a failed/interrupted capture

Use mi help recovery if a recording stopped unexpectedly. Raw recordings stay
local under ~/Movies/MeetingIntel Recordings/. A forced lid close is not a
supported way to stop recording.
""",
    "recovery": """If a step fails

Phone not ready: mi sync status; allow the 10-minute settlement interval.
Fetch problem: mi sync scheduler diagnose (does not run transcription).
Analysis service problem: mi ollama status.
Interrupted laptop recording: mi recordings diagnose last.

For a clean capture awaiting normalization: mi recordings normalize-last.
For eligible terminal source loss only: mi recordings recover-last.
To keep an eligible prefix: mi recordings recover-last --keep-segments N.
Read the recovery preview; RECOVER attests that the meeting ended before loss.

After fixing the reported problem, return to mi inbox and select the exact
unprocessed source. Do not use mi all or --refresh-existing as a retry shortcut.
Accepted completed laptop/phone transcription can be reused on an exact retry
after source, model/configuration and artifact checks. Incomplete/rejected ASR
is not accepted. There is no general partial-ASR resume command. Keep originals.
""",
    "advanced": """Advanced commands

LEGACY MEETILY PROCESSING LANE
  mi / mi today / mi last / mi new
These do not include phone or guarded laptop recordings.
mi all deliberately reprocesses old meetings; it is not ordinary recovery.

  mi --audio-first new list          Read-only phone inventory
  mi --audio-first last              Process latest selected phone source
  mi phone review today              Expert grouping review

Optional fetch schedule (requires Google authorization):
  mi sync scheduler install         At login and every two hours
  mi sync scheduler status
  mi sync scheduler remove

  mi voice status                   Read owner Voice ID status
  mi voice automatic on|off         Enable/disable existing-owner matching
  mi voice automatic last           Present latest completed laptop/phone stage
  mi voice revoke FULL_FINGERPRINT  Keep this meeting anonymous
  mi voice review last              Explicit owner-only measurement/review
Automatic matches are separate from human confirmations. Unknown or ambiguous
voices stay anonymous. Existing enrollment consent is required; no new person
is enrolled by automatic processing. Names do not enter analysis prompts.
""",
    "setup": """Public setup

  docs/SETUP.md             Tools, models, settings and single-service Ollama
  docs/AI_SETUP.md          Instructions for your coding AI
  docs/PHONE_SETUP.md       Phone recorder, transfer, settlement and grouping
  docs/COMMANDS.md          Everyday commands and recovery
  docs/VOICE_ID.md          Optional calibration, consent and owner matching

Start with python3.11 scripts/setup_check.py --mode demo.
Qwen is the phone/laptop default. Select mi --transcription-backend whisper
for an explicit Whisper run; no hidden fallback. Meetily keeps Whisper.
Download model assets using your own accepted terms and credentials.
""",
}


def help_text(topic: str | None = None) -> str:
    if topic is None:
        return OVERVIEW
    if topic == "all":
        return OVERVIEW + "\n" + "\n".join(TOPICS.values())
    if topic not in TOPICS:
        raise ValueError("Unknown help topic. Choose: " + ", ".join(TOPICS) + ", all")
    return TOPICS[topic]
