"""Operator help, kept separate from processing and service initialization."""
from __future__ import annotations

OVERVIEW = """MeetingIntel — start with mi inbox

EVERYDAY USE
  mi ui                       Open the local browser inbox
  mi inbox                    Find a recording in Terminal
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

Phone transcription uses Qwen; Gemma writes the analysis and brief.
Laptop and Meetily transcription use Whisper. Opening the inbox is read-only;
processing requires selecting and confirming an item. Keep the lid open while
recording. Do not use mi all for routine recovery: it reprocesses old meetings.

MORE HELP
  mi help phone               Phone delivery, processing and Whisper override
  mi help recordings          Laptop recording and recovery
  mi help advanced            Legacy commands, schedules and owner Voice ID
  mi help setup               Installation, backup and restoring another Mac
  mi help all                 Show every topic

Written guide: docs/SETUP.md
"""

TOPICS = {
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
There is no general ASR checkpoint/resume command. Keep original recordings.
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
  mi voice review last              Explicit owner-only measurement/review
Voice ID is separate from anonymous diarization and needs its own consent.
""",
    "setup": """Public contributor setup

  docs/SETUP.md              Model-free demo and optional real-runtime setup
  docs/VOICE_ID.md           Voice ID API, calibration and consent boundaries
  docs/USAGE_LEARNINGS.md    Problems solved and remaining limits

Start with python3.11 app/demo_ui.py and python3.11 app/demo_voice_id.py.
Real capture/transcription require your own models, configuration and permissions.
This repository includes no private recordings, credentials or model weights.
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
