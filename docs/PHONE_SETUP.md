# Phone recordings: recorder to your computer

The supported capture lane is **ASR Voice Recorder on Android, recording in-person
meetings as `.m4a`**. It is not a cellular or VoIP call recorder. Ask which phone and
recorder the user has before setting anything up. iPhone/other-recorder exports
need a separate format/timestamp/continuity check; do not promise automatic grouping.

The flow is:

**Phone records → recorder uploads to Drive → `mi sync` downloads → files settle →
you review grouping → canonical audio → you approve transcription and notes.**

Phone filename timestamps currently use Asia/Kolkata (+05:30). If the recorder
uses another timezone, stop before importing: a verified timezone-aware mapping
is not included in this release. Do not relabel times or weaken source validation.

## 1. Set up and test the recorder

Use the recorder's microphone/audio permissions and choose `.m4a`. Its known
15-minute split pattern can help preserve long meetings; keep the real recording
start timestamps in filenames. The conservative parser handles names such as
`2026_11_05_12_00_00.m4a`. Do not relabel exports with invented dates or infer that
nearby files must be one meeting. Make a short test, play it and confirm that the
phone actually captured the intended sound before relying on it.

Configure the recorder's Drive upload to a dedicated folder, for example
`PhoneRecordings`. Check the phone's upload/background/battery settings and verify
the actual completed upload. The AI may explain menus, but the user completes
phone permissions and Google login. Cloud transfer means those recordings exist
in the user's Drive; transcription and report generation happen on their computer.

## 2. Authorize downloads on the computer

Install rclone and run `rclone config`. Create a Google Drive remote named
`meetingintel-drive` using the user's own account and **drive.readonly** scope.
The user completes the browser login; never copy the author's remote or tokens.
See the [official Drive backend guide](https://rclone.org/drive/).

Drive read-only scope is account-wide, not a Google-enforced folder-only grant.
MeetingIntel limits its calls to the configured recorder folder. Explain this
before authorization; a dedicated account/shared recorder folder is an option.

Save the folder in `app/.local/settings.json` alongside runtime settings:

```json
{
  "MI_PHONE_REMOTE": "meetingintel-drive:PhoneRecordings"
}
```

Merge that key into existing settings; do not replace other configuration. Match
the actual uploaded folder. Do not point it at the whole Drive or a mixed archive.

## 3. Fetch, settle and review

```sh
mi sync --dry-run
mi sync
mi sync status
```

Dry-run lists without downloading; it still accesses the authorized remote.
Files become ready only after **two unchanged successful fetch observations at
least ten minutes apart**, with local size/hash checks. After the interval:

```sh
mi sync
mi sync status
mi inbox
```

Review which chunks form each meeting. The code distinguishes joining, separating
and discarding from review; discarding does not delete the original staged file.
The supported continuity rule uses approximately 900-second chunks and conservative
gap bounds. Ambiguous sequences need your decision. Never join unrelated calls
just because their times are close. Originals remain retained.

Review creates canonical `audio.m4a` plus provenance under the data root's
`ingest/phone/`. Only selected normalized sources enter processing. Select the
exact item in `mi inbox` or approve ready recordings using `mi queue today`.
Fetching, grouping and processing are separate operations.

## 4. Optional delivery schedule

After a real on-demand fetch works, the user can choose:

```sh
mi sync scheduler install
mi sync scheduler status
mi sync scheduler diagnose
```

This macOS schedule fetches at login/every two hours. It does not group, transcribe
or summarize. A later fetch must still establish settlement. Install it only if
wanted. `mi sync scheduler remove` removes the schedule and preserves recordings.

## When delivery stops

- Pending settlement: wait, then fetch again; an upload change restarts the timer.
- Login expired: repair the user's rclone authorization, then retry delivery.
- Same remote ID moved or conflicting local file: stop and reconcile the exact
  object; do not delete the ledger or copy files around to hide the conflict.
- Unsupported filename/recorder: retain the original and identify its real time
  and chunk format before choosing a separate import path. The standalone
  [phone-ingest toolkit](https://github.com/St-Gonne/meetingintel-phone-ingest)
  is useful for testing this boundary, but also has explicit format limits.

A new phone setup is complete only after a real short recording has uploaded,
settled, been reviewed and produced checked local results. Desktop tests do not
prove phone background uploads or microphone continuity.
