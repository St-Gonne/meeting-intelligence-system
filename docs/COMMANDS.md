# MeetingIntel command sheet

These examples use `mi` after setup. From the checkout, replace `mi` with
`/absolute/path/to/whisper-env/bin/python /absolute/path/to/repo/app/mi` if you
have not added a launcher to PATH. Use the same interpreter that has the runtime.

| Task | Command / action |
|---|---|
| Help / local inbox | `mi help` / `mi ui` |
| Select a recording in Terminal | `mi inbox` |
| Record laptop audio | `mi record`; keep lid and Terminal open; Ctrl-C to stop |
| Process latest clean capture | `mi recordings process last` |
| Known speaker count | `mi recordings process SHORT_ID --speakers 2` |
| Choose Whisper for this run | `mi --transcription-backend whisper inbox` |
| Choose Qwen for this run | `mi --transcription-backend qwen inbox` |
| Preview / fetch phone delivery | `mi sync --dry-run` / `mi sync` |
| Check phone readiness | `mi sync status` |
| Review phone grouping | `mi inbox` or `mi phone review today` |
| Approve today's ready set | `mi queue today` |
| Queue state / continue pending work | `mi queue status` / `mi queue resume` |
| Preview and approve failed retries | `mi queue resume --retry-failed` |
| Diagnose capture / analysis service | `mi recordings diagnose last` / `mi ollama status` |
| Retry clean normalization | `mi recordings normalize-last` |
| Preview eligible interrupted recovery | `mi recordings recover-last` |
| Owner Voice ID / automatic policy | `mi voice status` / `mi voice automatic off` |

Only exact lowercase `y` starts selected processing. Phone settlement needs two
unchanged fetch observations at least ten minutes apart. Fetching never starts
transcription. A recovery `RECOVER` statement is a separate factual attestation;
read it and do not make it if the meeting continued after capture stopped.

Keep Terminal open and the Mac awake for a queue. No automatic crash restart or
failed-job retry. A resumed laptop job retains its originally selected backend.
Use exact-source recovery rather than `mi all` or refreshing historical meetings.
Results live in the configured data root's `output/`; use the inbox to open the
transcript, report and daily brief. Keep originals and back up private data.
