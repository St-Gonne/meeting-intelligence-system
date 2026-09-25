# Setup: demos first, real models separately

The public release targets contributors comfortable with Python and local models. The model-free inbox and Voice ID demos are the tested first-run path. A complete new-machine recording-to-analysis setup has not yet passed a public clean-room test.

## Demos and tests

Use Python 3.11. The two demos need only the standard library:

```sh
python3.11 app/mi help
python3.11 app/demo_ui.py
python3.11 app/demo_voice_id.py
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/test_public.py
```

`demo_ui.py --no-browser --port 8765` prints its per-session loopback URL. Use the entire printed URL; the fragment carries the local session token. Do not publish that URL. The demo has a read-only synthetic adapter; no processing action is enabled.

## Runtime paths

`app/mi_paths.py` replaces the original machine paths. Application data defaults to the ignored `app/.local/` tree. Moving or deleting that directory can destroy your locally created state; keep it outside Git and back it up privately if you use real data.

| Variable | Purpose |
|---|---|
| `MI_DATA_ROOT` | Absolute private directory for runtime data |
| `MI_PHONE_SOURCE` | Local folder containing phone recording inputs |
| `MI_MEETILY_ROOT` | Completed Meetily export folders |
| `MI_QWEN_PYTHON` | Python in the dedicated Qwen runtime environment |
| `MI_QWEN_MODEL_DIR` | Locally installed pinned Qwen model snapshot |
| `MI_WHISPER_PYTHON` | Python in the separate Whisper/diarization environment |
| `MI_WHISPERMLX_BIN` | Installed `whispermlx` executable |
| `MI_OBS_APP` | OBS application path, default `/Applications/OBS.app` |

Use the `app/mi` launcher: it also isolates UI, operation and learning state. Do not point the public checkout at an existing private install during evaluation. Library scripts are developer APIs and can have their own explicit output arguments; inspect `--help` before invoking them.

## Optional real processing

1. Install FFmpeg and ffprobe on PATH. Laptop capture additionally needs Node, OBS, its local control setup, microphone/screen permissions and an awake Mac. The capture scripts are included, but automatic OBS provisioning is not.
2. Build **separate** Python environments for Qwen and Whisper/pyannote. `app/requirements-transcription-qwen.txt` records the current Qwen pins. These are an adopted baseline, not a promise that every wheel works on every machine.
3. Download your own permitted model assets. The phone path expects `Qwen/Qwen3-ASR-1.7B-hf` at revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`. Whisper/pyannote assets and access terms are separate. No token or model weights ship here.
4. Inspect `app/config/meetingintel.restore.modelfile` for the current analysis recipe. The app expects the `meetingintel` alias on exact loopback port **11435** and verifies its pinned digest. It does not silently pull or replace a model. A different digest requires a deliberate evaluated configuration change in `ollama_endpoint.py`.
5. Read `gpu_lock.py` before running inference. The adopted guard currently expects the two loopback services on 11434/11435, one loaded model per server and verified lifecycle ownership. It fails closed when that setup is missing. The single-service portable setup is an open contribution target; do not bypass the guard to make a demo appear successful.
6. For phone Drive delivery, install rclone and configure your own read-only remote and source folder. Google authorization is user-controlled. For a narrower, independently released ingestion tool, start with [meetingintel-phone-ingest](https://github.com/St-Gonne/meetingintel-phone-ingest).
7. Run read-only help, status and source listing before selecting one consented recording. Confirm the exact recording; keep raw media. Compare source duration, transcript coverage and named facts manually before relying on the report.

## Voice ID setup

The portable review API accepts your own evaluation report and model digest. The owner review command still expects a separately prepared calibration, retained owner enrollment and completed supported phone diarization. It is not a one-command enrollment wizard. [Voice ID documentation](VOICE_ID.md) explains the API and the unfinished onboarding work.

## Model licensing

The source license does not grant rights to model weights, OBS, rclone or dependencies. Obtain them under their own upstream terms. No hosted inference API is required by the published demo. Offline flags in optional adapters control model fetching; they are not a network sandbox.

On Linux, run `.venv/bin/python scripts/test_public.py --portable`. The full suite also checks Mac-only capture, AppKit and launchd integration fixtures. Linux core/demo success is not Linux recorder support.
