# Setup on your computer

Ask your coding AI to follow [AI_SETUP.md](AI_SETUP.md), or use these steps.
Demos need Python 3.11 and run on macOS/Linux. Full processing targets an Apple
Silicon Mac. Windows, full Linux processing and a universal minimum RAM are not
established. Check available memory/disk before downloading the large analysis
model; model files, duplicate caches and recordings need substantial space.

## 1. Run the demos first

```sh
git clone https://github.com/St-Gonne/meeting-intelligence-system.git
cd meeting-intelligence-system
python3.11 scripts/setup_check.py --mode demo
python3.11 app/mi help
python3.11 app/demo_ui.py
python3.11 app/demo_voice_id.py
```

Stop the inbox demo with Ctrl-C. It uses invented items and cannot process them.
For automated checks, create `.venv`, install requirements-test.txt and run
`.venv/bin/python scripts/test_public.py` on macOS (`--portable` on Linux).

## 2. Install tools and separate runtimes

Use official [Homebrew](https://brew.sh/), [Ollama](https://ollama.com/download),
[OBS](https://obsproject.com/download) and [rclone](https://rclone.org/install/)
sources. With Homebrew available, the AI can install Python 3.11 and FFmpeg;
phone delivery additionally needs rclone, and laptop capture needs Node 22+
(with WebSocket) and signed OBS. Keep your existing installations and model stores.
The user grants microphone and screen/system-audio permissions themselves.

Rebuild the recorded environments before attempting unreviewed package upgrades:

```sh
python3.11 -m venv "$HOME/.local/share/meetingintel/runtimes/whisper"
"$HOME/.local/share/meetingintel/runtimes/whisper/bin/python" -m pip install -r app/config/requirements-whispermlx-snapshot.txt
python3.11 -m venv "$HOME/.local/share/meetingintel/runtimes/qwen"
"$HOME/.local/share/meetingintel/runtimes/qwen/bin/python" -m pip install -r app/config/requirements-qwen-snapshot.txt
```

Qwen and Whisper use different Torch/Transformers versions. Keep both environments
separate. The Whisper environment also supplies alignment and diarization when
Qwen transcribes. If choosing only Whisper, the Qwen environment/download is optional.
These snapshots describe a working baseline; package availability on a fresh
machine is not guaranteed. If installation fails, retain the error and report the
specific version. Do not silently replace the model/backend to claim success.

## 3. Download your model assets

The AI can run downloads under your direction. No weights, credentials or
enrollment material ship with the code. You handle any host login and model terms.

| Role | Asset |
|---|---|
| Default phone/laptop transcription | [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |
| Explicit Whisper / Meetily transcription | `mlx-community/whisper-large-v3-mlx` |
| Anonymous speaker separation | [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| English/Hindi alignment | TorchAudio `WAV2VEC2_ASR_BASE_960H`; `theainerd/Wav2Vec2-large-xlsr-hindi` |
| Report and brief | Included `app/config/meetingintel.restore.modelfile`; baseline/digest in `app/config/runtime-versions.json` |

Keep the standard Hugging Face cache for the alignment/Whisper adapters, which
validate cached model provenance. For the selected Qwen backend:

```sh
"$HOME/.local/share/meetingintel/runtimes/qwen/bin/hf" download Qwen/Qwen3-ASR-1.7B-hf --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3
```

Use the printed snapshot directory as MI_QWEN_MODEL_DIR. Its final directory name
must remain the revision hash; do not rename it to `qwen`. The runtime validates
model identity, asset hashes and exact-source lineage.

Download shared runtime assets using the Whisper environment's `hf`. The first
command downloads Whisper transcription weights and can be omitted for a Qwen-only
setup that does not process Meetily exports:

```sh
"$HOME/.local/share/meetingintel/runtimes/whisper/bin/hf" download mlx-community/whisper-large-v3-mlx
"$HOME/.local/share/meetingintel/runtimes/whisper/bin/hf" download theainerd/Wav2Vec2-large-xlsr-hindi
"$HOME/.local/share/meetingintel/runtimes/whisper/bin/hf" download pyannote/speaker-diarization-community-1
"$HOME/.local/share/meetingintel/runtimes/whisper/bin/python" -c 'import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()'
```

Community-1 can require its referenced segmentation/embedding assets too; have the
AI check the downloaded pipeline configuration and resolve those permitted assets
using the installed pyannote runtime before the first meeting. A download of one
repository alone is not proof that its referenced assets are ready. The final short
trial is required. Existing asset caches and your own model licenses remain separate.

The CLI requires HF_TOKEN for processing. Authenticate locally, then place your
own token in the configured private data root's `.secrets/meetingintel.env` as
`HF_TOKEN=...`. Restrict that directory/file to 700/600. Never put a token in chat,
settings.json, Git, issue logs or a generated tutorial.

## 4. Save settings and a launcher

The launcher reads ignored `app/.local/settings.json`, or the file selected by
MI_CONFIG_FILE. Existing environment values take precedence. Have the AI write
actual absolute paths, for example:

```json
{
  "MI_DATA_ROOT": "/absolute/private/MeetingIntel-data",
  "MI_WHISPER_PYTHON": "/absolute/runtimes/whisper/bin/python",
  "MI_WHISPERMLX_BIN": "/absolute/runtimes/whisper/bin/whispermlx",
  "MI_QWEN_PYTHON": "/absolute/runtimes/qwen/bin/python",
  "MI_QWEN_MODEL_DIR": "/absolute/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
  "MI_TRANSCRIPTION_BACKEND": "qwen",
  "MI_GPU_MODE": "standalone",
  "MI_TIMEZONE": "Europe/London"
}
```

Select your actual timezone. The compatibility default is Asia/Kolkata; source-day
fields retain their existing schema names. MI_DATA_ROOT contains private outputs,
state, staging, canonical input and capture files. Keep it outside Git/cloud sync,
with private permissions and a separate backup. MI_VOICE_ROOT can select a separate
private, non-repository/non-cloud parent for optional voice enrollment; its default
is `~/Library/Application Support/MeetingIntel-public`.

Other supported keys: MI_PHONE_REMOTE, MI_PHONE_SOURCE (expert local normalizer
input), MI_MEETILY_ROOT, MI_OBS_APP. Paths used for production must be absolute.
Phone Drive review uses its managed staging and ledger, not an arbitrary folder.

Run the public launcher with the Whisper interpreter. Create a small `mi` wrapper
on PATH that executes that absolute interpreter and this checkout's `app/mi`,
forwarding `"$@"`. Preserve an existing `mi` command; use a different wrapper name
if another installation owns it. Settings persist across new terminals and UI workers.
Do not point this checkout at an existing private installation during evaluation.

Qwen is the default for phone and guarded laptop recordings. Explicit alternatives:

```sh
mi --transcription-backend whisper inbox
mi --transcription-backend qwen recordings process last
```

The choice applies to phone/laptop work in that invocation. A phone-only legacy
flag remains supported. Meetily exports retain Whisper. A Qwen failure stops;
there is no hidden fallback. Queue admissions freeze their backend, so a later
settings change does not silently switch an approved laptop/phone job.

## 5. Set up one Ollama service

A normal MeetingIntel install needs **one** managed service on **127.0.0.1:11435**.
Qwen/Whisper are separate transcription runtimes, not second Ollama servers.
Use `mi ollama --help`, install/start the managed service, then create its model:

```sh
mi ollama install
mi ollama start
OLLAMA_HOST=127.0.0.1:11435 ollama create meetingintel -f app/config/meetingintel.restore.modelfile
mi ollama status
```

The [Ollama import guide](https://docs.ollama.com/import) explains the create step.
The service must have OLLAMA_MAX_LOADED_MODELS=1; the managed installer supplies it.
The alias digest must match the recorded baseline. Upstream tags/Ollama versions
can change the result. Recover the recorded baseline or review a deliberate model
change; do not patch out the digest check. The recorded Ollama version is 0.32.14.

If another local AI app uses the GPU, choose `MI_GPU_MODE: shared` and coordinate
both exact-loopback services on 11434/11435 through the same guardian/lock, one
loaded model per service. Standalone mode is for a dedicated MeetingIntel runtime;
it does not coordinate arbitrary simultaneous GPU users. Never create a dummy
second service or disable cleanup to make a check pass.

## 6. Finish the selected recording lane

For phone recordings, follow [PHONE_SETUP.md](PHONE_SETUP.md). For laptop capture,
install signed OBS, select the intended microphone and system-audio source, and
prepare/verify the dedicated profile and authenticated control configuration using
`mi record` help and `app/obs_capture_probe.py --help`. Source/device settings vary;
automatic OBS device provisioning is not promised. The AI should inspect the guard's
preflight error and help configure OBS, rather than weakening it. Keep the lid open.

```sh
python3.11 scripts/setup_check.py --mode runtime --phone --capture
```

Omit unneeded lane flags. Run this with the same settings/MI_CONFIG_FILE as the
launcher. For Qwen, run the selected Qwen interpreter on
`app/meetingintel_qwen_asr_runtime.py --check --model-dir YOUR_SNAPSHOT` to verify
Metal, FFmpeg and the actual assets. No inference is run by that dependency check.

Finish with one explicitly selected short trial. Verify capture sound, duration,
transcript coverage, speaker turns, report and brief. Leave a private setup receipt
and personalized [tutorial](TUTORIAL.md)/[command sheet](COMMANDS.md). Full new-machine
end-to-end readiness remains unproven until these actual steps pass on that machine.
Optional Voice ID follows [VOICE_ID.md](VOICE_ID.md); ordinary notes work anonymously.
