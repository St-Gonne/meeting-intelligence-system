# Dependency security — 6 November 2026

Full runtime setup is **not security-cleared**. The model-free demos and synthetic
suite do not require the affected runtime. Qwen transcription still depends on
the Whisper environment for alignment and diarization, so selecting Qwen does not
remove the shared-runtime risk.

## Compatible maintenance

The security-updated baseline requirements retain the existing ML core versions
and update Certifi, Filelock, Pip, Setuptools and urllib3 in both environments;
msgpack in Qwen; and aiohttp, Lightning/PyTorch Lightning, Mako, multidict, NLTK
and Pillow in Whisper. The complete tightly coupled OpenTelemetry family moves
to 1.45.1 / 0.66b1. New required companion packages and yarl 1.25.1 are pinned.

Certifi is 2026.10.23, newer than the advisory's first patched release. These files
are derived from the September working snapshots, not exact installed-environment
freezes. No installed production environment is changed by editing them.

## Remaining blocker: Torch and Transformers

[Issue #15](https://github.com/St-Gonne/meeting-intelligence-system/issues/15)
tracks the remaining advisories. The current WhisperMLX 3.12.2 and latest 3.14.0
PyPI metadata both require Torch `~=2.8.0`, TorchAudio `~=2.8.0` and TorchVision
`~=0.23.0`. Current open Torch advisories require releases beyond that constraint.
The Whisper environment's Transformers 4.57.6 also has open advisories.

Dependabot PR #14 proposes a major Torch/Transformers upgrade while leaving old
TorchAudio/TorchVision and Hub/tokenizers pins. Updating only OpenTelemetry API
also conflicts with its pinned SDK. That grouped proposal must not be merged as
an ordinary security patch. Newer Qwen Torch/Transformers pins do not need those
unrelated major upgrades for the currently observed alerts.

Do not force incompatible versions with `--no-deps`, dismiss the remaining alerts
as false positives, load untrusted model artifacts, or describe a Qwen-only
transcription choice as avoiding this blocker. A maintained compatible upstream
combination or a scoped replacement needs independent validation first.

## Validation and limits

- Both complete updated requirement lists passed pip `--dry-run --ignore-installed`
  resolution on Apple Silicon / Python 3.11.15. Resolution is not installation.
- All 25 changed/new pins were installed into a disposable library overlay;
  smoke checks use the unchanged installed ML core. This does not establish a
  clean full installation, model quality or real-audio inference. All 13 upstream
  speaker-assignment/export synthetic tests passed with this overlay.
- Required synthetic macOS/Linux CI and the source publication guard must pass
  before merging. They do not install or exercise these ML runtimes.
- Recheck live Dependabot alerts after the default branch updates. Keep any
  unresolved alerts open; advisory counts and package availability can change.

Before recommending full setup, resolve issue #15, install into new Apple Silicon
Python 3.11 environments, check dependencies/imports and speaker integration, then
conduct a separately authorized short real-audio trial. Preserve existing apps,
models, recordings and account configuration throughout.

Primary metadata: [WhisperMLX 3.12.2](https://pypi.org/pypi/whispermlx/3.12.2/json),
[WhisperMLX 3.14.0](https://pypi.org/pypi/whispermlx/3.14.0/json),
[OpenTelemetry SDK 1.45.1](https://pypi.org/pypi/opentelemetry-sdk/1.45.1/json).
