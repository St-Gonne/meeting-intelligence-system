# Security status and reporting

This is an early local application. A green synthetic build does not certify the
security of its model runtimes or establish real-call readiness.

Read [current dependency status](docs/DEPENDENCY_SECURITY.md) before installing.
The shared Whisper runtime has unresolved Torch/Transformers advisories tracked
in [issue #15](https://github.com/St-Gonne/meeting-intelligence-system/issues/15).
Qwen transcription still uses that environment for alignment and diarization.

Use GitHub private vulnerability reporting if it is offered on the Security tab
for a sensitive report. Do not post credentials, private audio, transcripts,
voiceprints, personal identifiers or exploit material containing private data in
public issues. Ordinary dependency/setup issues should use sanitized version
information and synthetic reproduction steps.

Preserve existing working environments. Test dependency changes in disposable,
separate environments, respect upstream constraints, and retain unresolved alerts
until a compatible remediation is verified. Never bypass dependency resolution
or disable TLS verification to make setup appear successful.
