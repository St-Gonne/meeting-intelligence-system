# A first run without recordings or models

You need Python 3.11 on macOS or Linux. This walkthrough does not install models or use your microphone.

```sh
git clone https://github.com/St-Gonne/meeting-intelligence-system.git
cd meeting-intelligence-system
python3.11 app/demo_ui.py
```

1. The browser opens the actual inbox using invented items. Choose the **interrupted recording**.
2. Check the separate recording, transcript, report and brief states. Is it clear that a saved transcript does not mean the whole meeting was captured?
3. Open the available transcript. It is an invented two-line example. Processing is deliberately disabled in this demo.
4. Stop the server with Ctrl-C. Then try the standalone Voice ID lifecycle:

```sh
python3.11 app/demo_voice_id.py
```

5. Look for the difference between a candidate match, confirming it for one meeting, retaining an enrollment and revoking it. This uses invented vectors: it tests decisions and state transitions, not recognition accuracy.

[Report your first run](https://github.com/St-Gonne/meeting-intelligence-system/issues/new?template=first-run.yml).
Please include your OS, Python version and commit. Do not paste the token-bearing browser URL, recordings, voiceprints or private paths.

If you want to contribute, start with [clean-machine setup documentation](https://github.com/St-Gonne/meeting-intelligence-system/issues/10).
For local-AI work, [measure time by stage](https://github.com/St-Gonne/meeting-intelligence-system/issues/6) before optimizing.
The real transcription/capture runtime is a separate, still-manual [setup](SETUP.md).
