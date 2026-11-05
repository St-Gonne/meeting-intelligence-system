# Your first meeting

First follow SETUP.md, or ask your coding AI to follow AI_SETUP.md. Have it adapt
this page and COMMANDS.md to your own launcher, folders, phone and timezone.

1. Open `mi ui`. Notice that recording, transcript, report and brief have separate
   states. Seeing a file or transcript does not establish a complete meeting.
2. For a laptop test, run `mi record`, say a short consented test phrase and stop
   with Ctrl-C. Keep the lid open. Play the retained recording and confirm that
   the intended microphone/system audio is audible before trusting capture.
3. Select the exact test recording and type `y` to process it. Qwen transcribes,
   speaker separation labels voices, and the local analysis model writes notes.
   Use `--transcription-backend whisper` explicitly if that is your chosen runtime.
4. Open the transcript, report and brief. Check duration, names/numbers, speaker
   turns and action ownership against what you actually said. Labels can be wrong;
   diarization alone does not identify a person.
5. For a phone test, follow PHONE_SETUP.md. Fetch once, wait for settlement and
   fetch again. Review which chunks belong together. Then select exact processing.
6. When several ready recordings exist, `mi queue today` shows the ordered set.
   Review it and type `y` once. Inspect with `mi queue status`; pending work can be
   continued with `mi queue resume`. Failed work needs a separate retry approval.
7. If a step stops, keep originals. Use `mi recordings diagnose last` for capture
   or `mi ollama status` for analysis. Return to the exact item after fixing the
   cause. Accepted transcription may be reused after an analysis failure, but an
   incomplete/failed transcript is never accepted to make a report succeed.

Start with ordinary anonymous speaker labels. Optional Voice ID needs a separate
calibration/enrollment process and consent; it is not required to get useful notes.
Your AI should leave a private setup receipt saying which demos, preflights and
actual lane tests passed and what remains unfinished.
