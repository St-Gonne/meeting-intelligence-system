"""Synthetic integration coverage for whole completed phone-stage reuse."""

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import meetingintel_pipeline as pipeline


def eligibility_summary():
    profiles = [{
        "speaker_label": label, "turn_count": 2, "word_count": 30,
        "speech_share": 0.5, "material_for_operator_prompt": True,
    } for label in ("SPEAKER_00", "SPEAKER_01")]
    unknown = {"present": False, "substantive": False, "turn_count": 0,
               "word_count": 0, "speech_share": 0.0}
    routing = pipeline.build_dominant_two_routing_evidence(profiles, unknown)
    return {
        "observed_raw_speaker_labels": [p["speaker_label"] for p in profiles],
        "speaker_profiles": profiles, "unknown_speaker": unknown,
        "material_speaker_labels": [p["speaker_label"] for p in profiles],
        "material_speaker_count": 2, "routing_eligibility": routing,
        "eligible_diarized_1to1_candidate": routing["eligible"],
    }


class _SyntheticSession:
    def __init__(self, *_args, **_kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *_args): return False


class StageReuseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / "phone-source"
        self.folder.mkdir(mode=0o700)
        self.audio = self.folder / "audio.m4a"
        self.audio.write_bytes(b"synthetic phone audio")
        identity = "a" * 64
        self.source = pipeline.ProductionSource(
            source_kind="phone_recording", source_identity=identity,
            source_folder=self.folder, source_audio_path=self.audio,
            source_audio_fingerprint=hashlib.sha256(self.audio.read_bytes()).hexdigest(),
            created_at=datetime.fromisoformat("2026-10-04T00:00:00+05:30"),
            duration_seconds=60.0, display_name="Synthetic phone source",
            metadata={"meeting_name": "Synthetic phone source"},
            fingerprint=pipeline.build_audio_first_fingerprint(identity),
        )
        self.stage_root = self.root / "private-staging"
        self.cache_root = self.root / "cache"

    def tearDown(self):
        self.temp.cleanup()

    def _write_completed_stage(self, output: Path):
        output.mkdir(mode=0o700)
        source_hash = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        qwen = {
            "status": "success", "source_sha256": source_hash,
            "model": {"identifier": pipeline.QWEN_MODEL_IDENTIFIER,
                      "revision": pipeline.QWEN_MODEL_REVISION},
            "chunk_count": 1, "all_generations_ended_with_eos": True,
        }
        (output / "qwen_asr_manifest.json").write_text(json.dumps(qwen))
        (output / "audio.json").write_text(json.dumps({"segments": [{
            "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
            "text": "Synthetic words for the cached stage.",
        }]}))
        (output / "audio.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_00]: Synthetic words\n"
        )
        (output / "audio.txt").write_text("Synthetic words for the cached stage.\n")
        manifest = {
            "status": "success", "exit_code": 0,
            "completed_at": "2026-10-04T00:01:00+05:30",
            "source_folder": str(self.folder), "source_audio": str(self.audio),
            "source_mode": "audio_first", "source_identity": self.source.source_identity,
            "output_folder": str(output),
            "transcription_mode": "qwen3-asr-1.7b-contiguous",
            "model": pipeline.QWEN_MODEL_IDENTIFIER,
            "candidate_stage_provenance": {
                "quality_assessment": {"status": "not_assessed", "eligible": False},
                "qwen_asr": qwen,
            },
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_downstream_failure_reuses_stage_and_contract_drift_runs_helper(self):
        calls, helper_stages, shadow_stages = [], [], []
        shadow_calls = 0
        backend = None if self.source.source_kind == 'laptop_capture' else 'qwen'

        def fake_command(command, stage_name):
            nonlocal shadow_calls
            calls.append(stage_name)
            if stage_name == "Qwen transcription preflight":
                return subprocess.CompletedProcess(command, 0, stdout="ready", stderr="")
            if stage_name == "Diarized staging helper":
                output = Path(command[command.index("--private-output") + 1])
                helper_stages.append(output)
                manifest = self._write_completed_stage(output)
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"Manifest: {manifest}\n", stderr=""
                )
            self.assertEqual("Eligibility shadow", stage_name)
            shadow_stages.append(Path(command[command.index("--diarization-folder") + 1]))
            shadow_calls += 1
            if shadow_calls == 1:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="synthetic failure")
            output = self.root / f"shadow-{shadow_calls}"
            output.mkdir()
            summary_path = output / "pre_confirmation_eligibility.json"
            summary_path.write_text(json.dumps(eligibility_summary()))
            return subprocess.CompletedProcess(command, 0, stdout=(
                f"Eligibility summary: {summary_path}\nOutput folder: {output}\n"), stderr="")

        with patch.dict(os.environ, {
            "MI_STAGING_CACHE_ROOT": str(self.cache_root), "MI_DISABLE_STAGE_REUSE": "0",
        }), patch.object(pipeline, "QWEN_STAGING_ROOT", self.stage_root), patch.object(
            pipeline, "run_captured_command", side_effect=fake_command
        ), patch.object(
            pipeline.gpu, "Session", _SyntheticSession
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(pipeline.ProcessingFallback, "eligibility_shadow:exit_1"):
                pipeline.stage_diarized_eligibility(
                    self.folder, self.source, transcription_backend=backend
                )
            first_stage = helper_stages[0]

            reused, _shadow, _summary = pipeline.stage_diarized_eligibility(
                self.folder, self.source, transcription_backend=backend
            )
            self.assertEqual(first_stage, reused)
            self.assertEqual(["Eligibility shadow"], calls[3:4])
            self.assertEqual(first_stage, shadow_stages[1])

            with patch.object(pipeline, "WHISPERMLX_PYTHON", Path("/synthetic/changed-python")):
                fresh, _shadow, _summary = pipeline.stage_diarized_eligibility(
                    self.folder, self.source, transcription_backend=backend
                )

        self.assertEqual(2, len(helper_stages))
        self.assertNotEqual(first_stage, fresh)
        self.assertEqual(helper_stages[1], fresh)
        self.assertEqual(helper_stages, [shadow_stages[0], shadow_stages[2]])
        self.assertEqual([
            "Qwen transcription preflight", "Diarized staging helper", "Eligibility shadow",
            "Eligibility shadow",
            "Qwen transcription preflight", "Diarized staging helper", "Eligibility shadow",
        ], calls)

    def test_laptop_default_runs_qwen_and_reuses_only_matching_completed_stage(self):
        self.source = replace(self.source, source_kind='laptop_capture',
            fingerprint=pipeline.build_laptop_capture_fingerprint(self.source.source_identity))
        self.test_downstream_failure_reuses_stage_and_contract_drift_runs_helper()

    def test_known_count_changes_cache_contract_and_reaches_diarizer(self):
        self.assertNotEqual(pipeline.staging_contract(self.source, 'qwen'),
                            pipeline.staging_contract(self.source, 'qwen', 2))
        for invalid in (0, 9, True, 2.0, '2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pipeline.staging_contract(self.source, 'qwen', invalid)
        commands = []
        def run(command, stage_name):
            if stage_name == 'Qwen transcription preflight':
                return subprocess.CompletedProcess(command, 0, stdout='ready', stderr='')
            if stage_name == 'Diarized staging helper':
                commands.append(command)
                output = Path(command[command.index('--private-output') + 1])
                manifest = self._write_completed_stage(output)
                return subprocess.CompletedProcess(command, 0, stdout=f'Manifest: {manifest}\n', stderr='')
            shadow = self.root / 'shadow'
            shadow.mkdir(exist_ok=True)
            summary = shadow / 'pre_confirmation_eligibility.json'
            summary.write_text(json.dumps(eligibility_summary()))
            return subprocess.CompletedProcess(command, 0,
                stdout=f'Eligibility summary: {summary}\nOutput folder: {shadow}\n', stderr='')
        with patch.dict(os.environ, {'MI_STAGING_CACHE_ROOT': str(self.cache_root),
                                     'MI_DISABLE_STAGE_REUSE': '0'}), \
             patch.object(pipeline, 'QWEN_STAGING_ROOT', self.stage_root), \
             patch.object(pipeline, 'run_captured_command', side_effect=run), \
             patch.object(pipeline.gpu, 'Session', _SyntheticSession), redirect_stdout(io.StringIO()):
            first = pipeline.stage_diarized_eligibility(self.folder, self.source, transcription_backend='qwen')[0]
            constrained = pipeline.stage_diarized_eligibility(self.folder, self.source,
                transcription_backend='qwen', speaker_count=2)[0]
            reused = pipeline.stage_diarized_eligibility(self.folder, self.source,
                transcription_backend='qwen', speaker_count=2)[0]
        self.assertEqual(2, len(commands))
        self.assertEqual('2', commands[1][commands[1].index('--min-speakers')+1])
        self.assertEqual('2', commands[1][commands[1].index('--max-speakers')+1])
        self.assertNotEqual(first, constrained)
        self.assertEqual(constrained, reused)


if __name__ == "__main__":
    unittest.main()
