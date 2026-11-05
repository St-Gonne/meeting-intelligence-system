import hashlib
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import meetingintel_identity as identity


class IdentityDecisionTests(unittest.TestCase):
    def test_unique_raw_score_and_minimum_evidence(self):
        rows = [{"speaker_label": "SPEAKER_00", "eligible_seconds": 40, "score": .6},
                {"speaker_label": "SPEAKER_01", "eligible_seconds": 39, "score": .9}]
        self.assertEqual((None, "multiple_candidates"), identity.decide(rows, .5))
        rows[1]["score"] = None  # The worker does not embed insufficient speech.
        self.assertEqual(("SPEAKER_00", "matched"), identity.decide(rows, .5))
        rows[1]["score"] = .9
        rows[1]["eligible_seconds"] = 40
        self.assertEqual((None, "multiple_candidates"), identity.decide(rows, .5))
        rows[0]["score"] = float("nan")
        rows[1]["score"] = .4
        self.assertEqual((None, "below_threshold"), identity.decide(rows, .5))
        self.assertEqual((None, "insufficient_speech"), identity.decide([], .5))

    def test_overlap_removed_and_own_intervals_not_double_counted(self):
        segments = [{"speaker": "A", "start": 0, "end": 50},
                    {"speaker": "A", "start": 20, "end": 60},
                    {"speaker": "B", "start": 10, "end": 30},
                    {"speaker": "B", "start": 45, "end": 55}]
        self.assertEqual([(0., 10.), (30., 45.), (55., 60.)], identity.clean_ranges(segments, "A"))

    def test_crop_limit_never_emits_rounding_residue_or_subminimum_fragment(self):
        rows = identity.bounded_track_segments([(0.,179.99999999999994),(800.,802.)])
        self.assertEqual(1,len(rows))
        self.assertTrue(all(r["end"]-r["start"]>=.5 for r in rows))
        self.assertLessEqual(sum(r["end"]-r["start"] for r in rows),180.)
        self.assertEqual([(0.,10.),(20.,60.)],identity.clean_ranges([
            {"speaker":"SPEAKER_00","start":0,"end":60},
            {"speaker":None,"start":10,"end":20}],"SPEAKER_00"))

    def test_name_replaces_only_speaker_prefix_and_retains_machine_provenance(self):
        srt = "[SPEAKER_00]: said [SPEAKER_00] aloud\n[SPEAKER_01]: other\n"
        rendered = identity.render(srt, "Owner", "SPEAKER_00", "matched")
        self.assertIn("[SPEAKER_00 — Owner]: said [SPEAKER_00] aloud", rendered)
        self.assertIn("[SPEAKER_01]: other", rendered)
        self.assertIn("not human-confirmed", rendered)
        with self.assertRaises(ValueError):
            identity.render(srt, "Owner\nInjected", "SPEAKER_00", "matched")


class IdentityPresentationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "gate5"
        self.output = self.root / "output"
        self.fp = "a" * 64
        self.state.mkdir(mode=0o700)
        self.enrollment = self.state / "voiceprint_candidates.json"
        self.enrollment.write_text("synthetic-enrollment-state")
        self.enrollment.chmod(0o600)
        metadata = self.state / "gate5_lifecycle.json"
        metadata.write_text("synthetic-name-metadata")
        metadata.chmod(0o600)
        identity._atomic(self.state / identity.POLICY_FILE, {"schema_version": 1, "enabled": True})
        directory = self.output / "transcripts/identity"
        directory.mkdir(parents=True)
        self.named = directory / "version.md"
        self.anonymous = directory / "version-anonymous.md"
        self.named.write_text("Owner automatically recognized")
        self.anonymous.write_text("Anonymous speakers")
        self.record = {"schema_version": 1, "fingerprint": self.fp,
                       "status": "matched", "reason": "matched", "method": "auto_match_v1",
                       "presentation_path": str(self.named), "presentation_sha256": identity._hash(self.named),
                       "anonymous_path": str(self.anonymous), "anonymous_sha256": identity._hash(self.anonymous),
                       "enrollment_state_sha256": identity._hash(self.enrollment),
                       "enrollment_metadata_sha256": identity._hash(metadata)}
        self.metadata = directory / (self.fp + ".json")
        identity._atomic(self.metadata, self.record)

    def displayed(self):
        return identity.presentation(self.fp, self.output, state_root=self.state)

    def test_enabled_named_then_policy_off_or_enrollment_changed_abstains(self):
        self.assertEqual(str(self.named), self.displayed()["path"])
        identity.set_enabled(False, state_root=self.state)
        self.assertEqual(str(self.anonymous), self.displayed()["path"])
        identity._atomic(self.state / identity.POLICY_FILE, {"schema_version": 1, "enabled": True})
        self.enrollment.write_text("enrollment revoked")
        self.assertEqual(str(self.anonymous), self.displayed()["path"])
        self.enrollment.unlink()
        self.assertEqual(str(self.anonymous), self.displayed()["path"])

    def test_revocation_retains_versions_and_prevents_future_named_display(self):
        before = self.named.read_bytes()
        identity.revoke(self.fp, self.output, state_root=self.state)
        result = self.displayed()
        self.assertEqual("revoked", result["status"])
        self.assertEqual(str(self.anonymous), result["path"])
        self.assertEqual(before, self.named.read_bytes())

    def test_name_metadata_change_abstains_without_changing_enrollment(self):
        before = self.enrollment.read_bytes()
        (self.state / "gate5_lifecycle.json").write_text("name removed")
        self.assertEqual("anonymous", self.displayed()["status"])
        self.assertEqual(before, self.enrollment.read_bytes())

    def test_failed_rematch_replaces_named_projection_with_anonymous_version(self):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "audio.srt").write_text("[SPEAKER_00]: synthetic speech\n")
        source = SimpleNamespace(source_folder=Path("/source"), fingerprint=self.fp)
        with patch.object(identity, "enabled", return_value=True), \
             patch("meetingintel_gpu.run", return_value=SimpleNamespace(returncode=1, stdout="")):
            result = identity.create_presentation(source, stage, self.output)
        self.assertEqual("worker_exit_failed", result["reason"])
        self.assertEqual("anonymous", self.displayed()["status"])
        self.assertNotEqual(str(self.named), self.displayed()["path"])
        self.assertTrue(self.named.is_file())

    def test_tamper_escape_and_public_metadata_fail_closed(self):
        self.named.write_text("tampered")
        self.assertEqual({}, self.displayed())
        self.record["presentation_path"] = str(self.enrollment)
        self.record["presentation_sha256"] = identity._hash(self.enrollment)
        identity._atomic(self.metadata, self.record)
        self.assertEqual({}, self.displayed())
        self.metadata.chmod(0o644)
        self.assertEqual({}, self.displayed())


class IdentityWorkerTests(unittest.TestCase):
    def setUp(self):
        controlled = patch.dict(os.environ, {"HOMEBREW_PREFIX": "/nonexistent/synthetic-homebrew"})
        controlled.start(); self.addCleanup(controlled.stop)

    def test_scheduler_path_fallback_reaches_frozen_embedding_decoder(self):
        import voiceprint_provider_pyannote as provider
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as temporary:
            root = Path(temporary); binary = root / "homebrew/bin/ffmpeg"
            binary.parent.mkdir(parents=True); binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o700)
            original_path = str(root / "empty-bin")
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, b"\0" * (8000 * 4), b"")
            with patch.dict(os.environ, {"PATH": original_path}), patch.object(identity.Path, "home", return_value=root):
                with identity._embedding_audio_environment():
                    samples = provider.decode_mono_16k(root / "synthetic.wav", runner=run)
                self.assertEqual(original_path, os.environ["PATH"])
            self.assertEqual(8000, len(samples))
            self.assertEqual(str(binary), commands[0][0])

    def test_embedding_path_scope_restores_after_failure_and_preserves_existing_tool(self):
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as temporary:
            root = Path(temporary); binary = root / "homebrew/bin/ffmpeg"
            binary.parent.mkdir(parents=True); binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o700)
            with patch.dict(os.environ, {"PATH": str(Path(temporary) / "empty-bin")}), patch.object(identity.Path, "home", return_value=root):
                with self.assertRaisesRegex(ValueError, "synthetic failure"):
                    with identity._embedding_audio_environment():
                        raise ValueError("synthetic failure")
                self.assertEqual(str(root / "empty-bin"), os.environ["PATH"])
            with patch.dict(os.environ, {"PATH": str(binary.parent)}):
                with identity._embedding_audio_environment():
                    self.assertEqual(str(binary.parent), os.environ["PATH"])

    def test_missing_embedding_tool_has_existing_safe_reason(self):
        from voiceprint_gate5_preview import PreviewError
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as temporary, \
             patch.dict(os.environ, {"PATH": str(Path(temporary) / "empty-bin")}), \
             patch.object(identity.Path, "home", return_value=Path(temporary)):
            with self.assertRaisesRegex(PreviewError, "^ffmpeg is unavailable$"):
                with identity._embedding_audio_environment():
                    pass

    def test_parent_distinguishes_worker_boundaries_without_retaining_private_output(self):
        source = SimpleNamespace(source_folder=Path("/source"), fingerprint="a" * 64)
        cases = [
            (OSError("private path"), "worker_launch_failed"),
            (subprocess.TimeoutExpired("private command", 600, output="secret"), "worker_timeout"),
            (SimpleNamespace(returncode=-signal.SIGALRM, stdout=""), "worker_timeout"),
            (SimpleNamespace(returncode=-signal.SIGKILL, stdout="secret"), "worker_terminated"),
            (SimpleNamespace(returncode=1, stdout="secret"), "worker_exit_failed"),
            (SimpleNamespace(returncode=0, stdout="secret"), "worker_output_invalid"),
            (SimpleNamespace(returncode=0, stdout='[]'), "worker_output_invalid"),
            (SimpleNamespace(returncode=0, stdout='{"reason":"secret","status":"matched"}'), "worker_output_invalid"),
        ]
        for value, reason in cases:
            with self.subTest(reason=reason), patch.object(identity, "enabled", return_value=True), \
                 patch("meetingintel_gpu.run", **({"side_effect": value} if isinstance(value, Exception) else {"return_value": value})), \
                 patch.object(identity, "anonymous_presentation", side_effect=lambda _s, _o, _f, r="unavailable": {"status":"anonymous", "reason":r}):
                result = identity.create_presentation(source, Path("/stage"), Path("/output"))
                self.assertEqual({"status":"anonymous", "reason":reason}, result)
                self.assertIn(reason, identity.REASONS)

    def test_worker_failure_stdout_has_only_specific_allowlisted_reason(self):
        from voiceprint_gate5_preview import PreviewError
        cases = {
            "ffmpeg is unavailable": "ffmpeg_unavailable",
            "ffprobe is unavailable": "ffprobe_unavailable",
            "speaker track is too short for a measured voice match": "speaker_track_too_short",
            "speaker track extraction failed": "audio_extraction_failed",
            "speaker track concatenation failed": "audio_concatenation_failed",
            "diarization manifest is not successful": "stage_not_successful",
            "diarization source lineage does not match the selected source": "stage_lineage_mismatch",
            "diarization artifacts are malformed": "stage_artifacts_malformed",
            "diarization JSON has no segments": "stage_segments_missing",
            "diarization JSON has no valid speaker segments": "stage_segments_invalid",
            "diarization folder is missing or unsafe": "stage_folder_unsafe",
            "diarization manifest is missing or unsafe": "stage_artifact_missing_or_unsafe",
            "diarization JSON path contains a symlink": "stage_artifact_symlink",
            "diarized SRT is not user-owned": "stage_artifact_not_owned",
            "diarized SRT must not be cloud-synced or staging storage": "stage_artifact_disallowed_location",
            "/private/secret/person/transcript UNKNOWN": "invalid_stage",
        }
        arguments = ["identity", "match", "--source-folder", "/unused-source",
                      "--diarization-folder", "/unused-stage", "--output-root", "/unused-output",
                      "--fingerprint", "a" * 64]
        for message, reason in cases.items():
            with self.subTest(reason=reason), patch("sys.argv", arguments), \
                 patch("signal.alarm"), patch("sys.stdout", new_callable=io.StringIO) as out, \
                 patch.object(identity, "match_and_present", side_effect=PreviewError(message)):
                self.assertEqual(1, identity.main())
                self.assertEqual({"status": "unavailable", "reason": reason}, json.loads(out.getvalue()))
                self.assertIn(reason, identity.REASONS)

    def test_unknown_worker_exception_does_not_disclose_private_text(self):
        arguments = ["identity", "match", "--source-folder", "/unused-source",
                      "--diarization-folder", "/unused-stage", "--output-root", "/unused-output",
                      "--fingerprint", "a" * 64]
        with patch("sys.argv", arguments), patch("signal.alarm"), \
             patch("sys.stdout", new_callable=io.StringIO) as out, \
             patch.object(identity, "match_and_present", side_effect=ValueError("private name /secret/audio")):
            self.assertEqual(1, identity.main())
            self.assertEqual({"status": "unavailable", "reason": "worker_exception"}, json.loads(out.getvalue()))

    def test_parent_uses_guardian_supported_arguments_and_never_returns_names(self):
        source = SimpleNamespace(source_folder=Path("/source"), fingerprint="a" * 64)
        def run(command, *, capture_output, text):
            self.assertTrue(capture_output and text)
            return SimpleNamespace(returncode=0, stdout='{"status":"matched","reason":"matched","presentation_path":"/private-result"}')
        with patch.object(identity, "enabled", return_value=True), patch("meetingintel_gpu.run", side_effect=run):
            self.assertEqual("matched", identity.create_presentation(source, Path("/stage"), Path("/output"))["status"])

    def test_automatic_presentation_runs_after_anonymous_analysis(self):
        self._check_analysis_and_identity_order()

    def test_failed_analysis_still_matches_owner_without_claiming_completion(self):
        self._check_analysis_and_identity_order(analysis_fails=True)

    def test_identity_error_cannot_mask_original_analysis_failure(self):
        self._check_analysis_and_identity_order(analysis_fails=True, identity_fails=True)

    def _check_analysis_and_identity_order(self, analysis_fails=False, identity_fails=False):
        import meetingintel_pipeline as pipeline
        from test_meetingintel_pipeline import valid_eligibility_summary
        source = SimpleNamespace(source_kind="laptop_capture", source_folder=Path("/source"),
                                 fingerprint="a"*64, display_name="Synthetic", created_at=None,
                                 metadata={})
        args = SimpleNamespace(output_dir=Path("/output"), diarization_speakers=2)
        summary = valid_eligibility_summary()
        events = []
        selected = pipeline.SelectedMeetingResult("Anonymous summary", "/report", {}, "flat")
        original_error = pipeline.ProcessingFallback('synthetic analysis failed')
        def analysis(*arguments, **kwargs):
            events.append("analysis")
            self.assertEqual(Path("/stage"), arguments[3])
            if analysis_fails:
                raise original_error
            return selected
        def presentation(*arguments):
            events.append("identity")
            if identity_fails:
                raise RuntimeError('synthetic identity error')
            return {"status": "matched", "reason": "matched", "presentation_path": "/named-result"}
        with patch.object(pipeline, "stage_diarized_eligibility", return_value=(Path("/stage"), Path("/shadow"), summary)) as stage, \
             patch.object(pipeline, "load_json", return_value={
                 'transcription_mode': 'qwen3-asr-1.7b-contiguous',
                 'candidate_stage_provenance': {'qwen_asr': {'model': {
                     'identifier': pipeline.QWEN_MODEL_IDENTIFIER,
                     'revision': pipeline.QWEN_MODEL_REVISION,
                 }}},
             }), \
             patch.object(pipeline, "validate_eligibility_summary", return_value=False), \
             patch.object(pipeline, "routing_decision_snapshot", return_value={}), \
             patch.object(pipeline, "load_staged_transcript", return_value=([], Path("/stage/audio.srt"))), \
             patch.object(pipeline, "run_staged_speaker_meeting", side_effect=analysis), \
             patch.object(identity, "create_presentation", side_effect=presentation):
            if analysis_fails:
                with self.assertRaises(pipeline.ProcessingFallback) as caught:
                    pipeline._select_audio_first_gpu.__wrapped__(args, source, "Layer2 template", "Layer3 template")
                self.assertIs(original_error, caught.exception)
            else:
                result = pipeline._select_audio_first_gpu.__wrapped__(args, source, "Layer2 template", "Layer3 template")
        self.assertEqual(["analysis", "identity"], events)
        self.assertEqual(2, stage.call_args.kwargs['speaker_count'])
        if not analysis_fails:
            self.assertEqual("Anonymous summary", result.summary_text)
            self.assertEqual("/stage/audio.srt", result.transcript_path)

    def test_worker_unique_match_writes_separate_provenance_without_confirmation(self):
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as temporary:
            root = Path(temporary)
            state = root / "gate5"
            state.mkdir()
            identity._atomic(state / identity.POLICY_FILE, {"schema_version": 1, "enabled": True})
            audio = root / "audio.m4a"
            audio.write_bytes(b"synthetic audio only")
            stage = root / "stage"
            stage.mkdir()
            source = SimpleNamespace(source_identity="b" * 64, source_folder=root / "source", canonical_audio_path=audio)
            manifest = {"source_folder": str(source.source_folder), "source_audio": str(audio)}
            (stage / "run_manifest.json").write_text(json.dumps(manifest))
            srt = stage / "audio.srt"
            srt.write_text("[SPEAKER_00]: synthetic\n[SPEAKER_01]: synthetic\n")
            segments = [{"speaker": "SPEAKER_00", "start": 0, "end": 60},
                        {"speaker": "SPEAKER_01", "start": 60, "end": 120}]
            enrollment = state / "voiceprint_candidates.json"
            enrollment.write_text("synthetic")
            inbox = SimpleNamespace(state_path=enrollment,
                private_display_name=lambda _: "Owner",
                provenance=SimpleNamespace(threshold=.5, provider_id="pinned", model_asset_sha256="d"*64, aggregation_recipe="frozen"))
            owner = SimpleNamespace(identity_id="owner", vector=(1., 0.))
            binary = root / "homebrew/bin/ffmpeg"
            binary.parent.mkdir(parents=True); binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o700)
            vectors = iter([(1., 0.), (0., 1.)])
            def embed(*args):
                self.assertEqual(str(binary), identity.shutil.which("ffmpeg"))
                return next(vectors)
            fp = hashlib.sha256(f"meetingintel|audio_first|v1|{source.source_identity}".encode()).hexdigest()
            with patch.dict(os.environ, {"PATH": str(Path(temporary) / "empty-bin")}), \
                 patch.object(identity.Path, "home", return_value=root), \
                 patch.object(identity, "_source", return_value=source), \
                 patch.object(identity, "_owner", return_value=(inbox, [owner], root / "model")), \
                 patch("voiceprint_gate5_preview._load_diarization", return_value=(manifest, segments, stage / "audio.json", srt)), \
                 patch("voiceprint_gate5_preview._make_track"), \
                 patch("voiceprint_provider_pyannote.embed", side_effect=embed):
                result = identity.match_and_present(source.source_folder, stage, root / "output", fp, state_root=state)
                identity.revoke(fp, root / "output", state_root=state)
                revoked = identity.match_and_present(source.source_folder, stage, root / "output", fp, state_root=state)
                self.assertEqual("revoked", revoked["status"])
            self.assertEqual("matched", result["status"])
            # Bind assertions to the matched version, not filesystem order;
            # revocation creates a separate JSON marker in the same directory.
            receipt = json.loads((state / "automatic" / (Path(result["presentation_path"]).stem + ".json")).read_text())
            self.assertFalse(receipt["human_confirmed"])
            self.assertEqual("auto_match_v1", receipt["method"])
            self.assertEqual("SPEAKER_00", receipt["named_label"])
            self.assertEqual("synthetic", enrollment.read_text())
            self.assertNotIn("Owner", json.dumps(result))
            self.assertIn("Owner", Path(result["presentation_path"]).read_text())
            self.assertFalse((state / "voiceprint_gate5_inbox.json").exists())


if __name__ == "__main__":
    unittest.main()
