import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from meetingintel_staging_cache import load, save


class StagingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir(mode=0o700)
        self.audio = self.source / "meeting.wav"
        self.audio.write_bytes(b"private synthetic audio")
        self.contract = {
            "source_identity": "1" * 64,
            "source_audio_sha256": hashlib.sha256(self.audio.read_bytes()).hexdigest(),
            "source_folder": str(self.source),
            "source_audio": str(self.audio),
            "backend": "legacy",
            "model_identifier": "large-v3",
            "revision": None,
            "config_sha256": "2" * 64,
        }
        self.cache = self.base / "cache"
        self.fingerprint = "3" * 64

    def tearDown(self):
        self.temp.cleanup()

    def make_stage(self, backend="legacy", *, repetition=False, runtime=True, runtime_quality=True, success=True):
        stage = self.base / f"proof-{backend}-{len(list(self.base.glob('proof-*')))}"
        stage.mkdir(mode=0o700)
        contract = dict(self.contract, backend=backend)
        if backend == "qwen":
            contract.update(model_identifier="Qwen/Qwen3-ASR-1.7B-hf", revision="rev-abc")
        text = "loop " * 24 if repetition else "A concise synthetic utterance."
        (stage / "audio.json").write_text(json.dumps({"segments": [{
            "start": 0.0, "end": 1.2, "speaker": "SPEAKER_00", "text": text.strip(),
        }]}))
        (stage / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,200\nSynthetic line\n")
        (stage / "audio.txt").write_text("Synthetic line\n")
        manifest = {
            "status": "success" if success else "failure", "exit_code": 0 if success else 1,
            "completed_at": "2026-10-04T01:00:00+05:30", "source_mode": "audio_first",
            "source_identity": contract["source_identity"], "source_folder": contract["source_folder"],
            "source_audio": contract["source_audio"], "output_folder": str(stage),
            "transcription_mode": "legacy" if backend == "legacy" else "qwen3-asr-1.7b-contiguous",
            "model": contract["model_identifier"],
        }
        if backend == "legacy" and runtime:
            (stage / "transcription_runtime.json").write_text(json.dumps({
                "completed": True, "chunks": [{"quality": {"accepted": runtime_quality}}],
            }))
        if backend == "qwen":
            qwen = {
                "status": "success", "source_sha256": contract["source_audio_sha256"],
                "model": {"identifier": contract["model_identifier"], "revision": contract["revision"]},
                "chunk_count": 1, "all_generations_ended_with_eos": True,
            }
            (stage / "qwen_asr_manifest.json").write_text(json.dumps(qwen))
            manifest["candidate_stage_provenance"] = {
                "quality_assessment": {"status": "not_assessed", "eligible": False},
                "qwen_asr": qwen,
            }
        (stage / "run_manifest.json").write_text(json.dumps(manifest))
        return stage, contract

    def test_completed_legacy_and_qwen_stages_load(self):
        for backend in ("legacy", "qwen"):
            with self.subTest(backend=backend):
                stage, contract = self.make_stage(backend)
                self.assertTrue(save(self.fingerprint, contract, stage, root=self.cache))
                self.assertEqual(stage, load(self.fingerprint, contract, root=self.cache))
                entry = self.cache / f"{self.fingerprint}.json"
                self.assertEqual(0o600, entry.stat().st_mode & 0o777)

    def test_valid_unassigned_speech_remains_anonymous_and_cacheable(self):
        stage, contract = self.make_stage("qwen")
        path = stage/"audio.json"
        body=json.loads(path.read_text())
        body["segments"].append({"speaker":None,"start":1.2,"end":2.0,"text":"Unassigned synthetic speech."})
        path.write_text(json.dumps(body))
        self.assertTrue(save(self.fingerprint,contract,stage,root=self.cache))
        self.assertEqual(stage,load(self.fingerprint,contract,root=self.cache))
        del body["segments"][-1]["speaker"]
        path.write_text(json.dumps(body))
        self.assertTrue(save(self.fingerprint,contract,stage,root=self.cache))
        body["segments"][-1]["speaker"]="arbitrary_label"
        path.write_text(json.dumps(body))
        self.assertFalse(save(self.fingerprint,contract,stage,root=self.cache))

    def test_source_and_config_changes_invalidate_contract(self):
        stage, contract = self.make_stage()
        self.assertTrue(save(self.fingerprint, contract, stage, root=self.cache))
        self.audio.write_bytes(b"changed audio")
        self.assertIsNone(load(self.fingerprint, contract, root=self.cache))
        self.audio.write_bytes(b"private synthetic audio")
        contract = dict(contract, source_audio_sha256=hashlib.sha256(self.audio.read_bytes()).hexdigest())
        changed_config = dict(contract, config_sha256="4" * 64)
        self.assertIsNone(load(self.fingerprint, changed_config, root=self.cache))

    def test_checksum_tampering_invalidates_stage(self):
        stage, contract = self.make_stage()
        self.assertTrue(save(self.fingerprint, contract, stage, root=self.cache))
        (stage / "audio.txt").write_text("tampered")
        self.assertIsNone(load(self.fingerprint, contract, root=self.cache))

    def test_failed_or_missing_legacy_runtime_is_not_saved(self):
        stage, contract = self.make_stage(runtime=False)
        self.assertFalse(save(self.fingerprint, contract, stage, root=self.cache))
        rejected_stage, rejected_contract = self.make_stage(runtime_quality=False)
        self.assertFalse(save("6" * 64, rejected_contract, rejected_stage, root=self.cache))
        failed_stage, failed_contract = self.make_stage(success=False)
        self.assertFalse(save("5" * 64, failed_contract, failed_stage, root=self.cache))

    def test_repetition_rejection_is_not_saved(self):
        stage, contract = self.make_stage(repetition=True)
        self.assertFalse(save(self.fingerprint, contract, stage, root=self.cache))

    def test_qwen_embedded_manifest_is_verified_and_hashed(self):
        stage, contract = self.make_stage("qwen")
        self.assertTrue(save(self.fingerprint, contract, stage, root=self.cache))
        self.assertEqual(stage, load(self.fingerprint, contract, root=self.cache))
        (stage / "qwen_asr_manifest.json").write_text("{}")
        self.assertIsNone(load(self.fingerprint, contract, root=self.cache))


if __name__ == "__main__":
    unittest.main()
