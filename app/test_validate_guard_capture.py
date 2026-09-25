"""Local file and setup safety checks; no OBS or capture is launched."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("validate_guard_capture", Path(__file__).parent/"scripts"/"validate_guard_capture.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class GuardProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = self.root/"obs-config"
        self.config.mkdir()
        (self.config/"global.ini").write_bytes(b"original global synthetic bytes\n")
        (self.config/"user.ini").write_bytes(b"original selected profile synthetic bytes\n")
        basic = self.config/"basic"/"profiles"/"Original"
        basic.mkdir(parents=True)
        (basic/"basic.ini").write_bytes(b"original profile synthetic bytes\n")
        self.patch = patch.object(proof.obs, "DEFAULT_OBS_CONFIG_ROOT", self.config)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        previous_umask = os.umask(0o077)
        self.addCleanup(os.umask, previous_umask)

    def test_original_config_restored_and_new_proof_config_preserved(self):
        before = proof.backup_config(self.root/"before")
        (self.config/"global.ini").write_text("proof changed settings")
        created = self.config/"basic"/"scenes"/"UniqueProof.json"
        created.parent.mkdir()
        created.write_text('{"synthetic":true}')
        with patch.object(proof.guard, "obs_app_is_running", return_value=False), \
             patch.object(proof.obs, "websocket_reachable", return_value=False):
            self.assertTrue(proof.restore_config(self.root/"before", before, self.root/"after"))
        self.assertEqual(b"original global synthetic bytes\n", (self.config/"global.ini").read_bytes())
        self.assertTrue(created.exists())
        self.assertEqual("proof changed settings", (self.root/"after"/"global.ini").read_text())
        self.assertTrue((self.root/"after"/"basic"/"scenes"/"UniqueProof.json").exists())

    def test_restore_never_writes_beneath_a_live_obs(self):
        before = proof.backup_config(self.root/"before")
        (self.config/"global.ini").write_text("live configuration")
        with patch.object(proof.guard, "obs_app_is_running", return_value=True):
            with self.assertRaisesRegex(proof.GateError, "restore_deferred"):
                proof.restore_config(self.root/"before", before, self.root/"after")
        self.assertEqual("live configuration", (self.config/"global.ini").read_text())

    def test_symlinked_original_config_rejected(self):
        path = self.config/"user.ini"
        path.unlink()
        path.symlink_to(self.config/"global.ini")
        with self.assertRaisesRegex(proof.GateError, "unsafe_path"):
            proof.backup_config(self.root/"before")

    def test_failed_mute_verification_prevents_start_record(self):
        run = self.root/"proof-worker"
        run.mkdir()
        (run/"tone.wav").write_bytes(b"synthetic bytes; no media command executes")
        requested = []
        def request(name, data=None):
            requested.append(name)
            if name == "GetInputList":
                return {"inputs": [{"inputName": "Capture Guard Microphone", "inputKind": "coreaudio_input_capture"}]}
            if name == "GetSpecialInputs":
                return {}
            if name == "GetRecordStatus":
                return {"outputActive": False}
            if name == "GetInputMute":
                return {"inputMuted": False}  # Acknowledged writes did not take effect.
            if name == "GetInputAudioTracks":
                return {"inputAudioTracks": {str(i): False for i in range(1, 7)}}
            return {}
        def pretend_guard(config):
            proof.guard.configure_guard_profile(config)
            proof.guard.request_json("StartRecord")
            return 0
        with patch.object(proof.guard, "configure_guard_profile"), \
             patch.object(proof.guard, "request_json", side_effect=request), \
             patch.object(proof.guard, "run_guard", side_effect=pretend_guard):
            with self.assertRaisesRegex(proof.GateError, "non_tone_audio_not_excluded"):
                proof.worker(run, "complete")
        self.assertNotIn("StartRecord", requested)
        self.assertFalse((run/"active.json").exists())


if __name__ == "__main__":
    unittest.main()
