from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import laptop_capture_contract as contract


GOOD_PROBE = contract.MediaProbe(True, True, 900.0)


class CaptureContractFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "segments").mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_segment(
        self,
        index: int,
        *,
        state: str = "finalized",
        name: str | None = None,
        content: bytes | None = None,
        duration_seconds: float = 900.0,
    ) -> dict[str, object]:
        relative = f"segments/{name or f'segment-{index:06}.mkv'}"
        path = self.root / relative
        path.write_bytes(content if content is not None else f"segment-{index}".encode())
        segment: dict[str, object] = {
            "index": index,
            "path": relative,
            "state": state,
            "size_bytes": path.stat().st_size,
            "duration_seconds": duration_seconds,
        }
        if state == "finalized":
            segment["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return segment

    def manifest(self, segments: list[dict[str, object]], **changes: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "capture_id": "f8e62a90-63e2-4c73-99b8-527289668885",
            "source_kind": "laptop_capture",
            "status": "complete",
            "started_at": "2026-07-25T09:15:00+05:30",
            "ended_at": "2026-07-25T12:15:00+05:30",
            "segment_seconds": 900,
            "segments": segments,
        }
        payload.update(changes)
        return payload

    def audit(
        self,
        manifest: dict[str, object],
        probe: contract.MediaProbe = GOOD_PROBE,
    ) -> contract.SessionAudit:
        return contract.audit_capture_session(self.root, manifest, lambda _: probe)


class CaptureSessionAuditTests(CaptureContractFixture):
    def test_complete_ordered_segments_are_merge_ready(self) -> None:
        audit = self.audit(self.manifest([self.add_segment(0), self.add_segment(1)]))
        self.assertTrue(audit.accepted)
        self.assertTrue(audit.merge_ready)
        self.assertTrue(audit.recovery_ready)

    def test_our_incident_recording_status_with_no_media_is_rejected(self) -> None:
        (self.root / ".recording.lock").touch()
        audit = self.audit(
            self.manifest(
                [],
                status="recording",
                ended_at=None,
            )
        )
        self.assertIn("no_segments", audit.codes())
        self.assertIn("active_segment_count_invalid", audit.codes())
        self.assertFalse(audit.recovery_ready)

    def test_meetily_223_sidebar_metadata_without_saved_recording_is_rejected(self) -> None:
        audit = self.audit(self.manifest([]))
        self.assertIn("no_segments", audit.codes())
        self.assertFalse(audit.merge_ready)

    def test_zero_byte_checkpoint_is_rejected(self) -> None:
        audit = self.audit(self.manifest([self.add_segment(0, content=b"")]))
        self.assertIn("segment_empty", audit.codes())

    def test_media_without_audio_stream_is_rejected(self) -> None:
        audit = self.audit(
            self.manifest([self.add_segment(0)]),
            contract.MediaProbe(True, False, 900.0),
        )
        self.assertIn("segment_audio_missing", audit.codes())

    def test_corrupt_finalized_segment_is_rejected(self) -> None:
        audit = self.audit(
            self.manifest([self.add_segment(0)]),
            contract.MediaProbe(False, False, 0.0),
        )
        self.assertIn("segment_not_decodable", audit.codes())
        self.assertFalse(audit.recovery_ready)

    def test_hash_tampering_after_finalize_is_rejected(self) -> None:
        segment = self.add_segment(0)
        (self.root / str(segment["path"])).write_bytes(b"changed-after-finalize")
        segment["size_bytes"] = len(b"changed-after-finalize")
        audit = self.audit(self.manifest([segment]))
        self.assertIn("segment_hash_mismatch", audit.codes())

    def test_duplicate_indexes_are_rejected(self) -> None:
        audit = self.audit(
            self.manifest(
                [
                    self.add_segment(0, name="first.mkv"),
                    self.add_segment(0, name="second.mkv"),
                ]
            )
        )
        self.assertIn("segment_index_duplicate", audit.codes())
        self.assertFalse(audit.recovery_ready)

    def test_index_gap_is_rejected(self) -> None:
        audit = self.audit(self.manifest([self.add_segment(0), self.add_segment(2)]))
        self.assertIn("segment_index_gap", audit.codes())
        self.assertFalse(audit.recovery_ready)

    def test_manifest_order_not_filename_order_controls_recovery(self) -> None:
        first = self.add_segment(0, name="z-last-lexically.mkv")
        second = self.add_segment(1, name="a-first-lexically.mkv")
        audit = self.audit(self.manifest([second, first]))
        recovered = contract.recovery_order(audit)
        self.assertEqual("z-last-lexically.mkv", recovered[0].name)
        self.assertEqual("a-first-lexically.mkv", recovered[1].name)

    def test_path_escape_is_rejected(self) -> None:
        outside = self.root.parent / "outside.mkv"
        outside.write_bytes(b"outside")
        segment = {
            "index": 0,
            "path": "../outside.mkv",
            "state": "finalized",
            "size_bytes": outside.stat().st_size,
            "duration_seconds": 900.0,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
        audit = self.audit(self.manifest([segment]))
        self.assertIn("segment_missing_or_outside_session", audit.codes())

    def test_interrupted_session_recovers_finalized_segments_not_missing_active(self) -> None:
        finalized = self.add_segment(0)
        missing_active = {
            "index": 1,
            "path": "segments/segment-000001.mkv",
            "state": "active",
            "size_bytes": 100,
            "duration_seconds": 100.0,
        }
        (self.root / ".recording.lock").touch()
        audit = self.audit(
            self.manifest(
                [finalized, missing_active],
                status="interrupted",
                ended_at=None,
            )
        )
        self.assertTrue(audit.recovery_ready)
        self.assertEqual(1, len(contract.recovery_order(audit)))
        self.assertIn("segment_missing_or_outside_session", audit.codes())
        self.assertIn("stale_recording_lock", audit.codes())

    def test_complete_session_with_stale_lock_is_rejected(self) -> None:
        (self.root / ".recording.lock").touch()
        audit = self.audit(self.manifest([self.add_segment(0)]))
        self.assertIn("stale_recording_lock", audit.codes())
        self.assertFalse(audit.merge_ready)

    def test_recording_session_requires_one_active_segment_and_lock(self) -> None:
        active = self.add_segment(0, state="active")
        audit = self.audit(
            self.manifest([active], status="recording", ended_at=None)
        )
        self.assertIn("recording_lock_missing", audit.codes())
        (self.root / ".recording.lock").touch()
        second_audit = self.audit(
            self.manifest([active], status="recording", ended_at=None)
        )
        self.assertNotIn("recording_lock_missing", second_audit.codes())

    def test_sleep_source_loss_without_recovery_rejects_silent_containers(self) -> None:
        audit = self.audit(
            self.manifest(
                [self.add_segment(0), self.add_segment(1)],
                source_events=[
                    {"type": "sleep", "at": "2026-07-25T10:00:00+05:30"},
                    {"type": "source_lost", "at": "2026-07-25T10:00:01+05:30"},
                    {"type": "wake", "at": "2026-07-25T10:00:30+05:30"},
                ],
            )
        )
        self.assertIn("source_not_recovered", audit.codes())
        self.assertFalse(audit.accepted)
        self.assertFalse(audit.merge_ready)
        self.assertFalse(audit.recovery_ready)

    def test_explicit_source_recovery_is_retained_as_a_warning(self) -> None:
        audit = self.audit(
            self.manifest(
                [self.add_segment(0)],
                source_events=[
                    {"type": "source_lost", "at": "2026-07-25T10:00:00+05:30"},
                    {"type": "source_recovered", "at": "2026-07-25T10:00:02+05:30"},
                ],
            )
        )
        self.assertIn("source_interruption_recovered", audit.codes())
        self.assertTrue(audit.accepted)
        self.assertTrue(audit.merge_ready)

    def test_source_recovery_without_loss_is_rejected(self) -> None:
        audit = self.audit(
            self.manifest(
                [self.add_segment(0)],
                source_events=[
                    {"type": "source_recovered", "at": "2026-07-25T10:00:00+05:30"}
                ],
            )
        )
        self.assertIn("source_recovery_without_loss", audit.codes())
        self.assertFalse(audit.accepted)


class PreflightFailureTests(unittest.TestCase):
    def test_healthy_preflight_allows_start(self) -> None:
        self.assertEqual((), contract.evaluate_preflight(contract.PreflightSnapshot()))

    def test_meetily_509_permission_or_backend_drift_blocks_start(self) -> None:
        actions = contract.evaluate_preflight(
            replace(contract.PreflightSnapshot(), capture_permission=False)
        )
        self.assertIn("capture_permission_missing", contract.action_codes(actions))
        self.assertTrue(all(item.action == "block_start" for item in actions))

    def test_meetily_450_bluetooth_microphone_not_reporting_blocks_start(self) -> None:
        actions = contract.evaluate_preflight(
            replace(
                contract.PreflightSnapshot(),
                sources_reporting_frames=("system_audio",),
            )
        )
        self.assertIn("source_not_reporting:microphone", contract.action_codes(actions))

    def test_meetily_481_output_path_mismatch_blocks_start(self) -> None:
        actions = contract.evaluate_preflight(
            replace(contract.PreflightSnapshot(), output_path_matches=False)
        )
        self.assertIn("output_path_mismatch", contract.action_codes(actions))

    def test_unwritable_output_and_control_auth_failure_block_start(self) -> None:
        actions = contract.evaluate_preflight(
            replace(
                contract.PreflightSnapshot(),
                websocket_authenticated=False,
                output_writable=False,
            )
        )
        self.assertEqual(
            {"control_auth_failed", "output_not_writable"},
            contract.action_codes(actions),
        )

    def test_low_disk_warns_but_critical_disk_blocks(self) -> None:
        warning = contract.evaluate_preflight(
            replace(
                contract.PreflightSnapshot(),
                disk_free_bytes=contract.LOW_DISK_WARNING_BYTES,
            )
        )
        critical = contract.evaluate_preflight(
            replace(
                contract.PreflightSnapshot(),
                disk_free_bytes=contract.CRITICAL_DISK_BYTES,
            )
        )
        self.assertIn("disk_low", contract.action_codes(warning))
        self.assertIn("disk_critical", contract.action_codes(critical))


class WatchdogFailureTests(unittest.TestCase):
    def test_healthy_recording_has_no_actions(self) -> None:
        self.assertEqual((), contract.evaluate_watchdog(contract.WatchdogSnapshot()))

    def test_recording_indicator_without_output_fails_within_ten_seconds(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                output_exists=False,
                elapsed_seconds=contract.FIRST_OUTPUT_DEADLINE_SECONDS,
            )
        )
        self.assertIn("no_output_file", contract.action_codes(actions))

    def test_meetily_594_process_crash_after_audio_starts_marks_interrupted(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(contract.WatchdogSnapshot(), recorder_process_alive=False)
        )
        self.assertIn("recorder_process_died", contract.action_codes(actions))
        self.assertIn("mark_interrupted", {item.action for item in actions})

    def test_obs_8928_sleep_wake_marks_session_interrupted(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(contract.WatchdogSnapshot(), sleep_detected=True)
        )
        self.assertIn("sleep_detected", contract.action_codes(actions))

    def test_frozen_capture_file_stops_and_finalizes(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                seconds_since_output_growth=contract.OUTPUT_STALL_SECONDS,
            )
        )
        self.assertIn("output_stalled", contract.action_codes(actions))

    def test_audio_callback_stall_is_not_confused_with_ordinary_silence(self) -> None:
        healthy_silence = contract.evaluate_watchdog(contract.WatchdogSnapshot())
        stalled_callbacks = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                seconds_since_audio_callback=contract.AUDIO_CALLBACK_STALL_SECONDS,
            )
        )
        self.assertEqual((), healthy_silence)
        self.assertIn("audio_callbacks_stalled", contract.action_codes(stalled_callbacks))

    def test_audio_device_disappearing_stops_and_finalizes(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                sources_reporting_frames=("system_audio",),
            )
        )
        self.assertIn("source_lost:microphone", contract.action_codes(actions))

    def test_low_battery_warns_then_critical_battery_finalizes(self) -> None:
        warning = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                on_ac_power=False,
                battery_percent=contract.LOW_BATTERY_WARNING_PERCENT,
            )
        )
        critical = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                on_ac_power=False,
                battery_percent=contract.CRITICAL_BATTERY_PERCENT,
            )
        )
        self.assertIn("battery_low", contract.action_codes(warning))
        self.assertIn("battery_critical", contract.action_codes(critical))
        self.assertIn("stop_and_finalize", {item.action for item in critical})

    def test_disk_exhaustion_stops_and_finalizes(self) -> None:
        actions = contract.evaluate_watchdog(
            replace(
                contract.WatchdogSnapshot(),
                disk_free_bytes=contract.CRITICAL_DISK_BYTES,
            )
        )
        self.assertIn("disk_critical", contract.action_codes(actions))


class CleanupAndManifestTests(CaptureContractFixture):
    def test_merge_failure_never_authorizes_raw_segment_deletion(self) -> None:
        incomplete_gate = contract.CleanupGate(
            session_status="complete",
            canonical_exists=True,
            canonical_decodable=False,
            canonical_hash_matches=False,
            operator_confirmed=True,
        )
        self.assertFalse(contract.raw_segment_cleanup_allowed(incomplete_gate))

    def test_raw_segment_deletion_requires_explicit_operator_confirmation(self) -> None:
        gate = contract.CleanupGate("complete", True, True, True, False)
        self.assertFalse(contract.raw_segment_cleanup_allowed(gate))
        self.assertTrue(
            contract.raw_segment_cleanup_allowed(replace(gate, operator_confirmed=True))
        )

    def test_manifest_replacement_never_leaves_partial_json(self) -> None:
        path = self.root / "capture_manifest.json"
        first = self.manifest([self.add_segment(0)])
        contract.write_manifest_atomically(path, first)
        second = dict(first)
        second["status"] = "interrupted"
        second["ended_at"] = None
        contract.write_manifest_atomically(path, second)
        loaded = contract.load_manifest(path)
        self.assertEqual("interrupted", loaded["status"])
        self.assertFalse(path.with_name(".capture_manifest.json.tmp").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
class SyntheticMediaSmokeTest(unittest.TestCase):
    def test_tiny_mkv_with_audio_passes_real_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic.mkv"
            result = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:r=1:d=0.2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=0.2",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "aac",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            probe = contract.ffprobe_media(output, shutil.which("ffprobe") or "ffprobe")
            self.assertTrue(probe.decodable)
            self.assertTrue(probe.has_audio)
            self.assertGreater(probe.duration_seconds, 0)

    def test_obs_short_tail_without_format_duration_uses_audio_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "short-tail.mkv"
            result = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=880:sample_rate=48000:duration=0.1",
                    "-c:a",
                    "aac",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            with mock.patch(
                "laptop_capture_contract.json.loads",
                return_value={
                    "streams": [{"codec_type": "audio"}],
                    "format": {"duration": "N/A"},
                },
            ):
                probe = contract.ffprobe_media(
                    output,
                    shutil.which("ffprobe") or "ffprobe",
                )
            self.assertTrue(probe.decodable)
            self.assertTrue(probe.has_audio)
            self.assertGreater(probe.duration_seconds, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
