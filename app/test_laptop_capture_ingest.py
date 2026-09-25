from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from laptop_capture_contract import ffprobe_media
from laptop_capture_ingest import (
    CANONICAL_AUDIO_FILENAME,
    LEDGER_FILENAME,
    METADATA_FILENAME,
    NormalizeConfig,
    normalize_capture,
    sha256_file,
)
from laptop_capture_meeting_source import (
    LaptopSourceValidationError,
    load_laptop_meeting_source,
)


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
class LaptopCaptureIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.capture_root = self.root / "captures"
        self.ingest_root = self.root / "ingest"
        self.capture_root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _media(self, path: Path, frequency: int = 440) -> tuple[float, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(FFMPEG),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=0.45",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-y",
                str(path),
            ],
            check=True,
        )
        probe = ffprobe_media(path, str(FFPROBE))
        self.assertTrue(probe.decodable)
        self.assertTrue(probe.has_audio)
        return probe.duration_seconds, sha256_file(path)

    def _session(
        self,
        *,
        status: str = "complete",
        two_segments: bool = True,
    ) -> tuple[Path, dict[str, object]]:
        session = self.capture_root / "Meeting 2026-07-30_10-00-00"
        segment_root = session / "segments"
        segment_root.mkdir(parents=True)

        # Lexical names intentionally oppose manifest index order.
        first = segment_root / "z-first.mkv"
        first_duration, first_hash = self._media(first, 440)
        segment_items: list[dict[str, object]] = [
            {
                "index": 0,
                "path": "segments/z-first.mkv",
                "state": "finalized",
                "size_bytes": first.stat().st_size,
                "duration_seconds": first_duration,
                "sha256": first_hash,
            }
        ]
        if two_segments:
            second = segment_root / "a-second.mkv"
            second_duration, second_hash = self._media(second, 660)
            segment_items.append(
                {
                    "index": 1,
                    "path": "segments/a-second.mkv",
                    "state": "finalized",
                    "size_bytes": second.stat().st_size,
                    "duration_seconds": second_duration,
                    "sha256": second_hash,
                }
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "capture_id": "capture-test-001",
            "source_kind": "laptop_capture",
            "status": status,
            "started_at": "2026-07-30T04:30:00+00:00",
            "ended_at": "2026-07-30T04:30:02+00:00",
            "source_events": [],
            "stop_reason": "operator_stop",
            "segments": segment_items,
        }
        _write_json(session / "capture_manifest.json", manifest)
        return session, manifest

    def _config(self, session: Path, dry_run: bool = False) -> NormalizeConfig:
        return NormalizeConfig(
            session_root=session,
            capture_root=self.capture_root,
            ingest_root=self.ingest_root,
            dry_run=dry_run,
        )

    def test_normalizes_complete_manifest_in_manifest_order(self) -> None:
        session, manifest = self._session()
        original_hashes = [
            sha256_file(session / str(item["path"]))
            for item in manifest["segments"]  # type: ignore[index]
        ]

        result = normalize_capture(self._config(session))

        self.assertEqual(result.status, "normalized")
        self.assertEqual(result.segment_count, 2)
        audio = result.target_root / CANONICAL_AUDIO_FILENAME
        metadata_path = result.target_root / METADATA_FILENAME
        self.assertTrue(audio.is_file())
        self.assertTrue(metadata_path.is_file())
        self.assertGreater(ffprobe_media(audio, str(FFPROBE)).duration_seconds, 0.8)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_kind"], "laptop_capture")
        self.assertEqual(metadata["normalization_policy"], "guarded_obs_manifest_v1")
        self.assertEqual(
            [item["index"] for item in metadata["source_segments"]],
            [0, 1],
        )
        self.assertEqual(
            [Path(item["source_path"]).name for item in metadata["source_segments"]],
            ["z-first.mkv", "a-second.mkv"],
        )
        self.assertEqual(metadata["canonical_audio_sha256"], sha256_file(audio))
        self.assertEqual(
            original_hashes,
            [
                sha256_file(session / str(item["path"]))
                for item in manifest["segments"]  # type: ignore[index]
            ],
        )
        self.assertTrue((self.ingest_root / LEDGER_FILENAME).is_file())
        self.assertEqual(stat_mode(audio), 0o600)

    def test_repeat_is_exactly_once_skip(self) -> None:
        session, _ = self._session(two_segments=False)
        first = normalize_capture(self._config(session))
        first_hash = sha256_file(first.target_root / CANONICAL_AUDIO_FILENAME)

        second = normalize_capture(self._config(session))

        self.assertEqual(second.status, "skipped")
        self.assertEqual(first.target_root, second.target_root)
        self.assertEqual(
            first_hash,
            sha256_file(second.target_root / CANONICAL_AUDIO_FILENAME),
        )
        targets = [
            path
            for path in self.ingest_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        self.assertEqual(len(targets), 1)

    def test_dry_run_writes_nothing(self) -> None:
        session, _ = self._session()

        result = normalize_capture(self._config(session, dry_run=True))

        self.assertEqual(result.status, "planned")
        self.assertFalse(self.ingest_root.exists())

    def test_recording_or_interrupted_capture_fails_closed(self) -> None:
        for status in ("recording", "interrupted"):
            with self.subTest(status=status):
                if self.capture_root.exists():
                    shutil.rmtree(self.capture_root)
                self.capture_root.mkdir()
                session, _ = self._session(status=status, two_segments=False)
                if status == "recording":
                    (session / ".recording.lock").write_text("", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not eligible"):
                    normalize_capture(self._config(session))
                self.assertFalse(self.ingest_root.exists())

    def test_changed_segment_fails_before_writes(self) -> None:
        session, manifest = self._session(two_segments=False)
        segment = session / str(manifest["segments"][0]["path"])  # type: ignore[index]
        with segment.open("ab") as handle:
            handle.write(b"changed")

        with self.assertRaisesRegex(ValueError, "segment_"):
            normalize_capture(self._config(session))

        self.assertFalse(self.ingest_root.exists())

    def test_session_must_be_direct_safe_child(self) -> None:
        session, _ = self._session(two_segments=False)
        nested = session / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, "direct child"):
            normalize_capture(self._config(nested))

        link = self.capture_root / "linked-session"
        try:
            link.symlink_to(session, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "symlink"):
            normalize_capture(self._config(link))

    def test_existing_target_with_changed_manifest_fails_closed(self) -> None:
        session, manifest = self._session(two_segments=False)
        result = normalize_capture(self._config(session))
        manifest["stop_reason"] = "changed_after_normalization"
        _write_json(session / "capture_manifest.json", manifest)

        with self.assertRaisesRegex(ValueError, "does not match"):
            normalize_capture(self._config(session))

        self.assertTrue((result.target_root / CANONICAL_AUDIO_FILENAME).is_file())

    def test_remux_failure_leaves_no_target_ledger_or_temporary_directory(self) -> None:
        session, _ = self._session(two_segments=False)

        with patch(
            "laptop_capture_ingest._remux_audio",
            side_effect=RuntimeError("synthetic remux failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic remux failure"):
                normalize_capture(self._config(session))

        self.assertFalse((self.ingest_root / LEDGER_FILENAME).exists())
        self.assertEqual(list(self.ingest_root.iterdir()), [])

    def test_symlinked_ingest_root_is_rejected_before_existing_target_check(self) -> None:
        session, _ = self._session(two_segments=False)
        real_ingest = self.root / "real-ingest"
        real_ingest.mkdir()
        try:
            self.ingest_root.symlink_to(real_ingest, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")

        with self.assertRaisesRegex(ValueError, "ingest root is unsafe"):
            normalize_capture(self._config(session))

        self.assertEqual(list(real_ingest.iterdir()), [])

    def test_normalized_source_loader_revalidates_capture_and_canonical_audio(self) -> None:
        session, _ = self._session()
        result = normalize_capture(self._config(session))

        source = load_laptop_meeting_source(
            result.target_root, capture_root=self.capture_root
        )

        self.assertEqual(source.source_identity, result.source_identity)
        self.assertEqual(source.source_kind, "laptop_capture")
        self.assertEqual(source.source_segment_count, 2)
        self.assertEqual(source.display_name, session.name)
        self.assertEqual(source.capture_manifest_path.parent, session.resolve())

    def test_loader_rejects_changed_canonical_audio(self) -> None:
        session, _ = self._session(two_segments=False)
        result = normalize_capture(self._config(session))
        with (result.target_root / CANONICAL_AUDIO_FILENAME).open("ab") as handle:
            handle.write(b"changed")

        with self.assertRaisesRegex(
            LaptopSourceValidationError, "canonical audio checksum"
        ):
            load_laptop_meeting_source(
                result.target_root, capture_root=self.capture_root
            )

    def test_loader_rejects_changed_raw_capture_evidence(self) -> None:
        session, manifest = self._session(two_segments=False)
        result = normalize_capture(self._config(session))
        raw = session / str(manifest["segments"][0]["path"])  # type: ignore[index]
        with raw.open("ab") as handle:
            handle.write(b"changed")

        with self.assertRaisesRegex(
            LaptopSourceValidationError, "no longer eligible"
        ):
            load_laptop_meeting_source(
                result.target_root, capture_root=self.capture_root
            )

    def test_terminal_source_loss_requires_explicit_attestation_and_preserves_truth(self) -> None:
        session, manifest = self._session(status="interrupted")
        manifest["stop_reason"] = "source_lost"
        manifest["source_events"] = [
            {"type": "source_lost", "at": "2026-07-30T04:30:01+00:00"}
        ]
        _write_json(session / "capture_manifest.json", manifest)

        with self.assertRaisesRegex(ValueError, "source_not_recovered"):
            normalize_capture(self._config(session))

        result = normalize_capture(
            NormalizeConfig(
                session_root=session,
                capture_root=self.capture_root,
                ingest_root=self.ingest_root,
                allow_terminal_source_loss_recovery=True,
                operator_attested_meeting_ended_before_source_loss=True,
            )
        )
        metadata = json.loads(
            (result.target_root / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["capture_status"], "interrupted")
        self.assertEqual(metadata["stop_reason"], "source_lost")
        self.assertEqual(
            metadata["normalization_policy"],
            "guarded_obs_terminal_source_loss_operator_recovery_v1",
        )
        self.assertEqual(
            metadata["operator_recovery"]["attestation"],
            "meeting_ended_before_terminal_source_loss",
        )
        self.assertTrue(metadata["operator_recovery"]["source_loss_preserved"])

        source = load_laptop_meeting_source(
            result.target_root, capture_root=self.capture_root
        )
        self.assertEqual(source.source_identity, result.source_identity)

    def test_terminal_source_loss_can_recover_verified_prefix_only(self) -> None:
        session, manifest = self._session(status="interrupted")
        manifest["stop_reason"] = "source_lost"
        manifest["source_events"] = [
            {"type": "source_lost", "at": "2026-07-30T04:30:01+00:00"}
        ]
        _write_json(session / "capture_manifest.json", manifest)
        original_hashes = [
            sha256_file(session / str(item["path"]))
            for item in manifest["segments"]  # type: ignore[index]
        ]

        result = normalize_capture(
            NormalizeConfig(
                session_root=session,
                capture_root=self.capture_root,
                ingest_root=self.ingest_root,
                allow_terminal_source_loss_recovery=True,
                operator_attested_meeting_ended_before_source_loss=True,
                keep_finalized_segment_count=1,
            )
        )

        metadata = json.loads(
            (result.target_root / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(result.segment_count, 1)
        self.assertEqual(
            metadata["normalization_policy"],
            "guarded_obs_terminal_source_loss_prefix_trim_recovery_v1",
        )
        self.assertEqual(len(metadata["source_segments"]), 1)
        self.assertEqual(
            metadata["operator_recovery"]["retained_prefix_segment_count"], 1
        )
        self.assertEqual(
            metadata["operator_recovery"]["excluded_tail_segment_count"], 1
        )
        self.assertEqual(
            original_hashes,
            [
                sha256_file(session / str(item["path"]))
                for item in manifest["segments"]  # type: ignore[index]
            ],
        )
        source = load_laptop_meeting_source(
            result.target_root, capture_root=self.capture_root
        )
        self.assertEqual(source.source_segment_count, 1)
        self.assertEqual(source.source_identity, result.source_identity)

    def test_prefix_recovery_rejects_invalid_count_before_writes(self) -> None:
        session, manifest = self._session(status="interrupted")
        manifest["stop_reason"] = "source_lost"
        manifest["source_events"] = [
            {"type": "source_lost", "at": "2026-07-30T04:30:01+00:00"}
        ]
        _write_json(session / "capture_manifest.json", manifest)

        for count in (0, 3):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "outside"):
                    normalize_capture(
                        NormalizeConfig(
                            session_root=session,
                            capture_root=self.capture_root,
                            ingest_root=self.ingest_root,
                            allow_terminal_source_loss_recovery=True,
                            operator_attested_meeting_ended_before_source_loss=True,
                            keep_finalized_segment_count=count,
                        )
                    )
                self.assertFalse(self.ingest_root.exists())

    def test_nonterminal_or_complex_source_loss_is_not_recoverable(self) -> None:
        for events in (
            [{"type": "source_lost", "at": "2026-07-30T04:29:30+00:00"}],
            [
                {"type": "source_lost", "at": "2026-07-30T04:29:50+00:00"},
                {"type": "source_recovered", "at": "2026-07-30T04:29:55+00:00"},
                {"type": "source_lost", "at": "2026-07-30T04:30:01+00:00"},
            ],
        ):
            with self.subTest(events=events):
                if self.capture_root.exists():
                    shutil.rmtree(self.capture_root)
                self.capture_root.mkdir()
                session, manifest = self._session(status="interrupted")
                manifest["stop_reason"] = "source_lost"
                manifest["source_events"] = events
                _write_json(session / "capture_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "not eligible"):
                    normalize_capture(
                        NormalizeConfig(
                            session_root=session,
                            capture_root=self.capture_root,
                            ingest_root=self.ingest_root,
                            allow_terminal_source_loss_recovery=True,
                            operator_attested_meeting_ended_before_source_loss=True,
                        )
                    )

    def test_loader_rejects_metadata_contract_drift(self) -> None:
        session, _ = self._session(two_segments=False)
        result = normalize_capture(self._config(session))
        metadata_path = result.target_root / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_kind"] = "phone_recording"
        _write_json(metadata_path, metadata)

        with self.assertRaisesRegex(
            LaptopSourceValidationError, "unsupported source_kind"
        ):
            load_laptop_meeting_source(
                result.target_root, capture_root=self.capture_root
            )

    def test_loader_rejects_valid_manifest_copy_outside_approved_capture_root(self) -> None:
        session, _ = self._session(two_segments=False)
        result = normalize_capture(self._config(session))
        outside = self.root / "outside" / session.name
        outside.mkdir(parents=True)
        copied_manifest = outside / "capture_manifest.json"
        shutil.copy2(session / "capture_manifest.json", copied_manifest)
        metadata_path = result.target_root / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["capture_manifest_path"] = str(copied_manifest)
        _write_json(metadata_path, metadata)

        with self.assertRaisesRegex(
            LaptopSourceValidationError, "outside the approved recording root"
        ):
            load_laptop_meeting_source(
                result.target_root, capture_root=self.capture_root
            )


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
