from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import meetingintel_pipeline as pipeline
from laptop_capture_meeting_source import LaptopMeetingSource


IST = ZoneInfo("Asia/Kolkata")


class LaptopCaptureAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_folder = self.root / "canonical"
        self.source_folder.mkdir()
        self.audio = self.source_folder / "audio.m4a"
        self.audio.write_bytes(b"synthetic canonical audio")
        self.prompt = self.root / "prompt.txt"
        self.layer2_prompt = self.root / "layer2.txt"
        self.prompt.write_text("Prompt", encoding="utf-8")
        self.layer2_prompt.write_text("Layer 2 {TRANSCRIPT}", encoding="utf-8")
        self.ledger = self.root / "processed_ledger.json"
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source(self, identity: str = "a" * 64) -> pipeline.ProductionSource:
        return pipeline.ProductionSource(
            source_kind="laptop_capture",
            source_identity=identity,
            source_folder=self.source_folder,
            source_audio_path=self.audio,
            source_audio_fingerprint="b" * 64,
            created_at=datetime(2026, 7, 30, 10, 0, tzinfo=IST),
            duration_seconds=120.0,
            display_name="Meeting 2026-07-30_10-00-00",
            metadata={
                "meeting_name": "Meeting 2026-07-30_10-00-00",
                "created_at": "2026-07-30T10:00:00+05:30",
                "duration_seconds": 120.0,
                "status": "completed",
                "source_kind": "laptop_capture",
            },
            fingerprint=pipeline.build_laptop_capture_fingerprint(identity),
        )

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            ledger_path=self.ledger,
            output_dir=self.output,
            prompt_path=self.prompt,
            layer2_prompt_path=self.layer2_prompt,
            context_pack=None,
            context_pack_snapshot=None,
            brief_date="2026-07-30",
            dry_run=False,
            refresh_existing=False,
            ollama_url="http://127.0.0.1:11435/api/generate",
            model=pipeline.DEFAULT_MODEL,
        )

    def selected_result(self) -> pipeline.SelectedMeetingResult:
        return pipeline.SelectedMeetingResult(
            summary_text="Synthetic summary",
            layer2_report_path=str(self.output / "layer2.md"),
            ollama_metrics={"eval_count": 1},
            processing_mode="diarized_1to1",
            transcript_path="",
            context_pack=None,
            layer2_transcript_input="SPEAKER_00: synthetic",
        )

    def test_adapter_uses_distinct_laptop_namespace_and_contract(self) -> None:
        identity = "c" * 64
        source = LaptopMeetingSource(
            source_folder=self.source_folder,
            source_kind="laptop_capture",
            source_identity=identity,
            display_name="Meeting 2026-07-30_10-00-00",
            capture_id="capture-1",
            capture_manifest_path=self.root / "capture_manifest.json",
            capture_manifest_sha256="d" * 64,
            canonical_audio_path=self.audio,
            canonical_audio_sha256="e" * 64,
            created_at=datetime(2026, 7, 30, 10, 0, tzinfo=IST),
            duration_seconds=120.0,
            ingested_at=datetime(2026, 7, 30, 10, 5, tzinfo=IST),
            audio_handling="extract_copy",
            source_segment_count=1,
        )

        adapted = pipeline.production_source_from_laptop_capture(source)

        self.assertEqual(adapted.source_kind, "laptop_capture")
        self.assertEqual(adapted.source_identity, identity)
        self.assertEqual(adapted.source_audio_path, self.audio)
        self.assertEqual(
            adapted.fingerprint,
            pipeline.build_laptop_capture_fingerprint(identity),
        )
        self.assertNotEqual(
            adapted.fingerprint,
            pipeline.build_audio_first_fingerprint(identity),
        )

    def test_explicit_source_processes_once_without_any_discovery(self) -> None:
        source = self.source()
        result = self.selected_result()

        with patch.object(
            pipeline, "select_audio_first_result", return_value=result
        ) as select, patch.object(
            pipeline, "discover_audio_first_sources"
        ) as phone_discovery, patch.object(
            pipeline, "list_meeting_dirs"
        ) as meetily_discovery:
            processed, brief = pipeline.process_explicit_laptop_sources(
                self.args(), (source,)
            )
            repeated, repeated_brief = pipeline.process_explicit_laptop_sources(
                self.args(), (source,)
            )

        self.assertEqual(processed, 1)
        self.assertEqual(repeated, 0)
        self.assertEqual(brief, repeated_brief)
        select.assert_called_once()
        self.assertIs(select.call_args.args[1], source)
        phone_discovery.assert_not_called()
        meetily_discovery.assert_not_called()

        saved = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(list(saved["records"]), [source.fingerprint])
        record = saved["records"][source.fingerprint]
        self.assertEqual(record["source_kind"], "laptop_capture")
        self.assertEqual(record["source_identity"], source.source_identity)
        self.assertEqual(record["source_audio_path"], str(self.audio))
        self.assertEqual(record["processing_mode"], "diarized_1to1")
        transcript = Path(record["transcript_path"])
        self.assertTrue(transcript.is_file())
        self.assertIn("SPEAKER_00", transcript.read_text(encoding="utf-8"))

    def test_non_laptop_and_duplicate_sources_fail_before_processing_or_writes(self) -> None:
        source = self.source()
        phone = pipeline.ProductionSource(
            **{**source.__dict__, "source_kind": "phone_recording"}
        )
        for sources in ((phone,), (source, source)):
            with self.subTest(count=len(sources)), patch.object(
                pipeline, "select_audio_first_result"
            ) as select:
                with self.assertRaises(ValueError):
                    pipeline.process_explicit_laptop_sources(self.args(), sources)
                select.assert_not_called()
                self.assertFalse(self.ledger.exists())
                self.assertFalse(self.output.exists())

    def test_processing_failure_does_not_persist_a_meeting_record(self) -> None:
        source = self.source()
        with patch.object(
            pipeline,
            "select_audio_first_result",
            side_effect=pipeline.ProcessingFallback("synthetic failure"),
        ):
            processed, _ = pipeline.process_explicit_laptop_sources(
                self.args(), (source,)
            )

        self.assertEqual(processed, 0)
        self.assertFalse(self.ledger.exists())


if __name__ == "__main__":
    unittest.main()
