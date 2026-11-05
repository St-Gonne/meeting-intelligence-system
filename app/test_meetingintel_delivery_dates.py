import tempfile
import unittest
import json
import os
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import meetingintel_cli as cli
import meetingintel_pipeline as pipeline
from test_meetingintel_pipeline import legacy_record


class DeliveryDateTests(unittest.TestCase):
    def test_exact_already_saved_retry_repairs_source_brief_without_model_work(self):
        record = legacy_record(fingerprint="a"*64)
        record["meeting_date_ist"] = "2026-10-02"
        ledger = {"records": {record["fingerprint"]: record}}
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            saved = root/"ledger.json"
            saved.write_text(json.dumps(ledger))
            before = saved.read_bytes()
            output = root/"output"
            with patch.object(pipeline, "write_brief", side_effect=OSError("synthetic write failure")):
                with self.assertRaises(OSError):
                    pipeline.write_delivery_briefs(output, ledger, date(2026,10,4), [record["fingerprint"]])
            candidate = cli.SourceCandidate(record["fingerprint"], datetime(2026,10,2,tzinfo=pipeline.IST), "laptop_capture", "Synthetic")
            with patch.dict(os.environ,{"MI_RUNTIME_ROOT":str(root/"runtime"), "MI_LEARNING_ROOT":str(root/"learning")}), \
                 patch.object(cli,"LEDGER_PATH",saved), patch.object(cli,"OUTPUT_DIR",output), \
                 patch.object(cli,"load_secret_environment",side_effect=AssertionError("credentials must not load")), \
                 patch.object(cli,"preflight_ollama",side_effect=AssertionError("model must not run")):
                code = cli._run_production((candidate,),scope="exact laptop recording",processed={record["fingerprint"]},
                    clock=lambda:datetime(2026,10,4,tzinfo=pipeline.IST),
                    pipeline_runner=lambda args: self.fail("ASR/analysis must not run"),ollama_manager=None,audio_first_only=True)
                self.assertEqual(0,code)
            self.assertEqual(before,saved.read_bytes())
            self.assertIn("Meetings captured: 1",(output/"brief_input_2026-10-02.md").read_text())
            self.assertTrue((output/"brief_input_2026-10-04.md").is_file())
            source_mtime=(output/"brief_input_2026-10-02.md").stat().st_mtime_ns
            pipeline.write_delivery_briefs(output,ledger,date(2026,10,4),[record["fingerprint"]])
            self.assertEqual(source_mtime,(output/"brief_input_2026-10-02.md").stat().st_mtime_ns)

    def test_old_source_gets_its_own_brief_when_processed_today(self):
        record = legacy_record()
        record["meeting_date_ist"] = "2026-10-02"
        ledger = {"records": {record["fingerprint"]: record}}
        before = dict(record)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            output = Path(temporary)
            requested = pipeline.write_delivery_briefs(output, ledger, date(2026, 10, 4), [record["fingerprint"]])
            self.assertEqual(output / "brief_input_2026-10-04.md", requested)
            source_brief = output / "brief_input_2026-10-02.md"
            self.assertIn("Meetings captured: 1", source_brief.read_text())
            self.assertIn("Summary", source_brief.read_text())
            self.assertEqual(before, record)
            self.assertEqual(2, len(list(output.glob("brief_input_*.md"))))


if __name__ == "__main__":
    unittest.main()
