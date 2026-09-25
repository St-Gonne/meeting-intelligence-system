#!/usr/bin/env python3

import argparse
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import context_pack
import diarized_layer2_shadow as shadow
import meetingintel_pipeline as pipeline


VALID = """pack_version: 1
generated_at: 2026-07-08T10:00:00+05:30

## OPERATOR
Alex
## ROSTER
Jordan
## ENTITIES
LightFury
## LIVE THREADS
Example
"""


class ContextPackTests(unittest.TestCase):
    def write(self, root, body=VALID, raw=None):
        path = Path(root) / "pack.md"
        path.write_bytes(raw if raw is not None else body.encode("utf-8"))
        return path

    def test_valid_pack_and_exact_byte_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root)
            snapshot, diagnostic = context_pack.load_context_pack(path)
        self.assertIsNone(diagnostic)
        self.assertEqual(snapshot.sha256, hashlib.sha256(VALID.encode()).hexdigest())
        self.assertEqual(snapshot.raw_bytes, VALID.encode())

    def test_invalid_variants_are_ignored(self):
        variants = {
            "oversized": "x" * 6001,
            "wrong_version": VALID.replace("pack_version: 1", "pack_version: 2"),
            "missing_section": VALID.replace("## ROSTER", "## PEOPLE"),
            "naive_date": VALID.replace("2026-07-08T10:00:00+05:30", "2026-07-08T10:00:00"),
            "malformed_section_order": VALID.replace("## ROSTER\nJordan\n## ENTITIES", "## ENTITIES\nLightFury\n## ROSTER"),
            "extra_section": VALID + "\n## NOTES\nextra\n",
        }
        for name, body in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                snapshot, diagnostic = context_pack.load_context_pack(self.write(root, body))
                self.assertIsNone(snapshot)
                self.assertTrue(diagnostic.startswith("context_pack_ignored:"))
        with tempfile.TemporaryDirectory() as root:
            snapshot, diagnostic = context_pack.load_context_pack(self.write(root, raw=b"\xff"))
            self.assertIsNone(snapshot)
            self.assertEqual(diagnostic, "context_pack_ignored:invalid_utf8")

    def test_anonymous_speaker_map_removes_entire_roster_section(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        view = context_pack.context_view_for_speaker_map(
            snapshot,
            [{"speaker_label": "SPEAKER_00", "identity": None, "status": "unknown"}],
        )
        self.assertIn("pack_version: 1", view.content)
        self.assertIn("generated_at: 2026-07-08T10:00:00+05:30", view.content)
        self.assertIn("## ENTITIES\nLightFury", view.content)
        self.assertNotIn("## OPERATOR", view.content)
        self.assertNotIn("## LIVE THREADS", view.content)
        self.assertNotIn("## ROSTER", view.content)
        self.assertNotIn("Jordan", view.content)

    def test_confirmed_material_speaker_does_not_unlock_roster(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        view = context_pack.context_view_for_speaker_map(
            snapshot,
            [
                {
                    "speaker_label": "SPEAKER_00",
                    "identity": "Jordan Kansal",
                    "status": "confirmed",
                    "material_for_operator_prompt": True,
                }
            ],
        )
        self.assertIn("## ENTITIES\nLightFury", view.content)
        self.assertNotIn("## ROSTER", view.content)

    def test_context_pack_cannot_create_confirmation_condition(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        self.assertFalse(context_pack.has_confirmed_material_speaker_identity([]))
        self.assertFalse(
            context_pack.has_confirmed_material_speaker_identity(
                [{"speaker_label": "SPEAKER_00", "identity": None, "status": "unknown"}]
            )
        )
        self.assertNotIn(
            "## ROSTER",
            context_pack.context_view_for_speaker_map(snapshot, []).content,
        )

    def test_non_material_confirmed_label_does_not_unlock_roster(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        view = context_pack.context_view_for_speaker_map(
            snapshot,
            [
                {
                    "speaker_label": "SPEAKER_02",
                    "identity": "Background",
                    "status": "confirmed",
                    "material_for_operator_prompt": False,
                }
            ],
        )
        self.assertNotIn("## ROSTER", view.content)

    def test_source_provenance_hashes_original_bytes_after_filtering(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        view = context_pack.rosterless_view_snapshot(snapshot)
        self.assertEqual(view.raw_bytes, VALID.encode())
        self.assertEqual(view.provenance, snapshot.provenance)
        self.assertEqual(view.sha256, hashlib.sha256(VALID.encode()).hexdigest())
        self.assertNotEqual(view.view_sha256, view.sha256)

    def test_filtered_view_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        first = context_pack.rosterless_view_snapshot(snapshot)
        second = context_pack.rosterless_view_snapshot(snapshot)
        self.assertEqual(first.content.encode("utf-8"), second.content.encode("utf-8"))
        self.assertEqual(first.view_sha256, second.view_sha256)

    def test_entity_spelling_view_permits_entity_alias_only(self):
        body = VALID.replace("LightFury", "Craftin -> Krafton")
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root, body))
        view = context_pack.entity_spelling_view_snapshot(snapshot)
        self.assertIn("Craftin -> Krafton", view.content)
        self.assertNotIn("OPERATOR", view.content)
        self.assertNotIn("ROSTER", view.content)
        self.assertNotIn("LIVE THREADS", view.content)

    def test_advisory_identity_and_role_injection_are_not_model_visible(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        view = context_pack.context_view_for_speaker_map(
            snapshot,
            [{
                "speaker_label": "SPEAKER_00",
                "identity": None,
                "status": "unknown",
                "confirmation_source": "cli",
            }],
        )
        self.assertNotIn("Jordan", view.content)
        self.assertNotIn("CEO", view.content)

    def test_missing_and_symlink_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "missing.md"
            self.assertIsNone(context_pack.load_context_pack(missing)[0])
            target = self.write(root)
            link = Path(root) / "link.md"
            link.symlink_to(target)
            self.assertEqual(context_pack.load_context_pack(link)[1], "context_pack_ignored:unsafe_file_shape")

    def test_no_context_preserves_exact_prompt(self):
        metadata = {"meeting_name": "M", "duration_seconds": 10}
        when = datetime(2026, 7, 8, tzinfo=timezone.utc)
        expected = (
            "X Meeting: M | Date/Time: 2026-07-08 00:00:00 IST | Duration: 10s Y "
            + pipeline.untrusted_content_block("evidence", "transcript")
        )
        self.assertEqual(
            pipeline.build_layer2_prompt("X {METADATA} Y {TRANSCRIPT}", metadata, when, "evidence"),
            expected,
        )

    def test_context_reaches_layer2_but_not_layer3_directly(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, _ = context_pack.load_context_pack(self.write(root))
        metadata = {"meeting_name": "M", "duration_seconds": 10}
        when = datetime(2026, 7, 8, tzinfo=timezone.utc)
        l2 = pipeline.build_layer2_prompt("{METADATA}\n{TRANSCRIPT}", metadata, when, "evidence", snapshot)
        l3 = pipeline.build_llm_prompt("L3", metadata, when, "LAYER2_ONLY")
        self.assertIn("## ENTITIES\nLightFury", l2)
        self.assertNotIn("## ROSTER", l2)
        self.assertNotIn(VALID, l3)
        self.assertIn("LAYER2_ONLY", l3)

    def test_flat_pipeline_uses_rosterless_view_without_persisting_body(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot, _ = context_pack.load_context_pack(self.write(root))
            args = argparse.Namespace(
                dry_run=False,
                output_dir=root / "out",
                ollama_url="",
                model="",
                context_pack_snapshot=snapshot,
            )
            seen_prompts = []

            def fake_summarize(prompt, _url, _model):
                seen_prompts.append(prompt)
                return ("L2" if len(seen_prompts) == 1 else "L3", {})

            with patch.object(pipeline, "summarize_with_ollama", side_effect=fake_summarize):
                result = pipeline.run_flat_meeting(
                    args,
                    {"meeting_name": "M", "duration_seconds": 1},
                    datetime(2026, 7, 8, tzinfo=timezone.utc),
                    "transcript",
                    "M",
                    "abc12345",
                    "{METADATA}\n{TRANSCRIPT}",
                    "L3 {METADATA} {TRANSCRIPT}",
                )
            self.assertEqual(result.context_pack, snapshot.provenance)
            self.assertNotIn("body", json.dumps(result.context_pack))
            self.assertNotIn("## OPERATOR", seen_prompts[0])
            self.assertNotIn("## ROSTER", seen_prompts[0])
            self.assertIn("## ENTITIES", seen_prompts[0])
            self.assertNotIn("Jordan", seen_prompts[0])
            self.assertNotIn("ADVISORY CONTEXT", seen_prompts[1])

    def test_shadow_layer3_builder_receives_layer2_only(self):
        prompt = shadow.build_layer3_prompt("{METADATA}\n{LAYER2_REPORT}", {}, datetime.now(timezone.utc), "L2", [])
        self.assertEqual(prompt.splitlines()[-1], "================ END UNTRUSTED LAYER 2 REPORT ================")
        self.assertIn("L2", prompt)
        self.assertNotIn("ADVISORY CONTEXT", prompt)

    def test_no_context_record_has_no_persisted_provenance(self):
        record = pipeline.make_meeting_record(
            Path("/tmp"), {"created_at": "2026-07-08T00:00:00+00:00", "duration_seconds": 1},
            "hash", "body", datetime.now(timezone.utc), "summary"
        )
        normalized = pipeline.normalize_record_for_v2(asdict(record))
        self.assertNotIn("context_pack", normalized)

    def test_context_record_has_slim_provenance_without_body(self):
        provenance = {"pack_version": 1, "generated_at": "x", "sha256": "abc"}
        raw = pipeline.normalize_record_for_v2({"context_pack": provenance})
        self.assertEqual(raw["context_pack"], provenance)
        self.assertNotIn("body", json.dumps(raw))

    def test_invalid_pack_does_not_fail_processing_or_change_mode(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bad = self.write(root, VALID.replace("pack_version: 1", "pack_version: 9"))
            args = argparse.Namespace(
                layer2_prompt_path=root / "l2.txt", ledger_path=root / "ledger.json",
                prompt_path=root / "l3.txt", output_dir=root / "out", meetings_root=root / "meetings",
                audio_first_root=None, context_pack=bad, selected_fingerprint=[], brief_date="2026-07-08",
                refresh_existing=False, dry_run=False,
            )
            args.layer2_prompt_path.write_text("{METADATA} {TRANSCRIPT}")
            args.prompt_path.write_text("L3")
            with patch.object(pipeline, "write_brief", return_value=root / "brief.md"):
                count, _ = pipeline.process_meetings(args)
            self.assertEqual(count, 0)
            self.assertIsNone(args.context_pack_snapshot)

    def test_pack_is_loaded_once_per_pipeline_run(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = argparse.Namespace(
                layer2_prompt_path=root / "l2.txt", context_pack=root / "pack.md",
                ledger_path=root / "ledger.json", prompt_path=root / "l3.txt",
                output_dir=root / "out", meetings_root=root / "meetings",
                audio_first_root=None, selected_fingerprint=[], brief_date="2026-07-08",
                refresh_existing=False, dry_run=False,
            )
            args.layer2_prompt_path.write_text("L2")
            args.prompt_path.write_text("L3")
            with patch.object(pipeline, "load_context_pack", return_value=(None, None)) as load:
                pipeline.process_meetings(args)
            load.assert_called_once_with(args.context_pack)


if __name__ == "__main__":
    unittest.main()

# Model calls in this suite are synthetic. GPU lifecycle is tested separately.
from test_meetingintel_pipeline import _SyntheticSession

def setUpModule():
    global _public_gpu_patch
    _public_gpu_patch = patch("meetingintel_gpu.Session", _SyntheticSession)
    _public_gpu_patch.start()

def tearDownModule():
    _public_gpu_patch.stop()
