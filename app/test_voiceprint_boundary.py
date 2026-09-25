from __future__ import annotations

import ast
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import context_pack
import diarized_layer2_shadow
import meetingintel_cli as cli
import meetingintel_pipeline as pipeline
from voiceprint_synthetic_candidate import SyntheticVoiceprintStore


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_MODULES = (
    "audio_first_meeting_source.py",
    "context_pack.py",
    "diarized_layer2_shadow.py",
    "meeting_artifacts.py",
    "meetingintel_cli.py",
    "meetingintel_pipeline.py",
    "meetingintel_shadow_workflow.py",
    "ollama_endpoint.py",
    "phone_drive_fetch.py",
    "phone_recording_ingest.py",
    "shadow_artifact_pipeline.py",
    "truth_layer_normalizer.py",
    "whispermlx_diarization_helper.py",
)
VOICEPRINT_MODULES = {
    "voiceprint_synthetic_candidate",
    "voiceprint_feasibility_probe",
    "voiceprint_owner_bakeoff",
    "voiceprint_provider_pyannote",
    "voiceprint_provider_speechbrain",
}
VOICEPRINT_SYMBOLS = {
    "SyntheticVoiceprintStore",
    "SyntheticVoiceprintError",
    "InvalidSyntheticVector",
    "InvalidStateRoot",
    "InvalidLifecycleState",
    "CandidateNotFound",
    "EnrollmentRecord",
    "MatchCandidate",
    "TrustedMapEntry",
    "cosine_similarity",
}
EXPECTED_PROCESSING_MODES = {
    "flat",
    "flat_fallback",
    "diarized_1to1",
    "flat_from_diarized_transcript",
    "flat_fallback_from_diarized_transcript",
}


def candidate(name: str, kind: str) -> cli.SourceCandidate:
    return cli.SourceCandidate(
        fingerprint=f"fingerprint-{name}",
        created_at=datetime(2026, 7, 19, 10, 0, tzinfo=pipeline.IST),
        source_kind=kind,
        display_name=name,
    )


class VoiceprintBoundaryTests(unittest.TestCase):
    def test_normal_runtime_sources_have_no_voiceprint_imports_or_symbols(self):
        for module_name in RUNTIME_MODULES:
            source = (PROJECT_ROOT / module_name).read_text(encoding="utf-8")
            lowered = source.casefold()
            self.assertNotIn("voiceprint", lowered, module_name)
            self.assertNotIn("voiceprint_feasibility_probe", lowered, module_name)
            tree = ast.parse(source, filename=module_name)
            imported_names: set[str] = set()
            referenced_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_names.add(node.module)
                    imported_names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Name):
                    referenced_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    referenced_names.add(node.attr)
            self.assertTrue(VOICEPRINT_MODULES.isdisjoint(imported_names), module_name)
            self.assertTrue(VOICEPRINT_SYMBOLS.isdisjoint(referenced_names), module_name)

    def test_operator_interfaces_remain_normal_and_phone_scope_is_explicit(self):
        meetily = candidate("meetily-new", "meetily")
        phone = candidate("phone-new", "phone_recording")

        normal_runner = Mock(return_value=(1, Path("/tmp/brief.md")))
        normal_discovery = cli.Discovery((meetily,), ())
        with (
            patch.object(cli, "discover_candidates", return_value=normal_discovery) as discover,
            patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}),
            patch.object(cli, "load_secret_environment", return_value=True),
            patch.object(cli, "resolve_ollama_endpoint", return_value=pipeline.DEFAULT_OLLAMA_URL),
            patch.object(cli, "preflight_ollama"),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                0,
                cli.run_cli(
                    ["new"],
                    clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=pipeline.IST),
                    pipeline_runner=normal_runner,
                ),
            )
        self.assertEqual({"audio_first_root": None, "audio_first_only": False}, discover.call_args.kwargs)
        self.assertFalse(normal_runner.call_args.args[0].audio_first_only)
        self.assertEqual([meetily.fingerprint], normal_runner.call_args.args[0].selected_fingerprint)

        audio_runner = Mock(return_value=(1, Path("/tmp/brief.md")))
        audio_discovery = cli.Discovery((meetily, phone), ())
        with (
            patch.object(cli, "discover_candidates", return_value=audio_discovery) as discover,
            patch.object(cli.pipeline, "load_ledger", return_value={"records": {}}),
            patch.object(cli, "load_secret_environment", return_value=True),
            patch.object(cli, "resolve_ollama_endpoint", return_value=pipeline.DEFAULT_OLLAMA_URL),
            patch.object(cli, "preflight_ollama"),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                0,
                cli.run_cli(
                    ["--audio-first", "new"],
                    clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=pipeline.IST),
                    pipeline_runner=audio_runner,
                ),
            )
        self.assertEqual(
            {"audio_first_root": cli.AUDIO_FIRST_ROOT, "audio_first_only": True},
            discover.call_args.kwargs,
        )
        self.assertTrue(audio_runner.call_args.args[0].audio_first_only)
        self.assertEqual([phone.fingerprint], audio_runner.call_args.args[0].selected_fingerprint)

    def test_voiceprint_activation_flags_are_rejected_and_no_voiceprint_env_activation_exists(self):
        for argv in (("--voiceprint",), ("--voiceprint-enable",), ("--voiceprint-root", "/tmp/x")):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    cli.parse_scope(argv)

        for argv in (("--voiceprint",), ("--voiceprint-enable",), ("--voiceprint-root", "/tmp/x")):
            with self.subTest(argv=argv):
                with patch.object(sys, "argv", ["meetingintel_pipeline.py", *argv]):
                    with self.assertRaises(SystemExit):
                        pipeline.parse_args()

        self.assertEqual("today", cli.parse_scope([]))
        self.assertEqual("new", cli.parse_scope(["--audio-first", "new"]))
        runtime_text = "\n".join(
            (PROJECT_ROOT / module_name).read_text(encoding="utf-8")
            for module_name in RUNTIME_MODULES
        ).casefold()
        self.assertNotIn("meetingintel_voiceprint", runtime_text)
        self.assertNotIn("voiceprint_enable", runtime_text)

    def test_synthetic_state_is_explicit_and_invisible_to_normal_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory) / "disposable-voiceprint-state"
            store = SyntheticVoiceprintStore(
                state_root,
                clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
            )
            store.enroll("synthetic-only", (1.0, 0.0))
            state_path = store.state_path
            before = state_path.read_bytes()

            discovery = cli.discover_candidates(
                meetings_root=state_root,
                audio_first_root=state_root,
                audio_first_only=False,
            )

            self.assertEqual((), discovery.candidates)
            self.assertEqual(before, state_path.read_bytes())
            self.assertEqual(("synthetic-only",), tuple(record.identity_id for record in store.all_records()))
            self.assertEqual(state_root.resolve(), state_path.parent)
            self.assertFalse((PROJECT_ROOT / "voiceprint_candidates.json").exists())

    def test_prompt_routing_ledger_brief_and_diarization_contracts_have_no_voiceprint_surface(self):
        for prompt_path in (PROJECT_ROOT / "prompts").glob("*.txt"):
            prompt = prompt_path.read_text(encoding="utf-8").casefold()
            self.assertNotIn("voiceprint", prompt_path.name.casefold())
            self.assertNotIn("voiceprint", prompt)
            self.assertNotIn("embedding", prompt)

        record_fields = {field.name.casefold() for field in fields(pipeline.MeetingRecord)}
        self.assertFalse(any("voiceprint" in field or "embedding" in field for field in record_fields))
        pipeline_source = (PROJECT_ROOT / "meetingintel_pipeline.py").read_text(encoding="utf-8")
        self.assertFalse(any(token in pipeline_source.casefold() for token in ("voiceprint", "embedding")))

        processing_modes = set()
        for node in ast.walk(ast.parse(pipeline_source)):
            if isinstance(node, ast.keyword) and node.arg == "processing_mode":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    processing_modes.add(node.value.value)
        self.assertEqual(EXPECTED_PROCESSING_MODES, processing_modes)
        self.assertNotIn("voiceprint", inspect_source(diarized_layer2_shadow))


def inspect_source(module: object) -> str:
    module_path = Path(getattr(module, "__file__")).resolve()
    return module_path.read_text(encoding="utf-8").casefold()


if __name__ == "__main__":
    unittest.main()
