#!/usr/bin/env python3
from __future__ import annotations

import unittest

from shadow_artifact_pipeline import sanitize_layer2_identity_labels, validate_and_sanitize_outputs


class ShadowArtifactSanitizerTests(unittest.TestCase):
    def test_first_person_tomorrow_becomes_review_safe(self) -> None:
        artifact = {}
        layer2 = "## ANALYST NOTES\nNone."
        layer3 = (
            "MEETING_TYPE: Mixed/other\n"
            "SIGNAL: test\n"
            "INTEL: none\n"
            "ACTION: I will meet JetSynthesys global head tomorrow\n"
            "URGENCY: tomorrow\n"
            "ROLODEX_FLAG: null\n"
        )

        safety, _layer2_s, layer3_s = validate_and_sanitize_outputs(artifact, layer2, layer3)
        self.assertIn("ACTION: needs review", layer3_s)
        self.assertIn("unclear owner: meet JetSynthesys global head", layer3_s)
        self.assertIn("timing stated as “tomorrow” relative to meeting date", layer3_s)
        self.assertIn("URGENCY: needs-review", layer3_s)
        self.assertTrue(safety["first_person_action_detected"])
        self.assertTrue(safety["extended_relative_timing_detected"])
        self.assertTrue(safety["layer3_action_rewritten_for_review"])

    def test_named_owner_and_5_6_days_becomes_review_safe(self) -> None:
        artifact = {}
        layer2 = "## ANALYST NOTES\nNone."
        layer3 = (
            "MEETING_TYPE: Mixed/other\n"
            "SIGNAL: test\n"
            "INTEL: none\n"
            "ACTION: Jordan to prepare internal demo in 5-6 days\n"
            "URGENCY: this-week\n"
            "ROLODEX_FLAG: Jordan + warm relationship\n"
        )

        safety, _layer2_s, layer3_s = validate_and_sanitize_outputs(artifact, layer2, layer3)
        self.assertIn("ACTION: needs review", layer3_s)
        self.assertIn("unclear owner: prepare internal demo", layer3_s)
        self.assertIn("timing stated as “5-6 days", layer3_s)
        self.assertIn("URGENCY: needs-review", layer3_s)
        self.assertTrue(safety["unsupported_named_owner_detected"])
        self.assertTrue(safety["extended_relative_timing_detected"])
        self.assertTrue(safety["layer3_action_rewritten_for_review"])

    def test_urgency_tomorrow_becomes_needs_review(self) -> None:
        artifact = {}
        layer2 = "## ANALYST NOTES\nNone."
        layer3 = (
            "MEETING_TYPE: Mixed/other\n"
            "SIGNAL: test\n"
            "INTEL: none\n"
            "ACTION: follow up tomorrow\n"
            "URGENCY: tomorrow\n"
            "ROLODEX_FLAG: null\n"
        )

        safety, _layer2_s, layer3_s = validate_and_sanitize_outputs(artifact, layer2, layer3)
        self.assertIn("URGENCY: needs-review", layer3_s)
        self.assertTrue(safety["extended_relative_timing_detected"])

    def test_speaker_00_likely_alex_is_sanitized(self) -> None:
        sanitized, warnings = sanitize_layer2_identity_labels(
            "## PEOPLE\n- **SPEAKER_00 (likely Alex [uncertain])** — context",
            has_speaker_map=False,
        )
        self.assertIn("SPEAKER_00", sanitized)
        self.assertNotIn("likely Alex", sanitized)
        self.assertTrue(warnings)

    def test_speaker_01_likely_jordan_is_sanitized(self) -> None:
        sanitized, warnings = sanitize_layer2_identity_labels(
            "## PEOPLE\n- **SPEAKER_01 (likely Jordan)** — context",
            has_speaker_map=False,
        )
        self.assertIn("SPEAKER_01", sanitized)
        self.assertNotIn("likely Jordan", sanitized)
        self.assertTrue(warnings)

    def test_detection_flags_are_set(self) -> None:
        artifact = {}
        layer2 = "## PEOPLE\n- **SPEAKER_00 (likely Alex [uncertain])** — context"
        layer3 = (
            "MEETING_TYPE: Mixed/other\n"
            "SIGNAL: test\n"
            "INTEL: none\n"
            "ACTION: I will meet X tomorrow; Jordan to prepare internal demo in 5-6 days\n"
            "URGENCY: tomorrow\n"
            "ROLODEX_FLAG: Jordan + warm relationship\n"
        )

        safety, _layer2_s, layer3_s = validate_and_sanitize_outputs(artifact, layer2, layer3)
        self.assertTrue(safety["first_person_action_detected"])
        self.assertTrue(safety["unsupported_named_owner_detected"])
        self.assertTrue(safety["extended_relative_timing_detected"])
        self.assertTrue(safety["layer3_action_rewritten_for_review"])
        self.assertTrue(safety["speaker_identity_inference_detected"])
        self.assertIn("ACTION: needs review", layer3_s)


if __name__ == "__main__":
    unittest.main()
