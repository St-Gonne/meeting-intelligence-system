"""Conservative repetition checks; language/script is never a rejection signal."""

from __future__ import annotations

def repetition_loop(text: str) -> bool:
    """Detect a contiguous word/short-phrase loop, not normal repeated vocabulary.

    Require at least eight copies and at least 24 words. This is a catastrophic
    output guard, not a transcription-accuracy score or automatic text repair.
    """
    # Whitespace tokenization preserves Indic combining marks and complete words.
    words = [word.strip(".,!?;:\"'()[]{}—–") for word in text.casefold().split()]
    words = [word for word in words if word]
    for width in range(1, 5):
        copies = max(8, (24 + width - 1) // width)
        span = width * copies
        for start in range(max(0, len(words) - span + 1)):
            unit = words[start : start + width]
            if words[start : start + span] == unit * copies:
                return True
    return False


def assess_segments(segments: list[dict]) -> dict:
    flagged = sum(repetition_loop(str(segment.get("text", ""))) for segment in segments)
    return {
        "policy": "contiguous_repetition_v1",
        "segment_count": len(segments),
        "flagged_segments": flagged,
        "accepted": flagged == 0,
    }


def assess_backend_result(result: dict) -> dict:
    """Also screen the assembled chunk: backend timestamps can split a loop."""
    segments = result.get("segments", [])
    report = assess_segments(segments)
    assembled = result.get("text")
    if assembled is None:
        assembled = " ".join(str(segment.get("text", "")) for segment in segments)
    report["assembled_text_flagged"] = repetition_loop(str(assembled))
    report["accepted"] = report["accepted"] and not report["assembled_text_flagged"]
    return report
