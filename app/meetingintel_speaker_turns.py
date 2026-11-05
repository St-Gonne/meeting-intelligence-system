"""Preserve aligned word-level speaker changes before transcript export.

No inference: split only when original text and timed words agree completely.
An ambiguous mixed segment keeps its text and is presented as unknown. The
input, including retained alignment evidence, is never modified in place.
"""
from copy import deepcopy
import math
from numbers import Real
import re

POLICY = "aligned_word_speaker_turns_v1"


def preserve_overlap_uncertainty(result, intervals):
    """Abstain when acoustic overlap gives multiple speakers the same maximum.

    This mirrors upstream's overlap totals, changing only ties at floating-point
    precision. Input row order cannot justify a speaker. No confidence threshold
    or voice identity is inferred, and all timing/text evidence is retained.
    """
    projected = deepcopy(result)
    valid = [r for r in intervals if _label(r.get("speaker")) != "UNKNOWN"
             and _number(r.get("start")) and _number(r.get("end")) and r["end"] > r["start"]]
    ambiguous, timed = 0, 0
    flat = [w for s in projected["segments"] for w in s.get("words", [])]
    for word in flat:
        start, end = word.get("start"), word.get("end")
        if not _number(start) or not _number(end) or end <= start:
            continue
        timed += 1
        totals = {}
        for row in valid:
            overlap = min(end, row["end"]) - max(start, row["start"])
            if overlap > 0:
                label = row["speaker"]
                totals[label] = totals.get(label, 0.) + overlap
        if len(totals) < 2:
            continue
        maximum = max(totals.values())
        tied = sorted(label for label, seconds in totals.items()
                      if math.isclose(seconds, maximum, rel_tol=1e-9, abs_tol=1e-8))
        if len(tied) > 1:
            word.update(speaker="UNKNOWN", speaker_attribution="equal_acoustic_overlap",
                        overlap_speakers=tied)
            ambiguous += 1
    # JSON loads detach the optional flat index from segment word objects. Keep
    # its corresponding ambiguity fields consistent without replacing metadata.
    index = projected.get("word_segments")
    if (isinstance(index, list) and len(index) == len(flat)
            and all(a.get("word") == b.get("word") for a, b in zip(index, flat))):
        for indexed, word in zip(index, flat):
            if word.get("speaker_attribution") == "equal_acoustic_overlap":
                for key in ("speaker", "speaker_attribution", "overlap_speakers"):
                    indexed[key] = deepcopy(word[key])
    projected["word_speaker_uncertainty"] = {
        "policy": "equal_acoustic_overlap_abstention_v1",
        "timed_words": timed, "equal_overlap_words": ambiguous,
    }
    return projected


def _label(value):
    return value if isinstance(value, str) and re.fullmatch(r"SPEAKER_\d+", value) else "UNKNOWN"


def _number(value):
    # Live alignment returns NumPy real scalars; JSON checkpoints turn them
    # into builtins. Accept both without accepting booleans or string times.
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _partition(segment):
    """Return word positions only if text coverage and timing are complete."""
    text, words = segment["text"], segment["words"]
    start, end = segment.get("start"), segment.get("end")
    if not isinstance(text, str) or not _number(start) or not _number(end) or end < start:
        return None
    positions, cursor, prior = [], 0, start
    for word in words:
        token = word.get("word")
        a, b = word.get("start"), word.get("end")
        if (not isinstance(token, str) or not token.strip() or not _number(a)
                or not _number(b) or a < prior or b < a or b > end):
            return None
        token = token.strip()
        position = text.find(token, cursor)
        if position < 0 or text[cursor:position].strip():
            return None
        positions.append(position)
        cursor, prior = position + len(token), a
    return positions if not text[cursor:].strip() else None


def project_speaker_turns(result):
    projected = deepcopy(result)
    output, corrected, split, abstained = [], 0, 0, 0
    for segment in projected["segments"]:
        words = segment.get("words")
        if not words or not isinstance(words, list) or not all(isinstance(w, dict) for w in words):
            output.append(segment)
            continue
        labels = [_label(w.get("speaker")) for w in words]
        if all(label == _label(segment.get("speaker")) for label in labels):
            output.append(segment)
            continue
        positions = _partition(segment)
        if positions is None:
            segment["speaker"] = "UNKNOWN"
            output.append(segment)
            abstained += 1
            continue
        boundaries = [0] + [i for i in range(1, len(words)) if labels[i] != labels[i - 1]] + [len(words)]
        for a, b in zip(boundaries, boundaries[1:]):
            turn = dict(segment)
            turn.update(
                speaker=labels[a], words=words[a:b],
                text=segment["text"][0 if a == 0 else positions[a]:positions[b] if b < len(words) else len(segment["text"])],
                start=segment["start"] if a == 0 else words[a]["start"],
                end=segment["end"] if b == len(words) else max(w["end"] for w in words[a:b]),
            )
            output.append(turn)
        corrected += 1
        split += len(boundaries) > 2
    projected["segments"] = output
    projected["speaker_turn_projection"] = {
        "policy": POLICY, "input_segments": len(result["segments"]),
        "output_segments": len(output), "corrected_segments": corrected,
        "split_segments": split, "ambiguous_segments": abstained,
    }
    return projected
