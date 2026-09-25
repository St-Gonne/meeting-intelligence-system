#!/usr/bin/env python3
from __future__ import annotations
from mi_paths import public_path

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meetingintel_pipeline import (
    DEFAULT_LAYER2_PROMPT_PATH,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROMPT_PATH,
    build_llm_prompt,
    load_prompt,
    strip_thinking_tokens,
    summarize_with_ollama,
)


DEFAULT_OUTPUT_DIR = Path(str(public_path('project/output/shadow_artifacts')))
DEFAULT_CHUNK_MAX_CHARS = 45000
RELATIVE_TIMING_PATTERNS = [
    r"\btoday\b",
    r"\btonight\b",
    r"\btomorrow\b",
    r"\byesterday\b",
    r"\bsame[- ]day\b",
    r"\bnext[- ]week\b",
    r"\bthis[- ]week\b",
    r"\bnext month\b",
    r"\bin\s+\d+\s*-\s*\d+\s+days\b",
    r"\bin\s+\d+\s+days\b",
    r"\b\d+\s*-\s*\d+\s+days\b",
    r"\b\d+\s+days\b",
    r"\b\d+\s*-\s*\d+\s+years\b",
]
UNSUPPORTED_SPEAKER_IDENTITY_PATTERNS = [
    r"likely Alex",
    r"likely Jordan",
    r"Alex is SPEAKER",
    r"SPEAKER_00 \(likely",
    r"SPEAKER_01 \(likely",
]
FIRST_PERSON_ACTION_PATTERNS = [
    r"^\s*I will\b",
    r"^\s*I’ll\b",
    r"^\s*I'll\b",
    r"^\s*my\b",
]
UNSUPPORTED_NAMED_OWNER_PATTERNS = [
    r"^\s*Alex\b",
    r"^\s*Jordan\b",
    r"\bAlex\s+to\b",
    r"\bJordan\s+to\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MeetingIntel Layer 2 and Layer 3 in shadow mode against one normalized artifact.")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--sanitize-existing-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--layer2-prompt-path", type=Path, default=DEFAULT_LAYER2_PROMPT_PATH)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS)
    parser.add_argument("--force-chunk", action="store_true")
    parser.add_argument("--no-chunk", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[shadow-artifact-pipeline] {message}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value).strip("_")
    return cleaned[:60] or "artifact"


def ensure_artifact_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Artifact path is not a directory: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file for {label}: {path}")
    return path


def build_metadata_line(artifact: dict[str, Any], quality: dict[str, Any]) -> str:
    parts = [
        f"Meeting: {artifact.get('meeting_name') or 'Unnamed meeting'}",
        f"Date (IST): {artifact.get('meeting_date_ist', 'unknown')}",
        f"Time (IST): {artifact.get('meeting_time_ist', 'unknown')}",
        f"Source type: {artifact.get('source_type', 'unknown')}",
        f"Diarization quality note: {quality.get('confidence_label', 'unknown')}",
        f"Speaker labels: {', '.join(artifact.get('final_speaker_labels', [])) or 'none'}",
    ]
    return " | ".join(parts)


def build_shadow_layer2_prompt(prompt_template: str, metadata_line: str, transcript_text: str) -> str:
    filled = prompt_template.replace("{METADATA}", metadata_line)
    filled = filled.replace("{TRANSCRIPT}", transcript_text)
    return filled


def build_temporal_safety_block(meeting_date: str, meeting_time: str) -> str:
    return (
        "=== TEMPORAL SAFETY RULES ===\n"
        f"- Meeting date/time is the source of truth: {meeting_date} {meeting_time} IST.\n"
        "- Relative timing words like today, tonight, tomorrow, this week, and next week are relative to the meeting date, not the processing date.\n"
        "- Do not output stale relative timing as if it is current.\n"
        "- Preserve explicitly stated timing as \"STATED: tomorrow relative to meeting date\" or similar when needed.\n"
        "- If a stated date/time has already passed or cannot be resolved cleanly from the meeting date, mark it as stale / needs human review.\n"
    )


def build_speaker_identity_safety_block(artifact: dict[str, Any]) -> str:
    speaker_map = artifact.get("speaker_map")
    speaker_map_text = json.dumps(speaker_map, ensure_ascii=False) if speaker_map else "none provided"
    return (
        "=== SPEAKER IDENTITY SAFETY RULES ===\n"
        "- Speaker labels are diarization labels, not identity labels.\n"
        "- Do not map SPEAKER_00 or SPEAKER_01 to Alex unless explicitly provided in artifact metadata.\n"
        "- Use SPEAKER_00 / SPEAKER_01 style labels unless a speaker_map exists in artifact metadata.\n"
        "- Do not infer a real identity from conversational style, context, role, or prior assumptions alone.\n"
        f"- Artifact speaker_map: {speaker_map_text}.\n"
    )


def build_speaker_aware_transcript(
    transcript_text: str,
    quality: dict[str, Any],
    artifact: dict[str, Any],
    meeting_date: str,
    meeting_time: str,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    quality_note = quality.get("confidence_label", "unknown")
    warnings = artifact.get("warnings") or quality.get("warnings") or []
    warning_lines = "\n".join(f"- {warning}" for warning in warnings)
    instruction_block = (
        build_temporal_safety_block(meeting_date, meeting_time)
        + "\n"
        + build_speaker_identity_safety_block(artifact)
        + "\n"
        + "=== SPEAKER-AWARE INSTRUCTIONS ===\n"
        "- Speaker labels are useful evidence, not absolute truth.\n"
        "- Attribute signals, actions, commitments, or ownership to speaker labels only when supported by the wording.\n"
        "- If speaker ownership is unclear, mark it as unclear instead of guessing.\n"
        f"- In ANALYST NOTES, preserve this diarization quality note: {quality_note}.\n"
    )
    if chunk_index is not None and total_chunks is not None:
        instruction_block += (
            f"- This is chunk {chunk_index} of {total_chunks}. Treat it as one part of the same meeting and do not assume it contains the full conversation.\n"
        )
    if warning_lines:
        instruction_block += "Diarization warnings to preserve in Analyst Notes when relevant:\n"
        instruction_block += f"{warning_lines}\n"
    return f"{instruction_block}\n{transcript_text}"


def build_merge_prompt(
    metadata_line: str,
    chunk_reports: list[str],
    quality: dict[str, Any],
    artifact: dict[str, Any],
    meeting_date: str,
    meeting_time: str,
) -> str:
    quality_note = quality.get("confidence_label", "unknown")
    warnings = artifact.get("warnings") or quality.get("warnings") or []
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None."
    chunk_sections = []
    for index, report in enumerate(chunk_reports, start=1):
        chunk_sections.append(f"===== CHUNK REPORT {index} =====\n{report}\n===== END CHUNK REPORT {index} =====")

    return (
        "You are merging multiple partial Layer 2 meeting intelligence reports from different chunks of the SAME diarized meeting transcript.\n\n"
        "Output ONLY one final report in the exact same format used by the Layer 2 prompt:\n"
        "===== LAYER 2 INTELLIGENCE REPORT ===== through ===== END REPORT =====.\n\n"
        "Merge rules:\n"
        "- Deduplicate entities, people, actions, signals, and open threads across chunks.\n"
        "- Preserve uncertainty flags such as [uncertain] exactly when present.\n"
        "- Preserve the speaker attribution caution and diarization quality note in ANALYST NOTES.\n"
        "- Preserve temporal safety: meeting date/time is the source of truth, and relative words like tomorrow or tonight are relative to the meeting date, not the processing date.\n"
        "- Preserve stale or ambiguous timing as stale / needs human review rather than making it current.\n"
        "- Do not infer real-world speaker identities unless an explicit speaker_map is provided in artifact metadata.\n"
        "- Do not invent any new facts, names, companies, dates, or conclusions.\n"
        "- If two chunks provide overlapping evidence, keep the strongest supported wording without creating contradictions.\n"
        "- If speaker ownership is unclear, keep it unclear rather than guessing.\n"
        "- Treat all chunk reports as parts of one meeting, not separate meetings.\n"
        "- Keep the final output phone-readable and faithful to the required Layer 2 format.\n\n"
        f"Meeting metadata: {metadata_line}\n"
        f"Meeting source-of-truth date/time: {meeting_date} {meeting_time} IST\n"
        f"Diarization quality note: {quality_note}\n"
        + build_speaker_identity_safety_block(artifact)
        + "\n"
        f"Diarization warnings:\n{warning_lines}\n\n"
        + "\n\n".join(chunk_sections)
    )


def build_shadow_layer3_prompt(
    prompt_template: str,
    meeting_name: str,
    meeting_datetime: datetime,
    layer2_report: str,
) -> str:
    temporal_block = build_temporal_safety_block(
        meeting_datetime.strftime("%Y-%m-%d"),
        meeting_datetime.strftime("%H:%M"),
    )
    return (
        f"{temporal_block}\n"
        "=== LAYER 3 SAFETY RULES ===\n"
        "- Meeting date/time is the source of truth for all action timing.\n"
        "- Do not phrase stale timing as a current first-person action like \"I will meet X tomorrow\" unless you explicitly mark it as relative to the meeting date.\n"
        "- If timing from the meeting is now stale or ambiguous, mark it as stale / needs human review.\n"
        "- Do not convert unverified diarization labels into real identities.\n\n"
        f"{build_llm_prompt(prompt_template, {'meeting_name': meeting_name, 'duration_seconds': 0}, meeting_datetime, layer2_report)}"
    )


def unique_output_path(base_dir: Path, stem: str, suffix: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = base_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_header_and_body(transcript_text: str) -> tuple[list[str], list[str]]:
    lines = transcript_text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        if line.strip() == "## Speaker-Labeled Transcript":
            body_start = index + 1
            break
    return lines[:body_start], lines[body_start:]


def chunk_transcript_lines(transcript_text: str, chunk_max_chars: int) -> list[str]:
    header_lines, body_lines = split_header_and_body(transcript_text)
    if not body_lines:
        return [transcript_text]

    header_text = "\n".join(header_lines).strip()
    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = 0

    for line in body_lines:
        line_with_newline = line + "\n"
        if current_lines and current_size + len(line_with_newline) > chunk_max_chars:
            body_text = "\n".join(current_lines).strip()
            chunk_text = f"{header_text}\n\n{body_text}".strip() if header_text else body_text
            chunks.append(chunk_text)
            current_lines = []
            current_size = 0

        current_lines.append(line)
        current_size += len(line_with_newline)

        if len(line_with_newline) > chunk_max_chars and len(current_lines) == 1:
            body_text = "\n".join(current_lines).strip()
            chunk_text = f"{header_text}\n\n{body_text}".strip() if header_text else body_text
            chunks.append(chunk_text)
            current_lines = []
            current_size = 0

    if current_lines:
        body_text = "\n".join(current_lines).strip()
        chunk_text = f"{header_text}\n\n{body_text}".strip() if header_text else body_text
        chunks.append(chunk_text)

    return chunks or [transcript_text]


def should_chunk(transcript_text: str, args: argparse.Namespace) -> bool:
    if args.no_chunk:
        return False
    if args.force_chunk:
        return True
    return len(transcript_text) > args.chunk_max_chars


def build_meeting_datetime(meeting_date: str, meeting_time: str) -> datetime:
    return datetime.strptime(f"{meeting_date} {meeting_time}", "%Y-%m-%d %H:%M")


def text_has_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def collect_relative_timing_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in RELATIVE_TIMING_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append(match.group(0))
    return sorted(set(hits), key=str.lower)


def collect_identity_inference_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in UNSUPPORTED_SPEAKER_IDENTITY_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append(match.group(0))
    return sorted(set(hits), key=str.lower)


def strip_relative_timing_phrases(text: str) -> str:
    sanitized = text
    for pattern in RELATIVE_TIMING_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r"\s*[,;]+\s*", "; ", sanitized)
    return sanitized.strip(" ,;-")


def strip_unsupported_action_owner(text: str) -> tuple[str, bool, bool]:
    sanitized = text.strip()
    first_person_detected = text_has_any_pattern(sanitized, FIRST_PERSON_ACTION_PATTERNS)
    named_owner_detected = text_has_any_pattern(sanitized, UNSUPPORTED_NAMED_OWNER_PATTERNS)

    sanitized = re.sub(r"^\s*(I will|I’ll|I'll)\s+", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"^\s*my\s+", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"^\s*(Alex|Jordan)\s+to\s+", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"^\s*(Alex|Jordan)\s+", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b(Alex|Jordan)\s+to\s+", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" ,;-")

    return sanitized, first_person_detected, named_owner_detected


def sanitize_layer2_identity_labels(layer2_report: str, has_speaker_map: bool) -> tuple[str, list[str]]:
    if has_speaker_map:
        return layer2_report, []

    sanitized = layer2_report
    warnings: list[str] = []
    replacements = [
        (r"SPEAKER_00\s*\(likely[^)]*\)", "SPEAKER_00"),
        (r"SPEAKER_01\s*\(likely[^)]*\)", "SPEAKER_01"),
    ]
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        if updated != sanitized:
            warnings.append(f"Sanitized unsupported speaker identity label via pattern: {pattern}")
            sanitized = updated

    return sanitized, warnings


def sanitize_layer3_brief(layer3_brief: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    lines = layer3_brief.splitlines()
    sanitized_lines: list[str] = []
    first_person_action_detected = False
    unsupported_named_owner_detected = False
    extended_relative_timing_detected = False
    layer3_action_rewritten_for_review = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ACTION:"):
            action_text = stripped[len("ACTION:"):].strip()
            hits = collect_relative_timing_hits(action_text)
            if hits:
                extended_relative_timing_detected = True
            parts = [part.strip() for part in action_text.split(";") if part.strip()] or [action_text]
            rewritten_parts: list[str] = []
            action_needs_rewrite = False

            for part in parts:
                part_hits = collect_relative_timing_hits(part)
                if part_hits:
                    extended_relative_timing_detected = True

                cleaned_owner_part, part_first_person, part_named_owner = strip_unsupported_action_owner(part)
                if part_first_person:
                    first_person_action_detected = True
                if part_named_owner:
                    unsupported_named_owner_detected = True

                part_needs_rewrite = bool(part_hits or part_first_person or part_named_owner)
                action_needs_rewrite = action_needs_rewrite or part_needs_rewrite

                base_fragment = strip_relative_timing_phrases(cleaned_owner_part) if part_needs_rewrite else cleaned_owner_part
                base_fragment = re.sub(r"\s+", " ", base_fragment).strip(" ,;-")
                if not base_fragment:
                    base_fragment = "action mentioned"

                if part_needs_rewrite:
                    fragment = f"unclear owner: {base_fragment}"
                    if part_hits:
                        fragment += f" — timing stated as “{' / '.join(part_hits)}” relative to meeting date; review if still relevant"
                    rewritten_parts.append(fragment)
                else:
                    rewritten_parts.append(f"unclear owner: {base_fragment}")

            if action_needs_rewrite:
                layer3_action_rewritten_for_review = True
                sanitized_lines.append(
                    "ACTION: needs review — meeting-relative or identity-unsafe actions were detected: "
                    + "; ".join(rewritten_parts)
                )
                continue
        if stripped.startswith("URGENCY:"):
            urgency_text = stripped[len("URGENCY:"):].strip()
            if collect_relative_timing_hits(urgency_text) or layer3_action_rewritten_for_review:
                if collect_relative_timing_hits(urgency_text):
                    extended_relative_timing_detected = True
                sanitized_lines.append("URGENCY: needs-review")
                continue
        sanitized_lines.append(line)

    if first_person_action_detected:
        warnings.append("first_person_action_detected")
    if unsupported_named_owner_detected:
        warnings.append("unsupported_named_owner_detected")
    if extended_relative_timing_detected:
        warnings.append("extended_relative_timing_detected")
    if layer3_action_rewritten_for_review:
        warnings.append("layer3_action_rewritten_for_review")

    return "\n".join(sanitized_lines), warnings


def validate_and_sanitize_outputs(
    artifact: dict[str, Any],
    layer2_report: str,
    layer3_brief: str,
) -> tuple[dict[str, Any], str, str]:
    has_speaker_map = bool(artifact.get("speaker_map"))
    relative_timing_detected = text_has_any_pattern(layer2_report, RELATIVE_TIMING_PATTERNS) or text_has_any_pattern(
        layer3_brief, RELATIVE_TIMING_PATTERNS
    )
    identity_detected = (
        not has_speaker_map and text_has_any_pattern(layer2_report, UNSUPPORTED_SPEAKER_IDENTITY_PATTERNS)
    )

    safety_warnings: list[str] = []
    relative_hits_layer2 = collect_relative_timing_hits(layer2_report)
    relative_hits_layer3 = collect_relative_timing_hits(layer3_brief)
    identity_hits = collect_identity_inference_hits(layer2_report)

    if relative_hits_layer2:
        safety_warnings.append("Relative timing detected in Layer 2: " + ", ".join(relative_hits_layer2))
    if relative_hits_layer3:
        safety_warnings.append("Relative timing detected in Layer 3: " + ", ".join(relative_hits_layer3))
    if identity_detected and identity_hits:
        safety_warnings.append("Unsupported speaker identity inference detected in Layer 2: " + ", ".join(identity_hits))

    sanitized_layer2, layer2_sanitize_warnings = sanitize_layer2_identity_labels(layer2_report, has_speaker_map)
    sanitized_layer3, layer3_sanitize_warnings = sanitize_layer3_brief(layer3_brief)
    safety_warnings.extend(layer2_sanitize_warnings)
    safety_warnings.extend(layer3_sanitize_warnings)

    first_person_action_detected = "first_person_action_detected" in layer3_sanitize_warnings
    unsupported_named_owner_detected = "unsupported_named_owner_detected" in layer3_sanitize_warnings
    extended_relative_timing_detected = "extended_relative_timing_detected" in layer3_sanitize_warnings
    layer3_action_rewritten_for_review = "layer3_action_rewritten_for_review" in layer3_sanitize_warnings

    return (
        {
            "relative_timing_detected": relative_timing_detected,
            "speaker_identity_inference_detected": identity_detected,
            "first_person_action_detected": first_person_action_detected,
            "unsupported_named_owner_detected": unsupported_named_owner_detected,
            "extended_relative_timing_detected": extended_relative_timing_detected,
            "layer3_action_rewritten_for_review": layer3_action_rewritten_for_review,
            "safety_warnings": safety_warnings,
        },
        sanitized_layer2,
        sanitized_layer3,
    )


def sanitize_from_existing_manifest(args: argparse.Namespace) -> int:
    manifest_path = require_file(args.sanitize_existing_manifest.expanduser().resolve(), "existing manifest")
    manifest = load_json(manifest_path)
    artifact_dir = ensure_artifact_dir(Path(manifest["artifact_path"]))
    artifact = load_json(require_file(artifact_dir / "artifact.json", "artifact.json"))

    layer2_path = require_file(Path(manifest["output_paths"]["layer2_report"]), "Layer 2 report")
    layer3_path = require_file(Path(manifest["output_paths"]["layer3_brief"]), "Layer 3 brief")
    layer2_report = layer2_path.read_text(encoding="utf-8")
    layer3_brief = layer3_path.read_text(encoding="utf-8")

    print_step(f"Sanitizing existing manifest: {manifest_path}")
    print_step(f"Using existing Layer 2 report: {layer2_path}")
    print_step(f"Using existing Layer 3 brief: {layer3_path}")

    safety_result, sanitized_layer2_report, sanitized_layer3_brief = validate_and_sanitize_outputs(
        artifact,
        layer2_report,
        layer3_brief,
    )

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    layer2_stem = f"{layer2_path.stem}__resanitized_{stamp}"
    brief_stem = f"{layer3_path.stem}__resanitized_{stamp}"
    sanitized_layer2_output_path = unique_output_path(args.output_dir.expanduser().resolve() / "layer2_sanitized", layer2_stem, ".md")
    sanitized_brief_output_path = unique_output_path(args.output_dir.expanduser().resolve() / "briefs_sanitized", brief_stem, ".md")

    write_text(sanitized_layer2_output_path, sanitized_layer2_report)
    write_text(sanitized_brief_output_path, sanitized_layer3_brief)

    manifest.setdefault("output_paths", {})
    manifest["output_paths"]["sanitized_layer2_report"] = str(sanitized_layer2_output_path)
    manifest["output_paths"]["sanitized_layer3_brief"] = str(sanitized_brief_output_path)
    manifest["sanitized_layer2_path"] = str(sanitized_layer2_output_path)
    manifest["sanitized_layer3_path"] = str(sanitized_brief_output_path)
    manifest["safety_warnings"] = safety_result["safety_warnings"]
    manifest["relative_timing_detected"] = safety_result["relative_timing_detected"]
    manifest["speaker_identity_inference_detected"] = safety_result["speaker_identity_inference_detected"]
    manifest["first_person_action_detected"] = safety_result["first_person_action_detected"]
    manifest["unsupported_named_owner_detected"] = safety_result["unsupported_named_owner_detected"]
    manifest["extended_relative_timing_detected"] = safety_result["extended_relative_timing_detected"]
    manifest["layer3_action_rewritten_for_review"] = safety_result["layer3_action_rewritten_for_review"]
    manifest.setdefault("timestamps", {})
    manifest["timestamps"]["sanitized_at_utc"] = now_utc().isoformat()
    write_json(manifest_path, manifest)

    print_step(f"Sanitized Layer 2 output path: {sanitized_layer2_output_path}")
    print_step(f"Sanitized Layer 3 output path: {sanitized_brief_output_path}")
    print_step("Safety warnings: " + (" | ".join(safety_result["safety_warnings"]) if safety_result["safety_warnings"] else "none"))
    print_step(f"Updated manifest: {manifest_path}")
    return 0


def run_shadow_pipeline(args: argparse.Namespace) -> int:
    artifact_dir = ensure_artifact_dir(args.artifact_dir)
    output_dir = args.output_dir.expanduser().resolve()
    prompt_path = require_file(args.prompt_path.expanduser().resolve(), "Layer 3 prompt")
    layer2_prompt_path = require_file(args.layer2_prompt_path.expanduser().resolve(), "Layer 2 prompt")

    artifact_json_path = require_file(artifact_dir / "artifact.json", "artifact.json")
    transcript_path = require_file(artifact_dir / "transcript_diarized.md", "transcript_diarized.md")
    quality_path = require_file(artifact_dir / "quality.json", "quality.json")

    artifact = load_json(artifact_json_path)
    quality = load_json(quality_path)
    transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript_text:
        raise RuntimeError(f"Transcript is empty: {transcript_path}")

    artifact_id = artifact.get("artifact_id") or artifact_dir.name
    meeting_name = artifact.get("meeting_name") or artifact_dir.name
    meeting_date = artifact.get("meeting_date_ist") or "unknown-date"
    meeting_time = artifact.get("meeting_time_ist") or "unknown-time"
    source_audio = artifact.get("source_audio_path")
    meeting_datetime = build_meeting_datetime(meeting_date, meeting_time)
    run_started_at = now_utc()
    run_stamp = run_started_at.strftime("%Y%m%dT%H%M%SZ")
    output_stem = f"{meeting_date}_{safe_name(meeting_name)}_{artifact_id}_{run_stamp}"

    layer2_output_path = unique_output_path(output_dir / "layer2", output_stem, ".md")
    brief_output_path = unique_output_path(output_dir / "briefs", output_stem, ".md")
    sanitized_layer2_output_path = unique_output_path(output_dir / "layer2_sanitized", output_stem, ".md")
    sanitized_brief_output_path = unique_output_path(output_dir / "briefs_sanitized", output_stem, ".md")
    manifest_output_path = unique_output_path(output_dir / "manifests", f"{output_stem}__run_manifest", ".json")

    run_manifest: dict[str, Any] = {
        "artifact_path": str(artifact_dir),
        "artifact_id": artifact_id,
        "source_audio": source_audio,
        "model": args.model,
        "prompt_paths": {
            "layer3_prompt": str(prompt_path),
            "layer2_prompt": str(layer2_prompt_path),
        },
        "output_paths": {
            "layer2_report": str(layer2_output_path),
            "layer3_brief": str(brief_output_path),
            "sanitized_layer2_report": str(sanitized_layer2_output_path),
            "sanitized_layer3_brief": str(sanitized_brief_output_path),
            "run_manifest": str(manifest_output_path),
            "layer2_chunks_dir": str(output_dir / "layer2_chunks"),
        },
        "timestamps": {
            "started_at_utc": run_started_at.isoformat(),
            "completed_at_utc": None,
        },
        "chunking": {
            "enabled": False,
            "chunk_max_chars": args.chunk_max_chars,
            "chunk_count": 0,
            "chunk_output_paths": [],
        },
        "safety_warnings": [],
        "relative_timing_detected": False,
        "speaker_identity_inference_detected": False,
        "first_person_action_detected": False,
        "unsupported_named_owner_detected": False,
        "extended_relative_timing_detected": False,
        "layer3_action_rewritten_for_review": False,
        "sanitized_layer2_path": str(sanitized_layer2_output_path),
        "sanitized_layer3_path": str(sanitized_brief_output_path),
        "success": False,
        "error": None,
    }

    try:
        print_step(f"Artifact: {artifact_dir}")
        print_step(f"Meeting: {meeting_name} ({meeting_date} {meeting_time} IST)")
        print_step(f"Reading transcript: {transcript_path}")
        print_step(f"Transcript size: {len(transcript_text)} chars")

        layer2_prompt_template = load_prompt(layer2_prompt_path)
        layer3_prompt_template = load_prompt(prompt_path)
        metadata_line = build_metadata_line(artifact, quality)

        chunked = should_chunk(transcript_text, args)
        transcript_chunks = chunk_transcript_lines(transcript_text, args.chunk_max_chars) if chunked else [transcript_text]
        run_manifest["chunking"]["enabled"] = chunked
        run_manifest["chunking"]["chunk_count"] = len(transcript_chunks)

        print_step(f"Chunking enabled: {chunked}")
        print_step(f"Chunk count: {len(transcript_chunks)}")

        chunk_reports: list[str] = []
        if chunked:
            for index, transcript_chunk in enumerate(transcript_chunks, start=1):
                print_step(f"Building Layer 2 prompt for chunk {index}/{len(transcript_chunks)}")
                speaker_aware_chunk = build_speaker_aware_transcript(
                    transcript_chunk,
                    quality,
                    artifact,
                    meeting_date,
                    meeting_time,
                    chunk_index=index,
                    total_chunks=len(transcript_chunks),
                )
                chunk_metadata_line = f"{metadata_line} | Chunk: {index}/{len(transcript_chunks)}"
                chunk_prompt = build_shadow_layer2_prompt(
                    layer2_prompt_template,
                    chunk_metadata_line,
                    speaker_aware_chunk,
                )
                print_step(f"Calling Ollama for chunk {index}/{len(transcript_chunks)}")
                chunk_raw = summarize_with_ollama(chunk_prompt, args.ollama_url, args.model)
                chunk_report = strip_thinking_tokens(chunk_raw)
                if not chunk_report:
                    raise RuntimeError(f"Chunk {index}/{len(transcript_chunks)} returned an empty Layer 2 report")
                chunk_output_path = unique_output_path(
                    output_dir / "layer2_chunks",
                    f"{output_stem}__chunk_{index:02d}_of_{len(transcript_chunks):02d}",
                    ".md",
                )
                write_text(chunk_output_path, chunk_report)
                run_manifest["chunking"]["chunk_output_paths"].append(str(chunk_output_path))
                chunk_reports.append(chunk_report)
                print_step(f"Chunk {index}/{len(transcript_chunks)} report saved: {chunk_output_path}")

            print_step("Merging chunk reports into final Layer 2 report")
            merge_prompt = build_merge_prompt(
                metadata_line,
                chunk_reports,
                quality,
                artifact,
                meeting_date,
                meeting_time,
            )
            merged_raw = summarize_with_ollama(merge_prompt, args.ollama_url, args.model)
            layer2_report = strip_thinking_tokens(merged_raw)
            if not layer2_report:
                raise RuntimeError("Merged Layer 2 report was empty")
        else:
            print_step("Building Layer 2 prompt without chunking")
            speaker_aware_transcript = build_speaker_aware_transcript(
                transcript_text,
                quality,
                artifact,
                meeting_date,
                meeting_time,
            )
            layer2_prompt = build_shadow_layer2_prompt(
                layer2_prompt_template,
                metadata_line,
                speaker_aware_transcript,
            )
            print_step("Calling Ollama for Layer 2")
            layer2_raw = summarize_with_ollama(layer2_prompt, args.ollama_url, args.model)
            layer2_report = strip_thinking_tokens(layer2_raw)
            if not layer2_report:
                raise RuntimeError("Layer 2 report was empty")

        write_text(layer2_output_path, layer2_report)
        print_step(f"Final merge output path: {layer2_output_path}")

        print_step("Building Layer 3 prompt from the final Layer 2 report")
        layer3_prompt = build_shadow_layer3_prompt(
            layer3_prompt_template,
            meeting_name,
            meeting_datetime,
            layer2_report,
        )

        print_step("Calling Ollama for Layer 3")
        layer3_raw = summarize_with_ollama(layer3_prompt, args.ollama_url, args.model)
        layer3_brief = strip_thinking_tokens(layer3_raw)
        if not layer3_brief:
            raise RuntimeError("Layer 3 brief was empty")
        write_text(brief_output_path, layer3_brief)
        print_step(f"Layer 3 output path: {brief_output_path}")

        print_step("Running deterministic post-generation safety validation")
        safety_result, sanitized_layer2_report, sanitized_layer3_brief = validate_and_sanitize_outputs(
            artifact,
            layer2_report,
            layer3_brief,
        )
        run_manifest["safety_warnings"] = safety_result["safety_warnings"]
        run_manifest["relative_timing_detected"] = safety_result["relative_timing_detected"]
        run_manifest["speaker_identity_inference_detected"] = safety_result["speaker_identity_inference_detected"]
        run_manifest["first_person_action_detected"] = safety_result["first_person_action_detected"]
        run_manifest["unsupported_named_owner_detected"] = safety_result["unsupported_named_owner_detected"]
        run_manifest["extended_relative_timing_detected"] = safety_result["extended_relative_timing_detected"]
        run_manifest["layer3_action_rewritten_for_review"] = safety_result["layer3_action_rewritten_for_review"]
        write_text(sanitized_layer2_output_path, sanitized_layer2_report)
        write_text(sanitized_brief_output_path, sanitized_layer3_brief)
        print_step(f"Sanitized Layer 2 output path: {sanitized_layer2_output_path}")
        print_step(f"Sanitized Layer 3 output path: {sanitized_brief_output_path}")
        if run_manifest["safety_warnings"]:
            print_step("Safety warnings: " + " | ".join(run_manifest["safety_warnings"]))
        else:
            print_step("Safety warnings: none")

        run_manifest["success"] = True
    except Exception as exc:
        run_manifest["error"] = str(exc)
        print_step(f"ERROR: {exc}")
        raise
    finally:
        run_manifest["timestamps"]["completed_at_utc"] = now_utc().isoformat()
        write_json(manifest_output_path, run_manifest)
        print_step(f"Run manifest saved: {manifest_output_path}")

    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.sanitize_existing_manifest:
            return sanitize_from_existing_manifest(args)
        if args.artifact_dir is None:
            raise RuntimeError("Either --artifact-dir or --sanitize-existing-manifest is required")
        return run_shadow_pipeline(args)
    except Exception as exc:
        print(f"[shadow-artifact-pipeline] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
