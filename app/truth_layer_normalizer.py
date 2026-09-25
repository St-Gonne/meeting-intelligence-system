#!/usr/bin/env python3
from __future__ import annotations
from mi_paths import public_path

import argparse
from pathlib import Path
from typing import Any

from meeting_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    SPEAKER_ATTRIBUTION_WARNING,
    TRUTH_LAYER_ENGINE,
    artifact_dir_name,
    build_quality_warnings,
    choose_single_file,
    collect_source_files,
    final_speaker_labels,
    infer_confidence_label,
    load_json,
    meeting_id_from_fields,
    new_artifact_id,
    normalize_segments,
    render_transcript_markdown,
    save_json,
    sha256_file,
    unclear_segment_count,
    utc_now_iso,
)


DEFAULT_OUTPUT_ROOT = Path(str(public_path('project/artifacts/meetings')))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an existing WhisperMLX output folder into a MeetingIntel artifact.")
    parser.add_argument("--whispermlx-output-dir", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--meeting-name", required=True)
    parser.add_argument("--meeting-date-ist", required=True)
    parser.add_argument("--meeting-time-ist", required=True)
    parser.add_argument("--source-type", required=True)
    parser.add_argument("--speaker-count-mode", required=True)
    parser.add_argument("--detected-speakers-before-forcing", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[truth-layer-normalizer] {message}")


def require_input_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"WhisperMLX output directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"WhisperMLX output path is not a directory: {path}")
    return path


def build_quality_payload(
    warnings: list[str],
    final_speaker_count: int,
    detected_speakers_before_forcing: int | None,
    speaker_count_mode: str,
    has_segments: bool,
) -> dict[str, Any]:
    return {
        "confidence_label": infer_confidence_label(final_speaker_count, has_segments),
        "warnings": warnings,
        "final_speaker_count": final_speaker_count,
        "detected_speakers_before_forcing": detected_speakers_before_forcing,
        "speaker_count_mode": speaker_count_mode,
        "human_review_recommended": True,
    }


def main() -> int:
    args = parse_args()
    output_dir = require_input_dir(args.whispermlx_output_dir.expanduser().resolve())
    source_audio = args.source_audio.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    print_step(f"Scanning WhisperMLX output folder: {output_dir}")
    source_json_path = choose_single_file(output_dir, ".json")
    source_txt_path = choose_single_file(output_dir, ".txt")
    if source_json_path is None:
        raise FileNotFoundError(f"No .json file found in {output_dir}")
    if source_txt_path is None:
        raise FileNotFoundError(f"No .txt file found in {output_dir}")

    print_step(f"Loading segment data from: {source_json_path.name}")
    whisper_payload = load_json(source_json_path)
    segments = normalize_segments(whisper_payload)
    labels = final_speaker_labels(segments)
    final_speaker_count = len(labels)
    unclear_segments = unclear_segment_count(segments)
    warnings = build_quality_warnings(
        speaker_count_mode=args.speaker_count_mode,
        detected_speakers_before_forcing=args.detected_speakers_before_forcing,
        final_speaker_count=final_speaker_count,
        unclear_segments=unclear_segments,
    )
    quality = build_quality_payload(
        warnings=warnings,
        final_speaker_count=final_speaker_count,
        detected_speakers_before_forcing=args.detected_speakers_before_forcing,
        speaker_count_mode=args.speaker_count_mode,
        has_segments=bool(segments),
    )

    artifact_id = new_artifact_id()
    meeting_id = meeting_id_from_fields(args.meeting_date_ist, args.meeting_time_ist, args.meeting_name)
    artifact_name = artifact_dir_name(args.meeting_date_ist, args.meeting_time_ist, args.meeting_name, artifact_id)
    artifact_dir = output_root / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    print_step(f"Created artifact folder: {artifact_dir}")

    transcript_raw_text = source_txt_path.read_text(encoding="utf-8")
    transcript_md = render_transcript_markdown(
        meeting_name=args.meeting_name,
        meeting_date_ist=args.meeting_date_ist,
        meeting_time_ist=args.meeting_time_ist,
        source_type=args.source_type,
        quality_note=quality["confidence_label"],
        segments=segments,
    )

    transcript_raw_path = artifact_dir / "transcript_raw.txt"
    transcript_md_path = artifact_dir / "transcript_diarized.md"
    segments_path = artifact_dir / "segments.json"
    quality_path = artifact_dir / "quality.json"
    source_files_path = artifact_dir / "source_files.json"
    artifact_json_path = artifact_dir / "artifact.json"

    transcript_raw_path.write_text(transcript_raw_text, encoding="utf-8")
    transcript_md_path.write_text(transcript_md, encoding="utf-8")
    save_json(segments_path, segments)
    save_json(quality_path, quality)

    source_files_payload = {
        "source_audio_path": str(source_audio),
        "source_audio_sha256": sha256_file(source_audio) if source_audio.exists() and source_audio.is_file() else None,
        "source_output_folder": str(output_dir),
        "files": collect_source_files(output_dir),
    }
    save_json(source_files_path, source_files_payload)

    artifact_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "meeting_id": meeting_id,
        "meeting_name": args.meeting_name,
        "meeting_date_ist": args.meeting_date_ist,
        "meeting_time_ist": args.meeting_time_ist,
        "source_type": args.source_type,
        "source_audio_path": str(source_audio),
        "truth_layer_engine": TRUTH_LAYER_ENGINE,
        "speaker_count_mode": args.speaker_count_mode,
        "detected_speakers_before_forcing": args.detected_speakers_before_forcing,
        "final_speaker_labels": labels,
        "speaker_names_resolved": False,
        "generated_at_utc": utc_now_iso(),
        "generated_files": {
            "artifact_json": str(artifact_json_path),
            "transcript_diarized_md": str(transcript_md_path),
            "transcript_raw_txt": str(transcript_raw_path),
            "segments_json": str(segments_path),
            "quality_json": str(quality_path),
            "source_files_json": str(source_files_path),
        },
        "warnings": warnings,
    }
    save_json(artifact_json_path, artifact_payload)

    print_step(f"Wrote transcript: {transcript_md_path}")
    print_step(f"Wrote segments: {segments_path}")
    print_step(f"Wrote quality summary: {quality_path}")
    print_step(f"Wrote source file manifest: {source_files_path}")
    print_step(f"Wrote artifact descriptor: {artifact_json_path}")
    print_step(
        "Completed successfully with "
        f"{len(segments)} segments, {final_speaker_count} final speaker labels, "
        f"and {unclear_segments} unclear segment(s)."
    )
    print_step(f"Artifact ready at: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
