"""Validated, advisory Context Pack v1 snapshots for Layer 2 prompts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MAX_CONTEXT_PACK_BYTES = 6000
REQUIRED_SECTIONS = (
    "## OPERATOR",
    "## ROSTER",
    "## ENTITIES",
    "## LIVE THREADS",
)


@dataclass(frozen=True)
class ContextPackSnapshot:
    content: str
    raw_bytes: bytes
    pack_version: int
    generated_at: str
    sha256: str
    view_sha256: str | None = None

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "pack_version": self.pack_version,
            "generated_at": self.generated_at,
            "sha256": self.sha256,
        }


SECTION_HEADING_RE = re.compile(r"^## [^\n\r]+$", re.MULTILINE)
SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_\d+$")


def _section_spans(content: str) -> tuple[dict[str, tuple[int, int, int]], str | None]:
    matches = list(SECTION_HEADING_RE.finditer(content))
    headings = [match.group(0).strip() for match in matches]
    if headings != list(REQUIRED_SECTIONS):
        return {}, "context_pack_ignored:unexpected_section_structure"
    spans: dict[str, tuple[int, int, int]] = {}
    for index, match in enumerate(matches):
        body_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        spans[match.group(0).strip()] = (match.start(), body_start, section_end)
    return spans, None


def source_view_snapshot(snapshot: ContextPackSnapshot) -> ContextPackSnapshot:
    """Return the original producer snapshot as the full trusted-context view."""
    return ContextPackSnapshot(
        content=snapshot.content,
        raw_bytes=snapshot.raw_bytes,
        pack_version=snapshot.pack_version,
        generated_at=snapshot.generated_at,
        sha256=snapshot.sha256,
        view_sha256=hashlib.sha256(snapshot.content.encode("utf-8")).hexdigest(),
    )


def rosterless_view_snapshot(snapshot: ContextPackSnapshot) -> ContextPackSnapshot:
    """Return a deterministic view with the complete ROSTER section removed."""
    spans, diagnostic = _section_spans(snapshot.content)
    if diagnostic:
        raise ValueError(diagnostic)
    roster_start, _roster_body_start, roster_end = spans["## ROSTER"]
    filtered = snapshot.content[:roster_start].rstrip() + "\n\n" + snapshot.content[roster_end:].lstrip()
    return ContextPackSnapshot(
        content=filtered,
        raw_bytes=snapshot.raw_bytes,
        pack_version=snapshot.pack_version,
        generated_at=snapshot.generated_at,
        sha256=snapshot.sha256,
        view_sha256=hashlib.sha256(filtered.encode("utf-8")).hexdigest(),
    )


def entity_spelling_view_snapshot(snapshot: ContextPackSnapshot) -> ContextPackSnapshot:
    """Return the only context view allowed to reach model prompts.

    The storage format remains unchanged.  This view deliberately exposes only
    the ENTITIES section so aliases and spelling variants cannot act as person,
    role, ownership, or meeting-status evidence.
    """
    spans, diagnostic = _section_spans(snapshot.content)
    if diagnostic:
        raise ValueError(diagnostic)
    entities_start, _entities_body_start, entities_end = spans["## ENTITIES"]
    header = snapshot.content[: snapshot.content.find("## OPERATOR")].rstrip()
    entities = snapshot.content[entities_start:entities_end].strip()
    filtered = f"{header}\n\n{entities}\n"
    return ContextPackSnapshot(
        content=filtered,
        raw_bytes=snapshot.raw_bytes,
        pack_version=snapshot.pack_version,
        generated_at=snapshot.generated_at,
        sha256=snapshot.sha256,
        view_sha256=hashlib.sha256(filtered.encode("utf-8")).hexdigest(),
    )


def has_confirmed_material_speaker_identity(speaker_map: list[dict[str, object]] | None) -> bool:
    """Use only existing trusted speaker-map evidence to unlock named roster context."""
    for entry in speaker_map or []:
        if not SPEAKER_LABEL_RE.fullmatch(str(entry.get("speaker_label", ""))):
            continue
        if entry.get("status") != "confirmed":
            continue
        identity = entry.get("identity")
        if not isinstance(identity, str) or not identity.strip():
            continue
        if entry.get("material_for_operator_prompt") is False:
            continue
        return True
    return False


def context_view_for_speaker_map(
    snapshot: ContextPackSnapshot,
    speaker_map: list[dict[str, object]] | None,
) -> ContextPackSnapshot:
    # Identity trust is supplied only by the operator speaker map itself.  A
    # speaker map must never unlock advisory person/role context.
    del speaker_map
    return entity_spelling_view_snapshot(snapshot)


def load_context_pack(path: Path | None) -> tuple[ContextPackSnapshot | None, str | None]:
    """Read at most once and return a validated frozen snapshot or safe diagnostic."""
    if path is None:
        return None, None
    try:
        expanded = path.expanduser()
        if expanded.is_symlink() or not expanded.is_file():
            return None, "context_pack_ignored:unsafe_file_shape"
        raw = expanded.read_bytes()
        if len(raw) > MAX_CONTEXT_PACK_BYTES:
            return None, "context_pack_ignored:oversized"
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "context_pack_ignored:invalid_utf8"
    except OSError:
        return None, "context_pack_ignored:unreadable"

    fields: dict[str, str] = {}
    for line in content.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"pack_version", "generated_at"}:
                fields[key.strip()] = value.strip()
    if fields.get("pack_version") != "1":
        return None, "context_pack_ignored:unsupported_version"
    generated_at = fields.get("generated_at", "")
    try:
        parsed = datetime.fromisoformat(generated_at)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except ValueError:
        return None, "context_pack_ignored:invalid_generated_at"
    if any(section not in content for section in REQUIRED_SECTIONS):
        return None, "context_pack_ignored:missing_required_section"
    _spans, diagnostic = _section_spans(content)
    if diagnostic:
        return None, diagnostic
    return ContextPackSnapshot(
        content=content,
        raw_bytes=raw,
        pack_version=1,
        generated_at=generated_at,
        sha256=hashlib.sha256(raw).hexdigest(),
    ), None


def advisory_context_block(snapshot: ContextPackSnapshot) -> str:
    return f"""================ ADVISORY CONTEXT PACK V1 ================
The Markdown below is advisory entity/spelling background, not meeting evidence.
- Use ENTITIES and explicit aliases/mishearings only to normalize spelling when meeting evidence is reasonably consistent.
- This view contains no trusted person, role, ownership, attribution, confirmation, stage, commitment, urgency, or current-status evidence.
- Never infer or hint at a real identity for any diarization label from this context.
- The pack is not exhaustive. Preserve new people/entities; absence does not imply uncertainty.
- Meeting evidence wins. Preserve a conflicting meeting claim and note a material discrepancy in ANALYST NOTES.
- Use [uncertain] only when meeting evidence itself is genuinely uncertain.
- Do not force a canonical match merely from phonetic similarity.

{snapshot.content.rstrip()}
================ END ADVISORY CONTEXT PACK V1 ================"""
