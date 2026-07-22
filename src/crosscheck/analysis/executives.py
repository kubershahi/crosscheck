"""Filter executive transcript chunks for claim extraction."""

from __future__ import annotations

import re

from crosscheck.models import Chunk

_EXECUTIVE_PATTERN = re.compile(
    r"\b(ceo|cfo|chief executive|chief financial|president|chairman)\b",
    re.IGNORECASE,
)


def is_executive_chunk(chunk: Chunk) -> bool:
    """True for C-suite transcript turns (role-based; section is ignored)."""
    if chunk.doc_type != "transcript":
        return False
    haystack = " ".join(x for x in (chunk.speaker_role, chunk.speaker_name) if x)
    if not haystack.strip():
        return False
    return bool(_EXECUTIVE_PATTERN.search(haystack))


def _format_block(chunk: Chunk) -> str:
    speaker = chunk.speaker_name or "Unknown"
    role = chunk.speaker_role or ""
    label = f"{speaker}, {role}" if role else speaker
    return f"[{label}]\n{chunk.text.strip()}"


def executive_source_text(chunks: list[Chunk]) -> str:
    """Stitch all C-suite transcript turns for the LLM in document order."""
    exec_chunks = [c for c in chunks if is_executive_chunk(c)]
    if not exec_chunks:
        return ""
    return "\n\n".join(_format_block(c) for c in exec_chunks)


def executive_prepared_text(chunks: list[Chunk]) -> str:
    """Backward-compatible alias for :func:`executive_source_text`."""
    return executive_source_text(chunks)
