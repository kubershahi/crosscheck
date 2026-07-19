"""Speaker-turn chunking for earnings-call transcript plain text.

Splits cleaned Motley Fool / IR transcripts on speaker headers
(``Name -- Title``, ``Name:``, ``Operator:``, …) and tags prepared remarks
vs Q&A when detectable.

Preamble blocks (Date, Call participants, Industry glossary) stay in the
``.txt`` for humans / meta but are not turned into chunks. Participant titles
from that roster are attached to later turns as ``speaker_title``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from crosscheck.ingest.motley_fool import (
    TITLE_CUES,
    lookup_speaker_title,
    parse_participant_line,
    parse_participant_roster,
)
from crosscheck.models import Chunk, DocumentMeta

SectionType = Literal["prepared_remarks", "qa", "unknown"]

# Motley Fool / IR / Seeking Alpha style speaker lines.
SPEAKER_PATTERNS: list[re.Pattern[str]] = [
    # "Tim Cook -- Chief Executive Officer"
    re.compile(
        r"^(?P<speaker>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*[-–—]{1,2}\s+(?P<title>.+)$"
    ),
    # "Operator:" / "Tim Cook:" / "Analyst:"
    re.compile(
        r"^(?P<speaker>Operator|Analyst|[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*:\s*(?P<rest>.*)$"
    ),
    # Bare "Operator" (older Fool layout)
    re.compile(r"^(?P<speaker>Operator)\s*$", re.I),
    # ALL CAPS "TIM COOK:"
    re.compile(r"^(?P<speaker>[A-Z][A-Z .'()\-]{1,40})\s*:\s*(?P<rest>.*)$"),
]

QA_MARKERS = re.compile(
    r"\b(question[- ]and[- ]answer|q\s*&\s*a|questions?\s+and\s+answers?)\b",
    re.I,
)
PREPARED_MARKERS = re.compile(
    r"\b(prepared\s+remarks|opening\s+remarks|operator\s+instructions)\b",
    re.I,
)

PREAMBLE_HEADERS = re.compile(
    r"^(date|call\s+participants|industry\s+glossary)\s*:?\s*$",
    re.I,
)
CALL_START_HEADERS = re.compile(
    r"^(full\s+conference\s+call\s+transcript|conference\s+call\s+transcript|"
    r"prepared\s+remarks|questions?\s*(?:&|and)\s*answers?)\s*:?\s*$",
    re.I,
)

SKIP_SPEAKERS = {
    "company representatives",
    "participants",
    "conference call participants",
    "duration",
    "contents",
    "date",
    "call participants",
    "industry glossary",
}

# Single-token "names" that are almost always glossary / product labels
_NON_PERSON_TOKENS = re.compile(
    r"^(optimus|cybercab|powerwall|bedrock|copilot|agent|fabric|gemini|"
    r"epyc|instinct|rackscale|rocm|xpu|dsp|vcf|scb|nvfi|arpu|threads|"
    r"upgraders|installed|metaverified|metai)$",
    re.I,
)


def _normalize_speaker(name: str) -> str:
    """Normalize speaker names (strip junk; title-case ALL CAPS)."""
    name = re.sub(r"\s+", " ", name).strip(" -:.")
    if name.isupper() and len(name) > 3:
        name = name.title()
    return name


def _looks_like_person_name(name: str) -> bool:
    """Heuristic: plausible human speaker (not glossary term / product label)."""
    name = name.strip()
    if not name:
        return False
    if name.lower() in {"operator", "analyst"}:
        return True
    if TITLE_CUES.search(name):
        return False
    if _NON_PERSON_TOKENS.match(name.replace(" ", "")):
        return False
    parts = name.split()
    if not 1 <= len(parts) <= 4:
        return False
    for part in parts:
        letters = re.sub(r"[^A-Za-z]", "", part)
        # Allow initials (C.J., J.P.); reject bare acronyms (AI, TPU, HBM)
        if letters.isupper() and 1 <= len(letters) <= 5:
            if not re.match(r"^(?:[A-Z]\.){1,3}[A-Z]?\.?$", part):
                return False
        if not re.match(r"^[A-Z](?:[a-zA-Z.'\-]*[A-Za-z.]|[.])$", part):
            return False
        if re.search(r"-[a-z]", part):  # tariff-related
            return False
    # Single short token is usually a product/acronym, not a speaker
    if len(parts) == 1 and len(re.sub(r"[^A-Za-z]", "", parts[0])) <= 4:
        return False
    return True


_DIALOGUE_STARTERS = re.compile(
    r"^(thank|thanks|good|hi|hello|yes|no|sure|well|okay|ok|great|"
    r"all right|alright|let me|i |we |our |as |today|afternoon|"
    r"morning|evening|and |on the|welcome|ladies)\b",
    re.I,
)


def _match_speaker(line: str) -> tuple[str, str, str | None] | None:
    """If ``line`` is a speaker header, return ``(speaker, rest, title_or_none)``."""
    line = line.strip()
    if not line:
        return None
    looks_like_inline = ":" in line[:80] or " -- " in line[:80] or " — " in line[:80]
    if not looks_like_inline and len(line) > 160 and line.lower() != "operator":
        return None

    # Name -- Title (only when left side is the person name)
    parsed = parse_participant_line(line)
    if parsed and re.search(r"[-–—]", line) and ":" not in line.split("—")[0].split("–")[0][:40]:
        name, title = parsed
        left = re.split(r"\s*[-–—]{1,2}\s*", line, maxsplit=1)[0].strip()
        if left.lower() == name.lower() and _looks_like_person_name(name):
            if title and len(title.split()) > 14:
                return None
            return _normalize_speaker(name), "", title

    for pat in SPEAKER_PATTERNS:
        m = pat.match(line)
        if not m:
            continue
        speaker = _normalize_speaker(m.group("speaker"))
        if speaker.lower() in SKIP_SPEAKERS:
            return None
        if not _looks_like_person_name(speaker):
            return None
        groups = m.groupdict()
        rest = (groups.get("rest") or "").strip()
        title = (groups.get("title") or "").strip() or None
        if title and len(title.split()) > 14:
            return None
        # Industry-glossary style "Term: The/A/An definition…"
        if rest and re.match(r"^(The|A|An|Direct|Existing|Total|Risk|Meta|Apple|"
                             r"Alphabet|Proposed|Application|Original|High|"
                             r"Revenue|Tensor)\b", rest):
            if not _DIALOGUE_STARTERS.match(rest):
                return None
        return speaker, rest, title
    return None


def _detect_section(line: str, current: SectionType) -> SectionType:
    """Update prepared-remarks vs Q&A state from marker phrases on ``line``."""
    if QA_MARKERS.search(line):
        return "qa"
    if PREPARED_MARKERS.search(line):
        return "prepared_remarks"
    return current


def _is_strong_call_turn(line: str) -> bool:
    """True for Operator / dialogue ``Name: remark`` — ends Fool preamble."""
    s = line.strip()
    if re.match(r"^Operator\b", s, re.I):
        return True
    m = re.match(
        r"^(?P<speaker>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*:\s*(?P<rest>\S.*)$",
        s,
    )
    if not m:
        return False
    speaker = m.group("speaker")
    rest = m.group("rest").strip()
    if not _looks_like_person_name(speaker):
        return False
    # Require a conversational open — glossary defs look like "Term: The/An/…"
    return bool(_DIALOGUE_STARTERS.match(rest))


def chunk_transcript(text: str, meta: DocumentMeta) -> list[Chunk]:
    """Split a transcript into speaker-turn chunks with inherited metadata.

    Each chunk is one continuous speaker block; metadata includes
    ``speaker_name``, optional ``speaker_title``, and ``section_type``.
    """
    roster = parse_participant_roster(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[Chunk] = []
    section: SectionType = "prepared_remarks"
    speaker: str | None = None
    speaker_title: str | None = None
    buf: list[str] = []
    in_preamble = True
    in_glossary = False

    def flush() -> None:
        nonlocal buf, speaker, speaker_title
        body = "\n".join(buf).strip()
        buf = []
        if not body or speaker is None:
            return
        title = speaker_title or lookup_speaker_title(speaker, roster)
        chunks.append(
            Chunk(
                text=body,
                doc_type="transcript",
                ticker=meta.ticker,
                company_name=meta.company_name,
                fiscal_year=meta.fiscal_year,
                fiscal_quarter=meta.fiscal_quarter,
                chunk_index=len(chunks),
                speaker_name=speaker,
                speaker_title=title,
                section_type=section,
                source_path=meta.source_path,
            )
        )

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if buf and not in_preamble:
                buf.append("")
            continue

        if PREAMBLE_HEADERS.match(stripped):
            in_preamble = True
            in_glossary = bool(re.match(r"^industry\s+glossary", stripped, re.I))
            continue
        if CALL_START_HEADERS.match(stripped):
            in_preamble = False
            in_glossary = False
            section = _detect_section(stripped, section)
            continue

        if in_preamble:
            # Glossary defs are never call turns; dialogue ``Name: Thanks…`` ends preamble
            if in_glossary and not _is_strong_call_turn(stripped):
                continue
            if _is_strong_call_turn(stripped):
                in_preamble = False
                in_glossary = False
            else:
                continue

        section = _detect_section(stripped, section)
        hit = _match_speaker(stripped)
        if hit:
            flush()
            speaker, rest, inline_title = hit
            speaker_title = inline_title or lookup_speaker_title(speaker, roster)
            if inline_title:
                roster.setdefault(speaker.lower(), inline_title)
            buf = [rest] if rest else []
            continue

        if speaker is None:
            continue
        buf.append(stripped)

    flush()
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def chunk_transcript_path(path: Path | str, meta: DocumentMeta) -> list[Chunk]:
    """Load transcript text from disk and run :func:`chunk_transcript`."""
    path = Path(path)
    meta = meta.model_copy(update={"source_path": str(path)})
    text = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_transcript(text, meta)
