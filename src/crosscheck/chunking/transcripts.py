"""Speaker-turn chunking for earnings-call transcript plain text.

Splits cleaned Motley Fool / ROIC / IR transcripts on speaker headers:

- Motley Fool: ``Name -- Title``, ``Name:``, ``Operator:``
- ROIC.ai: bare name alone on a line (``Sundar Pichai``); role left null
  unless known from a Call participants roster

Preamble blocks (Date, Call participants, Industry glossary) stay in the
``.txt`` for humans / meta but are **not** turned into chunks.

Transcript chunks do **not** set ``section`` (Prepared Remarks vs Q&A is
unreliable across hosts). Claim extraction reads the full cleaned ``.txt``
(not chunk/speaker filters). Embeddings key off ``speaker_name`` /
``speaker_role`` when present. Filing chunks still use ``section`` for
Item / MD&A labels.
"""

from __future__ import annotations

import re
from pathlib import Path

from crosscheck.ingest.transcript import (
    SPEAKER_BARE_NAME,
    TITLE_CUES,
    lookup_speaker_title,
    parse_participant_line,
    parse_participant_roster,
)
from crosscheck.models import (
    Chunk,
    DocumentMeta,
    fiscal_period_from_quarter,
    make_chunk_id,
)

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

# Words that can look Title Case but are dialogue, not speaker names
_BARE_NAME_BLOCKLIST = {
    "yeah",
    "yes",
    "yep",
    "no",
    "sure",
    "well",
    "okay",
    "ok",
    "look",
    "great",
    "thanks",
    "thank",
    "right",
    "alright",
    "hello",
    "hi",
    "good",
    "and",
    "so",
    "now",
    "next",
    "first",
}

# Single-token "names" that are almost always glossary / product labels
_NON_PERSON_TOKENS = re.compile(
    r"^(optimus|cybercab|powerwall|bedrock|copilot|agent|fabric|gemini|"
    r"epyc|instinct|rackscale|rocm|xpu|dsp|vcf|scb|nvfi|arpu|threads|"
    r"upgraders|installed|metaverified|metai)$",
    re.I,
)


def _is_bare_speaker_name(line: str) -> bool:
    """True for ROIC-style speaker headers: a person name alone on a line."""
    s = line.strip()
    if not s or s.lower() in SKIP_SPEAKERS:
        return False
    # Dialogue / sentence fragments, not names
    if s[-1] in ".!?,:;":
        return False
    if s.lower() != "operator" and not SPEAKER_BARE_NAME.match(s):
        return False
    if not _looks_like_person_name(s):
        return False
    tokens = re.sub(r"[^A-Za-z\s]", " ", s).lower().split()
    if any(t in _BARE_NAME_BLOCKLIST for t in tokens):
        return False
    return True


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
    """If ``line`` is a speaker header, return ``(speaker, rest, title_or_none)``.

    Supports Motley Fool ``Name -- Title`` / ``Name:``, bare ``Operator``, and
    ROIC-style bare names alone on a line (``Sundar Pichai``).
    """
    line = line.strip()
    if not line:
        return None
    looks_like_inline = ":" in line[:80] or " -- " in line[:80] or " — " in line[:80]
    if not looks_like_inline and len(line) > 160 and line.lower() != "operator":
        return None

    # Name -- Title (only when left side is the person name)
    parsed = parse_participant_line(line)
    if (
        parsed
        and re.search(r"[-–—]", line)
        and ":" not in line.split("—")[0].split("–")[0][:40]
    ):
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
        if rest and re.match(
            r"^(The|A|An|Direct|Existing|Total|Risk|Meta|Apple|"
            r"Alphabet|Proposed|Application|Original|High|"
            r"Revenue|Tensor)\b",
            rest,
        ):
            if not _DIALOGUE_STARTERS.match(rest):
                return None
        return speaker, rest, title

    # ROIC / bare-name layout: speaker alone on a line (no colon / dash title)
    if not looks_like_inline and _is_bare_speaker_name(line):
        return _normalize_speaker(line), "", None
    return None


def _is_strong_call_turn(line: str) -> bool:
    """True for Operator / ``Name:`` / bare-name speaker headers — ends preamble."""
    hit = _match_speaker(line)
    if not hit:
        return False
    speaker, rest, _title = hit
    if speaker.lower() == "operator":
        return True
    if rest:
        return bool(_DIALOGUE_STARTERS.match(rest))
    # Bare name with no inline rest (ROIC) is a real turn header
    return True


def chunk_transcript(text: str, meta: DocumentMeta) -> list[Chunk]:
    """Split a transcript into speaker-turn chunks with inherited metadata.

    Each chunk is one continuous speaker block. ``section`` is left unset;
    metadata includes ``speaker_name``, optional ``speaker_role``, and
    ``call_date``.
    """
    roster = parse_participant_roster(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[Chunk] = []
    speaker: str | None = None
    speaker_role: str | None = None
    buf: list[str] = []
    in_preamble = True
    in_glossary = False
    fiscal_period = fiscal_period_from_quarter(meta.fiscal_quarter)

    def flush() -> None:
        nonlocal buf, speaker, speaker_role
        body = "\n".join(buf).strip()
        buf = []
        if not body or speaker is None:
            return
        role = speaker_role or lookup_speaker_title(speaker, roster)
        # Keep role unset when neither the header nor the roster has a title.
        if role is not None and not str(role).strip():
            role = None
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(
                    ticker=meta.ticker,
                    fiscal_year=meta.fiscal_year,
                    fiscal_period=fiscal_period,
                    doc_type="transcript",
                    index=idx,
                ),
                ticker=meta.ticker.upper(),
                company_name=meta.company_name,
                doc_type="transcript",
                fiscal_year=meta.fiscal_year,
                fiscal_period=fiscal_period,
                is_table=False,
                text=body,
                section=None,
                speaker_name=speaker,
                speaker_role=role,
                call_date=meta.call_date,
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
            continue

        if in_preamble:
            if in_glossary and not _is_strong_call_turn(stripped):
                continue
            if _is_strong_call_turn(stripped):
                in_preamble = False
                in_glossary = False
            else:
                continue

        hit = _match_speaker(stripped)
        if hit:
            flush()
            speaker, rest, inline_title = hit
            speaker_role = inline_title or lookup_speaker_title(speaker, roster)
            if inline_title:
                roster.setdefault(speaker.lower(), inline_title)
            buf = [rest] if rest else []
            continue

        if speaker is None:
            continue

        buf.append(stripped)

    flush()
    for i, c in enumerate(chunks):
        c.chunk_id = make_chunk_id(
            ticker=c.ticker,
            fiscal_year=c.fiscal_year,
            fiscal_period=c.fiscal_period,
            doc_type=c.doc_type,
            index=i,
        )
    return chunks


def chunk_transcript_path(path: Path | str, meta: DocumentMeta) -> list[Chunk]:
    """Load transcript text from disk and run :func:`chunk_transcript`."""
    path = Path(path)
    meta = meta.model_copy(update={"source_path": str(path)})
    text = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_transcript(text, meta)
