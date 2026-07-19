"""Fetch and clean earnings-call transcript pages.

Primary source is Motley Fool; other hosts (e.g. TickerTrends) use a generic
DOM / JSON extraction pass with the same speaker-turn formatting.

Motley Fool pages often include preamble blocks (Date, Call participants,
Industry glossary). Those are kept in the cleaned ``.txt``; AI Takeaways /
Summary / Risks are dropped. Call-body start is flexible: full conference
marker, prepared remarks, or the first speaker / Operator line.

Writes::

    data/raw/transcripts/{fiscal_year}/{TICKER}/FY{YYYY}_Q{N}.fool.html
    data/raw/transcripts/{fiscal_year}/{TICKER}/FY{YYYY}_Q{N}.txt
    data/raw/transcripts/{fiscal_year}/{TICKER}/FY{YYYY}_Q{N}.meta.json
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

from crosscheck.config import transcript_dir
from crosscheck.manifest import CompanyPeriod

_MIN_INTERVAL_S = 1.5
_last_request_at = 0.0

MIN_TRANSCRIPT_CHARS = 1500

# "Tim Cook -- CEO" / "Tim Cook — CEO" (name first)
SPEAKER_DASH = re.compile(
    r"^[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4}\s*[-–—]{1,2}\s+.+$"
)
# "Tim Cook: Thanks everyone…" / "Operator: …"
SPEAKER_COLON = re.compile(
    r"^(?:Operator|Analyst|[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4})\s*:\s*\S"
)
OPERATOR_LINE = re.compile(r"^Operator\b", re.I)

# Job-title cues (helps split "Title — Name" vs "Name -- Title")
TITLE_CUES = re.compile(
    r"\b(chief|officer|president|chairman|chair|director|svp|evp|vp|"
    r"ceo|cfo|coo|cto|cmo|general\s+counsel|secretary|manager|"
    r"analyst|founder|co-founder)\b",
    re.I,
)

# Section headers on Motley Fool transcript pages
_HDR_DATE = re.compile(r"^date:?$", re.I)
_HDR_PARTICIPANTS = re.compile(r"^call\s+participants:?$", re.I)
_HDR_GLOSSARY = re.compile(r"^industry\s+glossary:?$", re.I)
_HDR_TAKEAWAYS = re.compile(r"^takeaways?:?$", re.I)
_HDR_SUMMARY = re.compile(r"^summary:?$", re.I)
_HDR_RISKS = re.compile(r"^risks?:?$", re.I)
_HDR_CONTENTS = re.compile(r"^contents?:?$", re.I)
_HDR_FULL = re.compile(
    r"^(full\s+conference\s+call\s+transcript|conference\s+call\s+transcript|"
    r"transcript\s+of\s+the\s+(?:earnings\s+)?call)\s*:?\s*$",
    re.I,
)
_HDR_PREPARED = re.compile(r"^prepared\s+remarks?:?\s*$", re.I)
_HDR_QA = re.compile(r"^questions?\s*(?:&|and)\s*answers?:?\s*$", re.I)

# Keep in cleaned text
KEEP_PREAMBLE = ("date", "call_participants", "industry_glossary")
# Drop AI editorial blocks
DROP_SECTIONS = {"takeaways", "summary", "risks", "contents"}

NOISE_LINE = [
    re.compile(r"^accessibility menu$", re.I),
    re.compile(r"^image source:\s*the motley fool", re.I),
    re.compile(r"^the motley fool has a disclosure", re.I),
    re.compile(r"^this article is a transcript", re.I),
    re.compile(r"^join the motley fool", re.I),
    re.compile(r"^stock advisor", re.I),
    re.compile(r"^continue reading$", re.I),
    re.compile(r"^duration\s*[-–—:]", re.I),
    re.compile(r"^need a quote from a motley fool", re.I),
    re.compile(r"^---%$"),
]

SKIP_SPEAKER_NAMES = {
    "duration",
    "contents",
    "image source",
    "accessibility menu",
}


@dataclass
class TranscriptExtract:
    """Cleaned transcript text plus Motley Fool preamble metadata."""

    text: str
    call_date: str | None = None
    participants: list[dict[str, str]] = field(default_factory=list)


def _headers() -> dict[str, str]:
    """Browser-like headers for transcript page requests."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36 CrosscheckResearch/0.1"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _get(url: str, *, timeout: float = 60) -> requests.Response:
    """GET with polite inter-request delay."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    _last_request_at = time.monotonic()
    resp.raise_for_status()
    return resp


def transcript_stem(period: CompanyPeriod) -> str:
    """Return filename stem ``FY{year}_Q{quarter}``."""
    return f"FY{period.fiscal_year}_Q{period.fiscal_quarter}"


def transcript_paths(period: CompanyPeriod) -> dict[str, Path]:
    """Resolve raw HTML / cleaned txt / meta paths for a company-period."""
    out_dir = transcript_dir(period.ticker, period.fiscal_year)
    stem = transcript_stem(period)
    return {
        "dir": out_dir,
        "txt": out_dir / f"{stem}.txt",
        "html": out_dir / f"{stem}.fool.html",
        "meta": out_dir / f"{stem}.meta.json",
    }


def is_speaker_line(line: str) -> bool:
    """True if ``line`` looks like a speaker header (dash or colon styles)."""
    s = line.strip()
    if not s:
        return False
    if OPERATOR_LINE.match(s) and (":" in s or s.lower() == "operator"):
        return True
    if SPEAKER_COLON.match(s):
        return True
    if SPEAKER_DASH.match(s):
        name = re.split(r"\s*[-–—]{1,2}\s*", s, maxsplit=1)[0].strip().lower()
        return name not in SKIP_SPEAKER_NAMES
    return False


def count_speaker_headers(text: str) -> int:
    """Count speaker-header lines in cleaned transcript text."""
    return sum(1 for line in text.splitlines() if is_speaker_line(line))


def parse_participant_line(line: str) -> tuple[str, str] | None:
    """Parse ``Name -- Title`` or ``Title — Name`` into ``(name, title)``."""
    s = line.strip()
    if not s or not re.search(r"[-–—]", s):
        return None
    left, right = re.split(r"\s*[-–—]{1,2}\s*", s, maxsplit=1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None

    left_titleish = bool(TITLE_CUES.search(left))
    right_titleish = bool(TITLE_CUES.search(right))
    left_words, right_words = left.split(), right.split()

    # "Chief Executive Officer — Timothy Donald Cook"
    if left_titleish and not right_titleish and 1 <= len(right_words) <= 5:
        return right, left
    # "Timothy Cook -- Chief Executive Officer"
    if right_titleish and not left_titleish and 1 <= len(left_words) <= 5:
        return left, right
    # Prefer short left as name when ambiguous
    if 1 <= len(left_words) <= 4 and len(right) > len(left):
        return left, right
    if 1 <= len(right_words) <= 4 and left_titleish:
        return right, left
    return None


def parse_participant_roster(text: str) -> dict[str, str]:
    """Build ``normalized_name → title`` from a Call participants block / dash headers."""
    roster: dict[str, str] = {}
    in_participants = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if in_participants:
                # blank line ends a short participants block only if we already have entries
                continue
            continue
        if _HDR_PARTICIPANTS.match(line):
            in_participants = True
            continue
        if in_participants:
            if _classify_header(line):
                in_participants = False
            else:
                parsed = parse_participant_line(line)
                if parsed:
                    name, title = parsed
                    roster[_norm_name(name)] = title
                elif is_speaker_line(line) and not parse_participant_line(line):
                    in_participants = False
                continue
        # Also harvest titles from in-call "Name -- Title" headers
        if is_speaker_line(line):
            parsed = parse_participant_line(line)
            if parsed:
                name, title = parsed
                roster.setdefault(_norm_name(name), title)
    return roster


def lookup_speaker_title(speaker: str, roster: dict[str, str]) -> str | None:
    """Resolve a colloquial / partial speaker name against a participant roster."""
    if not speaker or not roster:
        return None
    key = _norm_name(speaker)
    if key in roster:
        return roster[key]
    parts = key.split()
    if not parts:
        return None
    # First + last
    if len(parts) >= 2:
        for rk, title in roster.items():
            rp = rk.split()
            if len(rp) >= 2 and rp[0] == parts[0] and rp[-1] == parts[-1]:
                return title
    # Unique last-name match
    last = parts[-1]
    hits = [(rk, title) for rk, title in roster.items() if rk.split()[-1] == last]
    if len(hits) == 1:
        return hits[0][1]
    return None


def _norm_name(name: str) -> str:
    """Lowercase / collapse whitespace for roster keys."""
    return re.sub(r"\s+", " ", name).strip().lower()


def _classify_header(line: str) -> str | None:
    """Return a section key if ``line`` is a known Motley Fool section header."""
    s = line.strip()
    if _HDR_DATE.match(s):
        return "date"
    if _HDR_PARTICIPANTS.match(s):
        return "call_participants"
    if _HDR_GLOSSARY.match(s):
        return "industry_glossary"
    if _HDR_TAKEAWAYS.match(s):
        return "takeaways"
    if _HDR_SUMMARY.match(s):
        return "summary"
    if _HDR_RISKS.match(s):
        return "risks"
    if _HDR_CONTENTS.match(s):
        return "contents"
    if _HDR_FULL.match(s):
        return "full_transcript"
    if _HDR_PREPARED.match(s):
        return "prepared_remarks"
    if _HDR_QA.match(s):
        return "qa"
    return None


def _is_noise_line(line: str) -> bool:
    """Site chrome / ads noise (not structural section headers we keep)."""
    s = line.strip()
    if not s or len(s) < 2:
        return True
    return any(p.search(s) for p in NOISE_LINE)


def _article_root(soup: BeautifulSoup) -> Tag:
    """Pick the best DOM node that likely contains the transcript body."""
    for selector in (
        "div.article-body",
        "div.transcript-content",
        "div.tailwind-article-body",
        "article .article-content",
        "article",
        "main",
    ):
        node = soup.select_one(selector)
        if node is not None:
            return node
    body = soup.body
    if body is None:
        raise ValueError("Page has no <body>; cannot extract transcript")
    return body


def _strip_noise(root: Tag) -> None:
    """Remove nav/ads/scripts in-place from the article subtree."""
    for tag in root.find_all(
        ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg"]
    ):
        tag.decompose()
    for tag in root.find_all(
        class_=re.compile(
            r"(ad-|advert|promo|newsletter|related|share|social|paywall|cookie)",
            re.I,
        )
    ):
        tag.decompose()


def _lines_from_dom(root: Tag) -> list[str]:
    """Collect cleaned text lines from common block tags under ``root``."""
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "p", "div", "li"]):
        if el.name == "div" and el.find(["p", "h1", "h2", "h3", "li"]):
            continue
        text = " ".join(el.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if _is_noise_line(text):
            continue
        # Skip TOC-only echo of section names without trailing colon handled as headers later
        lines.append(text)
    return _dedupe_consecutive(lines)


def _dedupe_consecutive(lines: list[str]) -> list[str]:
    """Drop consecutive duplicate lines."""
    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    return deduped


def _split_fool_sections(lines: list[str]) -> dict[str, list[str]]:
    """Partition Motley Fool article lines into named sections."""
    sections: dict[str, list[str]] = {"_pre": []}
    current = "_pre"
    for line in lines:
        header = _classify_header(line)
        if header:
            current = header
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _call_body_lines(sections: dict[str, list[str]], all_lines: list[str]) -> list[str]:
    """Pick the earnings-call body with flexible start markers."""
    # Prefer explicit full transcript block
    if sections.get("full_transcript"):
        return list(sections["full_transcript"])

    # Prepared remarks (+ Q&A if present) — common older Fool layout
    if sections.get("prepared_remarks"):
        body = list(sections["prepared_remarks"])
        if sections.get("qa"):
            body.append("")
            body.append("Questions & Answers:")
            body.extend(sections["qa"])
        return body

    # Fall back: scan all lines, skip dropped editorial sections, start at first speaker
    current_drop = False
    collected: list[str] = []
    started = False
    for line in all_lines:
        header = _classify_header(line)
        if header:
            if header in DROP_SECTIONS or header in KEEP_PREAMBLE:
                current_drop = True
                continue
            if header in {"full_transcript", "prepared_remarks", "qa"}:
                current_drop = False
                started = True
                if header == "qa":
                    collected.append("Questions & Answers:")
                continue
            current_drop = False
            continue
        if current_drop:
            continue
        if not started:
            if is_speaker_line(line):
                started = True
                collected.append(line)
            continue
        collected.append(line)
    return collected


def _format_speaker_turns(lines: list[str]) -> list[str]:
    """Insert blank lines around speaker headers for the chunker."""
    formatted: list[str] = []
    for line in lines:
        if is_speaker_line(line):
            if formatted and formatted[-1] != "":
                formatted.append("")
            formatted.append(line)
            formatted.append("")
        else:
            formatted.append(line)
    return formatted


def _participants_as_dicts(section_lines: list[str]) -> list[dict[str, str]]:
    """Convert Call participants lines to ``[{name, title}, ...]``."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in section_lines:
        parsed = parse_participant_line(line)
        if not parsed:
            continue
        name, title = parsed
        key = _norm_name(name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "title": title})
    return out


def _fallback_period_label(fiscal_year: int, fiscal_quarter: int) -> str:
    """Human-readable period when the page has no Date section."""
    return f"Q{fiscal_quarter} {fiscal_year}"


def _harvest_participants_from_body(body: list[str]) -> list[dict[str, str]]:
    """Collect Name -- Title pairs from in-call speaker headers (meta only)."""
    rebuilt: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in body:
        if not is_speaker_line(line):
            continue
        parsed = parse_participant_line(line)
        if not parsed:
            continue
        name, title = parsed
        key = _norm_name(name)
        if key in seen:
            continue
        seen.add(key)
        rebuilt.append({"name": name, "title": title})
    return rebuilt


def assemble_transcript_document(
    *,
    call_date: str | None,
    participants: list[dict[str, str]],
    glossary_lines: list[str],
    body_lines: list[str],
    fiscal_year: int,
    fiscal_quarter: int,
    participants_section_found: bool,
    glossary_section_found: bool,
) -> TranscriptExtract:
    """Build the canonical cleaned ``.txt`` layout.

    Order::

        Date
        …
        Call participants
        …
        Industry glossary
        …
        Full Conference Call Transcript
        …
    """
    out: list[str] = []

    # 1) Date
    out.append("Date")
    if call_date:
        out.append(call_date)
        date_meta = call_date
    else:
        date_meta = _fallback_period_label(fiscal_year, fiscal_quarter)
        out.append(date_meta)
    out.append("")

    # 2) Call participants
    out.append("Call participants")
    if participants_section_found and participants:
        for p in participants:
            out.append(f"{p['name']} — {p['title']}")
    else:
        out.append("section not found")
    out.append("")

    # 3) Industry glossary
    out.append("Industry glossary")
    if glossary_section_found and glossary_lines:
        out.extend(glossary_lines)
    else:
        out.append("section not found")
    out.append("")

    # 4) Call body
    out.append("Full Conference Call Transcript")
    out.append("")
    out.extend(_format_speaker_turns(body_lines))

    text = "\n".join(out).strip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Meta still gets titles from in-call dash headers when the roster block was missing
    meta_participants = list(participants) if participants else _harvest_participants_from_body(
        body_lines
    )

    return TranscriptExtract(
        text=text,
        call_date=date_meta,
        participants=meta_participants,
    )


def extract_motley_fool(
    html: str,
    *,
    fiscal_year: int,
    fiscal_quarter: int,
) -> TranscriptExtract:
    """Section-aware Motley Fool extraction (preamble + flexible call body)."""
    soup = BeautifulSoup(html, "lxml")
    root = _article_root(soup)
    _strip_noise(root)
    lines = _lines_from_dom(root)
    sections = _split_fool_sections(lines)

    call_date: str | None = None
    for line in sections.get("date") or []:
        if line.strip():
            call_date = line.strip()
            break

    participant_lines = [
        ln for ln in (sections.get("call_participants") or []) if ln.strip()
    ]
    participants = _participants_as_dicts(participant_lines)
    # Section "found" only when we can parse at least one Name — Title row
    participants_section_found = bool(participants)

    glossary_lines = [
        ln for ln in (sections.get("industry_glossary") or []) if ln.strip()
    ]
    glossary_section_found = bool(glossary_lines)

    body = _call_body_lines(sections, lines)
    body = [ln for ln in body if not _is_noise_line(ln) or is_speaker_line(ln)]

    return assemble_transcript_document(
        call_date=call_date,
        participants=participants,
        glossary_lines=glossary_lines,
        body_lines=body,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        participants_section_found=participants_section_found,
        glossary_section_found=glossary_section_found,
    )


def _body_lines_from_plain(text: str) -> list[str]:
    """Normalize a pre-extracted plain-text blob into call-body lines."""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not _is_noise_line(ln)]
    lines = _dedupe_consecutive(lines)
    start = 0
    for i, line in enumerate(lines):
        if _HDR_FULL.match(line) or _HDR_PREPARED.match(line):
            start = i + 1
            break
    else:
        for i, line in enumerate(lines):
            if is_speaker_line(line):
                start = i
                break
    return lines[start:]


def _extract_from_next_data(html: str) -> list[str] | None:
    """Pull transcript body lines from Next.js ``__NEXT_DATA__`` when present."""
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("script", id="__NEXT_DATA__")
    if node is None or not node.string:
        return None
    try:
        data = json.loads(node.string)
    except json.JSONDecodeError:
        return None

    candidates: list[str] = []

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    isinstance(value, str)
                    and len(value) >= MIN_TRANSCRIPT_CHARS
                    and (
                        key.lower() in {"transcript", "transcripttext", "content", "body"}
                        or "operator:" in value.lower()[:2000]
                    )
                ):
                    candidates.append(value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    if not candidates:
        return None
    return _body_lines_from_plain(max(candidates, key=len))


def looks_like_motley_fool(html: str, url: str | None = None) -> bool:
    """Heuristic: Fool host or article-body with Fool section headers."""
    if url and "fool.com" in urlparse(url).netloc.lower():
        return True
    low = html.lower()
    return "article-body" in low and (
        "call participants" in low or "full conference call transcript" in low
        or "prepared remarks" in low
    )


def extract_transcript(
    html: str,
    *,
    url: str | None = None,
    fiscal_year: int,
    fiscal_quarter: int,
) -> TranscriptExtract:
    """Parse transcript HTML into the canonical cleaned ``.txt`` layout."""
    if looks_like_motley_fool(html, url):
        extracted = extract_motley_fool(
            html, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter
        )
        if len(extracted.text) >= MIN_TRANSCRIPT_CHARS:
            return extracted

    next_body = _extract_from_next_data(html)
    if next_body and sum(len(x) for x in next_body) >= MIN_TRANSCRIPT_CHARS:
        return assemble_transcript_document(
            call_date=None,
            participants=[],
            glossary_lines=[],
            body_lines=next_body,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            participants_section_found=False,
            glossary_section_found=False,
        )

    soup = BeautifulSoup(html, "lxml")
    root = _article_root(soup)
    _strip_noise(root)
    lines = _lines_from_dom(root)
    start = 0
    for i, line in enumerate(lines):
        if _HDR_FULL.match(line) or _HDR_PREPARED.match(line):
            start = i + 1
            break
    else:
        for i, line in enumerate(lines):
            if is_speaker_line(line):
                start = i
                break
    return assemble_transcript_document(
        call_date=None,
        participants=[],
        glossary_lines=[],
        body_lines=lines[start:],
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        participants_section_found=False,
        glossary_section_found=False,
    )


def validate_transcript_text(text: str) -> None:
    """Fail loudly if extraction looks like a paywall/block page."""
    if len(text) < MIN_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Extracted transcript too short ({len(text)} chars); "
            "page may be paywalled, blocked, or HTML structure changed."
        )
    speaker_hits = count_speaker_headers(text)
    if speaker_hits < 2:
        raise ValueError(
            f"Expected speaker headers like 'Name:' or 'Name -- Title'; "
            f"found {speaker_hits}. Extraction likely failed."
        )


def source_label(url: str) -> str:
    """Short host-based source tag for meta.json."""
    host = urlparse(url).netloc.lower()
    if "fool.com" in host:
        return "motley_fool"
    if "tickertrends" in host:
        return "tickertrends"
    return host or "unknown"


def fetch_transcript(
    period: CompanyPeriod,
    *,
    force: bool = False,
    allow_existing_txt: bool = True,
) -> Path:
    """Fetch transcript URL for a company period; return cleaned .txt path."""
    url = period.transcript_url_str()
    paths = transcript_paths(period)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if url is None:
        if allow_existing_txt and paths["txt"].exists():
            return paths["txt"]
        raise ValueError(
            f"No transcript_url for {period.ticker} FY{period.fiscal_year} "
            f"Q{period.fiscal_quarter} and no existing {paths['txt']}"
        )

    if paths["txt"].exists() and not force:
        return paths["txt"]

    # Re-parse saved HTML when present (avoids re-hitting the site after failed validate)
    if paths["html"].exists() and not force:
        html = paths["html"].read_text(encoding="utf-8")
    else:
        resp = _get(url)
        html = resp.text
        paths["html"].write_text(html, encoding="utf-8")

    extracted = extract_transcript(
        html,
        url=url,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
    )
    validate_transcript_text(extracted.text)
    paths["txt"].write_text(extracted.text, encoding="utf-8")

    meta = {
        "source": source_label(url),
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ticker": period.ticker,
        "company_name": period.name,
        "fiscal_year": period.fiscal_year,
        "fiscal_quarter": period.fiscal_quarter,
        "chars": len(extracted.text),
        "call_date": extracted.call_date,
        "call_participants": extracted.participants,
        "raw_html": str(paths["html"]),
        "cleaned_txt": str(paths["txt"]),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths["txt"]
