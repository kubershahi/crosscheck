"""Section-aware + table-atomic chunking for SEC filing HTML.

Structure (10-Q / 10-K HTML):
- ``section``: lines that start with ``Item`` (also ``PART``).
- ``subsection`` / ``subsubsection``: short right-aligned headers when present;
  Apple-style filings often use bold + justify for the same role, so those are
  accepted as a fallback. Kept as metadata on every chunk; also prepended into
  **prose** ``text`` (tables already embed titles/captions in ``text`` via the
  centered / short-preface path — that table logic is left unchanged).
- Tables: contiguous **centered** preceding div texts are prepended as the
  table title/subtitle. When no centered preface exists (typical for Notes),
  short immediately-preceding title/caption lines are prepended instead.
  The full ``<table>`` is one chunk; short footers are appended when present.

Emits **stateless** :class:`~crosscheck.models.Chunk` rows (local ``chunk_id``
only — no ``global_id``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from crosscheck.models import (
    Chunk,
    DocumentMeta,
    filing_doc_type,
    filing_fiscal_period,
    make_chunk_id,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MAX_TEXT_CHUNK_CHARS = 3500
MIN_TEXT_CHUNK_CHARS = 80
SUBSECTION_MAX_CHARS = 80
TABLE_CAPTION_MAX_CHARS = 400
TABLE_FOOTER_MAX_CHARS = 400

_ITEM_SECTION_RE = re.compile(r"^item\s+\d+[a-z]?\.?\s*", re.I)
_PART_SECTION_RE = re.compile(r"^part\s+[ivx]+\b", re.I)
_NOTE_RE = re.compile(r"^Note\s+\d+\b", re.I)
_PAGE_CHROME_RE = re.compile(
    r"Form\s+10-[QK]\s*\|\s*\d+|\|\s*Q[1-4]\s+20\d{2}\s+Form\s+10-[QK]",
    re.I,
)
_FOOTER_START_RE = re.compile(
    r"^(\(\d+\)|\*+|†|‡|Note:|Includes\b|Total\b.+\binclude)",
    re.I,
)
_ALIGN_RE = re.compile(r"text-align\s*:\s*([a-z]+)", re.I)
_BOLD_RE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.I)


@dataclass
class Block:
    """One linearized filing block with HTML alignment hints."""

    kind: str  # "text" | "table"
    text: str
    align: str | None = None  # center | right | left | justify
    bold: bool = False


def _normalize_ws(text: str) -> str:
    """Collapse whitespace / NBSP from EDGAR extracted text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _style_align(style: str | None, align_attr: str | None = None) -> str | None:
    """Parse CSS ``text-align`` or HTML ``align`` into a lowercase token."""
    if style:
        match = _ALIGN_RE.search(style)
        if match:
            return match.group(1).lower()
    if align_attr:
        return align_attr.strip().lower() or None
    return None


def _element_align(tag: Tag) -> str | None:
    """Alignment declared on ``tag`` (not inherited)."""
    return _style_align(tag.get("style"), tag.get("align"))


def _element_bold(tag: Tag) -> bool:
    """True when ``tag`` or a nested span declares bold weight."""
    if _BOLD_RE.search(tag.get("style") or ""):
        return True
    if tag.find(["b", "strong"]):
        return True
    for span in tag.find_all("span"):
        if _BOLD_RE.search(span.get("style") or ""):
            return True
    return False


def _is_section_header(text: str) -> str | None:
    """Return cleaned section title for Item / PART headers only."""
    cleaned = _normalize_ws(text)
    if not cleaned or len(cleaned) > 200:
        return None
    cleaned = re.sub(r"^[\W\d_]{0,6}", "", cleaned)
    if _ITEM_SECTION_RE.match(cleaned) or _PART_SECTION_RE.match(cleaned):
        return cleaned[:160]
    return None


def _is_page_chrome(text: str) -> bool:
    """True for running headers like ``Apple Inc. | Q1 2025 Form 10-Q | 6``."""
    cleaned = _normalize_ws(text)
    if _PAGE_CHROME_RE.search(cleaned):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9 .,&'\-]+\s*\|\s*.+\|\s*\d+", cleaned))


def _is_subsection_header(block: Block) -> bool:
    """Short right-aligned (or bold fallback) subsection / subsubsection titles."""
    cleaned = _normalize_ws(block.text)
    if not cleaned or len(cleaned) > SUBSECTION_MAX_CHARS:
        return False
    if _is_section_header(cleaned) or _is_page_chrome(cleaned):
        return False
    # Table captions often end with ':' — keep those for the table preface.
    if cleaned.endswith(":"):
        return False
    if block.align == "right":
        return True
    # Many EDGAR HTML filings (e.g. Apple) use bold+justify instead of right.
    if block.bold and block.align in {"justify", "left", None}:
        if cleaned.endswith("."):
            return False
        return True
    return False


def _is_table_footer(text: str) -> bool:
    """True for short numbered / 'includes' lines that follow a table."""
    cleaned = _normalize_ws(text)
    if not cleaned or len(cleaned) > TABLE_FOOTER_MAX_CHARS:
        return False
    if _is_section_header(cleaned) or _is_page_chrome(cleaned):
        return False
    # Reject CamelCase product labels (iPhone, iPad, …).
    if re.match(r"^[a-z]+[A-Z]", cleaned):
        return False
    if _FOOTER_START_RE.match(cleaned):
        return True
    # Marked footnote continuations often start lowercase after (1)/(2).
    return bool(re.match(r"^[a-z]", cleaned)) and len(cleaned) <= 200


def _escape_md_cell(text: str) -> str:
    """Escape pipe characters so Markdown table cells stay intact."""
    return text.replace("|", "\\|")


def _table_to_markdown(table: Tag) -> str:
    """Convert an HTML ``<table>`` to a Markdown table (one string)."""
    grid: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [
            _normalize_ws(cell.get_text(" ", strip=True))
            for cell in tr.find_all(["th", "td"])
        ]
        if any(cells):
            grid.append(cells)
    if not grid:
        return ""

    width = max(len(row) for row in grid)
    rows = [row + [""] * (width - len(row)) for row in grid]
    header = rows[0]
    lines = [
        "| " + " | ".join(_escape_md_cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(_escape_md_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _strip_ixbrl_noise(soup: BeautifulSoup) -> None:
    """Remove scripts/styles and ``display:none`` XBRL chrome in-place."""
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()


_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr", "br"}


def _iter_blocks(soup: BeautifulSoup) -> list[Block]:
    """Return ordered text/table blocks with alignment + bold flags."""
    body = soup.body or soup
    blocks: list[Block] = []

    def walk(node: Tag) -> None:
        name = (node.name or "").lower()
        if name == "table":
            md = _table_to_markdown(node)
            if md.strip():
                blocks.append(Block(kind="table", text=md, align=_element_align(node)))
            return

        child_tags = [c for c in node.children if isinstance(c, Tag)]
        has_table_descendant = any(
            (c.name or "").lower() == "table" or c.find("table") for c in child_tags
        )
        has_block_child = any(
            (c.name or "").lower() in _BLOCK_TAGS | {"table"} for c in child_tags
        )

        if has_table_descendant or (
            has_block_child
            and name in {"body", "html", "div", "section", "td", "th", "span"}
        ):
            for child in child_tags:
                walk(child)
            return

        if name in _BLOCK_TAGS or name in {"font"}:
            text = _normalize_ws(node.get_text(" ", strip=True))
            if text:
                blocks.append(
                    Block(
                        kind="text",
                        text=text,
                        align=_element_align(node),
                        bold=_element_bold(node),
                    )
                )
            return

        for child in child_tags:
            walk(child)

    if isinstance(body, Tag):
        walk(body)

    deduped: list[Block] = []
    for block in blocks:
        if (
            deduped
            and deduped[-1].kind == block.kind
            and deduped[-1].text == block.text
            and len(block.text) < 200
        ):
            continue
        deduped.append(block)
    return deduped


def _make_filing_chunk(
    *,
    text: str,
    section: str,
    meta: DocumentMeta,
    is_table: bool,
    index: int,
    subsection: str | None = None,
    subsubsection: str | None = None,
) -> Chunk:
    doc_type = filing_doc_type(meta.form, meta.fiscal_quarter)
    fiscal_period = filing_fiscal_period(meta.form, meta.fiscal_quarter)
    return Chunk(
        chunk_id=make_chunk_id(
            ticker=meta.ticker,
            fiscal_year=meta.fiscal_year,
            fiscal_period=fiscal_period,
            doc_type=doc_type,
            index=index,
        ),
        ticker=meta.ticker.upper(),
        company_name=meta.company_name,
        doc_type=doc_type,
        fiscal_year=meta.fiscal_year,
        fiscal_period=fiscal_period,
        is_table=is_table,
        text=text,
        section=section,
        subsection=subsection,
        subsubsection=subsubsection,
        filing_date=meta.filing_date,
        report_date=meta.report_date,
        quarter_period_label=meta.quarter_period_label,
        quarter_months=meta.quarter_months,
    )


def _prepend_scope_titles(
    text: str,
    *,
    subsection: str | None,
    subsubsection: str | None,
) -> str:
    """Prepend sticky subsection titles onto prose so BM25/NLI see them in ``text``.

    Skips titles already present at the start (tables handle their own preface).
    """
    text = _normalize_ws(text)
    if not text:
        return text
    prefix: list[str] = []
    if subsection and not text.startswith(subsection):
        prefix.append(subsection)
    if subsubsection and subsubsection not in text[: max(len(subsubsection) + 20, 200)]:
        # Avoid duplicating when the body already opens with this title.
        if not text.startswith(subsubsection) and (
            not prefix or subsubsection != prefix[0]
        ):
            prefix.append(subsubsection)
    if not prefix:
        return text
    return "\n\n".join(prefix + [text])


def _flush_text(
    buf: list[str],
    *,
    section: str,
    meta: DocumentMeta,
    chunks: list[Chunk],
    subsection: str | None,
    subsubsection: str | None,
) -> None:
    """Flush accumulated prose into one or more sized chunks for ``section``."""
    text = _normalize_ws("\n\n".join(buf))
    buf.clear()
    if len(text) < MIN_TEXT_CHUNK_CHARS:
        return
    while len(text) > MAX_TEXT_CHUNK_CHARS:
        cut = text.rfind("\n\n", 0, MAX_TEXT_CHUNK_CHARS)
        if cut < MAX_TEXT_CHUNK_CHARS // 2:
            cut = MAX_TEXT_CHUNK_CHARS
        piece, text = text[:cut].strip(), text[cut:].strip()
        if len(piece) >= MIN_TEXT_CHUNK_CHARS:
            chunks.append(
                _make_filing_chunk(
                    text=_prepend_scope_titles(
                        piece,
                        subsection=subsection,
                        subsubsection=subsubsection,
                    ),
                    section=section,
                    meta=meta,
                    is_table=False,
                    index=len(chunks),
                    subsection=subsection,
                    subsubsection=subsubsection,
                )
            )
    if len(text) >= MIN_TEXT_CHUNK_CHARS:
        chunks.append(
            _make_filing_chunk(
                text=_prepend_scope_titles(
                    text,
                    subsection=subsection,
                    subsubsection=subsubsection,
                ),
                section=section,
                meta=meta,
                is_table=False,
                index=len(chunks),
                subsection=subsection,
                subsubsection=subsubsection,
            )
        )


def _update_subsection_scope(
    title: str,
    *,
    subsection: str | None,
    subsubsection: str | None,
) -> tuple[str | None, str | None]:
    """Update sticky subsection / subsubsection from a header line."""
    title = _normalize_ws(title)
    if _NOTE_RE.match(title):
        return title, None
    if subsection and _NOTE_RE.match(subsection):
        return subsection, title
    return title, None


def _take_centered_preface(preface_buf: list[Block]) -> list[str]:
    """Consume a trailing run of centered (non-chrome) texts as table titles."""
    taken: list[str] = []
    while preface_buf and preface_buf[-1].align == "center":
        block = preface_buf.pop()
        if _is_page_chrome(block.text):
            continue
        taken.insert(0, _normalize_ws(block.text))
    return [t for t in taken if t]


def _take_short_preface(preface_buf: list[Block]) -> list[str]:
    """Fallback preface for Note/MD&A tables without centered titles.

    Takes contiguous trailing short lines (≤400 chars each, ≤400 total) that look
    like titles or captions. Stops before a long body paragraph.
    """
    taken: list[str] = []
    total = 0
    while preface_buf:
        block = preface_buf[-1]
        text = _normalize_ws(block.text)
        if not text or _is_page_chrome(text):
            preface_buf.pop()
            continue
        if len(text) > TABLE_CAPTION_MAX_CHARS:
            break
        sep = 2 if taken else 0
        if total + sep + len(text) > TABLE_CAPTION_MAX_CHARS:
            break
        # Keep bold titles, caption lines (end with :), and "following table" lines.
        caption_like = (
            block.bold
            or text.endswith(":")
            or text.lower().startswith("the following table")
            or "(in millions" in text.lower()
            or "(dollars in" in text.lower()
        )
        if not caption_like and taken:
            break
        if not caption_like and not taken:
            # Single short non-caption line immediately before table — still useful.
            if len(text) > SUBSECTION_MAX_CHARS:
                break
        preface_buf.pop()
        taken.insert(0, text)
        total += sep + len(text)
    return taken


def _table_preface_lines(
    preface_buf: list[Block],
    *,
    subsection: str | None,
    subsubsection: str | None,
) -> list[str]:
    """Build ordered title/subtitle lines to prepend to a table chunk."""
    centered = _take_centered_preface(preface_buf)
    if centered:
        lines = centered
    else:
        lines = _take_short_preface(preface_buf)

    # Ensure sticky note/MD&A titles appear in table text for BM25 / NLI when the
    # HTML used bold+justify (not center) and those lines were diverted to metadata.
    prefix: list[str] = []
    joined = "\n".join(lines)
    if subsection and subsection not in joined:
        prefix.append(subsection)
    if subsubsection and subsubsection not in joined:
        prefix.append(subsubsection)
    return prefix + lines


def _table_footer_room(chunk: Chunk) -> int:
    """Remaining character budget for footnotes after the markdown table."""
    parts = chunk.text.split("\n\n")
    table_idx = next((i for i, p in enumerate(parts) if p.lstrip().startswith("|")), 0)
    suffix = "\n\n".join(parts[table_idx + 1 :]) if table_idx + 1 < len(parts) else ""
    return max(0, TABLE_FOOTER_MAX_CHARS - len(suffix))


def _attach_table_footer(chunk: Chunk, footer: str) -> bool:
    """Append a footnote to a table chunk when budget remains."""
    footer = _normalize_ws(footer)
    if not footer or len(footer) > _table_footer_room(chunk):
        return False
    chunk.text = f"{chunk.text.rstrip()}\n\n{footer}"
    return True


def chunk_filing(html: str, meta: DocumentMeta) -> list[Chunk]:
    """Split filing HTML into Item sections, sticky subsections, and table chunks."""
    soup = BeautifulSoup(html, "lxml")
    _strip_ixbrl_noise(soup)
    blocks = _iter_blocks(soup)

    chunks: list[Chunk] = []
    section = "Document"
    subsection: str | None = None
    subsubsection: str | None = None
    prose_buf: list[str] = []
    preface_buf: list[Block] = []  # recent text blocks eligible as table preface
    last_was_table = False

    def flush_prose() -> None:
        _flush_text(
            prose_buf,
            section=section,
            meta=meta,
            chunks=chunks,
            subsection=subsection,
            subsubsection=subsubsection,
        )

    def clear_preface_into_prose() -> None:
        """Move leftover preface blocks into normal prose before a scope change."""
        while preface_buf:
            block = preface_buf.pop(0)
            if _is_page_chrome(block.text):
                continue
            prose_buf.append(block.text)

    for block in blocks:
        if block.kind == "text":
            sec = _is_section_header(block.text)
            if sec:
                clear_preface_into_prose()
                flush_prose()
                section = sec
                subsection = None
                subsubsection = None
                last_was_table = False
                continue

            if last_was_table and chunks and chunks[-1].is_table and _is_table_footer(block.text):
                if _attach_table_footer(chunks[-1], block.text):
                    continue

            last_was_table = False

            if _is_subsection_header(block):
                clear_preface_into_prose()
                flush_prose()
                subsection, subsubsection = _update_subsection_scope(
                    block.text,
                    subsection=subsection,
                    subsubsection=subsubsection,
                )
                # Keep a copy in preface so a following table can still lift the
                # title into chunk text when HTML is not center-aligned.
                preface_buf.append(block)
                continue

            # Normal prose: keep a rolling preface window for the next table.
            preface_buf.append(block)
            # Also accumulate into prose; table emission will pull caption lines
            # back out of preface_buf and we must not double-write them.
            # Defer writing preface lines into prose until we know they are not
            # table titles — hold only in preface_buf, flush older ones as prose
            # when preface grows large.
            if sum(len(b.text) for b in preface_buf) > TABLE_CAPTION_MAX_CHARS * 2:
                older = preface_buf.pop(0)
                if not _is_page_chrome(older.text):
                    prose_buf.append(older.text)
                    if sum(len(x) for x in prose_buf) > MAX_TEXT_CHUNK_CHARS:
                        flush_prose()
            continue

        # ---- table ----
        last_was_table = False
        caption_lines = _table_preface_lines(
            preface_buf,
            subsection=subsection,
            subsubsection=subsubsection,
        )
        # Any preface not consumed is ordinary prose.
        clear_preface_into_prose()
        flush_prose()

        parts = [line for line in caption_lines if line]
        parts.append(block.text)
        table_text = "\n\n".join(parts)
        chunks.append(
            _make_filing_chunk(
                text=table_text,
                section=section,
                meta=meta,
                is_table=True,
                index=len(chunks),
                subsection=subsection,
                subsubsection=subsubsection,
            )
        )
        last_was_table = True

    clear_preface_into_prose()
    flush_prose()

    for i, chunk in enumerate(chunks):
        chunk.chunk_id = make_chunk_id(
            ticker=chunk.ticker,
            fiscal_year=chunk.fiscal_year,
            fiscal_period=chunk.fiscal_period,
            doc_type=chunk.doc_type,
            index=i,
        )
    return chunks


def chunk_filing_path(path: Path | str, meta: DocumentMeta) -> list[Chunk]:
    """Load filing HTML from disk and run :func:`chunk_filing`."""
    path = Path(path)
    meta = meta.model_copy(update={"source_path": str(path)})
    html = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_filing(html, meta)
