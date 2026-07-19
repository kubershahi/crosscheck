"""Section-aware + table-atomic chunking for SEC filing HTML.

Walks EDGAR HTML, splits prose on Item / MD&A-style headers, and keeps each
``<table>`` as a single atomic TSV chunk so row/column meaning is preserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from crosscheck.models import Chunk, DocumentMeta

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Common 10-K / 10-Q item headers (case-insensitive).
SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^item\s+1a\.?\s*[-–—.]?\s*risk\s+factors", re.I),
    re.compile(r"^item\s+1b\.?\s*[-–—.]?\s*unresolved\s+staff", re.I),
    re.compile(r"^item\s+1c\.?\s*[-–—.]?\s*cybersecurity", re.I),
    re.compile(r"^item\s+1\.?\s*[-–—.]?\s*business\b", re.I),
    re.compile(r"^item\s+2\.?\s*[-–—.]?\s*properties", re.I),
    re.compile(r"^item\s+3\.?\s*[-–—.]?\s*legal\s+proceedings", re.I),
    re.compile(r"^item\s+5\.?\s*[-–—.]?\s*market\s+for", re.I),
    re.compile(
        r"^item\s+7a?\.?\s*[-–—.]?\s*(management.?s?\s+discussion|quantitative)",
        re.I,
    ),
    re.compile(r"^item\s+8\.?\s*[-–—.]?\s*financial\s+statements", re.I),
    re.compile(r"^item\s+9a?\.?\s*[-–—.]?\s*", re.I),
    re.compile(r"^part\s+[ivx]+\b", re.I),
    re.compile(r"^management.?s?\s+discussion\s+and\s+analysis", re.I),
    re.compile(r"^liquidity\s+and\s+capital\s+resources", re.I),
    re.compile(r"^critical\s+accounting", re.I),
    re.compile(r"^consolidated\s+(statements?|balance\s+sheet)", re.I),
]

MAX_TEXT_CHUNK_CHARS = 3500
MIN_TEXT_CHUNK_CHARS = 80


def _normalize_ws(text: str) -> str:
    """Collapse whitespace / NBSP from EDGAR extracted text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_section_header(text: str) -> str | None:
    """Return a cleaned section title if ``text`` looks like an Item/MD&A header."""
    cleaned = _normalize_ws(text)
    if not cleaned or len(cleaned) > 180:
        return None
    # Strip leading junk common in iXBRL
    cleaned = re.sub(r"^[\W\d_]{0,6}", "", cleaned)
    for pat in SECTION_PATTERNS:
        if pat.search(cleaned):
            return cleaned[:160]
    return None


def _table_to_tsv(table: Tag) -> str:
    """Convert an HTML ``<table>`` to tab-separated rows (one string)."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            cells.append(_normalize_ws(cell.get_text(" ", strip=True)))
        if any(cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _strip_ixbrl_noise(soup: BeautifulSoup) -> None:
    """Remove scripts/styles and ``display:none`` XBRL chrome in-place."""
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    # Remove common XBRL hidden blocks
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()


_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr", "br"}


def _iter_blocks(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Return ordered (kind, text) blocks where kind is 'text' or 'table'."""
    body = soup.body or soup
    blocks: list[tuple[str, str]] = []

    def walk(node: Tag) -> None:
        name = (node.name or "").lower()
        if name == "table":
            tsv = _table_to_tsv(node)
            if tsv.strip():
                blocks.append(("table", tsv))
            return

        # Descend into containers that hold nested structure / tables.
        child_tags = [c for c in node.children if isinstance(c, Tag)]
        has_table_descendant = any(
            (c.name or "").lower() == "table" or c.find("table") for c in child_tags
        )
        has_block_child = any((c.name or "").lower() in _BLOCK_TAGS | {"table"} for c in child_tags)

        if has_table_descendant or (has_block_child and name in {"body", "html", "div", "section", "td", "th", "span"}):
            for child in child_tags:
                walk(child)
            return

        if name in _BLOCK_TAGS or name in {"font"}:
            text = _normalize_ws(node.get_text(" ", strip=True))
            if text:
                blocks.append(("text", text))
            return

        for child in child_tags:
            walk(child)

    if isinstance(body, Tag):
        walk(body)

    # Deduplicate consecutive identical short lines (common in EDGAR HTML)
    deduped: list[tuple[str, str]] = []
    for kind, text in blocks:
        if deduped and deduped[-1] == (kind, text) and len(text) < 200:
            continue
        deduped.append((kind, text))
    return deduped


def _flush_text(
    buf: list[str],
    *,
    section: str,
    meta: DocumentMeta,
    chunks: list[Chunk],
) -> None:
    """Flush accumulated prose into one or more sized chunks for ``section``."""
    text = _normalize_ws("\n\n".join(buf))
    buf.clear()
    if len(text) < MIN_TEXT_CHUNK_CHARS:
        return
    # Split oversized prose while keeping section metadata.
    while len(text) > MAX_TEXT_CHUNK_CHARS:
        cut = text.rfind("\n\n", 0, MAX_TEXT_CHUNK_CHARS)
        if cut < MAX_TEXT_CHUNK_CHARS // 2:
            cut = MAX_TEXT_CHUNK_CHARS
        piece, text = text[:cut].strip(), text[cut:].strip()
        if len(piece) >= MIN_TEXT_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    text=piece,
                    doc_type="filing",
                    ticker=meta.ticker,
                    company_name=meta.company_name,
                    fiscal_year=meta.fiscal_year,
                    fiscal_quarter=meta.fiscal_quarter,
                    chunk_index=len(chunks),
                    section_header=section,
                    is_table=False,
                    source_path=meta.source_path,
                )
            )
    if len(text) >= MIN_TEXT_CHUNK_CHARS:
        chunks.append(
            Chunk(
                text=text,
                doc_type="filing",
                ticker=meta.ticker,
                company_name=meta.company_name,
                fiscal_year=meta.fiscal_year,
                fiscal_quarter=meta.fiscal_quarter,
                chunk_index=len(chunks),
                section_header=section,
                is_table=False,
                source_path=meta.source_path,
            )
        )


def chunk_filing(html: str, meta: DocumentMeta) -> list[Chunk]:
    """Split filing HTML into section-aware prose chunks and atomic table chunks.

    Section headers become ``section_header`` metadata; each ``<table>`` is one
    chunk with ``is_table=True``.
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_ixbrl_noise(soup)
    blocks = _iter_blocks(soup)

    chunks: list[Chunk] = []
    section = "Document"
    prose_buf: list[str] = []

    for kind, text in blocks:
        if kind == "text":
            header = _is_section_header(text)
            if header:
                _flush_text(prose_buf, section=section, meta=meta, chunks=chunks)
                section = header
                continue
            prose_buf.append(text)
            # Opportunistically flush very large buffers
            if sum(len(x) for x in prose_buf) > MAX_TEXT_CHUNK_CHARS:
                _flush_text(prose_buf, section=section, meta=meta, chunks=chunks)
        else:
            _flush_text(prose_buf, section=section, meta=meta, chunks=chunks)
            chunks.append(
                Chunk(
                    text=text,
                    doc_type="filing",
                    ticker=meta.ticker,
                    company_name=meta.company_name,
                    fiscal_year=meta.fiscal_year,
                    fiscal_quarter=meta.fiscal_quarter,
                    chunk_index=len(chunks),
                    section_header=section,
                    is_table=True,
                    source_path=meta.source_path,
                )
            )

    _flush_text(prose_buf, section=section, meta=meta, chunks=chunks)

    # Re-index for stability
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def chunk_filing_path(path: Path | str, meta: DocumentMeta) -> list[Chunk]:
    """Load filing HTML from disk and run :func:`chunk_filing`."""
    path = Path(path)
    meta = meta.model_copy(update={"source_path": str(path)})
    html = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_filing(html, meta)
