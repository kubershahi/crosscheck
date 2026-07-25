"""BM25 keyword retrieval and reciprocal rank fusion (RRF) for hybrid search."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from crosscheck.models import IndexedChunk, as_fiscal_quarter

# Keep digits / $ / . so amounts like 124.3 and $2.40 stay searchable.
_TOKEN = re.compile(r"[a-z0-9$%]+(?:\.[0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens; preserve decimals and currency-ish fragments.

    Thousands separators are stripped so ``$124,300`` indexes as ``124300``.
    Leading ``$`` / trailing ``%`` are dropped so query digits align with filings.
    """
    cleaned = (text or "").replace(",", "")
    tokens: list[str] = []
    for match in _TOKEN.finditer(cleaned):
        tok = match.group(0).lower().strip("$%")
        if tok:
            tokens.append(tok)
    return tokens


def period_matches(chunk_period: str, fiscal_quarter: str | None) -> bool:
    """True when chunk ``fiscal_period`` matches the query quarter.

    10-Q chunks use ``Q1``…``Q4``. Annual 10-K chunks use ``FY`` and match
    only when the requested quarter is ``Q4``.
    """
    if fiscal_quarter is None:
        return True
    wanted = as_fiscal_quarter(fiscal_quarter)
    period = (chunk_period or "").strip().upper()
    if period == wanted:
        return True
    return period == "FY" and wanted == "Q4"


@dataclass
class Bm25Index:
    """In-memory BM25 over a corpus of :class:`IndexedChunk` rows."""

    bm25: BM25Okapi
    chunks: list[IndexedChunk]
    tokenized: list[list[str]]


def build_bm25_index(chunks: list[IndexedChunk]) -> Bm25Index:
    """Build BM25Okapi from chunk texts (same list as FAISS ``all_chunks``)."""
    tokenized = [tokenize(c.text) for c in chunks]
    # BM25Okapi needs at least one doc; empty corpus → empty scores at query time.
    if not tokenized:
        tokenized = [[]]
        chunks = []
    return Bm25Index(bm25=BM25Okapi(tokenized), chunks=chunks, tokenized=tokenized)


def bm25_retrieve(
    query: str,
    index: Bm25Index,
    *,
    k: int = 20,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    doc_types: set[str] | None = None,
) -> list[tuple[IndexedChunk, float]]:
    """Return top-k BM25 hits after metadata filters (ticker / year / quarter)."""
    if k < 1 or not index.chunks:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = index.bm25.get_scores(tokens)
    ticker_u = ticker.upper() if ticker else None
    # Rank by score descending (Okapi scores may be negative when some
    # query terms are absent; relative order still ranks keyword hits).
    ranked = sorted(range(len(index.chunks)), key=lambda i: scores[i], reverse=True)

    out: list[tuple[IndexedChunk, float]] = []
    for i in ranked:
        chunk = index.chunks[i]
        if ticker_u and chunk.ticker != ticker_u:
            continue
        if doc_types and chunk.doc_type not in doc_types:
            continue
        if fiscal_year is not None and chunk.fiscal_year != fiscal_year:
            continue
        if not period_matches(chunk.fiscal_period, fiscal_quarter):
            continue
        out.append((chunk, float(scores[i])))
        if len(out) >= k:
            break
    return out


def rrf_fuse(
    rankings: list[list[tuple[IndexedChunk, float]]],
    *,
    k: int = 60,
    top_n: int,
) -> list[tuple[IndexedChunk, float]]:
    """Fuse ranked lists with reciprocal rank fusion: ``Σ 1/(k + rank)``.

    ``rank`` is 1-based within each list. Chunks are keyed by ``global_id``.
    """
    if top_n < 1:
        return []

    fused: dict[int, float] = {}
    by_id: dict[int, IndexedChunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            gid = chunk.global_id
            by_id[gid] = chunk
            fused[gid] = fused.get(gid, 0.0) + 1.0 / (k + rank)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(by_id[gid], score) for gid, score in ordered[:top_n]]
