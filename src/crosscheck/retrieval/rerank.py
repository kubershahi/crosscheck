"""Cross-encoder reranking for dense-retrieved filing candidates."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from crosscheck.config import RERANKER_MODEL
from crosscheck.models import IndexedChunk
from crosscheck.retrieval.embeddings import resolve_device


def _rerank_text(chunk: IndexedChunk) -> str:
    """Build reranker text from chunk metadata and content."""
    section = chunk.section or "Unknown"
    header = (
        f"[TYPE: {chunk.doc_type} | PERIOD: {chunk.fiscal_period}{chunk.fiscal_year} | "
        f"SECTION: {section} | TABLE: {chunk.is_table}]"
    )
    return f"{header}\n{chunk.text.strip()}"


@lru_cache(maxsize=1)
def load_reranker(model_name: str | None = None) -> CrossEncoder:
    """Load and cache the cross-encoder reranker model."""
    name = model_name or RERANKER_MODEL
    device = resolve_device()
    print(f"[rerank] loading {name} on {device}", flush=True)
    return CrossEncoder(name, device=device)


def rerank_claim_passages(
    claim: str,
    candidates: list[tuple[IndexedChunk, float]],
    *,
    top_k: int,
    model: CrossEncoder,
) -> list[tuple[IndexedChunk, float]]:
    """Rerank candidate passages for one claim and return top_k by reranker score."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        return []

    pairs = [(claim, _rerank_text(chunk)) for chunk, _ in candidates]
    scores = model.predict(pairs)
    rescored = [
        (chunk, float(score))
        for (chunk, _dense_score), score in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]
