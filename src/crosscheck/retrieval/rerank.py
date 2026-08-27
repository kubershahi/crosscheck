"""Cross-encoder reranking for dense-retrieved filing candidates.

Default: Pinecone Inference ``bge-reranker-v2-m3`` (API).
Fallback: sentence-transformers ``CrossEncoder`` on MPS/CUDA/CPU via
``CROSSCHECK_EMBEDDING_DEVICE``.

Dense BGE-M3 embeddings stay local; only rerank is offloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from sentence_transformers import CrossEncoder

from crosscheck.config import (
    PINECONE_RERANK_MODEL,
    RERANKER_MODEL,
    get_pinecone_api_key,
    get_rerank_backend,
)
from crosscheck.models import IndexedChunk
from crosscheck.retrieval.embeddings import resolve_device

RerankerModel = Any

_active_rerank_backend: Literal["pinecone", "torch"] = "torch"


@dataclass(frozen=True)
class PineconeReranker:
    """Thin handle for Pinecone Inference rerank (same model family as local)."""

    client: Any
    model_name: str


def _rerank_text(chunk: IndexedChunk) -> str:
    """Build reranker text from chunk metadata and content.

    Lighter than the embedding header, but still grounds period / table /
    call-date when available.
    """
    period = chunk.quarter_period_label or f"{chunk.fiscal_period} {chunk.fiscal_year}"
    parts = [
        f"TYPE: {chunk.doc_type}",
        f"PERIOD: {period}",
    ]
    if chunk.doc_type == "transcript":
        if chunk.speaker_name:
            role = chunk.speaker_role or "Unknown"
            parts.append(f"SPEAKER: {chunk.speaker_name} ({role})")
        if chunk.call_date:
            parts.append(f"CALL_DATE: {chunk.call_date}")
    else:
        parts.append(f"SECTION: {chunk.section or 'Unknown'}")
        parts.append(f"TABLE: {chunk.is_table}")
        # subsection / subsubsection are already in chunk.text.
        if chunk.report_date:
            parts.append(f"REPORT_DATE: {chunk.report_date}")
        if chunk.quarter_months:
            parts.append(f"QUARTER_MONTHS: {','.join(chunk.quarter_months)}")
    header = "[" + " | ".join(parts) + "]"
    return f"{header}\n{chunk.text.strip()}"


def active_rerank_backend() -> Literal["pinecone", "torch"]:
    """Backend in use after ``load_reranker`` (post any Torch fallback)."""
    return _active_rerank_backend


def _load_torch_reranker(model_name: str | None = None) -> CrossEncoder:
    global _active_rerank_backend
    name = model_name or RERANKER_MODEL
    device = resolve_device()
    print(f"[rerank] loading {name} backend=torch on {device}", flush=True)
    model = CrossEncoder(name, device=device)
    _active_rerank_backend = "torch"
    return model


def _load_pinecone_reranker(model_name: str | None = None) -> PineconeReranker:
    global _active_rerank_backend
    from pinecone import Pinecone

    api_key = get_pinecone_api_key()
    name = model_name or PINECONE_RERANK_MODEL
    print(f"[rerank] loading {name} backend=pinecone (Inference API)", flush=True)
    client = Pinecone(api_key=api_key)
    # Fail fast with a tiny request so callers fall back before claim loops.
    _ = client.inference.rerank(
        model=name,
        query="pinecone smoke query",
        documents=["pinecone smoke passage"],
        top_n=1,
        return_documents=False,
        parameters={"truncate": "END"},
    )
    handle = PineconeReranker(client=client, model_name=name)
    _active_rerank_backend = "pinecone"
    return handle


@lru_cache(maxsize=1)
def load_reranker(model_name: str | None = None) -> RerankerModel:
    """Load and cache the reranker (Pinecone by default, Torch fallback).

    ``CROSSCHECK_RERANK_BACKEND=local`` / ``torch`` forces the local CrossEncoder.
    """
    backend = get_rerank_backend()
    if backend == "torch":
        return _load_torch_reranker(model_name)

    try:
        return _load_pinecone_reranker(model_name)
    except Exception as exc:
        print(
            f"[rerank] Pinecone failed ({type(exc).__name__}: {exc}); "
            "falling back to local torch …",
            flush=True,
        )
        return _load_torch_reranker(None)


def _rerank_pinecone(
    claim: str,
    candidates: list[tuple[IndexedChunk, float]],
    *,
    top_k: int,
    model: PineconeReranker,
) -> list[tuple[IndexedChunk, float]]:
    documents = [_rerank_text(chunk) for chunk, _ in candidates]
    result = model.client.inference.rerank(
        model=model.model_name,
        query=claim,
        documents=documents,
        top_n=top_k,
        return_documents=False,
        parameters={"truncate": "END"},
    )
    ranked: list[tuple[IndexedChunk, float]] = []
    for item in result.data:
        idx = int(item.index)
        if idx < 0 or idx >= len(candidates):
            continue
        ranked.append((candidates[idx][0], float(item.score)))
    return ranked


def _rerank_torch(
    claim: str,
    candidates: list[tuple[IndexedChunk, float]],
    *,
    top_k: int,
    model: CrossEncoder,
) -> list[tuple[IndexedChunk, float]]:
    pairs = [(claim, _rerank_text(chunk)) for chunk, _ in candidates]
    scores = model.predict(pairs)
    rescored = [
        (chunk, float(score))
        for (chunk, _dense_score), score in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]


def rerank_claim_passages(
    claim: str,
    candidates: list[tuple[IndexedChunk, float]],
    *,
    top_k: int,
    model: RerankerModel,
) -> list[tuple[IndexedChunk, float]]:
    """Rerank candidate passages for one claim and return top_k by reranker score."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        return []

    if isinstance(model, PineconeReranker):
        return _rerank_pinecone(claim, candidates, top_k=top_k, model=model)
    return _rerank_torch(claim, candidates, top_k=top_k, model=model)
