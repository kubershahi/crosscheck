"""Qdrant Cloud (or local path) store for hybrid retrieval.

Each point holds:
  - named vector ``dense`` (BGE-M3, Dot / IP on L2-normalized embeds)
  - named sparse vector ``sparse`` (Qdrant BM25 over chunk text)
  - full ``IndexedChunk`` payload (all metadata + text)

Filings and transcripts use separate collections (separate ``global_id`` spaces).

Query path: dual prefetch (dense + BM25) → RRF fusion → payload filters.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from crosscheck.config import (
    get_qdrant_api_key,
    get_qdrant_endpoint,
    get_qdrant_filings_collection,
    get_qdrant_path,
    get_qdrant_transcripts_collection,
)
from crosscheck.models import IndexedChunk, as_fiscal_quarter

EMBEDDING_DIMENSION = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_MODEL = "Qdrant/bm25"

CorpusKind = Literal["filings", "transcripts"]


def get_qdrant_client() -> QdrantClient:
    """Cloud client when ``QDRANT_ENDPOINT`` is set; else local path store."""
    endpoint = get_qdrant_endpoint()
    if endpoint:
        return QdrantClient(
            url=endpoint,
            api_key=get_qdrant_api_key(),
            timeout=120,
        )
    path = get_qdrant_path()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def collection_name(corpus: CorpusKind) -> str:
    """Return configured Qdrant collection name for ``corpus``."""
    if corpus == "filings":
        return get_qdrant_filings_collection()
    return get_qdrant_transcripts_collection()


def filings_collection_name() -> str:
    """Return configured filings collection name."""
    return collection_name("filings")


def transcripts_collection_name() -> str:
    """Return configured transcripts collection name."""
    return collection_name("transcripts")


def collection_ready(
    corpus: CorpusKind,
    client: QdrantClient | None = None,
    *,
    min_points: int = 1,
) -> bool:
    """True when the corpus collection exists and has at least ``min_points``."""
    client = client or get_qdrant_client()
    name = collection_name(corpus)
    try:
        if not client.collection_exists(name):
            return False
        info = client.get_collection(name)
        return int(info.points_count or 0) >= min_points
    except (UnexpectedResponse, ValueError, OSError):
        return False


def filings_collection_ready(
    client: QdrantClient | None = None,
    *,
    min_points: int = 1,
) -> bool:
    """True when the filings collection exists and has at least ``min_points``."""
    return collection_ready("filings", client, min_points=min_points)


def transcripts_collection_ready(
    client: QdrantClient | None = None,
    *,
    min_points: int = 1,
) -> bool:
    """True when the transcripts collection exists and has at least ``min_points``."""
    return collection_ready("transcripts", client, min_points=min_points)


def ensure_collection(
    corpus: CorpusKind,
    client: QdrantClient | None = None,
    *,
    force: bool = False,
    dim: int = EMBEDDING_DIMENSION,
) -> str:
    """Create (or recreate) a hybrid collection for ``corpus``; return its name."""
    client = client or get_qdrant_client()
    name = collection_name(corpus)
    if force and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dim,
                    distance=models.Distance.DOT,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
    for field_name, schema in (
        ("ticker", models.PayloadSchemaType.KEYWORD),
        ("fiscal_year", models.PayloadSchemaType.INTEGER),
        ("fiscal_period", models.PayloadSchemaType.KEYWORD),
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("chunk_id", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema,
            )
        except UnexpectedResponse:
            pass
    return name


def ensure_filings_collection(
    client: QdrantClient | None = None,
    *,
    force: bool = False,
    dim: int = EMBEDDING_DIMENSION,
) -> str:
    """Create (or recreate) the filings hybrid collection; return its name."""
    return ensure_collection("filings", client, force=force, dim=dim)


def ensure_transcripts_collection(
    client: QdrantClient | None = None,
    *,
    force: bool = False,
    dim: int = EMBEDDING_DIMENSION,
) -> str:
    """Create (or recreate) the transcripts hybrid collection; return its name."""
    return ensure_collection("transcripts", client, force=force, dim=dim)


def chunk_to_payload(chunk: IndexedChunk) -> dict[str, Any]:
    """Full IndexedChunk as Qdrant payload (all metadata + text)."""
    return chunk.model_dump(mode="json")


def payload_to_chunk(payload: dict[str, Any] | None) -> IndexedChunk:
    """Rebuild ``IndexedChunk`` from a Qdrant point payload."""
    if not payload:
        raise ValueError("Empty Qdrant payload; expected full IndexedChunk fields")
    return IndexedChunk.model_validate(payload)


def period_filter(
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    doc_types: set[str] | None = None,
) -> models.Filter | None:
    """Build Qdrant filter matching NLI period semantics (incl. FY ↔ Q4)."""
    must: list[models.Condition] = []
    if ticker:
        must.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper()),
            )
        )
    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=int(fiscal_year)),
            )
        )
    if fiscal_quarter is not None:
        wanted = as_fiscal_quarter(fiscal_quarter)
        if wanted == "Q4":
            must.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="fiscal_period",
                            match=models.MatchValue(value="Q4"),
                        ),
                        models.FieldCondition(
                            key="fiscal_period",
                            match=models.MatchValue(value="FY"),
                        ),
                    ]
                )
            )
        else:
            must.append(
                models.FieldCondition(
                    key="fiscal_period",
                    match=models.MatchValue(value=wanted),
                )
            )
    if doc_types:
        must.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=sorted(doc_types)),
            )
        )
    if not must:
        return None
    return models.Filter(must=must)


def retrieval_filter_from_plan(
    *,
    temporal_scope: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    doc_types: set[str] | None = None,
) -> models.Filter | None:
    """Build Qdrant filter from a query temporal scope (+ optional overrides).

    When ``doc_types`` is provided by the caller it is used only for
    ``STANDARD_QUARTER`` (legacy override). Temporal scopes always set
    doc_type / fiscal_period explicitly.
    """
    from crosscheck.retrieval.query_processor import TemporalScope

    must: list[models.Condition] = []
    if ticker:
        must.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper()),
            )
        )
    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=int(fiscal_year)),
            )
        )

    scope = (
        temporal_scope
        if isinstance(temporal_scope, TemporalScope)
        else TemporalScope(str(temporal_scope))
    )

    if scope == TemporalScope.FULL_YEAR_ONLY:
        must.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value="10-K"),
            )
        )
        must.append(
            models.FieldCondition(
                key="fiscal_period",
                match=models.MatchValue(value="FY"),
            )
        )
    elif scope == TemporalScope.Q4_COMPOSITE:
        # Composite Q4 uses dual-path retrieval (see retrieve_claim_passages).
        pass
    else:
        # STANDARD_QUARTER: 10-Q for the claim period (caller fiscal_quarter).
        if doc_types:
            must.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchAny(any=sorted(doc_types)),
                )
            )
        else:
            must.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value="10-Q"),
                )
            )
        if fiscal_quarter is not None:
            wanted = as_fiscal_quarter(fiscal_quarter)
            must.append(
                models.FieldCondition(
                    key="fiscal_period",
                    match=models.MatchValue(value=wanted),
                )
            )

    if not must:
        return None
    return models.Filter(must=must)


def composite_10k_fy_filter(
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
) -> models.Filter | None:
    """Q4 composite Path A: FY 10-K only."""
    must: list[models.Condition] = []
    if ticker:
        must.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper()),
            )
        )
    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=int(fiscal_year)),
            )
        )
    must.append(
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="10-K"),
        )
    )
    must.append(
        models.FieldCondition(
            key="fiscal_period",
            match=models.MatchValue(value="FY"),
        )
    )
    return models.Filter(must=must)


def composite_q3_10q_filter(
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
) -> models.Filter | None:
    """Q4 composite Path B: Q3 10-Q (9-month YTD) only."""
    must: list[models.Condition] = []
    if ticker:
        must.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper()),
            )
        )
    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=int(fiscal_year)),
            )
        )
    must.append(
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="10-Q"),
        )
    )
    must.append(
        models.FieldCondition(
            key="fiscal_period",
            match=models.MatchValue(value="Q3"),
        )
    )
    return models.Filter(must=must)


def upsert_corpus(
    corpus: CorpusKind,
    chunks: list[IndexedChunk],
    dense_vectors: np.ndarray,
    *,
    client: QdrantClient | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> int:
    """Upsert dense + BM25-sparse points with full chunk payloads.

    ``dense_vectors`` shape must be ``(len(chunks), EMBEDDING_DIMENSION)``.
    """
    if len(chunks) != dense_vectors.shape[0]:
        raise ValueError(
            f"chunks={len(chunks)} vs dense_vectors rows={dense_vectors.shape[0]}"
        )
    if dense_vectors.ndim != 2 or dense_vectors.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected dense shape (N, {EMBEDDING_DIMENSION}), got {dense_vectors.shape}"
        )

    client = client or get_qdrant_client()
    name = ensure_collection(corpus, client, force=force)
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points: list[models.PointStruct] = []
        for i in range(start, end):
            chunk = chunks[i]
            vec = dense_vectors[i].astype(np.float32, copy=False)
            points.append(
                models.PointStruct(
                    id=int(chunk.global_id),
                    vector={
                        DENSE_VECTOR_NAME: vec.tolist(),
                        SPARSE_VECTOR_NAME: models.Document(
                            text=chunk.text,
                            model=BM25_MODEL,
                        ),
                    },
                    payload=chunk_to_payload(chunk),
                )
            )
        client.upsert(collection_name=name, points=points, wait=True)
        print(
            f"  [qdrant] upserted {end}/{total} {corpus} points → {name}",
            flush=True,
        )
    return total


def upsert_filings(
    chunks: list[IndexedChunk],
    dense_vectors: np.ndarray,
    *,
    client: QdrantClient | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> int:
    """Upsert filings points (dense + BM25 + full payload)."""
    return upsert_corpus(
        "filings",
        chunks,
        dense_vectors,
        client=client,
        batch_size=batch_size,
        force=force,
    )


def upsert_corpus_from_memmap(
    corpus: CorpusKind,
    chunks_path,
    embeddings_path,
    *,
    n_chunks: int,
    client: QdrantClient | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> int:
    """Stream ``all_chunks.jsonl`` + ``embeddings.npy`` into Qdrant."""
    from crosscheck.chunking.store import load_indexed_chunks_jsonl

    chunks = load_indexed_chunks_jsonl(chunks_path)
    if len(chunks) != n_chunks:
        raise RuntimeError(
            f"all_chunks lines={len(chunks)} != n_chunks={n_chunks}"
        )
    vectors = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode="r",
        shape=(n_chunks, EMBEDDING_DIMENSION),
    )
    try:
        return upsert_corpus(
            corpus,
            chunks,
            np.asarray(vectors),
            client=client,
            batch_size=batch_size,
            force=force,
        )
    finally:
        del vectors


def upsert_filings_from_memmap(
    chunks_path,
    embeddings_path,
    *,
    n_chunks: int,
    client: QdrantClient | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> int:
    """Stream filings ``all_chunks.jsonl`` + ``embeddings.npy`` into Qdrant."""
    return upsert_corpus_from_memmap(
        "filings",
        chunks_path,
        embeddings_path,
        n_chunks=n_chunks,
        client=client,
        batch_size=batch_size,
        force=force,
    )


def upsert_transcripts_from_memmap(
    chunks_path,
    embeddings_path,
    *,
    n_chunks: int,
    client: QdrantClient | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> int:
    """Stream transcripts ``all_chunks.jsonl`` + ``embeddings.npy`` into Qdrant."""
    return upsert_corpus_from_memmap(
        "transcripts",
        chunks_path,
        embeddings_path,
        n_chunks=n_chunks,
        client=client,
        batch_size=batch_size,
        force=force,
    )


def hybrid_search(
    query_text: str,
    query_dense: np.ndarray,
    *,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    doc_types: set[str] | None = None,
    prefetch_k: int | None = None,
    client: QdrantClient | None = None,
    corpus: CorpusKind = "filings",
    query_filter: models.Filter | None = None,
) -> list[tuple[IndexedChunk, float]]:
    """Dense + BM25 prefetch with RRF fusion and period payload filters.

    Pass ``query_filter`` to override the default ``period_filter`` (used by
    temporal query preprocessing).
    """
    if k < 1:
        return []
    text = (query_text or "").strip()
    if not text:
        return []

    client = client or get_qdrant_client()
    name = collection_name(corpus)
    if not client.collection_exists(name):
        raise FileNotFoundError(
            f"Qdrant collection {name!r} missing. "
            f"Run: python scripts/build_indices.py --corpus {corpus} --force"
        )

    channel_k = prefetch_k if prefetch_k is not None else max(k, 20)
    vec = np.asarray(query_dense, dtype=np.float32).reshape(-1)
    if vec.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"query dense dim {vec.shape[0]} != {EMBEDDING_DIMENSION}"
        )

    q_filter = (
        query_filter
        if query_filter is not None
        else period_filter(
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            doc_types=doc_types,
        )
    )

    response = client.query_points(
        collection_name=name,
        prefetch=[
            models.Prefetch(
                query=vec.tolist(),
                using=DENSE_VECTOR_NAME,
                filter=q_filter,
                limit=channel_k,
            ),
            models.Prefetch(
                query=models.Document(text=text, model=BM25_MODEL),
                using=SPARSE_VECTOR_NAME,
                filter=q_filter,
                limit=channel_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=q_filter,
        limit=k,
        with_payload=True,
    )

    results: list[tuple[IndexedChunk, float]] = []
    for point in response.points:
        chunk = payload_to_chunk(point.payload)
        results.append((chunk, float(point.score or 0.0)))
    return results
