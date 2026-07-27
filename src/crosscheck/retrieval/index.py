"""Disk-streamed corpus assembly, embeddings, and Qdrant hybrid indexing.

Filings (10-K / 10-Q) and transcripts are indexed in separate Qdrant collections
so NLI can retrieve only SEC filing passages when checking transcript claims.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from crosscheck.chunking.store import (
    iter_company_chunk_files,
    load_indexed_chunks_jsonl,
)
from crosscheck.config import EMBEDDING_MODEL, INDICES_DIR
from crosscheck.models import Chunk, IndexedChunk
from crosscheck.retrieval.embeddings import (
    chunk_embedding_text,
    embed_texts,
    load_embedding_model,
    resolve_device,
)
from crosscheck.retrieval.hybrid import (
    Bm25Index,
    bm25_retrieve,
    build_bm25_index,
    period_matches,
    rrf_fuse,
)

EMBEDDING_DIMENSION = 1024

CorpusKind = Literal["filings", "transcripts"]

FILING_DOC_TYPES = frozenset({"10-K", "10-Q"})
TRANSCRIPT_DOC_TYPES = frozenset({"transcript"})

CORPUS_DOC_TYPES: dict[CorpusKind, frozenset[str]] = {
    "filings": FILING_DOC_TYPES,
    "transcripts": TRANSCRIPT_DOC_TYPES,
}


@dataclass
class CorpusIndex:
    """Loaded corpus for retrieval.

    Filings (Qdrant hybrid): ``index`` is None; dense+BM25+RRF run in Qdrant.
    Transcripts (FAISS): ``index`` is IndexFlatIP; optional local BM25 unused by NLI.
    """

    corpus: CorpusKind
    chunks: list[IndexedChunk]
    embedding_model: str
    index: faiss.IndexFlatIP | None = None
    bm25: Bm25Index | None = None
    backend: Literal["qdrant", "faiss"] = "faiss"


# Backward-compatible alias used by older call sites.
MasterIndex = CorpusIndex


def corpus_indices_dir(corpus: CorpusKind) -> Path:
    """Return ``data/indices/{filings|transcripts}/``."""
    return INDICES_DIR / corpus


def corpus_chunks_path(corpus: CorpusKind) -> Path:
    """Return corpus master JSONL path."""
    return corpus_indices_dir(corpus) / "all_chunks.jsonl"


def corpus_embeddings_path(corpus: CorpusKind) -> Path:
    """Return corpus embedding memmap path."""
    return corpus_indices_dir(corpus) / "embeddings.npy"


def corpus_index_path(corpus: CorpusKind) -> Path:
    """Return corpus FAISS path."""
    return corpus_indices_dir(corpus) / "index.faiss"


def corpus_manifest_path(corpus: CorpusKind) -> Path:
    """Return corpus manifest JSON path."""
    return corpus_indices_dir(corpus) / "manifest.json"


def filings_index_path() -> Path:
    """Return ``data/indices/filings/index.faiss``."""
    return corpus_index_path("filings")


def corpus_index_exists(corpus: CorpusKind) -> bool:
    """True when merge manifest + Qdrant collection exist for ``corpus``."""
    from crosscheck.retrieval.qdrant_store import collection_ready

    return corpus_manifest_path(corpus).exists() and collection_ready(corpus)


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty JSONL rows without loading the file."""
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def merge_corpus_chunks(corpus: CorpusKind, *, force: bool = False) -> Path:
    """Merge matching company JSONL files into a corpus ``all_chunks.jsonl``."""
    out = corpus_chunks_path(corpus)
    if out.exists() and not force:
        return out

    allowed = CORPUS_DOC_TYPES[corpus]
    company_files = iter_company_chunk_files()
    if not company_files:
        raise FileNotFoundError(
            f"No per-company chunk JSONL under {INDICES_DIR.parent / 'chunks'}. "
            "Run: python scripts/build_chunks.py"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    global_id = 0
    with out.open("w", encoding="utf-8") as fh:
        for path in company_files:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    chunk = Chunk.model_validate_json(line)
                    if chunk.doc_type not in allowed:
                        continue
                    row = IndexedChunk(
                        **chunk.model_dump(exclude_none=True),
                        global_id=global_id,
                    )
                    fh.write(row.model_dump_json(exclude_none=True))
                    fh.write("\n")
                    global_id += 1

    if global_id == 0:
        raise FileNotFoundError(
            f"No {corpus} chunks found under data/chunks. "
            "Run: python scripts/build_chunks.py"
        )

    print(
        f"  [{corpus}] merged {global_id} chunks from {len(company_files)} files → {out}",
        flush=True,
    )
    return out


def _write_embedding_batch(
    *,
    model: SentenceTransformer,
    texts: list[str],
    target: np.memmap,
    start_idx: int,
    device: str,
) -> int:
    """Encode one mini-batch and write it directly to the disk memmap."""
    if not texts:
        return start_idx

    end_idx = start_idx + len(texts)
    vecs: np.ndarray | None = None
    try:
        with torch.no_grad():
            vecs = model.encode(
                texts,
                batch_size=len(texts),
                device=device,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        vecs = np.asarray(vecs, dtype=np.float32)
        expected_shape = (len(texts), EMBEDDING_DIMENSION)
        if vecs.shape != expected_shape:
            raise RuntimeError(
                f"Expected embedding shape {expected_shape}, got {vecs.shape}"
            )
        target[start_idx:end_idx] = vecs
        target.flush()
        return end_idx
    finally:
        del vecs
        texts.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def stream_embeddings_to_memmap(
    *,
    model: SentenceTransformer,
    chunks_path: Path,
    output_path: Path,
    batch_size: int = 16,
    max_chunks: int | None = None,
    progress_desc: str = "Embedding",
) -> tuple[Path, int]:
    """Stream corpus JSONL into a disk-backed float32 embedding matrix."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    total_rows = count_jsonl_rows(chunks_path)
    total_chunks = min(total_rows, max_chunks) if max_chunks else total_rows
    if total_chunks == 0:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_chunks, EMBEDDING_DIMENSION),
    )
    device = resolve_device()
    batch_texts: list[str] = []
    written = 0

    print(
        f"  streaming {total_chunks} chunks → {output_path} "
        f"(batch_size={batch_size}, device={device})",
        flush=True,
    )

    try:
        with chunks_path.open("r", encoding="utf-8") as fh, tqdm(
            total=total_chunks,
            desc=progress_desc,
            unit="chunk",
            dynamic_ncols=True,
        ) as pbar:
            for line_number, line in enumerate(fh):
                if written + len(batch_texts) >= total_chunks:
                    break
                if not line.strip():
                    continue

                chunk = IndexedChunk.model_validate_json(line)
                expected_global_id = written + len(batch_texts)
                if chunk.global_id != expected_global_id:
                    raise RuntimeError(
                        f"global_id mismatch at streamed row {expected_global_id}: "
                        f"got {chunk.global_id} (file line {line_number})"
                    )
                batch_texts.append(chunk_embedding_text(chunk))

                if len(batch_texts) == batch_size:
                    n = len(batch_texts)
                    written = _write_embedding_batch(
                        model=model,
                        texts=batch_texts,
                        target=fp,
                        start_idx=written,
                        device=device,
                    )
                    pbar.update(n)

            if batch_texts:
                n = len(batch_texts)
                written = _write_embedding_batch(
                    model=model,
                    texts=batch_texts,
                    target=fp,
                    start_idx=written,
                    device=device,
                )
                pbar.update(n)

        if written != total_chunks:
            raise RuntimeError(
                f"Expected to write {total_chunks} embeddings, wrote {written}"
            )
        fp.flush()
    finally:
        del fp
        batch_texts.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    return output_path, total_chunks


def build_faiss_from_memmap(
    *,
    source_path: Path,
    output_path: Path,
    total_chunks: int,
) -> Path:
    """Load embeddings read-only and add the memmap directly to IndexFlatIP."""
    vectors = np.memmap(
        source_path,
        dtype=np.float32,
        mode="r",
        shape=(total_chunks, EMBEDDING_DIMENSION),
    )
    try:
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        index.add(vectors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
    finally:
        del vectors
        gc.collect()
    return output_path


def build_corpus_index(
    corpus: CorpusKind,
    *,
    model: SentenceTransformer | None = None,
    force: bool = False,
    batch_size: int = 16,
) -> Path:
    """Merge → contextual embed → Qdrant hybrid (filings or transcripts)."""
    if corpus_index_exists(corpus) and not force:
        return corpus_manifest_path(corpus)

    print(f"[build_indices] assembling {corpus} index …", flush=True)
    chunks_path = merge_corpus_chunks(corpus, force=True)
    model = model or load_embedding_model()
    disk_embeddings_path, total_chunks = stream_embeddings_to_memmap(
        model=model,
        chunks_path=chunks_path,
        output_path=corpus_embeddings_path(corpus),
        batch_size=batch_size,
        progress_desc=f"Embed {corpus}",
    )

    from crosscheck.config import (
        get_qdrant_endpoint,
        get_qdrant_filings_collection,
        get_qdrant_transcripts_collection,
    )
    from crosscheck.retrieval.qdrant_store import upsert_corpus_from_memmap

    collection = (
        get_qdrant_filings_collection()
        if corpus == "filings"
        else get_qdrant_transcripts_collection()
    )
    endpoint = get_qdrant_endpoint()
    target = endpoint or "local QDRANT_PATH"
    print(
        f"  [{corpus}] upserting {total_chunks} points to Qdrant "
        f"({target}, collection={collection!r}) …",
        flush=True,
    )
    upsert_corpus_from_memmap(
        corpus,
        chunks_path,
        disk_embeddings_path,
        n_chunks=total_chunks,
        batch_size=max(batch_size, 32),
        force=True,
    )
    path = corpus_manifest_path(corpus)
    manifest = {
        "corpus": corpus,
        "backend": "qdrant",
        "doc_types": sorted(CORPUS_DOC_TYPES[corpus]),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_dtype": "float32",
        "embedding_storage": "numpy.memmap",
        "n_chunks": total_chunks,
        "all_chunks_path": str(chunks_path),
        "embeddings_path": str(disk_embeddings_path),
        "qdrant_endpoint": endpoint,
        "qdrant_collection": collection,
        "index_type": "qdrant_hybrid_dense_bm25_rrf",
        "embedding_batch_size": batch_size,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"  [{corpus}] Qdrant hybrid ready ({total_chunks} points)",
        flush=True,
    )
    return path


def build_all_indices(
    *,
    model: SentenceTransformer | None = None,
    force: bool = False,
    batch_size: int = 16,
    corpora: list[CorpusKind] | None = None,
) -> dict[CorpusKind, Path]:
    """Build one or both corpus indices; share a loaded embedding model."""
    selected: list[CorpusKind] = corpora or ["filings", "transcripts"]
    model = model or load_embedding_model()
    outs: dict[CorpusKind, Path] = {}
    for corpus in selected:
        if corpus_index_exists(corpus) and not force:
            path = corpus_manifest_path(corpus)
            print(f"skip: {corpus} index exists at {path} (use --force)", flush=True)
            outs[corpus] = path
            continue
        outs[corpus] = build_corpus_index(
            corpus,
            model=model,
            force=True,
            batch_size=batch_size,
        )
    return outs


def load_corpus_index(corpus: CorpusKind) -> CorpusIndex:
    """Load corpus for retrieval (Qdrant hybrid for filings or transcripts)."""
    if corpus == "filings":
        return load_filings_index()
    return load_transcripts_index()


def load_filings_index() -> CorpusIndex:
    """Load filings corpus backed by Qdrant hybrid search.

    Does not load ``all_chunks.jsonl`` — hybrid hits carry full payloads from
    Qdrant. ``chunks`` stays empty on this path.
    """
    from crosscheck.retrieval.qdrant_store import collection_ready

    manifest_path = corpus_manifest_path("filings")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Filings index missing. Run: python scripts/build_indices.py "
            "--corpus filings --force\n"
            f"  expected {manifest_path}"
        )
    if not collection_ready("filings"):
        raise FileNotFoundError(
            "Qdrant filings collection empty or missing. "
            "Run: python scripts/build_indices.py --corpus filings --force"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CorpusIndex(
        corpus="filings",
        index=None,
        chunks=[],
        embedding_model=manifest.get("embedding_model", EMBEDDING_MODEL),
        bm25=None,
        backend="qdrant",
    )


def load_transcripts_index() -> CorpusIndex:
    """Load transcripts corpus backed by Qdrant hybrid search."""
    from crosscheck.retrieval.qdrant_store import collection_ready

    manifest_path = corpus_manifest_path("transcripts")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Transcripts index missing. Run: python scripts/build_indices.py "
            "--corpus transcripts --force\n"
            f"  expected {manifest_path}"
        )
    if not collection_ready("transcripts"):
        raise FileNotFoundError(
            "Qdrant transcripts collection empty or missing. "
            "Run: python scripts/build_indices.py --corpus transcripts --force"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CorpusIndex(
        corpus="transcripts",
        index=None,
        chunks=[],
        embedding_model=manifest.get("embedding_model", EMBEDDING_MODEL),
        bm25=None,
        backend="qdrant",
    )


def load_master_index() -> CorpusIndex:
    """Deprecated alias for :func:`load_filings_index`."""
    return load_filings_index()


def retrieve(
    query: str,
    master: CorpusIndex,
    model: SentenceTransformer,
    *,
    k: int = 5,
    ticker: str | None = None,
    doc_types: set[str] | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    candidate_multiplier: int = 20,
) -> list[tuple[IndexedChunk, float]]:
    """Dense retrieve (FAISS) or hybrid Qdrant when ``backend=qdrant``."""
    if master.backend == "qdrant":
        return hybrid_retrieve(
            query,
            master,
            model,
            k=k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            doc_types=doc_types,
            candidate_multiplier=candidate_multiplier,
        )

    if not master.chunks or master.index is None or master.index.ntotal == 0:
        return []

    search_k = min(master.index.ntotal, max(k * candidate_multiplier, k))
    query_vec = embed_texts(model, [query.strip()])
    scores, indices = master.index.search(query_vec, search_k)

    ticker_u = ticker.upper() if ticker else None
    results: list[tuple[IndexedChunk, float]] = []
    for idx, score in zip(indices[0], scores[0], strict=True):
        if idx < 0:
            continue
        chunk = master.chunks[int(idx)]
        if ticker_u and chunk.ticker != ticker_u:
            continue
        if doc_types and chunk.doc_type not in doc_types:
            continue
        if fiscal_year is not None and chunk.fiscal_year != fiscal_year:
            continue
        if not period_matches(chunk.fiscal_period, fiscal_quarter):
            continue
        results.append((chunk, float(score)))
        if len(results) >= k:
            break
    return results


def hybrid_retrieve(
    query: str,
    corpus: CorpusIndex,
    model: SentenceTransformer,
    *,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: str | None = None,
    doc_types: set[str] | None = None,
    candidate_multiplier: int = 20,
    rrf_k: int = 60,
) -> list[tuple[IndexedChunk, float]]:
    """Hybrid retrieve: Qdrant dense+BM25+RRF (filings or transcripts)."""
    if k < 1:
        return []

    if corpus.backend == "qdrant":
        from crosscheck.retrieval.qdrant_store import hybrid_search

        query_vec = embed_texts(model, [query.strip()])[0]
        return hybrid_search(
            query,
            query_vec,
            k=k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            doc_types=doc_types,
            prefetch_k=max(k, 20),
            corpus=corpus.corpus,
        )

    channel_k = max(k, 20)
    # Avoid recurse through retrieve() qdrant branch
    if not corpus.chunks or corpus.index is None or corpus.index.ntotal == 0:
        dense: list[tuple[IndexedChunk, float]] = []
    else:
        search_k = min(
            corpus.index.ntotal,
            max(channel_k * candidate_multiplier, channel_k),
        )
        query_vec = embed_texts(model, [query.strip()])
        scores, indices = corpus.index.search(query_vec, search_k)
        ticker_u = ticker.upper() if ticker else None
        dense = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            chunk = corpus.chunks[int(idx)]
            if ticker_u and chunk.ticker != ticker_u:
                continue
            if doc_types and chunk.doc_type not in doc_types:
                continue
            if fiscal_year is not None and chunk.fiscal_year != fiscal_year:
                continue
            if not period_matches(chunk.fiscal_period, fiscal_quarter):
                continue
            dense.append((chunk, float(score)))
            if len(dense) >= channel_k:
                break

    sparse: list[tuple[IndexedChunk, float]] = []
    if corpus.bm25 is not None:
        sparse = bm25_retrieve(
            query,
            corpus.bm25,
            k=channel_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            doc_types=doc_types,
        )

    if not dense and not sparse:
        return []
    if not sparse:
        return dense[:k]
    if not dense:
        return sparse[:k]
    return rrf_fuse([dense, sparse], k=rrf_k, top_n=k)
