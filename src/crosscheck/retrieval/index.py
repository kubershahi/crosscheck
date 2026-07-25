"""Disk-streamed corpus assembly, embeddings, and FAISS indexing.

Filings (10-K / 10-Q) and transcripts are indexed separately so NLI retrieves
only SEC filing passages when checking transcript claims.
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
    """FAISS index + aligned ``IndexedChunk`` rows (global_id == row)."""

    corpus: CorpusKind
    index: faiss.IndexFlatIP
    chunks: list[IndexedChunk]
    embedding_model: str
    bm25: Bm25Index | None = None


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
    """True when all artifacts for one corpus exist."""
    return (
        corpus_index_path(corpus).exists()
        and corpus_chunks_path(corpus).exists()
        and corpus_embeddings_path(corpus).exists()
        and corpus_manifest_path(corpus).exists()
    )


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
    """Merge → contextual embed → write FAISS for one corpus."""
    if corpus_index_exists(corpus) and not force:
        return corpus_index_path(corpus)

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
    path = build_faiss_from_memmap(
        source_path=disk_embeddings_path,
        output_path=corpus_index_path(corpus),
        total_chunks=total_chunks,
    )

    manifest = {
        "corpus": corpus,
        "doc_types": sorted(CORPUS_DOC_TYPES[corpus]),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_dtype": "float32",
        "embedding_storage": "numpy.memmap",
        "n_chunks": total_chunks,
        "all_chunks_path": str(chunks_path),
        "embeddings_path": str(disk_embeddings_path),
        "index_path": str(path),
        "index_type": "IndexFlatIP",
        "embedding_batch_size": batch_size,
    }
    corpus_manifest_path(corpus).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  [{corpus}] wrote {path} ({total_chunks} vectors)", flush=True)
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
            path = corpus_index_path(corpus)
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
    """Load FAISS + aligned master JSONL for one corpus."""
    if not corpus_index_exists(corpus):
        raise FileNotFoundError(
            f"{corpus} index missing. Run: python scripts/build_indices.py "
            f"--corpus {corpus}\n"
            f"  expected {corpus_index_path(corpus)}"
        )
    chunks = load_indexed_chunks_jsonl(corpus_chunks_path(corpus))
    index = faiss.read_index(str(corpus_index_path(corpus)))
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS ntotal={index.ntotal} != {corpus} all_chunks lines={len(chunks)}. "
            "Rebuild with --force."
        )
    manifest = json.loads(corpus_manifest_path(corpus).read_text(encoding="utf-8"))
    bm25 = build_bm25_index(chunks) if corpus == "filings" else None
    return CorpusIndex(
        corpus=corpus,
        index=index,
        chunks=chunks,
        embedding_model=manifest.get("embedding_model", EMBEDDING_MODEL),
        bm25=bm25,
    )


def load_filings_index() -> CorpusIndex:
    """Load the filings-only index used by NLI (includes in-memory BM25)."""
    return load_corpus_index("filings")


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
    """Embed a query and return top-k chunks, optionally filtered by metadata."""
    if not master.chunks or master.index.ntotal == 0:
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
    """Dense FAISS + BM25, each filtered, then RRF-fuse to top-k.

    Each channel overfetches ``max(k, 20)`` (or denser FAISS overfetch via
    ``candidate_multiplier``), then RRF merges the filtered ranked lists.
    """
    if k < 1:
        return []

    channel_k = max(k, 20)
    dense = retrieve(
        query,
        corpus,
        model,
        k=channel_k,
        ticker=ticker,
        doc_types=doc_types,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        candidate_multiplier=candidate_multiplier,
    )

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
