"""Disk-streamed master chunk assembly, embeddings, and FAISS indexing."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from crosscheck.chunking.store import (
    all_chunks_path,
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

EMBEDDING_DIMENSION = 1024


@dataclass
class MasterIndex:
    """Unified FAISS index + aligned ``IndexedChunk`` rows (global_id == row)."""

    index: faiss.IndexFlatIP
    chunks: list[IndexedChunk]
    embedding_model: str


def unified_index_path() -> Path:
    """Return ``data/indices/unified_master.faiss``."""
    return INDICES_DIR / "unified_master.faiss"


def unified_manifest_path() -> Path:
    """Return ``data/indices/unified_master.manifest.json``."""
    return INDICES_DIR / "unified_master.manifest.json"


def embeddings_path() -> Path:
    """Return the disk-backed embedding matrix path."""
    return INDICES_DIR / "embeddings.npy"


def master_index_exists() -> bool:
    """True when all master index artifacts exist."""
    return (
        unified_index_path().exists()
        and all_chunks_path().exists()
        and embeddings_path().exists()
        and unified_manifest_path().exists()
    )


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty JSONL rows without loading the file."""
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def merge_all_chunks(*, force: bool = False) -> Path:
    """Step A: merge company JSONL files into ``all_chunks.jsonl`` with global_id."""
    out = all_chunks_path()
    if out.exists() and not force:
        return out

    company_files = iter_company_chunk_files()
    if not company_files:
        raise FileNotFoundError(
            f"No per-company chunk JSONL under {out.parent}. "
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
                    row = IndexedChunk(
                        **chunk.model_dump(),
                        global_id=global_id,
                    )
                    fh.write(row.model_dump_json())
                    fh.write("\n")
                    global_id += 1

    print(
        f"  merged {global_id} chunks from {len(company_files)} files → {out}",
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
    chunks_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 16,
    max_chunks: int | None = None,
) -> tuple[Path, int]:
    """Stream master JSONL into a disk-backed float32 embedding matrix.

    ``max_chunks`` is intended for bounded smoke tests; production leaves it
    unset and processes every row.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    chunks_path = chunks_path or all_chunks_path()
    output_path = output_path or embeddings_path()
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
            desc="Embedding",
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
    source_path: Path | None = None,
    output_path: Path | None = None,
    total_chunks: int,
) -> Path:
    """Load embeddings read-only and add the memmap directly to IndexFlatIP."""
    source_path = source_path or embeddings_path()
    output_path = output_path or unified_index_path()
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


def build_unified_index(
    *,
    model: SentenceTransformer | None = None,
    force: bool = False,
    batch_size: int = 16,
) -> Path:
    """Steps A–C: merge → contextual embed → write ``unified_master.faiss``."""
    if master_index_exists() and not force:
        return unified_index_path()

    master_chunks_path = merge_all_chunks(force=True)
    model = model or load_embedding_model()
    disk_embeddings_path, total_chunks = stream_embeddings_to_memmap(
        model=model,
        chunks_path=master_chunks_path,
        output_path=embeddings_path(),
        batch_size=batch_size,
    )
    path = build_faiss_from_memmap(
        source_path=disk_embeddings_path,
        output_path=unified_index_path(),
        total_chunks=total_chunks,
    )

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_dtype": "float32",
        "embedding_storage": "numpy.memmap",
        "n_chunks": total_chunks,
        "all_chunks_path": str(all_chunks_path()),
        "embeddings_path": str(disk_embeddings_path),
        "index_path": str(path),
        "index_type": "IndexFlatIP",
        "embedding_batch_size": batch_size,
    }
    unified_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {path} ({total_chunks} vectors)", flush=True)
    return path


def load_master_index() -> MasterIndex:
    """Load unified FAISS + ``all_chunks.jsonl``."""
    if not master_index_exists():
        raise FileNotFoundError(
            f"Master index missing. Run: python scripts/build_indices.py\n"
            f"  expected {unified_index_path()}"
        )
    chunks = load_indexed_chunks_jsonl()
    index = faiss.read_index(str(unified_index_path()))
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS ntotal={index.ntotal} != all_chunks lines={len(chunks)}. Rebuild with --force."
        )
    manifest = json.loads(unified_manifest_path().read_text(encoding="utf-8"))
    return MasterIndex(
        index=index,
        chunks=chunks,
        embedding_model=manifest.get("embedding_model", EMBEDDING_MODEL),
    )


def retrieve(
    query: str,
    master: MasterIndex,
    model: SentenceTransformer,
    *,
    k: int = 5,
    ticker: str | None = None,
    doc_types: set[str] | None = None,
    fiscal_year: int | None = None,
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
        results.append((chunk, float(score)))
        if len(results) >= k:
            break
    return results
