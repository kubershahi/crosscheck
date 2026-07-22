"""Persist and load chunk JSONL under ``data/chunks/``.

Per-company outputs (stateless)::

    data/chunks/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_{doc_type}.jsonl

Master assembly (stateful index build)::

    data/indices/all_chunks.jsonl   # includes sequential global_id
"""

from __future__ import annotations

import json
from pathlib import Path

from crosscheck.config import CHUNKS_DIR, INDICES_DIR, chunks_dir
from crosscheck.models import Chunk, IndexedChunk


def chunk_output_path(
    ticker: str,
    *,
    doc_type: str,
    fiscal_year: int,
    fiscal_quarter: int,
) -> Path:
    """Return e.g. ``data/chunks/2025/AAPL/AAPL_FY2025_Q4_transcript.jsonl``."""
    ticker = ticker.upper()
    # Sanitize doc_type for filenames (10-K → 10-K is fine on Unix/macOS)
    safe = doc_type.replace("/", "-")
    name = f"{ticker}_FY{fiscal_year}_Q{fiscal_quarter}_{safe}.jsonl"
    return chunks_dir(ticker, fiscal_year) / name


def all_chunks_path() -> Path:
    """Return ``data/indices/all_chunks.jsonl``."""
    return INDICES_DIR / "all_chunks.jsonl"


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> Path:
    """Write one JSON object per line; create parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json())
            fh.write("\n")
    return path


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    """Load a JSONL chunk file into a list of ``Chunk`` models."""
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def load_indexed_chunks_jsonl(path: Path | None = None) -> list[IndexedChunk]:
    """Load the master ``all_chunks.jsonl`` (requires ``global_id``)."""
    path = path or all_chunks_path()
    chunks: list[IndexedChunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(IndexedChunk.model_validate(json.loads(line)))
    return chunks


def iter_company_chunk_files(root: Path | None = None) -> list[Path]:
    """Find per-company chunk JSONL files (excludes ``all_chunks.jsonl``)."""
    root = root or CHUNKS_DIR
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*.jsonl")
        if p.name != "all_chunks.jsonl"
    )
