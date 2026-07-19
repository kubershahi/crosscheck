"""Persist and load chunk JSONL under ``data/chunks/{year}/{TICKER}/``."""

from __future__ import annotations

import json
from pathlib import Path

from crosscheck.config import chunks_dir
from crosscheck.models import Chunk


def chunk_output_path(
    ticker: str,
    *,
    doc_type: str,
    fiscal_year: int,
    fiscal_quarter: int,
) -> Path:
    """Return e.g. ``data/chunks/2024/AAPL/AAPL_FY2024_Q4_filing.jsonl``."""
    ticker = ticker.upper()
    name = f"{ticker}_FY{fiscal_year}_Q{fiscal_quarter}_{doc_type}.jsonl"
    return chunks_dir(ticker, fiscal_year) / name


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
