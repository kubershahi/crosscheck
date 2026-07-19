"""Chunking: section/table-aware filings and speaker-turn transcripts."""

from crosscheck.chunking.filings import chunk_filing
from crosscheck.chunking.pipeline import (
    build_chunks_for_period,
    periods_from_manifest,
)
from crosscheck.chunking.store import load_chunks_jsonl, write_chunks_jsonl
from crosscheck.chunking.transcripts import chunk_transcript

__all__ = [
    "chunk_filing",
    "chunk_transcript",
    "build_chunks_for_period",
    "periods_from_manifest",
    "write_chunks_jsonl",
    "load_chunks_jsonl",
]
