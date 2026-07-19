#!/usr/bin/env python3
"""Sample random chunks from a filing + transcript and print text + metadata.

Periods come only from ``data/manifests/companies.yml``.

Examples::

    python scripts/sanity_chunks.py --ticker AAPL
    python scripts/sanity_chunks.py --ticker AAPL --n 3 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.chunking.pipeline import (  # noqa: E402
    build_chunks_for_period,
    periods_from_manifest,
)
from crosscheck.models import Chunk  # noqa: E402


def _preview(text: str, limit: int = 400) -> str:
    """Truncate text for terminal display."""
    text = text.replace("\t", " | ")
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _print_samples(label: str, chunks: list[Chunk], n: int, seed: int) -> None:
    """Print ``n`` random chunks with metadata for manual inspection."""
    print("=" * 72)
    print(f"{label}: {len(chunks)} chunks total — sampling {min(n, len(chunks))}")
    print("=" * 72)
    if not chunks:
        print("(no chunks produced)")
        return
    rng = random.Random(seed)
    sample = chunks if len(chunks) <= n else rng.sample(chunks, n)
    if label.startswith("FILING"):
        tables = [c for c in chunks if c.is_table]
        if tables and not any(c.is_table for c in sample):
            sample[-1] = rng.choice(tables)
    for i, chunk in enumerate(sample, 1):
        print(f"\n--- sample {i} ---")
        print("metadata:", json.dumps(chunk.metadata_dict(), indent=2))
        print("text:")
        print(_preview(chunk.text))


def main() -> None:
    """CLI entry: build chunks in memory and print random samples."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--n", type=int, default=5, help="Samples per document type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--filing", type=Path, default=None, help="Optional filing path")
    parser.add_argument(
        "--transcript", type=Path, default=None, help="Optional transcript path"
    )
    args = parser.parse_args()

    try:
        periods = periods_from_manifest(
            tickers=[args.ticker],
            manifest_path=args.manifest,
        )
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    period = periods[0]
    try:
        result = build_chunks_for_period(
            period,
            filing_path=args.filing,
            transcript_path=args.transcript,
            write=False,
        )
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    filing_chunks = result["filing_chunks"]
    transcript_chunks = result["transcript_chunks"]
    filing_path = result["filing_path"]
    transcript_path = result["transcript_path"]
    assert isinstance(filing_chunks, list)
    assert isinstance(transcript_chunks, list)

    n_tables = sum(1 for c in filing_chunks if c.is_table)
    speakers = sorted({c.speaker_name for c in transcript_chunks if c.speaker_name})
    print(f"Filing tables (atomic): {n_tables}")
    print(f"Transcript speakers ({len(speakers)}): {', '.join(speakers[:12])}")
    if len(speakers) > 12:
        print(f"  … +{len(speakers) - 12} more")

    _print_samples(f"FILING {Path(str(filing_path)).name}", filing_chunks, args.n, args.seed)
    _print_samples(
        f"TRANSCRIPT {Path(str(transcript_path)).name}",
        transcript_chunks,
        args.n,
        args.seed + 1,
    )


if __name__ == "__main__":
    main()
