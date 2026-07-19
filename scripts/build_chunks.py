#!/usr/bin/env python3
"""Chunk raw filings + transcripts and write JSONL under data/chunks/{year}/.

Periods come only from ``data/manifests/companies.yml`` (no hardcoded fallbacks).

Examples::

    # Chunk every company-period in the manifest
    python scripts/build_chunks.py

    # One (or more) tickers
    python scripts/build_chunks.py --ticker AAPL
    python scripts/build_chunks.py --ticker AAPL --ticker MSFT

Output::

    data/chunks/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_filing.jsonl
    data/chunks/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_transcript.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.chunking.pipeline import (  # noqa: E402
    build_chunks_for_period,
    periods_from_manifest,
)
from crosscheck.models import Chunk  # noqa: E402


def _summary(chunks: list[Chunk], label: str) -> None:
    """Print a one-line summary of chunk counts / speakers / sections."""
    n_tables = sum(1 for c in chunks if c.is_table)
    speakers = sorted({c.speaker_name for c in chunks if c.speaker_name})
    sections = sorted({c.section_header for c in chunks if c.section_header})
    print(f"  {label}: {len(chunks)} chunks", end="")
    if n_tables:
        print(f" ({n_tables} tables)", end="")
    if speakers:
        print(f" | speakers={len(speakers)}", end="")
    if sections and label == "filing":
        print(f" | sections≈{min(len(sections), 8)} shown later", end="")
    print()
    if speakers:
        print(f"    speakers: {', '.join(speakers[:10])}" + (" …" if len(speakers) > 10 else ""))
    if sections and label == "filing":
        print(f"    sample sections: {', '.join(sections[:6])}" + (" …" if len(sections) > 6 else ""))


def main() -> None:
    """CLI entry: chunk manifest periods and write JSONL."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Only chunk this ticker (repeatable). Default: all rows in the manifest.",
    )
    args = parser.parse_args()

    try:
        periods = periods_from_manifest(tickers=args.tickers)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    for i, period in enumerate(periods):
        if i:
            print()
        label = f"{period.ticker} FY{period.fiscal_year} Q{period.fiscal_quarter}"
        print(f"[{label}] chunking …")
        try:
            result = build_chunks_for_period(period, write=True)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}")
            continue

        filing_chunks = result["filing_chunks"]
        transcript_chunks = result["transcript_chunks"]
        assert isinstance(filing_chunks, list)
        assert isinstance(transcript_chunks, list)
        _summary(filing_chunks, "filing")
        _summary(transcript_chunks, "transcript")
        if result["filing_out"]:
            print(f"  wrote {result['filing_out']}")
            print(f"  wrote {result['transcript_out']}")


if __name__ == "__main__":
    main()
