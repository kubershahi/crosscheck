#!/usr/bin/env python3
"""Chunk raw filing/transcript pairs and write JSONL under data/chunks/{year}/.

By default only builds outputs that are missing. Use ``--force`` to rebuild.

Examples::

    # Chunk every complete raw pair that does not already have JSONL
    python scripts/build_chunks.py

    # By ticker
    python scripts/build_chunks.py --ticker AAPL
    python scripts/build_chunks.py --ticker AAPL,MSFT,NVDA

    # By year / quarter (alone or combined)
    python scripts/build_chunks.py --year 2025
    python scripts/build_chunks.py --quarter Q2
    python scripts/build_chunks.py --year 2025 --quarter Q1
    python scripts/build_chunks.py --ticker AAPL --year 2025 --quarter Q3

    # Rebuild even if chunk JSONL already exists
    python scripts/build_chunks.py --force
    python scripts/build_chunks.py --ticker AAPL --force
    python scripts/build_chunks.py --ticker AAPL --year 2025 --quarter Q2 --force

Output::

    data/chunks/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_10-K.jsonl   # or 10-Q
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
    discover_raw_periods,
)
from crosscheck.models import Chunk, quarter_number  # noqa: E402


def _parse_tickers(value: str | None) -> list[str] | None:
    """Expand ``--ticker AAPL,MSFT`` into a normalized ticker list."""
    if value is None:
        return None
    return list(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))


def _summary(chunks: list[Chunk], label: str) -> None:
    """Print a one-line summary of chunk counts / speakers / sections."""
    n_tables = sum(1 for c in chunks if c.is_table)
    speakers = sorted({c.speaker_name for c in chunks if c.speaker_name})
    sections = sorted({c.section for c in chunks if c.section})
    print(f"  {label}: {len(chunks)} chunks", end="")
    if n_tables:
        print(f" ({n_tables} tables)", end="")
    if speakers:
        print(f" | speakers={len(speakers)}", end="")
    if sections and label != "transcript":
        print(f" | sections≈{min(len(sections), 8)}", end="")
    print()
    if speakers:
        print(f"    speakers: {', '.join(speakers[:10])}" + (" …" if len(speakers) > 10 else ""))
    if sections and label != "transcript":
        print(f"    sample sections: {', '.join(sections[:6])}" + (" …" if len(sections) > 6 else ""))


def main() -> None:
    """CLI entry: discover and chunk requested raw ticker data."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker or comma-separated subset (e.g. AAPL,MSFT).",
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="years",
        help="Filter to fiscal year(s); repeatable (e.g. --year 2025).",
    )
    parser.add_argument(
        "--quarter",
        action="append",
        dest="quarters",
        help="Filter to fiscal quarter(s); repeatable (Q1–Q4 or 1–4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-chunk and overwrite JSONL that already exists.",
    )
    args = parser.parse_args()
    tickers = _parse_tickers(args.ticker)
    if args.ticker is not None and not tickers:
        parser.error("--ticker must contain at least one ticker")

    quarter_nums: list[int] | None = None
    if args.quarters:
        try:
            quarter_nums = [quarter_number(q) for q in args.quarters]
        except ValueError as exc:
            parser.error(str(exc))

    try:
        periods = discover_raw_periods(
            tickers=tickers,
            years=args.years,
            quarters=quarter_nums,
        )
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    for i, period in enumerate(periods):
        if i:
            print()
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        print(f"[{label}] chunking …")
        try:
            result = build_chunks_for_period(
                period, write=True, force=args.force
            )
        except FileNotFoundError as exc:
            print(f"  skip: {exc}")
            continue

        filing_chunks = result["filing_chunks"]
        transcript_chunks = result["transcript_chunks"]
        assert isinstance(filing_chunks, list)
        assert isinstance(transcript_chunks, list)

        if result["skipped_filing"]:
            print(f"  skip filing (exists): {result['filing_out']}")
        else:
            _summary(filing_chunks, "filing")
            if result["filing_out"]:
                print(f"  wrote {result['filing_out']}")

        if result["skipped_transcript"]:
            print(f"  skip transcript (exists): {result['transcript_out']}")
        else:
            _summary(transcript_chunks, "transcript")
            if result["transcript_out"]:
                print(f"  wrote {result['transcript_out']}")


if __name__ == "__main__":
    main()
