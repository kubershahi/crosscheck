#!/usr/bin/env python3
"""Fetch EDGAR filings + Motley Fool transcripts from the company manifest.

Examples::

    # Fetch filing + transcript for AAPL (from data/manifests/companies.yml)
    python scripts/fetch_corpus.py --ticker AAPL

    # All rows in the manifest
    python scripts/fetch_corpus.py

    # Only one side
    python scripts/fetch_corpus.py --ticker AAPL --filings-only
    python scripts/fetch_corpus.py --ticker AAPL --transcripts-only --force

Writes under year-first paths::

    data/raw/filings/{year}/{TICKER}/
    data/raw/transcripts/{year}/{TICKER}/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.ingest.edgar import fetch_filing  # noqa: E402
from crosscheck.ingest.motley_fool import fetch_transcript  # noqa: E402
from crosscheck.manifest import (  # noqa: E402
    default_manifest_path,
    filter_manifest,
    load_manifest,
)


def main() -> None:
    """CLI entry: iterate manifest rows and fetch filings and/or transcripts."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Path to companies.yml (default: {default_manifest_path()})",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Filter to ticker(s); repeatable. Default: all rows in manifest.",
    )
    parser.add_argument("--filings-only", action="store_true")
    parser.add_argument("--transcripts-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    if args.filings_only and args.transcripts_only:
        parser.error("Use only one of --filings-only / --transcripts-only")

    do_filings = not args.transcripts_only
    do_transcripts = not args.filings_only

    manifest = load_manifest(args.manifest)
    rows = filter_manifest(manifest, tickers=args.tickers)
    if not rows:
        print("No matching companies in manifest.")
        sys.exit(1)

    for period in rows:
        form = period.resolved_form()
        print(
            f"[{period.ticker} / {period.name}] FY{period.fiscal_year} "
            f"Q{period.fiscal_quarter} (form={form})"
        )

        if do_filings:
            try:
                path = fetch_filing(
                    period.ticker,
                    form=form,
                    fiscal_year=period.fiscal_year,
                    fiscal_quarter=period.fiscal_quarter,
                    company_name=period.name,
                    force=args.force,
                )
                print(f"  filing -> {path} ({path.stat().st_size / 1024:.1f} KB)")
            except Exception as exc:  # noqa: BLE001 — surface per-row failures
                print(f"  filing ERROR: {exc}")

        if do_transcripts:
            try:
                path = fetch_transcript(period, force=args.force)
                print(f"  transcript -> {path} ({path.stat().st_size / 1024:.1f} KB)")
            except Exception as exc:  # noqa: BLE001
                print(f"  transcript ERROR: {exc}")


if __name__ == "__main__":
    main()
