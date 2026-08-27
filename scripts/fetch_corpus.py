#!/usr/bin/env python3
"""Fetch EDGAR filings + earnings-call transcripts from the company manifest.

Examples::

    # fetch everyinclude: true companies / fetch: true quarters
    python scripts/fetch_corpus.py

    # By ticker (overrides include: false for that ticker)
    python scripts/fetch_corpus.py --ticker AAPL
    python scripts/fetch_corpus.py --ticker AAPL --ticker MSFT

    # By year / quarter (alone or combined)
    python scripts/fetch_corpus.py --year 2025
    python scripts/fetch_corpus.py --quarter Q2
    python scripts/fetch_corpus.py --year 2025 --quarter Q1
    python scripts/fetch_corpus.py --ticker AAPL --year 2025 --quarter Q3

    # Every manifest row (ignore include)
    python scripts/fetch_corpus.py --all
    python scripts/fetch_corpus.py --all --year 2025 --quarter Q4

    # Filings or transcripts only
    python scripts/fetch_corpus.py --filings-only
    python scripts/fetch_corpus.py --transcripts-only
    python scripts/fetch_corpus.py --ticker AAPL --filings-only
    python scripts/fetch_corpus.py --ticker AAPL --year 2025 --transcripts-only

    # Re-download even if files already exist
    python scripts/fetch_corpus.py --force
    python scripts/fetch_corpus.py --ticker AAPL --quarter Q1 --force
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
from crosscheck.ingest.transcript import fetch_transcript  # noqa: E402
from crosscheck.manifest import (  # noqa: E402
    default_manifest_path,
    filter_manifest,
    load_manifest,
)
from crosscheck.models import quarter_number  # noqa: E402


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
        help=(
            "Filter to ticker(s); repeatable. Overrides include: false. "
            "Default: rows with include: true."
        ),
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
        "--all",
        action="store_true",
        help="Fetch every manifest row, ignoring the include flag.",
    )
    parser.add_argument("--filings-only", action="store_true")
    parser.add_argument("--transcripts-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download HTML even if present. Sidecar .meta.json is "
            "refreshed on every fetch (with or without --force)."
        ),
    )
    args = parser.parse_args()

    if args.filings_only and args.transcripts_only:
        parser.error("Use only one of --filings-only / --transcripts-only")
    if args.all and args.tickers:
        parser.error("Use only one of --all / --ticker")

    quarter_nums: list[int] | None = None
    if args.quarters:
        try:
            quarter_nums = [quarter_number(q) for q in args.quarters]
        except ValueError as exc:
            parser.error(str(exc))

    do_filings = not args.transcripts_only
    do_transcripts = not args.filings_only

    manifest = load_manifest(args.manifest)
    rows = filter_manifest(
        manifest,
        tickers=args.tickers,
        years=args.years,
        quarters=quarter_nums,
        include_only=not args.all,
    )
    if not rows:
        print(
            "No matching periods in manifest. "
            "Set company include: true and quarters.*.fetch: true, "
            "or pass --ticker / --year / --quarter / --all."
        )
        sys.exit(1)

    labels = [f"{r.ticker} FY{r.fiscal_year} Q{r.fiscal_quarter}" for r in rows]
    print(
        f"Fetching {len(rows)} period"
        f"{'' if len(rows) == 1 else 's'}: {', '.join(labels)}"
    )

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
