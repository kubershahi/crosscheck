#!/usr/bin/env python3
"""Load saved claims, then run hybrid retrieval and NLI.

Examples::

    # All claim files under data/claims/
    python scripts/run_nli.py

    # All periods for one fiscal year
    python scripts/run_nli.py --year 2025

    # One ticker (all years/quarters on disk)
    python scripts/run_nli.py --ticker AAPL

    # One ticker-year or exact period
    python scripts/run_nli.py --ticker AAPL --year 2025
    python scripts/run_nli.py --ticker AAPL --year 2025 --quarter Q1

    python scripts/run_nli.py --ticker AAPL --top-k 5 --no-rerank

Output: ``data/reports/{year}/{TICKER}/{TICKER}_FY{y}_Q{n}_reports.json``

A 62s pause runs between claim files (not after the last) to stay under free-tier
per-minute LLM limits when processing many periods.

Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    python scripts/extract_claims.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.pipeline import run_pipeline  # noqa: E402
from crosscheck.config import CLAIMS_DIR, report_path  # noqa: E402
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    SavedTranscriptClaims,
    as_fiscal_quarter,
    quarter_number,
)

# Pause between claim JSON files so free-tier RPM budgets can recover.
CLAIM_FILE_GAP_SECONDS = 62


def _discover_claim_files(
    *,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> list[Path]:
    """Return claim JSON paths under ``data/claims`` matching optional filters.

    Layout: ``data/claims/{year}/{TICKER}/{TICKER}_FY{year}_{Qn}_claims.json``
    """
    if year is not None:
        year_roots = [CLAIMS_DIR / str(year)]
    else:
        year_roots = sorted(
            p for p in CLAIMS_DIR.iterdir() if p.is_dir() and p.name.isdigit()
        )

    files: list[Path] = []
    for year_dir in year_roots:
        if not year_dir.is_dir():
            continue
        if ticker:
            ticker_dirs = [year_dir / ticker.upper()]
        else:
            ticker_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())

        for ticker_dir in ticker_dirs:
            if not ticker_dir.is_dir():
                continue
            files.extend(sorted(ticker_dir.glob("*_claims.json")))

    if quarter is not None:
        wanted = as_fiscal_quarter(quarter)
        filtered: list[Path] = []
        for path in files:
            try:
                saved = SavedTranscriptClaims.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if saved.fiscal_quarter == wanted:
                filtered.append(path)
        files = filtered

    return files


def main() -> None:
    """CLI entry: run retrieval + NLI and write ``*_reports.json``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker filter (default: all tickers with saved claims).",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Optional fiscal year filter (default: all years under data/claims).",
    )
    parser.add_argument(
        "--quarter",
        help="Optional fiscal quarter filter: Q1–Q4 or 1–4.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of filing passages to retrieve per claim (default: 5).",
    )
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help=(
            "Dense candidate pool size before cross-encoder reranking "
            "(default: max(top_k*10, 20))."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking (still uses hybrid dense+BM25).",
    )
    parser.add_argument(
        "--profile",
        choices=("development", "production", "test", "dev"),
        help="Override CROSSCHECK_LLM_PROFILE (development|production).",
    )
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=CLAIM_FILE_GAP_SECONDS,
        help=(
            f"Seconds to sleep between claim files (default: {CLAIM_FILE_GAP_SECONDS}). "
            "Set 0 to disable."
        ),
    )
    args = parser.parse_args()

    if args.profile:
        profile = "development" if args.profile in {"test", "dev"} else args.profile
        os.environ["CROSSCHECK_LLM_PROFILE"] = profile

    if args.quarter is not None:
        quarter_number(args.quarter)

    ticker = args.ticker.strip().upper() if args.ticker else None
    claim_files = _discover_claim_files(
        ticker=ticker,
        year=args.year,
        quarter=args.quarter,
    )
    if not claim_files:
        scope = []
        if ticker:
            scope.append(f"ticker={ticker}")
        if args.year is not None:
            scope.append(f"year={args.year}")
        if args.quarter is not None:
            scope.append(f"quarter={as_fiscal_quarter(args.quarter)}")
        scope_s = ", ".join(scope) if scope else "no filters"
        print(f"No saved claims found under {CLAIMS_DIR} ({scope_s}).")
        sys.exit(1)

    print(f"Found {len(claim_files)} claim file(s) under {CLAIMS_DIR}")
    if args.gap_seconds > 0 and len(claim_files) > 1:
        print(
            f"Will pause {args.gap_seconds}s between claim files "
            "(rate-limit spacing)",
        )
    print()

    for index, claim_file in enumerate(claim_files):
        saved = SavedTranscriptClaims.model_validate_json(
            claim_file.read_text(encoding="utf-8")
        )
        period = DocumentMeta(
            ticker=saved.ticker,
            company_name=saved.company_name,
            fiscal_year=saved.fiscal_year,
            fiscal_quarter=saved.fiscal_quarter,
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        out = report_path(period.ticker, period.fiscal_year, period.fiscal_quarter)

        print(f"=== Crosscheck NLI: {label} ===")
        print(f"  claims source: {claim_file}")
        print(f"  report path: {out}")
        print()

        try:
            report = run_pipeline(
                period,
                top_k=args.top_k,
                use_reranker=not args.no_rerank,
                rerank_pool_k=args.rerank_pool_k,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        payload = report.model_dump(mode="json")
        text = json.dumps(payload, indent=2)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"=== Wrote {out} ===")

        if args.gap_seconds > 0 and index < len(claim_files) - 1:
            print(
                f"  sleeping {args.gap_seconds}s before next claim file …",
                flush=True,
            )
            time.sleep(args.gap_seconds)


if __name__ == "__main__":
    main()
