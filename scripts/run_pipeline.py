#!/usr/bin/env python3
"""Load saved claims, then run dense retrieval and NLI for one ticker.

Examples::

    python scripts/run_pipeline.py --ticker AAPL
    python scripts/run_pipeline.py --ticker AAPL --top-k 5

Output: JSON to terminal and ``data/reports/{year}/{TICKER}/`` file.

Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    python scripts/extract_claims.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.pipeline import run_pipeline  # noqa: E402
from crosscheck.config import CLAIMS_DIR, get_llm_profile, report_path, resolve_llm_models  # noqa: E402
from crosscheck.models import DocumentMeta, SavedTranscriptClaims  # noqa: E402


def main() -> None:
    """CLI entry: run end-to-end NLI pipeline and write report JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol whose saved claims should be processed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of filing passages to retrieve per claim (default: 5).",
    )
    parser.add_argument(
        "--profile",
        choices=("development", "production", "test", "dev"),
        help="Override CROSSCHECK_LLM_PROFILE (development|production).",
    )
    args = parser.parse_args()

    if args.profile:
        profile = "development" if args.profile in {"test", "dev"} else args.profile
        os.environ["CROSSCHECK_LLM_PROFILE"] = profile

    ticker = args.ticker.strip().upper()
    claim_files = sorted(CLAIMS_DIR.glob(f"*/{ticker}/*_claims.json"))
    if not claim_files:
        print(f"No saved claims found for {ticker!r} under {CLAIMS_DIR}.")
        sys.exit(1)

    for claim_file in claim_files:
        saved = SavedTranscriptClaims.model_validate_json(
            claim_file.read_text(encoding="utf-8")
        )
        period = DocumentMeta(
            ticker=saved.ticker,
            company_name=saved.company_name,
            fiscal_year=saved.fiscal_year,
            fiscal_quarter=saved.fiscal_quarter,
        )
        label = f"{period.ticker} FY{period.fiscal_year} Q{period.fiscal_quarter}"
        models = resolve_llm_models()
        out = report_path(period.ticker, period.fiscal_year, period.fiscal_quarter)

        print(f"=== Crosscheck pipeline: {label} ===")
        print(f"  claims source: {claim_file}")
        print(f"  llm profile: {get_llm_profile()}  models={', '.join(models)}")
        print(f"  retrieve top_k: {args.top_k}")
        print(f"  report path: {out}")
        print()

        try:
            report = run_pipeline(period, top_k=args.top_k)
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        print()
        print("=== Report JSON ===")
        payload = report.model_dump(mode="json")
        text = json.dumps(payload, indent=2)
        print(text)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\n=== Wrote {out} ===")


if __name__ == "__main__":
    main()
