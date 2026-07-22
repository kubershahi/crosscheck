#!/usr/bin/env python3
"""Extract and persist fixed transcript claims for one or all companies.

Examples::

    # Every transcript chunk file under data/chunks
    python scripts/extract_claims.py --n 3

    # One ticker or a comma-separated subset
    python scripts/extract_claims.py --ticker AAPL --n 3
    python scripts/extract_claims.py --ticker AAPL,MSFT --n 3 --force

Prerequisite: run ``python scripts/build_chunks.py`` first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.claims import extract_claims, save_claims  # noqa: E402
from crosscheck.analysis.executives import (  # noqa: E402
    executive_source_text,
    is_executive_chunk,
)
from crosscheck.chunking.store import iter_company_chunk_files, load_chunks_jsonl  # noqa: E402
from crosscheck.config import claims_path, get_llm_profile, resolve_llm_models  # noqa: E402
from crosscheck.models import DocumentMeta  # noqa: E402


def _parse_tickers(value: str | None) -> set[str] | None:
    """Expand ``--ticker AAPL,MSFT`` into a normalized ticker set."""
    if value is None:
        return None
    tickers = {part.strip().upper() for part in value.split(",") if part.strip()}
    return tickers or None


def main() -> None:
    """Extract stable claim sets from transcript chunks."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker or comma-separated subset (e.g. AAPL,MSFT).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        choices=range(1, 5),
        metavar="1-4",
        help="Maximum claims per company (default: 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace claims files that already exist.",
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

    wanted = _parse_tickers(args.ticker)
    if args.ticker is not None and not wanted:
        parser.error("--ticker must contain at least one ticker")

    transcript_files = [
        path
        for path in iter_company_chunk_files()
        if path.name.endswith("_transcript.jsonl")
        and (wanted is None or path.parent.name.upper() in wanted)
    ]
    if not transcript_files:
        print("No matching transcript chunk files under data/chunks.")
        sys.exit(1)

    models = resolve_llm_models()
    print(
        f"Claim extraction: profile={get_llm_profile()} "
        f"max_claims={args.n} files={len(transcript_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(models)}", flush=True)

    for file_idx, transcript_path in enumerate(transcript_files, start=1):
        chunks = load_chunks_jsonl(transcript_path)
        if not chunks:
            print(f"\n[skip] empty transcript chunks: {transcript_path}", flush=True)
            continue
        first = chunks[0]
        if not first.fiscal_period.startswith("Q"):
            print(f"\n[skip] invalid transcript fiscal period: {transcript_path}", flush=True)
            continue
        period = DocumentMeta(
            ticker=first.ticker,
            company_name=first.company_name or first.ticker,
            fiscal_year=first.fiscal_year,
            fiscal_quarter=int(first.fiscal_period[1:]),
        )
        label = f"{period.ticker} FY{period.fiscal_year} Q{period.fiscal_quarter}"
        out = claims_path(
            period.ticker,
            period.fiscal_year,
            period.fiscal_quarter,
        )
        print(
            f"\n[{file_idx}/{len(transcript_files)}] {label}",
            flush=True,
        )
        print(f"  source: {transcript_path}", flush=True)

        if out.exists() and not args.force:
            print(f"  skip: claims already exist at {out}", flush=True)
            continue

        executive_chunks = [c for c in chunks if is_executive_chunk(c)]
        executive_text = executive_source_text(chunks)
        if not executive_text:
            print("  skip: no CEO/CFO (or similar) executive turns found", flush=True)
            continue

        early = [
            f"{c.speaker_name} ({c.speaker_role or '?'})"
            for c in executive_chunks[:3]
        ]
        print(
            f"  executive turns: {len(executive_chunks)} "
            f"({len(executive_text):,} chars to LLM)",
            flush=True,
        )
        if early:
            print(f"  early speakers: {', '.join(early)}", flush=True)
        print("  calling LLM for structured claims …", flush=True)
        claims, model = extract_claims(
            period.company_name,
            executive_text,
            max_claims=args.n,
        )
        saved = save_claims(period, claims, model_used=model)
        print(
            f"  wrote {len(saved.claims)} claims via {model}: {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
