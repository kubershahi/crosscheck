#!/usr/bin/env python3
"""Extract and persist fixed transcript claims for one or all companies.

Passes the full cleaned transcript ``.txt`` to the LLM (no speaker-role
filtering). Chunk JSONL is used only to discover company-periods.

Examples::

    # Every transcript period discovered under data/chunks
    python scripts/extract_claims.py --n 5

    # By ticker
    python scripts/extract_claims.py --ticker AAPL --n 5
    python scripts/extract_claims.py --ticker AAPL,MSFT --n 5

    # By year / quarter (alone or combined)
    python scripts/extract_claims.py --year 2025 --n 5
    python scripts/extract_claims.py --quarter Q2 --n 5
    python scripts/extract_claims.py --year 2025 --quarter Q1 --n 5
    python scripts/extract_claims.py --ticker AAPL --year 2025 --quarter Q3 --n 5

    # Replace claims that already exist
    python scripts/extract_claims.py --force --n 5
    python scripts/extract_claims.py --ticker AAPL --year 2025 --quarter Q2 --force --n 5

Prerequisite: run ``python scripts/fetch_corpus.py`` (and ideally
``python scripts/build_chunks.py`` so periods are discoverable).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.claims import extract_claims, save_claims  # noqa: E402
from crosscheck.chunking.pipeline import resolve_transcript_path  # noqa: E402
from crosscheck.chunking.store import iter_company_chunk_files, load_chunks_jsonl  # noqa: E402
from crosscheck.config import claims_path, get_llm_profile, resolve_llm_models  # noqa: E402
from crosscheck.models import Chunk, DocumentMeta, as_fiscal_quarter  # noqa: E402


def _parse_tickers(value: str | None) -> set[str] | None:
    """Expand ``--ticker AAPL,MSFT`` into a normalized ticker set."""
    if value is None:
        return None
    tickers = {part.strip().upper() for part in value.split(",") if part.strip()}
    return tickers or None


def _load_transcript_text(
    period: DocumentMeta,
    chunks: list[Chunk],
) -> tuple[str, str]:
    """Return ``(text, source_label)`` preferring raw ``.txt``, else chunk join."""
    txt_path = resolve_transcript_path(
        period.ticker,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
    )
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text, str(txt_path)
    joined = "\n\n".join(c.text.strip() for c in chunks if c.text.strip()).strip()
    return joined, f"concatenated {len(chunks)} transcript chunks"


def main() -> None:
    """Extract stable claim sets from full transcripts."""
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
        "--n",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Maximum claims per company-period (default: 5).",
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

    year_set = set(args.years) if args.years else None
    quarter_set: set[str] | None = None
    if args.quarters:
        try:
            quarter_set = {as_fiscal_quarter(q) for q in args.quarters}
        except ValueError as exc:
            parser.error(str(exc))

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
        f"max_claims={args.n} candidates={len(transcript_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(models)}", flush=True)

    file_idx = 0
    for transcript_path in transcript_files:
        chunks = load_chunks_jsonl(transcript_path)
        if not chunks:
            print(f"\n[skip] empty transcript chunks: {transcript_path}", flush=True)
            continue
        first = chunks[0]
        if not first.fiscal_period.startswith("Q"):
            print(
                f"\n[skip] invalid transcript fiscal period: {transcript_path}",
                flush=True,
            )
            continue
        if year_set is not None and first.fiscal_year not in year_set:
            continue
        if quarter_set is not None and as_fiscal_quarter(first.fiscal_period) not in quarter_set:
            continue

        period = DocumentMeta(
            ticker=first.ticker,
            company_name=first.company_name or first.ticker,
            fiscal_year=first.fiscal_year,
            fiscal_quarter=first.fiscal_period,
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        out = claims_path(
            period.ticker,
            period.fiscal_year,
            period.fiscal_quarter,
        )
        file_idx += 1
        print(f"\n[{file_idx}] {label}", flush=True)

        if out.exists() and not args.force:
            print(f"  skip: claims already exist at {out}", flush=True)
            continue

        transcript_text, source = _load_transcript_text(period, chunks)
        if not transcript_text:
            print("  skip: empty transcript text", flush=True)
            continue

        print(f"  source: {source}", flush=True)
        print(
            f"  transcript: {len(transcript_text):,} chars to LLM",
            flush=True,
        )
        print("  calling LLM for structured claims …", flush=True)
        claims, model = extract_claims(
            period.company_name,
            transcript_text,
            max_claims=args.n,
        )
        saved = save_claims(period, claims, model_used=model)
        print(
            f"  wrote {len(saved.claims)} claims via {model}: {out}",
            flush=True,
        )

    if file_idx == 0:
        print("No periods matched --ticker / --year / --quarter filters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
