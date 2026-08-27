#!/usr/bin/env python3
"""Extract and persist fixed transcript claims for one or all companies.

Passes the full cleaned transcript ``.txt`` to the LLM (no speaker-role
filtering). Chunk JSONL is used only to discover company-periods.

Modes
-----
``create`` (default)
    Extract up to ``--n`` claims and write a flattened JSONL file
    (one claim object per line) under ``data/claims/`` with ``claim_id``
    and ``regenerate=false``. If a claims file already exists with fewer
    than ``--n`` claims, append new non-duplicate claims until the count
    matches ``--n``. Skip only when the file already has exactly ``--n``
    claims (unless ``--force``, which replaces the whole file).

``modify``
    Read existing ``data/claims/`` JSONL files. Keep claims with
    ``regenerate=false``. Regenerate only ``regenerate=true`` slots via
    the LLM, without duplicating other claims in the same file. Successful
    regenerations are written back with ``regenerate=false``.

Examples::

    python scripts/extract_claims.py --n 6
    python scripts/extract_claims.py --mode create --ticker AAPL --year 2025 --quarter Q3 --n 6
    python scripts/extract_claims.py --mode create --ticker AAPL --force --n 6
    python scripts/extract_claims.py --mode modify --ticker AAPL --year 2025 --quarter Q3

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

from crosscheck.analysis.claims import (  # noqa: E402
    extract_claims,
    load_saved_claims,
    modify_saved_claims,
    save_claims,
    top_up_claims,
)
from crosscheck.chunking.pipeline import resolve_transcript_path  # noqa: E402
from crosscheck.chunking.store import iter_company_chunk_files, load_chunks_jsonl  # noqa: E402
from crosscheck.config import (  # noqa: E402
    CLAIMS_DIR,
    claims_path,
    get_llm_profile,
    resolve_llm_models,
)
from crosscheck.models import Chunk, DocumentMeta, as_fiscal_quarter  # noqa: E402


def _parse_tickers(value: str | None) -> set[str] | None:
    """Expand ``--ticker AAPL,MSFT`` into a normalized ticker set."""
    if value is None:
        return None
    tickers = {part.strip().upper() for part in value.split(",") if part.strip()}
    return tickers or None


def _load_transcript_text(
    period: DocumentMeta,
    chunks: list[Chunk] | None = None,
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

    if chunks is None:
        chunks = []
        for chunk_path in iter_company_chunk_files():
            if not chunk_path.name.endswith("_transcript.jsonl"):
                continue
            if chunk_path.parent.name.upper() != period.ticker.upper():
                continue
            loaded = load_chunks_jsonl(chunk_path)
            if not loaded:
                continue
            first = loaded[0]
            if first.fiscal_year != period.fiscal_year:
                continue
            if as_fiscal_quarter(first.fiscal_period) != as_fiscal_quarter(
                period.fiscal_quarter
            ):
                continue
            chunks = loaded
            break

    joined = "\n\n".join(c.text.strip() for c in chunks if c.text.strip()).strip()
    return joined, f"concatenated {len(chunks)} transcript chunks"


def _discover_claim_files(
    *,
    wanted: set[str] | None,
    year_set: set[int] | None,
    quarter_set: set[str] | None,
) -> list[Path]:
    """Return claim JSONL (or legacy JSON) paths under ``data/claims``."""
    if not CLAIMS_DIR.is_dir():
        return []

    files: list[Path] = []
    for path in sorted(CLAIMS_DIR.rglob("*_claims.jsonl")) + sorted(
        CLAIMS_DIR.rglob("*_claims.json")
    ):
        try:
            year = int(path.parent.parent.name)
            ticker = path.parent.name.upper()
        except (ValueError, IndexError):
            continue
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            quarter = as_fiscal_quarter(parts[2])
        except ValueError:
            continue
        if wanted is not None and ticker not in wanted:
            continue
        if year_set is not None and year not in year_set:
            continue
        if quarter_set is not None and quarter not in quarter_set:
            continue
        # Prefer .jsonl when both exist for the same period.
        if path.suffix == ".json":
            jsonl = path.with_suffix(".jsonl")
            if jsonl.exists():
                continue
        files.append(path)
    return files


def _print_claims(claims: list) -> None:
    for claim in claims:
        preview = claim.claim[:88] + ("…" if len(claim.claim) > 88 else "")
        flag = "regen" if claim.regenerate else "keep"
        print(
            f"    [{claim.claim_id or '?'}] [{flag}] [{claim.speaker}] {preview}",
            flush=True,
        )


def _run_create(
    *,
    wanted: set[str] | None,
    year_set: set[int] | None,
    quarter_set: set[str] | None,
    max_claims: int,
    force: bool,
) -> int:
    transcript_files = [
        path
        for path in iter_company_chunk_files()
        if path.name.endswith("_transcript.jsonl")
        and (wanted is None or path.parent.name.upper() in wanted)
    ]
    if not transcript_files:
        print("No matching transcript chunk files under data/chunks.")
        return 1

    print(
        f"Claim extraction [create]: profile={get_llm_profile()} "
        f"max_claims={max_claims} candidates={len(transcript_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(resolve_llm_models())}", flush=True)

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
        legacy = out.with_suffix(".json")
        file_idx += 1
        print(f"\n[{file_idx}] {label}", flush=True)

        if (out.exists() or legacy.exists()) and not force:
            existing_path = out if out.exists() else legacy
            try:
                existing = load_saved_claims(period)
            except Exception as exc:
                print(f"  skip: cannot read existing claims ({exc})", flush=True)
                continue
            # Prefer company_name from the saved file when topping up.
            period = DocumentMeta(
                ticker=existing.ticker,
                company_name=existing.company_name or period.company_name,
                fiscal_year=existing.fiscal_year,
                fiscal_quarter=existing.fiscal_quarter,
            )
            have = len(existing.claims)
            if have == max_claims:
                print(
                    f"  skip: already have {have}/{max_claims} claims at "
                    f"{existing_path}",
                    flush=True,
                )
                continue
            if have > max_claims:
                print(
                    f"  skip: already have {have} claims (> --n {max_claims}) at "
                    f"{existing_path}",
                    flush=True,
                )
                continue

            transcript_text, source = _load_transcript_text(period, chunks)
            if not transcript_text:
                print("  skip: empty transcript text", flush=True)
                continue

            print(f"  source: {source}", flush=True)
            print(
                f"  existing claims: {have}/{max_claims} at {existing_path}; "
                f"topping up …",
                flush=True,
            )
            try:
                claims, model = top_up_claims(
                    existing,
                    transcript_text=transcript_text,
                    target_count=max_claims,
                )
            except ValueError as exc:
                print(f"  skip: {exc}", flush=True)
                continue
            saved, written = save_claims(period, claims, model_used=model)
            print(
                f"  wrote {len(saved.claims)} claims via {model}: {written}",
                flush=True,
            )
            _print_claims(saved.claims)
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
            max_claims=max_claims,
        )
        saved, written = save_claims(period, claims, model_used=model)
        print(
            f"  wrote {len(saved.claims)} claims via {model}: {written}",
            flush=True,
        )
        _print_claims(saved.claims)

    if file_idx == 0:
        print("No periods matched --ticker / --year / --quarter filters.")
        return 1
    return 0


def _run_modify(
    *,
    wanted: set[str] | None,
    year_set: set[int] | None,
    quarter_set: set[str] | None,
) -> int:
    claim_files = _discover_claim_files(
        wanted=wanted,
        year_set=year_set,
        quarter_set=quarter_set,
    )
    if not claim_files:
        print(
            f"No matching claim files under {CLAIMS_DIR}. "
            "Run --mode create first."
        )
        return 1

    print(
        f"Claim extraction [modify]: profile={get_llm_profile()} "
        f"candidates={len(claim_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(resolve_llm_models())}", flush=True)

    file_idx = 0
    for claim_path in claim_files:
        parts = claim_path.stem.split("_")
        ticker = parts[0]
        year_s = parts[1]
        if year_s.upper().startswith("FY"):
            year_s = year_s[2:]
        fiscal_year = int(year_s)
        fiscal_quarter = as_fiscal_quarter(parts[2])
        period = DocumentMeta(
            ticker=ticker,
            company_name=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        saved = load_saved_claims(period)
        period = DocumentMeta(
            ticker=saved.ticker,
            company_name=saved.company_name,
            fiscal_year=saved.fiscal_year,
            fiscal_quarter=saved.fiscal_quarter,
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        file_idx += 1
        print(f"\n[{file_idx}] {label}", flush=True)
        print(f"  claims: {claim_path}", flush=True)

        pending = [c for c in saved.claims if c.regenerate]
        if not pending:
            print("  skip: no claims with regenerate=true", flush=True)
            continue

        transcript_text, source = _load_transcript_text(period)
        if not transcript_text:
            print("  skip: empty transcript text", flush=True)
            continue

        print(f"  transcript: {source}", flush=True)
        print(
            f"  regenerating {len(pending)} claim(s); "
            f"keeping {len(saved.claims) - len(pending)} …",
            flush=True,
        )

        try:
            updated, model = modify_saved_claims(
                saved,
                transcript_text=transcript_text,
            )
        except ValueError as exc:
            print(f"  skip: {exc}", flush=True)
            continue

        written_saved, written = save_claims(period, updated, model_used=model)
        print(
            f"  wrote {len(written_saved.claims)} claims via {model}: {written}",
            flush=True,
        )
        _print_claims(written_saved.claims)

    if file_idx == 0:
        print("No periods matched --ticker / --year / --quarter filters.")
        return 1
    return 0


def main() -> None:
    """Extract (create) or refresh (modify) fixed claim sets."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("create", "modify"),
        default="create",
        help=(
            "create: extract claims when missing, or top up when count < --n "
            "(--force replaces); "
            "modify: regenerate only regenerate=true slots "
            "(default: create)."
        ),
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
        choices=range(1, 16),
        metavar="1-15",
        help="create mode: maximum claims per company-period (default: 5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "create mode only: replace claim files that already exist "
            "(instead of topping up when count < --n)."
        ),
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

    if args.force and args.mode != "create":
        parser.error("--force only applies to --mode create")

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

    if args.mode == "create":
        code = _run_create(
            wanted=wanted,
            year_set=year_set,
            quarter_set=quarter_set,
            max_claims=args.n,
            force=args.force,
        )
    else:
        code = _run_modify(
            wanted=wanted,
            year_set=year_set,
            quarter_set=quarter_set,
        )
    sys.exit(code)


if __name__ == "__main__":
    main()
