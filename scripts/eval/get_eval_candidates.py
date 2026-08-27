#!/usr/bin/env python3
"""Get labeled eval-candidate claims under ``data/eval/claims/``.

Modes
-----
``create`` (default)
    Read extracted claims from ``data/claims/`` (≥6 per period) and write a
    labeled 8-claim set when ``data/eval/claims/`` has no file yet (or
    ``--force``):

    - claims 1–4: copied as-is → intended_label=Consistent
    - claims 5–6: financial number slightly corrupted → Contradictory
    - claim 7-8: LLM rewrite grounded in the transcript → Unverifiable

    Each claim gets ``claim_id``, ``intended_label``, and
    ``is_golden_claim=false``.

``modify``
    Read existing files under ``data/eval/claims/``. Keep claims with
    ``is_golden_claim=true``. Rewrite only ``is_golden_claim=false`` slots
    using the same 3/2/1 rules above, without duplicating any golden claim
    text. Still uses ``data/claims/`` as the source pool for regenerations.

Examples::

    python scripts/eval/get_eval_candidates.py
    python scripts/eval/get_eval_candidates.py --mode create --ticker AAPL --year 2025 --quarter Q1
    python scripts/eval/get_eval_candidates.py --mode modify --ticker AAPL --year 2025 --quarter Q1
    python scripts/eval/get_eval_candidates.py --mode create --force

Prerequisite::

    python scripts/extract_claims.py --n 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.claims import load_saved_claims  # noqa: E402
from crosscheck.analysis.golden_claims import (  # noqa: E402
    build_golden_claim_set,
    load_eval_claims,
    modify_eval_claim_set,
    save_eval_claims,
)
from crosscheck.chunking.pipeline import resolve_transcript_path  # noqa: E402
from crosscheck.chunking.store import iter_company_chunk_files, load_chunks_jsonl  # noqa: E402
from crosscheck.config import (  # noqa: E402
    CLAIMS_DIR,
    EVAL_CLAIMS_DIR,
    claims_path,
    eval_claims_path,
    get_llm_profile,
    resolve_llm_models,
)
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    SavedTranscriptClaims,
    as_fiscal_quarter,
)


def _parse_tickers(value: str | None) -> set[str] | None:
    if value is None:
        return None
    tickers = {part.strip().upper() for part in value.split(",") if part.strip()}
    return tickers or None


def _discover_claim_files(
    root: Path,
    *,
    wanted: set[str] | None,
    year_set: set[int] | None,
    quarter_set: set[str] | None,
) -> list[Path]:
    """Return claims paths under ``root`` matching filters."""
    if not root.is_dir():
        return []

    pattern = "*_claims.jsonl"
    files: list[Path] = []
    for path in sorted(root.rglob(pattern)):
        try:
            year = int(path.parent.parent.name)
            ticker = path.parent.name.upper()
        except (ValueError, IndexError):
            continue
        # Filename: TICKER_FYyyyy_Qq_claims.jsonl
        parts = path.stem.split("_")
        if len(parts) < 4:
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
        files.append(path)
    # Legacy nested JSON under data/claims when JSONL is missing.
    if root == CLAIMS_DIR:
        for path in sorted(root.rglob("*_claims.json")):
            jsonl = path.with_suffix(".jsonl")
            if jsonl.exists():
                continue
            try:
                year = int(path.parent.parent.name)
                ticker = path.parent.name.upper()
            except (ValueError, IndexError):
                continue
            parts = path.stem.split("_")
            if len(parts) < 4:
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
            files.append(path)
    return files


def _load_transcript_text(period: DocumentMeta) -> tuple[str, str]:
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

    for chunk_path in iter_company_chunk_files():
        if not chunk_path.name.endswith("_transcript.jsonl"):
            continue
        if chunk_path.parent.name.upper() != period.ticker.upper():
            continue
        chunks = load_chunks_jsonl(chunk_path)
        if not chunks:
            continue
        first = chunks[0]
        if first.fiscal_year != period.fiscal_year:
            continue
        if as_fiscal_quarter(first.fiscal_period) != as_fiscal_quarter(
            period.fiscal_quarter
        ):
            continue
        joined = "\n\n".join(c.text.strip() for c in chunks if c.text.strip()).strip()
        if joined:
            return joined, f"concatenated {len(chunks)} transcript chunks"
    return "", ""


def _load_source_claims(period: DocumentMeta) -> SavedTranscriptClaims:
    path = claims_path(period.ticker, period.fiscal_year, period.fiscal_quarter)
    legacy = path.with_suffix(".json")
    if not path.exists() and not legacy.exists():
        raise FileNotFoundError(
            f"Source claims missing: {path}. Run: "
            f"python scripts/extract_claims.py --ticker {period.ticker} --n 6"
        )
    return load_saved_claims(period)


def _print_claims(saved: SavedTranscriptClaims) -> None:
    for i, claim in enumerate(saved.claims, start=1):
        preview = claim.claim[:88] + ("…" if len(claim.claim) > 88 else "")
        golden = "golden" if claim.is_golden_claim else "draft"
        print(
            f"    {i}. [{claim.intended_label}/{golden}] "
            f"[{claim.claim_id}] [{claim.speaker}] {preview}",
            flush=True,
        )


def _run_create(
    *,
    wanted: set[str] | None,
    year_set: set[int] | None,
    quarter_set: set[str] | None,
    force: bool,
) -> int:
    claim_files = _discover_claim_files(
        CLAIMS_DIR,
        wanted=wanted,
        year_set=year_set,
        quarter_set=quarter_set,
    )
    if not claim_files:
        print("No matching claim files under data/claims.")
        return 1

    print(
        f"Golden claims [create]: profile={get_llm_profile()} "
        f"candidates={len(claim_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(resolve_llm_models())}", flush=True)

    file_idx = 0
    for source_path in claim_files:
        parts = source_path.stem.split("_")
        ticker = parts[0]
        year_s = parts[1]
        if year_s.upper().startswith("FY"):
            year_s = year_s[2:]
        fiscal_year = int(year_s)
        fiscal_quarter = as_fiscal_quarter(parts[2])
        period_tmp = DocumentMeta(
            ticker=ticker,
            company_name=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        source = load_saved_claims(period_tmp)
        period = DocumentMeta(
            ticker=source.ticker,
            company_name=source.company_name,
            fiscal_year=source.fiscal_year,
            fiscal_quarter=source.fiscal_quarter,
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        out = eval_claims_path(
            period.ticker,
            period.fiscal_year,
            period.fiscal_quarter,
        )
        file_idx += 1
        print(f"\n[{file_idx}] {label}", flush=True)

        if out.exists() and not force:
            print(
                f"  skip: eval claims already exist at {out} "
                "(use --force or --mode modify)",
                flush=True,
            )
            continue

        if len(source.claims) < 6:
            print(
                f"  skip: need ≥6 source claims, found {len(source.claims)} "
                f"({source_path})",
                flush=True,
            )
            continue

        transcript_text, source_label = _load_transcript_text(period)
        if not transcript_text:
            print("  skip: empty transcript text", flush=True)
            continue

        print(f"  source claims: {source_path}", flush=True)
        print(f"  transcript: {source_label}", flush=True)
        print(
            "  building 4 Consistent + 2 Contradictory + 2 Unverifiable …",
            flush=True,
        )

        try:
            golden, model = build_golden_claim_set(
                source,
                transcript_text=transcript_text,
            )
        except ValueError as exc:
            print(f"  skip: {exc}", flush=True)
            continue

        saved, written = save_eval_claims(period, golden, model_used=model)
        print(f"  wrote {len(saved.claims)} claims via {model}: {written}", flush=True)
        _print_claims(saved)

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
        EVAL_CLAIMS_DIR,
        wanted=wanted,
        year_set=year_set,
        quarter_set=quarter_set,
    )
    if not claim_files:
        print(
            f"No matching eval claim files under {EVAL_CLAIMS_DIR}. "
            "Run --mode create first."
        )
        return 1

    print(
        f"Golden claims [modify]: profile={get_llm_profile()} "
        f"candidates={len(claim_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(resolve_llm_models())}", flush=True)

    file_idx = 0
    for eval_path in claim_files:
        # Eval-claims are stored as JSONL (one claim per line).
        # Load & re-hydrate into `SavedTranscriptClaims`.
        # We also derive the period metadata from the path.
        # Note: `eval_path` itself is not used for metadata; it is only the content.
        # Load uses `period` values from the file currently being processed.
        # To construct `period`, parse the file name via existing JSONL loader below.
        # We can parse from the filename structure:
        #   TICKER_FYyyyy_Qn_claims.jsonl
        parts = eval_path.stem.split("_")
        ticker = parts[0]
        year_s = parts[1]
        year_s = year_s.upper()
        if year_s.startswith("FY"):
            year_s = year_s[2:]
        fiscal_year = int(year_s)
        fiscal_quarter = as_fiscal_quarter(parts[2])
        period = DocumentMeta(
            ticker=ticker,
            company_name=ticker,  # overwritten after load if needed
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        # company_name is required by the model, but downstream modify uses only claims.
        eval_claims = load_eval_claims(period)
        period = DocumentMeta(
            ticker=eval_claims.ticker,
            company_name=eval_claims.company_name,
            fiscal_year=eval_claims.fiscal_year,
            fiscal_quarter=eval_claims.fiscal_quarter,
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        file_idx += 1
        print(f"\n[{file_idx}] {label}", flush=True)
        print(f"  eval claims: {eval_path}", flush=True)

        pending = [c for c in eval_claims.claims if c.is_golden_claim is not True]
        if not pending:
            print("  skip: all claims already is_golden_claim=true", flush=True)
            continue

        try:
            source = _load_source_claims(period)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", flush=True)
            continue

        transcript_text, source_label = _load_transcript_text(period)
        if not transcript_text:
            print("  skip: empty transcript text", flush=True)
            continue

        print(f"  source claims: {claims_path(period.ticker, period.fiscal_year, period.fiscal_quarter)}", flush=True)
        print(f"  transcript: {source_label}", flush=True)
        print(
            f"  rewriting {len(pending)} non-golden claim(s); "
            f"keeping {len(eval_claims.claims) - len(pending)} golden …",
            flush=True,
        )

        try:
            updated, model = modify_eval_claim_set(
                eval_claims,
                source,
                transcript_text=transcript_text,
            )
        except ValueError as exc:
            print(f"  skip: {exc}", flush=True)
            continue

        saved, written = save_eval_claims(period, updated, model_used=model)
        print(f"  wrote {len(saved.claims)} claims via {model}: {written}", flush=True)
        _print_claims(saved)

    if file_idx == 0:
        print("No periods matched --ticker / --year / --quarter filters.")
        return 1
    return 0


def main() -> None:
    """Build (create) or refresh (modify) labeled eval claims."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("create", "modify"),
        default="create",
        help=(
            "create: build eval claims from data/claims when missing; "
            "modify: rewrite only is_golden_claim=false slots in data/eval/claims "
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
        "--force",
        action="store_true",
        help="create mode only: replace eval claim files that already exist.",
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
