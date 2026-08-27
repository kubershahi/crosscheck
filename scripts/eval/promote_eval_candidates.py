#!/usr/bin/env python3
"""Promote matching NLI candidates into per-period files under ``data/eval/golden/``.

Reads per-period candidate files under ``data/eval/candidates/`` and copies rows where
``intended_label == nli_label`` into the curated golden set with schema::

    claim_id, ticker, company_name, fiscal_year, fiscal_quarter, speaker,
    claim, expected_nli_label, is_in_filing, is_in_table,
    ground_truth_reference

For Consistent / Contradictory claims, ``ground_truth_reference`` is written
via an LLM using NLI reasoning + retrieved (esp. matched) passages.
Unverifiable claims get a fixed reference and ``is_in_filing=false``.

Promoted claim_ids are marked ``is_golden_claim=true`` under
``data/eval/claims/``.

On HTTP 429 / rate-limit errors, sleeps 62s and retries (script-level, on top
of the shared LLM client's own backoff).

Examples::

    python scripts/eval/promote_eval_candidates.py
    python scripts/eval/promote_eval_candidates.py --ticker AAPL --year 2025 --quarter Q1
    python scripts/eval/promote_eval_candidates.py --ticker AAPL --year 2025 --quarter Q1 --force

Prerequisite::

    python scripts/eval/verify_eval_candidates.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.llm import complete_structured  # noqa: E402
from crosscheck.analysis.prompts import (  # noqa: E402
    GROUND_TRUTH_REFERENCE_SYSTEM,
    ground_truth_reference_user,
)
from crosscheck.config import (  # noqa: E402
    EVAL_DIR,
    eval_claims_path,
    get_llm_profile,
    resolve_llm_models,
)
from crosscheck.io.jsonl import (  # noqa: E402
    append_json_object,
    drop_json_object_ids,
    iter_json_objects,
    write_json_objects,
)
from crosscheck.models import (  # noqa: E402
    GoldenClaim,
    GroundTruthReference,
    SavedTranscriptClaims,
    as_fiscal_quarter,
    quarter_number,
)

CANDIDATES_ROOT = EVAL_DIR / "candidates"
GOLDEN_ROOT = EVAL_DIR / "golden"

UNVERIFIABLE_REFERENCE = (
    "Claim is unverifiable because it has no ground-truth reference in the filings."
)

# Script-level backoff when the LLM client still surfaces a 429 after its own retries.
_RATE_LIMIT_SLEEP_SECONDS = 62
_RATE_LIMIT_MAX_RETRIES = 8
_RATE_LIMIT_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "quota_exceeded",
    "too many requests",
)


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for row in iter_json_objects(path):
        cid = row.get("claim_id")
        if isinstance(cid, str) and cid:
            done.add(cid)
    return done


def _discover_candidate_files(
    *,
    root: Path,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> list[Path]:
    """Return per-period candidate JSONL paths under ``root`` matching filters."""
    if not root.is_dir():
        return []

    if year is not None:
        year_roots = [root / str(year)]
    else:
        year_roots = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())

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
            files.extend(sorted(ticker_dir.glob("*_candidates.jsonl")))

    if quarter is not None:
        wanted = as_fiscal_quarter(quarter)
        filtered: list[Path] = []
        for path in files:
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                q = as_fiscal_quarter(parts[2])
            except ValueError:
                continue
            if q == wanted:
                filtered.append(path)
        files = filtered

    return files


def _golden_period_path(row: dict, *, out_root: Path) -> Path:
    """Per-period golden output path for one candidate row."""
    ticker = str(row["ticker"]).upper()
    year = int(row["fiscal_year"])
    quarter = as_fiscal_quarter(row["fiscal_quarter"])
    return (
        out_root
        / str(year)
        / ticker
        / f"{ticker}_FY{year}_{quarter}_golden_claims.jsonl"
    )


def _append_golden(path: Path, row: GoldenClaim) -> None:
    append_json_object(path, row.model_dump(mode="json"))


def _drop_claim_ids(path: Path, drop: set[str]) -> int:
    return drop_json_object_ids(path, drop)


def _matched_passages(row: dict) -> list[dict]:
    """Return all matched passages by indices, falling back to single index or chunk_id."""
    retrieved = row.get("retrieved") or []
    if not isinstance(retrieved, list) or not retrieved:
        return []

    indices = row.get("matched_passage_indices")
    if isinstance(indices, list) and indices:
        results = []
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(retrieved):
                p = retrieved[idx - 1]
                if isinstance(p, dict):
                    results.append(p)
        if results:
            return results

    idx = row.get("matched_passage_index") or 0
    if isinstance(idx, int) and 1 <= idx <= len(retrieved):
        passage = retrieved[idx - 1]
        if isinstance(passage, dict):
            return [passage]

    chunk_id = row.get("matched_chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        for passage in retrieved:
            if isinstance(passage, dict) and passage.get("chunk_id") == chunk_id:
                return [passage]
    return []


def _resolve_matched_indices(row: dict) -> list[int]:
    """Get matched_passage_indices from the row, falling back to single index."""
    indices = row.get("matched_passage_indices")
    if isinstance(indices, list) and indices:
        return [i for i in indices if isinstance(i, int) and i >= 1]
    idx = row.get("matched_passage_index") or 0
    if isinstance(idx, int) and idx >= 1:
        return [idx]
    return []


def _is_rate_limit(exc: BaseException) -> bool:
    """True for HTTP 429 / quota / resource_exhausted style errors."""
    msg = str(exc).lower()
    return any(k in msg for k in _RATE_LIMIT_KEYWORDS)


def _generate_ground_truth_reference(row: dict, *, expected_label: str) -> tuple[str, str]:
    """Return ``(reference, model_used)``.

    On 429 / rate-limit errors, sleeps ``_RATE_LIMIT_SLEEP_SECONDS`` and retries
    (in addition to the shared LLM client's own rate-limit backoff).
    """
    if expected_label == "Unverifiable":
        return UNVERIFIABLE_REFERENCE, "rule"

    retrieved = row.get("retrieved") or []
    passages: list[dict] = [p for p in retrieved if isinstance(p, dict)]
    matched_indices = _resolve_matched_indices(row)
    messages = [
        {"role": "system", "content": GROUND_TRUTH_REFERENCE_SYSTEM},
        {
            "role": "user",
            "content": ground_truth_reference_user(
                claim=str(row.get("claim") or ""),
                speaker=str(row.get("speaker") or ""),
                ticker=str(row.get("ticker") or ""),
                company_name=str(row.get("company_name") or ""),
                fiscal_year=int(row.get("fiscal_year")),
                fiscal_quarter=str(row.get("fiscal_quarter") or ""),
                expected_nli_label=expected_label,
                nli_reasoning=str(row.get("nli_reasoning") or ""),
                matched_passage_indices=matched_indices,
                passages=passages,
            ),
        },
    ]

    rate_limit_retries = 0
    while True:
        try:
            result, model = complete_structured(
                response_model=GroundTruthReference,
                messages=messages,
            )
            ref = result.ground_truth_reference.strip()
            if not ref:
                raise ValueError("empty ground_truth_reference from LLM")
            return ref, model
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            rate_limit_retries += 1
            if rate_limit_retries > _RATE_LIMIT_MAX_RETRIES:
                raise
            print(
                f"  rate limit (429); sleep {_RATE_LIMIT_SLEEP_SECONDS}s "
                f"({rate_limit_retries}/{_RATE_LIMIT_MAX_RETRIES}) then retry …",
                flush=True,
            )
            time.sleep(_RATE_LIMIT_SLEEP_SECONDS)


def _mark_eval_claim_golden(claim_id: str) -> Path | None:
    """Set ``is_golden_claim=true`` for ``claim_id`` under data/eval/claims."""
    parts = claim_id.split("_")
    if len(parts) < 4:
        print(f"  warn: cannot parse claim_id={claim_id!r}", flush=True)
        return None
    ticker, year_s, quarter = parts[0], parts[1], parts[2]
    try:
        year = int(year_s)
        quarter = as_fiscal_quarter(quarter)
    except ValueError as exc:
        print(f"  warn: bad claim_id={claim_id!r}: {exc}", flush=True)
        return None

    path = eval_claims_path(ticker, year, quarter)
    if not path.exists():
        print(f"  warn: eval claims missing for {claim_id}: {path}", flush=True)
        return None

    rows = list(iter_json_objects(path))
    if not rows:
        print(f"  warn: eval claims empty for {claim_id}: {path}", flush=True)
        return None

    found = False
    updated = False
    out: list[dict] = []
    for obj in rows:
        if str(obj.get("claim_id") or "") == claim_id:
            found = True
            if obj.get("is_golden_claim") is not True:
                obj = {**obj, "is_golden_claim": True}
                updated = True
        out.append(obj)

    if not found:
        print(f"  warn: claim_id {claim_id} not found in {path}", flush=True)
        return None

    if updated:
        write_json_objects(path, out)
    return path


def _candidate_in_scope(
    row: dict,
    *,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> bool:
    if ticker and str(row.get("ticker") or "").upper() != ticker:
        return False
    if year is not None and row.get("fiscal_year") != year:
        return False
    if quarter is not None:
        try:
            if as_fiscal_quarter(row.get("fiscal_quarter")) != quarter:
                return False
        except ValueError:
            return False
    return True


def _build_golden_row(row: dict) -> tuple[GoldenClaim, str]:
    expected = str(row.get("nli_label") or "")
    if expected not in {"Consistent", "Contradictory", "Unverifiable"}:
        raise ValueError(f"bad nli_label={expected!r}")

    matched_list = _matched_passages(row)
    if expected == "Unverifiable":
        is_in_filing = False
        is_in_table = False
    else:
        is_in_filing = len(matched_list) > 0
        is_in_table = any(bool(p.get("is_table")) for p in matched_list)

    reference, model = _generate_ground_truth_reference(row, expected_label=expected)
    golden = GoldenClaim(
        claim_id=str(row["claim_id"]),
        ticker=str(row["ticker"]).upper(),
        company_name=str(row.get("company_name") or row["ticker"]),
        fiscal_year=int(row["fiscal_year"]),
        fiscal_quarter=as_fiscal_quarter(row["fiscal_quarter"]),
        speaker=str(row.get("speaker") or ""),
        claim=str(row.get("claim") or ""),
        expected_nli_label=expected,  # type: ignore[arg-type]
        is_in_filing=is_in_filing,
        is_in_table=is_in_table,
        ground_truth_reference=reference,
    )
    return golden, model


def main() -> None:
    """Promote label-matching candidates into the curated golden JSONL."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", help="Optional ticker filter.")
    parser.add_argument("--year", type=int, help="Optional fiscal year filter.")
    parser.add_argument("--quarter", help="Fiscal quarter: Q1–Q4 or 1–4.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop in-scope golden rows and re-promote matching candidates.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CANDIDATES_ROOT,
        help="Candidates root directory (default: data/eval/candidates/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_ROOT,
        help="Golden output root directory (default: data/eval/golden/).",
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

    if args.quarter is not None:
        quarter_number(args.quarter)

    ticker = args.ticker.strip().upper() if args.ticker else None
    quarter = as_fiscal_quarter(args.quarter) if args.quarter else None

    if not args.input.exists():
        print(
            f"No candidates under {args.input}. Run verify_eval_candidates.py first."
        )
        sys.exit(1)

    candidate_files = _discover_candidate_files(
        root=args.input,
        ticker=ticker,
        year=args.year,
        quarter=quarter,
    )
    if not candidate_files:
        print(
            f"No candidate files under {args.input} for the requested scope.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"Promote golden claims: profile={get_llm_profile()} "
        f"candidate_files={len(candidate_files)}",
        flush=True,
    )
    print(f"  model rank: {' → '.join(resolve_llm_models())}", flush=True)
    print(f"  input_root={args.input}", flush=True)
    print(f"  output_root={args.output}", flush=True)

    written = 0
    skipped = 0
    marked = 0
    matching_total = 0

    for file_idx, candidate_path in enumerate(candidate_files, start=1):
        rows = [
            row
            for row in iter_json_objects(candidate_path)
            if _candidate_in_scope(row, ticker=ticker, year=args.year, quarter=quarter)
        ]
        matching = [
            row
            for row in rows
            if row.get("intended_label")
            and row.get("intended_label") == row.get("nli_label")
            and row.get("claim_id")
        ]
        matching_total += len(matching)

        print(
            f"\n[{file_idx}/{len(candidate_files)}] {candidate_path} "
            f"rows={len(rows)} matching={len(matching)}",
            flush=True,
        )
        if not matching:
            continue

        out_path = _golden_period_path(matching[0], out_root=args.output)
        upcoming = {str(row["claim_id"]) for row in matching}
        if args.force:
            removed = _drop_claim_ids(out_path, upcoming)
            print(
                f"  --force: dropped {removed} existing golden row(s) in period file",
                flush=True,
            )
        done = _load_done_ids(out_path)

        for idx, row in enumerate(matching, start=1):
            cid = str(row["claim_id"])
            label = f"{row.get('ticker')} {row.get('fiscal_year')} {row.get('fiscal_quarter')}"
            print(f"  [{idx}/{len(matching)}] {cid}  ({label})", flush=True)
            print(
                f"    intended={row.get('intended_label')} nli={row.get('nli_label')}",
                flush=True,
            )

            if cid in done:
                skipped += 1
                print("    skip: already in period golden file", flush=True)
                continue

            try:
                golden, model = _build_golden_row(row)
            except Exception as exc:
                if _is_rate_limit(exc):
                    print(
                        f"    abort: rate limit persisted after retries: {exc}",
                        flush=True,
                    )
                    sys.exit(1)
                print(f"    skip: failed to build golden row: {exc}", flush=True)
                continue

            _append_golden(out_path, golden)
            done.add(cid)
            written += 1
            print(
                f"    → expected={golden.expected_nli_label} "
                f"is_in_filing={golden.is_in_filing} is_in_table={golden.is_in_table} "
                f"via={model}",
                flush=True,
            )
            print(f"      ref: {golden.ground_truth_reference[:120]}", flush=True)

            path = _mark_eval_claim_golden(cid)
            if path is not None:
                marked += 1
                print(f"    marked is_golden_claim=true in {path}", flush=True)

    print(
        f"\ndone  written={written}  skipped={skipped}  "
        f"matching={matching_total}  marked_eval_claims={marked}  → {args.output}"
    )


if __name__ == "__main__":
    main()
