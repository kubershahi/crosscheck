#!/usr/bin/env python3
"""Verify eval-candidate claims via the same retrieve + NLI path as run_nli.

Reads labeled claims from ``data/eval/claims/`` (produced by
``get_eval_candidates.py``), then writes scored candidates per ticker / year /
quarter.

Writes one JSON object per claim (fsync) to per-period files under
``data/eval/candidates/{fiscal_year}/{TICKER}/`` so
progress survives interruptions.

Re-runs skip a period that already has ≥8 candidates. Periods with fewer than
8 are filled from remaining claims (skipping claim_ids already present).
``--force`` drops in-scope candidates first, then regenerates up to 8/period.

Uses the development LLM profile. Rate limits (HTTP 429) sleep 62s in the
shared LLM client and retry — no proactive inter-period gap.

Does **not** touch ``data/claims/`` or ``run_nli.py`` outputs; those stay on
the pure extraction / report path.

Modes::

    create (default): existing behavior; process the period claim set.
    modify: only process claims where ``is_golden_claim`` is false.

Examples::

    python scripts/eval/verify_eval_candidates.py
    python scripts/eval/verify_eval_candidates.py --mode create --ticker AAPL --year 2025 --quarter Q1 --force
    python scripts/eval/verify_eval_candidates.py --mode modify --ticker AAPL --year 2025 --quarter Q1


Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    python scripts/extract_claims.py --n 6
    python scripts/eval/get_eval_candidates.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.nli import classify_claim, extract_passage_indices  # noqa: E402
from crosscheck.analysis.claims import load_saved_claims  # noqa: E402
from crosscheck.analysis.golden_claims import load_eval_claims  # noqa: E402
from crosscheck.config import EVAL_CLAIMS_DIR, EVAL_DIR  # noqa: E402
from crosscheck.io.jsonl import (  # noqa: E402
    append_json_object,
    drop_json_object_ids,
    iter_json_objects,
)
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    IndexedChunk,
    as_fiscal_quarter,
    make_claim_id,
    quarter_number,
)
from crosscheck.retrieval.embeddings import load_embedding_model  # noqa: E402
from crosscheck.retrieval.index import load_filings_index, retrieve_claim_passages  # noqa: E402
from crosscheck.retrieval.query_processor import (  # noqa: E402
    TemporalScope,
    fy_annual_retrieval_query,
    prepare_claim_query,
    q3_ytd_retrieval_query,
    retrieval_path_log,
)
from crosscheck.retrieval.rerank import load_reranker  # noqa: E402

PER_PERIOD_TARGET = 8
CANDIDATES_ROOT = EVAL_DIR / "candidates"

PeriodKey = tuple[str, int, str]  # ticker, fiscal_year, fiscal_quarter


def _discover_eval_claim_files(
    *,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> list[Path]:
    """Return claim JSON paths under ``data/eval/claims`` matching filters."""
    if not EVAL_CLAIMS_DIR.is_dir():
        return []

    if year is not None:
        year_roots = [EVAL_CLAIMS_DIR / str(year)]
    else:
        year_roots = sorted(
            p for p in EVAL_CLAIMS_DIR.iterdir() if p.is_dir() and p.name.isdigit()
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
            files.extend(sorted(ticker_dir.glob("*_claims.jsonl")))

    if quarter is not None:
        wanted = as_fiscal_quarter(quarter)
        filtered: list[Path] = []
        for path in files:
            # Filename: TICKER_FYyyyy_Qn_claims.jsonl
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


def _claim_id(ticker: str, year: int, quarter: str, index: int) -> str:
    """Stable id e.g. ``AAPL_2025_Q1_01``."""
    return make_claim_id(ticker, year, quarter, index)


def _period_key(ticker: str, year: int, quarter: str) -> PeriodKey:
    return (ticker.upper(), int(year), as_fiscal_quarter(quarter))


def _period_candidates_path(
    period: DocumentMeta,
    *,
    out_root: Path,
) -> Path:
    """Per-period candidate file path.

    Example:
        data/eval/candidates/2025/AAPL/AAPL_FY2025_Q1_candidates.jsonl
    """
    ticker = period.ticker.upper()
    q = as_fiscal_quarter(period.fiscal_quarter)
    return (
        out_root
        / str(period.fiscal_year)
        / ticker
        / f"{ticker}_FY{period.fiscal_year}_{q}_candidates.jsonl"
    )


def _preview(text: str, width: int = 64) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _load_candidates_index(
    path: Path,
) -> tuple[set[str], dict[PeriodKey, int]]:
    """Return done claim_ids and per-period candidate counts."""
    done: set[str] = set()
    counts: dict[PeriodKey, int] = defaultdict(int)
    for row in iter_json_objects(path):
        cid = row.get("claim_id")
        if isinstance(cid, str) and cid:
            done.add(cid)
        ticker = row.get("ticker")
        year = row.get("fiscal_year")
        quarter = row.get("fiscal_quarter")
        if (
            isinstance(ticker, str)
            and isinstance(year, int)
            and isinstance(quarter, str)
        ):
            counts[_period_key(ticker, year, quarter)] += 1
    return done, counts


def _append_candidate(path: Path, row: dict) -> None:
    """Append one candidate object (fsync for crash safety)."""
    append_json_object(path, row)


def _drop_claim_ids(path: Path, drop: set[str]) -> int:
    """Rewrite JSONL excluding ``drop`` claim_ids. Returns number removed."""
    return drop_json_object_ids(path, drop)


def _retrieved_payload(
    retrieved: list[tuple[IndexedChunk, float]],
) -> list[dict]:
    """Serialize ranked passages for the candidate record."""
    rows: list[dict] = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "global_id": int(chunk.global_id),
                "score": float(score),
                "section": chunk.section,
                "is_table": chunk.is_table,
                "text": chunk.text,
            }
        )
    return rows


def main() -> None:
    """CLI: retrieve + NLI each eval claim; append candidates for golden review."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("create", "modify"),
        default="create",
        help=(
            "create: default behavior; "
            "modify: only process claims with is_golden_claim=false."
        ),
    )
    parser.add_argument("--ticker", help="Optional ticker filter.")
    parser.add_argument("--year", type=int, help="Optional fiscal year filter.")
    parser.add_argument("--quarter", help="Fiscal quarter: Q1–Q4 or 1–4.")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            f"Drop in-scope candidates and regenerate up to "
            f"{PER_PERIOD_TARGET} per period."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CANDIDATES_ROOT,
        help="Output root directory for per-period candidates JSONL.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Passages kept after rerank.")
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Candidate pool before rerank (default: max(top_k*10, 20)).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking.",
    )
    args = parser.parse_args()

    os.environ["CROSSCHECK_LLM_PROFILE"] = "development"

    if args.quarter is not None:
        quarter_number(args.quarter)

    ticker = args.ticker.strip().upper() if args.ticker else None
    claim_files = _discover_eval_claim_files(
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
        print(
            f"No eval claims under {EVAL_CLAIMS_DIR} ({scope_s}). "
            "Run: python scripts/eval/get_eval_candidates.py",
        )
        sys.exit(1)

    out_root: Path = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"candidates_out_root={out_root}", flush=True)

    pool_k = (
        args.rerank_pool_k
        if args.rerank_pool_k is not None
        else max(args.top_k * 10, 20)
    )
    if pool_k < args.top_k:
        parser.error("--rerank-pool-k must be >= --top-k")

    print(f"eval_claim_files={len(claim_files)}  output_root={out_root}")
    print(f"mode={args.mode}  target={PER_PERIOD_TARGET}/period  llm_profile=development")
    print(f"top_k={args.top_k}  rerank_pool={pool_k}  rerank={not args.no_rerank}")
    print()

    print("loading qdrant + models (once) …", flush=True)
    filings = load_filings_index()
    embed_model = load_embedding_model()
    reranker = None if args.no_rerank else load_reranker()
    print(f"ready · backend={filings.backend}", flush=True)
    print()

    written = 0
    skipped_claims = 0
    skipped_periods = 0
    processed_periods = 0

    for file_idx, claim_file in enumerate(claim_files):
        try:
            year = int(claim_file.parent.parent.name)
            ticker = claim_file.parent.name.upper()
            parts = claim_file.stem.split("_")
            if len(parts) < 3:
                raise ValueError("bad eval claim filename")
            quarter = as_fiscal_quarter(parts[2])
        except (ValueError, IndexError) as exc:
            print(f"  warn: skip bad eval claims file {claim_file}: {exc}")
            continue

        # Derive company_name from the original extracted claims.
        period_temp = DocumentMeta(
            ticker=ticker,
            company_name=ticker,
            fiscal_year=year,
            fiscal_quarter=quarter,
        )
        source_saved = load_saved_claims(period_temp)
        period = DocumentMeta(
            ticker=ticker,
            company_name=source_saved.company_name,
            fiscal_year=year,
            fiscal_quarter=quarter,
        )

        saved = load_eval_claims(period)
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        claims = saved.claims
        # Per-period output file.
        period_out_path = _period_candidates_path(period, out_root=out_root)
        if not period_out_path.exists():
            period_out_path.parent.mkdir(parents=True, exist_ok=True)
            period_out_path.touch()

        indexed_claims = list(enumerate(claims, start=1))
        if args.mode == "modify":
            indexed_claims = [
                (i, claim) for i, claim in indexed_claims if claim.is_golden_claim is False
            ]
        else:
            indexed_claims = indexed_claims[:PER_PERIOD_TARGET]

        eligible_ids = {
            (claim.claim_id or _claim_id(period.ticker, period.fiscal_year, period.fiscal_quarter, i))
            for i, claim in indexed_claims
        }
        total_eligible = len(eligible_ids)

        existing = 0
        if args.mode == "modify":
            # Non-golden claims may have been rewritten — always overwrite candidates.
            if eligible_ids:
                removed = _drop_claim_ids(period_out_path, eligible_ids)
                print(
                    f"  modify: dropped {removed} existing candidate(s) "
                    f"for is_golden_claim=false claims",
                    flush=True,
                )
            done_ids: set[str] = set()
            need = total_eligible
        else:
            if args.force and eligible_ids:
                _drop_claim_ids(period_out_path, eligible_ids)
            period_done_ids, _ = _load_candidates_index(period_out_path)
            done_ids = period_done_ids
            existing = len(done_ids.intersection(eligible_ids))
            need = total_eligible - existing

        print("─" * 60)
        if args.mode == "modify":
            print(
                f"{label}  ({file_idx + 1}/{len(claim_files)})  "
                f"non_golden_claims={total_eligible}  (overwrite)"
            )
        else:
            print(
                f"{label}  ({file_idx + 1}/{len(claim_files)})  "
                f"eligible_claims={total_eligible}  "
                f"candidates={len(done_ids.intersection(eligible_ids))}/{total_eligible}"
            )

        if total_eligible == 0:
            skipped_periods += 1
            print("  skip period (no eligible claims for this mode)\n", flush=True)
            continue

        if args.mode != "modify" and need <= 0:
            skipped_periods += 1
            print(
                "  skip period (all eligible candidates already exist)\n",
                flush=True,
            )
            continue

        print(f"  processing {need} candidate(s)\n", flush=True)
        processed_periods += 1
        got = 0

        for i, claim in indexed_claims:
            if args.mode != "modify" and got >= need:
                break

            cid = claim.claim_id or _claim_id(
                period.ticker, period.fiscal_year, period.fiscal_quarter, i
            )
            if args.mode != "modify" and cid in done_ids:
                skipped_claims += 1
                print(f"  skip {cid} (already in candidates)", flush=True)
                continue

            print(f"  {cid}", flush=True)
            if claim.intended_label:
                print(f"    intended={claim.intended_label}", flush=True)
            print(f"    {_preview(claim.claim)}", flush=True)
            plan = prepare_claim_query(
                claim.claim, fiscal_quarter=period.fiscal_quarter
            )
            print(f"    {retrieval_path_log(plan)}", end=" ", flush=True)

            retrieved = retrieve_claim_passages(
                claim.claim,
                filings,
                embed_model,
                k=args.top_k,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                rerank_pool_k=pool_k,
                reranker=reranker,
                use_reranker=reranker is not None,
            )
            print(f"nli ({len(retrieved)} passages) …", flush=True)

            finding, nli_model, matched_idx = classify_claim(
                claim, retrieved, period=period
            )

            matched_chunk_id = None
            matched_global_id = None
            if matched_idx >= 1 and matched_idx <= len(retrieved):
                matched_chunk = retrieved[matched_idx - 1][0]
                matched_chunk_id = matched_chunk.chunk_id
                if isinstance(matched_chunk, IndexedChunk):
                    matched_global_id = int(matched_chunk.global_id)

            matched_indices = extract_passage_indices(
                finding.reasoning,
                primary_index=matched_idx,
                n_passages=len(retrieved),
            )

            if plan.temporal_scope == TemporalScope.Q4_COMPOSITE:
                query_a = fy_annual_retrieval_query(plan.processed_query)
                query_b = q3_ytd_retrieval_query(plan.processed_query)
            else:
                query_a = None
                query_b = None

            row = {
                "claim_id": cid,
                "ticker": period.ticker,
                "company_name": period.company_name,
                "fiscal_year": period.fiscal_year,
                "fiscal_quarter": period.fiscal_quarter,
                "speaker": claim.speaker,
                "claim": claim.claim,
                "intended_label": claim.intended_label,
                "nli_label": finding.classification,
                "nli_confidence": finding.confidence_score,
                "nli_reasoning": finding.reasoning,
                "nli_model": nli_model,
                "matched_passage_index": matched_idx,
                "matched_passage_indices": matched_indices,
                "matched_chunk_id": matched_chunk_id,
                "matched_global_id": matched_global_id,
                "query_A": query_a,
                "query_B": query_b,
                "retrieved": _retrieved_payload(retrieved),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_candidate(period_out_path, row)
            done_ids.add(cid)
            written += 1
            got += 1
            progress = got if args.mode == "modify" else existing + got
            print(
                f"    → nli={finding.classification}  "
                f"intended={claim.intended_label}  "
                f"matched_passage={matched_idx}  "
                f"chunk_id={matched_chunk_id}  "
                f"appended ({progress}/{total_eligible})",
                flush=True,
            )
            reason = " ".join(finding.reasoning.split())
            if len(reason) > 220:
                reason = reason[:219] + "…"
            print(f"      reasoning: {reason}", flush=True)
            print(flush=True)

        if got < need:
            print(
                f"  warn: only added {got}/{need} "
                f"(not enough unused claims in file)\n",
                flush=True,
            )

    print("─" * 60)
    print(
        f"done  written={written}  "
        f"skipped_claims={skipped_claims}  "
        f"skipped_periods={skipped_periods}  "
        f"processed_periods={processed_periods}  → {out_root}"
    )


if __name__ == "__main__":
    main()
