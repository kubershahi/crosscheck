#!/usr/bin/env python3
"""Build golden-set *candidates* via the same retrieve + NLI path as run_nli.

Writes up to ``PER_PERIOD_TARGET`` (5) candidates per ticker / year / quarter.
Appends one JSON object per claim to ``data/eval/candidates.jsonl`` (fsync) so
progress survives interruptions.

Re-runs skip a period that already has ≥5 candidates. Periods with fewer than
5 are filled from remaining claims (skipping claim_ids already present).
``--force`` drops in-scope candidates first, then regenerates up to 5/period.

Uses the development LLM profile. Default gap between periods is 12s.

Examples::

    python scripts/build_golden.py
    python scripts/build_golden.py --year 2025
    python scripts/build_golden.py --ticker AAPL --year 2025 --quarter Q1
    python scripts/build_golden.py --force

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.claims import load_saved_claims  # noqa: E402
from crosscheck.analysis.nli import classify_claim  # noqa: E402
from crosscheck.config import CLAIMS_DIR, EVAL_DIR  # noqa: E402
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    IndexedChunk,
    SavedTranscriptClaims,
    as_fiscal_quarter,
    quarter_number,
)
from crosscheck.retrieval.embeddings import load_embedding_model  # noqa: E402
from crosscheck.retrieval.index import hybrid_retrieve, load_filings_index  # noqa: E402
from crosscheck.retrieval.rerank import load_reranker, rerank_claim_passages  # noqa: E402

PER_PERIOD_TARGET = 5
CLAIM_FILE_GAP_SECONDS = 12
CANDIDATES_PATH = EVAL_DIR / "candidates.jsonl"

PeriodKey = tuple[str, int, str]  # ticker, fiscal_year, fiscal_quarter


def _discover_claim_files(
    *,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> list[Path]:
    """Return claim JSON paths under ``data/claims`` matching optional filters."""
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


def _claim_id(ticker: str, year: int, quarter: str, index: int) -> str:
    """Stable id e.g. ``AAPL_2025_Q1_01``."""
    return f"{ticker.upper()}_{year}_{as_fiscal_quarter(quarter)}_{index:02d}"


def _period_key(ticker: str, year: int, quarter: str) -> PeriodKey:
    return (ticker.upper(), int(year), as_fiscal_quarter(quarter))


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
    if not path.exists():
        return done, counts
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  warn: skip bad JSONL line {line_no} in {path}", flush=True)
                continue
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
    """Append one candidate object as a JSONL line (fsync for crash safety)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _drop_claim_ids(path: Path, drop: set[str]) -> int:
    """Rewrite JSONL excluding ``drop`` claim_ids. Returns number removed."""
    if not path.exists() or not drop:
        return 0
    kept: list[str] = []
    removed = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                kept.append(line if line.endswith("\n") else line + "\n")
                continue
            cid = row.get("claim_id")
            if isinstance(cid, str) and cid in drop:
                removed += 1
                continue
            kept.append(json.dumps(row, ensure_ascii=False) + "\n")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(kept)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    return removed


def _upcoming_claim_ids(claim_files: list[Path]) -> set[str]:
    """Claim ids in scope (first ``PER_PERIOD_TARGET`` claims per period)."""
    ids: set[str] = set()
    for path in claim_files:
        try:
            saved = SavedTranscriptClaims.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        n = min(len(saved.claims), PER_PERIOD_TARGET)
        for i in range(1, n + 1):
            ids.add(
                _claim_id(
                    saved.ticker, saved.fiscal_year, saved.fiscal_quarter, i
                )
            )
    return ids


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
    """CLI: retrieve + NLI each claim; append candidates for manual golden review."""
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
        help=(
            f"Drop in-scope candidates and regenerate up to "
            f"{PER_PERIOD_TARGET} per period."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CANDIDATES_PATH,
        help=f"Candidates JSONL path (default: {CANDIDATES_PATH}).",
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
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=CLAIM_FILE_GAP_SECONDS,
        help=f"Sleep between periods (default: {CLAIM_FILE_GAP_SECONDS}; 0=off).",
    )
    args = parser.parse_args()

    os.environ["CROSSCHECK_LLM_PROFILE"] = "development"

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
        print(f"No claims under {CLAIMS_DIR} ({scope_s}).")
        sys.exit(1)

    out_path = args.output
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        out_path.touch()
        print(f"created {out_path}", flush=True)

    upcoming = _upcoming_claim_ids(claim_files)
    if args.force:
        removed = _drop_claim_ids(out_path, upcoming)
        print(
            f"--force: dropped {removed} existing candidate(s) in scope "
            f"(up to {PER_PERIOD_TARGET}/period will be rewritten)",
            flush=True,
        )

    done_ids, period_counts = _load_candidates_index(out_path)

    pool_k = (
        args.rerank_pool_k
        if args.rerank_pool_k is not None
        else max(args.top_k * 10, 20)
    )
    if pool_k < args.top_k:
        parser.error("--rerank-pool-k must be >= --top-k")

    print(f"claim_files={len(claim_files)}  output={out_path}")
    print(
        f"target={PER_PERIOD_TARGET}/period  "
        f"existing_claim_ids={len(done_ids)}  "
        f"llm_profile=development  gap={args.gap_seconds}s"
    )
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
        saved = SavedTranscriptClaims.model_validate_json(
            claim_file.read_text(encoding="utf-8")
        )
        period = DocumentMeta(
            ticker=saved.ticker,
            company_name=saved.company_name,
            fiscal_year=saved.fiscal_year,
            fiscal_quarter=saved.fiscal_quarter,
        )
        pkey = _period_key(
            period.ticker, period.fiscal_year, period.fiscal_quarter
        )
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        claims = load_saved_claims(period).claims
        existing = period_counts.get(pkey, 0)

        print("─" * 60)
        print(
            f"{label}  ({file_idx + 1}/{len(claim_files)})  "
            f"claims_file={len(claims)}  candidates={existing}/{PER_PERIOD_TARGET}"
        )

        if existing >= PER_PERIOD_TARGET:
            skipped_periods += 1
            print(
                f"  skip period (≥{PER_PERIOD_TARGET} candidates already)\n",
                flush=True,
            )
            continue

        need = PER_PERIOD_TARGET - existing
        print(f"  need {need} more candidate(s)\n", flush=True)
        processed_periods += 1
        got = 0

        for i, claim in enumerate(claims, start=1):
            if got >= need:
                break

            cid = _claim_id(
                period.ticker, period.fiscal_year, period.fiscal_quarter, i
            )
            if cid in done_ids:
                skipped_claims += 1
                print(f"  skip {cid} (already in candidates)", flush=True)
                continue

            print(f"  {cid}", flush=True)
            print(f"    {_preview(claim.claim)}", flush=True)
            print("    retrieve …", end=" ", flush=True)

            retrieve_k = pool_k if reranker is not None else args.top_k
            hybrid_retrieved = hybrid_retrieve(
                claim.claim,
                filings,
                embed_model,
                k=retrieve_k,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
            )
            retrieved = hybrid_retrieved
            if reranker is not None:
                print(f"rerank ({len(hybrid_retrieved)}) …", end=" ", flush=True)
                retrieved = rerank_claim_passages(
                    claim.claim,
                    hybrid_retrieved,
                    top_k=args.top_k,
                    model=reranker,
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

            row = {
                "claim_id": cid,
                "ticker": period.ticker,
                "company_name": period.company_name,
                "fiscal_year": period.fiscal_year,
                "fiscal_quarter": period.fiscal_quarter,
                "speaker": claim.speaker,
                "claim": claim.claim,
                "nli_label": finding.classification,
                "nli_confidence": finding.confidence_score,
                "nli_reasoning": finding.reasoning,
                "nli_model": nli_model,
                "matched_passage_index": matched_idx,
                "matched_chunk_id": matched_chunk_id,
                "matched_global_id": matched_global_id,
                "retrieved": _retrieved_payload(retrieved),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_candidate(out_path, row)
            done_ids.add(cid)
            period_counts[pkey] = period_counts.get(pkey, 0) + 1
            written += 1
            got += 1
            print(
                f"    → {finding.classification}  "
                f"matched_passage={matched_idx}  "
                f"chunk_id={matched_chunk_id}  "
                f"appended ({period_counts[pkey]}/{PER_PERIOD_TARGET})",
                flush=True,
            )
            print(flush=True)

        if got < need:
            print(
                f"  warn: only added {got}/{need} "
                f"(not enough unused claims in file)\n",
                flush=True,
            )

        if (
            args.gap_seconds > 0
            and got > 0
            and file_idx < len(claim_files) - 1
        ):
            print(f"  … sleep {args.gap_seconds}s\n", flush=True)
            time.sleep(args.gap_seconds)

    print("─" * 60)
    print(
        f"done  written={written}  "
        f"skipped_claims={skipped_claims}  "
        f"skipped_periods={skipped_periods}  "
        f"processed_periods={processed_periods}  → {out_path}"
    )


if __name__ == "__main__":
    main()
