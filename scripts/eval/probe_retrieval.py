#!/usr/bin/env python3
"""Probe retrieve + NLI for ad-hoc claims in ``data/eval/claims/test_retrieval_claims.jsonl``.

Copy any eval/claims objects into the probe file (same fields). Period routing
uses each claim's ``ticker`` / ``fiscal_year`` / ``fiscal_quarter`` fields — no
CLI filters needed.

Writes candidate-schema rows to ``data/eval/candidates/test_retrieval_candidate.jsonl``.
On re-run, existing rows with the same ``claim_id`` are replaced (not duplicated).
``--force`` clears the entire output file first.

Examples::

    python scripts/eval/probe_retrieval.py
    python scripts/eval/probe_retrieval.py --force

Each input object needs at least::

    claim, speaker, claim_id, ticker, company_name, fiscal_year, fiscal_quarter

Optional:: intended_label, is_golden_claim
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.nli import classify_claim, extract_passage_indices  # noqa: E402
from crosscheck.config import EVAL_CLAIMS_DIR, EVAL_DIR  # noqa: E402
from crosscheck.io.jsonl import (  # noqa: E402
    append_json_object,
    drop_json_object_ids,
    iter_json_objects,
    write_json_objects,
)
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    FinancialClaim,
    IndexedChunk,
    as_fiscal_quarter,
    make_claim_id,
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

PROBE_CLAIMS_PATH = EVAL_CLAIMS_DIR / "test_retrieval_claims.jsonl"
PROBE_CANDIDATES_PATH = EVAL_DIR / "candidates" / "test_retrieval_candidate.jsonl"


def _preview(text: str, width: int = 64) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _retrieved_payload(
    retrieved: list[tuple[IndexedChunk, float]],
) -> list[dict]:
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


def _load_probe_claims(path: Path) -> list[dict]:
    rows = list(iter_json_objects(path))
    out: list[dict] = []
    for i, row in enumerate(rows, start=1):
        claim = str(row.get("claim") or "").strip()
        if not claim:
            print(f"  warn: skip row {i}: empty claim", flush=True)
            continue
        try:
            ticker = str(row.get("ticker") or "").strip().upper()
            year = int(row["fiscal_year"])
            quarter = as_fiscal_quarter(row.get("fiscal_quarter"))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  warn: skip row {i}: bad period fields ({exc})", flush=True)
            continue
        if not ticker:
            print(f"  warn: skip row {i}: missing ticker", flush=True)
            continue
        cid = str(row.get("claim_id") or "").strip() or make_claim_id(
            ticker, year, quarter, i
        )
        out.append(
            {
                **row,
                "claim": claim,
                "speaker": str(row.get("speaker") or ""),
                "claim_id": cid,
                "ticker": ticker,
                "company_name": str(row.get("company_name") or ticker),
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "intended_label": row.get("intended_label"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROBE_CLAIMS_PATH,
        help="Probe claims JSONL (default: data/eval/claims/test_retrieval_claims.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROBE_CANDIDATES_PATH,
        help="Candidates JSONL (default: data/eval/candidates/test_retrieval_candidate.jsonl).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear the entire output file before writing (otherwise replace by claim_id).",
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

    input_path: Path = args.input
    out_path: Path = args.output

    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.touch()
        print(
            f"No probe claims at {input_path}. "
            "Copy eval/claims objects into that file and re-run.",
            flush=True,
        )
        sys.exit(1)

    claims = _load_probe_claims(input_path)
    if not claims:
        print(f"No usable claims in {input_path}.", flush=True)
        sys.exit(1)

    pool_k = (
        args.rerank_pool_k
        if args.rerank_pool_k is not None
        else max(args.top_k * 10, 20)
    )
    if pool_k < args.top_k:
        parser.error("--rerank-pool-k must be >= --top-k")

    print(f"probe_input={input_path}  claims={len(claims)}", flush=True)
    print(f"candidates_out={out_path}", flush=True)
    print(f"top_k={args.top_k}  rerank_pool={pool_k}  rerank={not args.no_rerank}")
    print()

    print("loading qdrant + models (once) …", flush=True)
    filings = load_filings_index()
    embed_model = load_embedding_model()
    reranker = None if args.no_rerank else load_reranker()
    print(f"ready · backend={filings.backend}", flush=True)
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not out_path.exists():
        write_json_objects(out_path, [])
    else:
        probe_ids = {str(row["claim_id"]) for row in claims if row.get("claim_id")}
        removed = drop_json_object_ids(out_path, probe_ids)
        if removed:
            print(
                f"replaced {removed} existing candidate(s) by claim_id",
                flush=True,
            )
            print(flush=True)

    written = 0
    for idx, row in enumerate(claims, start=1):
        period = DocumentMeta(
            ticker=row["ticker"],
            company_name=row["company_name"],
            fiscal_year=row["fiscal_year"],
            fiscal_quarter=row["fiscal_quarter"],
        )
        claim = FinancialClaim(
            claim=row["claim"],
            speaker=row["speaker"],
            claim_id=row["claim_id"],
            intended_label=row.get("intended_label"),
        )
        cid = claim.claim_id or make_claim_id(
            period.ticker, period.fiscal_year, period.fiscal_quarter, idx
        )

        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        print("─" * 60)
        print(f"[{idx}/{len(claims)}] {cid}  ({label})", flush=True)
        if claim.intended_label:
            print(f"    intended={claim.intended_label}", flush=True)
        print(f"    {_preview(claim.claim)}", flush=True)

        plan = prepare_claim_query(claim.claim, fiscal_quarter=period.fiscal_quarter)
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

        candidate = {
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
        append_json_object(out_path, candidate)
        written += 1

        print(
            f"    → nli={finding.classification}  "
            f"intended={claim.intended_label}  "
            f"matched_passage={matched_idx}  "
            f"chunk_id={matched_chunk_id}",
            flush=True,
        )
        reason = " ".join(finding.reasoning.split())
        if len(reason) > 220:
            reason = reason[:219] + "…"
        print(f"      reasoning: {reason}", flush=True)
        print(flush=True)

    print("─" * 60)
    print(f"done  written={written}  → {out_path}")


if __name__ == "__main__":
    main()
