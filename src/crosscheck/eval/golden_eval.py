"""Live retrieve + NLI scoring for curated golden claims."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crosscheck.analysis.nli import classify_claim, extract_passage_indices
from crosscheck.eval.metrics import ClaimScoreRow, assign_bucket
from crosscheck.eval.retrieval_truth import score_retrieval_hit
from crosscheck.models import DocumentMeta, FinancialClaim, GoldenClaim
from crosscheck.retrieval.embeddings import load_embedding_model
from crosscheck.retrieval.index import load_filings_index, retrieve_claim_passages
from crosscheck.retrieval.query_processor import prepare_claim_query, retrieval_path_log
from crosscheck.retrieval.rerank import active_rerank_backend, load_reranker


def _preview(text: str, width: int = 64) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def golden_to_financial(claim: GoldenClaim) -> FinancialClaim:
    return FinancialClaim(
        claim=claim.claim,
        speaker=claim.speaker,
        claim_id=claim.claim_id,
        intended_label=claim.expected_nli_label,
        is_golden_claim=True,
    )


def _retrieved_payload(
    retrieved: list[tuple[Any, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "global_id": int(getattr(chunk, "global_id", -1)),
                "score": float(score),
                "section": chunk.section,
                "is_table": chunk.is_table,
                "text": chunk.text,
            }
        )
    return rows


@dataclass
class GoldenClaimResult:
    claim: GoldenClaim
    predicted: str
    nli_correct: bool
    retrieval_hit: bool | None
    bucket: str
    nli_confidence: float
    nli_reasoning: str
    nli_model: str
    matched_passage_indices: list[int]
    matched_figures: list[str] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GoldenPeriodResult:
    period: DocumentMeta
    claims: list[GoldenClaimResult]
    llm_models_used: set[str] = field(default_factory=set)
    reranker_backend: str = ""


def run_golden_period(
    period: DocumentMeta,
    claims: list[GoldenClaim],
    *,
    top_k: int = 5,
    use_reranker: bool = True,
    rerank_pool_k: int | None = None,
) -> GoldenPeriodResult:
    """Retrieve + NLI each golden claim for one company-period."""
    pool_k = rerank_pool_k if rerank_pool_k is not None else max(top_k * 10, 20)
    if pool_k < top_k:
        raise ValueError("rerank_pool_k must be >= top_k")

    print(f"  loading qdrant + models …", flush=True)
    filings = load_filings_index()
    model = load_embedding_model()
    reranker = load_reranker() if use_reranker else None
    backend = active_rerank_backend() if use_reranker else "none"
    n_claims = len(claims)
    print(
        f"  ready · {filings.backend} · {n_claims} golden claims · top_k={top_k}"
        f"{f' · rerank_pool={pool_k}' if use_reranker else ''}"
        f" · rerank={backend}",
        flush=True,
    )
    print(flush=True)

    results: list[GoldenClaimResult] = []
    models_used: set[str] = set()

    for i, golden in enumerate(claims, start=1):
        fin = golden_to_financial(golden)
        print(f"  claim {i}/{n_claims} [{golden.expected_nli_label}]", flush=True)
        print(f"    {_preview(golden.claim)}", flush=True)
        plan = prepare_claim_query(golden.claim, fiscal_quarter=period.fiscal_quarter)
        print(f"    {retrieval_path_log(plan)}", end=" ", flush=True)

        retrieved = retrieve_claim_passages(
            golden.claim,
            filings,
            model,
            k=top_k,
            ticker=period.ticker,
            fiscal_year=period.fiscal_year,
            fiscal_quarter=period.fiscal_quarter,
            rerank_pool_k=pool_k,
            reranker=reranker,
            use_reranker=use_reranker,
        )
        print(f"nli ({len(retrieved)} passages) …", flush=True)

        finding, nli_model, matched_idx = classify_claim(fin, retrieved, period=period)
        models_used.add(nli_model)
        matched_passage_indices = extract_passage_indices(
            finding.reasoning,
            primary_index=matched_idx,
            n_passages=len(retrieved),
        )

        if golden.expected_nli_label == "Unverifiable":
            hit: bool | None = None
            matched: list[str] = []
            missing: list[str] = []
        else:
            hit, _rate, matched, missing, _figs, _kws = score_retrieval_hit(
                golden.ground_truth_reference,
                golden.claim,
                finding.reasoning,
            )

        expected = golden.expected_nli_label
        predicted = finding.classification
        nli_correct = expected == predicted
        bucket = assign_bucket(
            expected=expected,
            predicted=predicted,
            retrieval_hit=hit,
        )

        results.append(
            GoldenClaimResult(
                claim=golden,
                predicted=predicted,
                nli_correct=nli_correct,
                retrieval_hit=hit,
                bucket=bucket,
                nli_confidence=finding.confidence_score,
                nli_reasoning=finding.reasoning,
                nli_model=nli_model,
                matched_passage_indices=matched_passage_indices,
                matched_figures=matched,
                missing_figures=missing,
                retrieved=_retrieved_payload(retrieved),
            )
        )
        print(
            f"    → pred={predicted}  expected={expected}  "
            f"retrieval_hit={'n/a' if hit is None else hit}  "
            f"bucket={bucket}  conf={finding.confidence_score:.2f}  "
            f"passages={matched_passage_indices}",
            flush=True,
        )
        print(flush=True)

    return GoldenPeriodResult(
        period=period,
        claims=results,
        llm_models_used=models_used,
        reranker_backend=backend,
    )


def format_claim_report(
    *,
    claim: GoldenClaim,
    predicted: str,
    nli_correct: bool,
    retrieval_hit: bool | None,
    bucket: str,
    matched_figures: list[str],
    missing_figures: list[str],
    nli_confidence: float,
    nli_reasoning: str,
    nli_model: str,
    matched_passage_indices: list[int],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Serialize one claim row; ``matched_passage_indices`` precedes ``retrieved``."""
    return {
        "claim_id": claim.claim_id,
        "speaker": claim.speaker,
        "claim": claim.claim,
        "expected_nli_label": claim.expected_nli_label,
        "predicted_nli_label": predicted,
        "nli_correct": nli_correct,
        "retrieval_hit": retrieval_hit,
        "bucket": bucket,
        "ground_truth_reference": claim.ground_truth_reference,
        "is_in_filing": claim.is_in_filing,
        "is_in_table": claim.is_in_table,
        "matched_figures": matched_figures,
        "missing_figures": missing_figures,
        "nli_confidence": nli_confidence,
        "nli_reasoning": nli_reasoning,
        "nli_model": nli_model,
        "matched_passage_indices": matched_passage_indices,
        "retrieved": retrieved,
    }


def period_result_to_score_rows(result: GoldenPeriodResult) -> list[ClaimScoreRow]:
    rows: list[ClaimScoreRow] = []
    for item in result.claims:
        g = item.claim
        rows.append(
            ClaimScoreRow(
                claim_id=g.claim_id,
                ticker=g.ticker,
                fiscal_year=g.fiscal_year,
                fiscal_quarter=str(g.fiscal_quarter),
                expected=g.expected_nli_label,
                predicted=item.predicted,  # type: ignore[arg-type]
                nli_correct=item.nli_correct,
                retrieval_hit=item.retrieval_hit,
                bucket=item.bucket,  # type: ignore[arg-type]
                nli_confidence=item.nli_confidence,
                matched_figures=item.matched_figures,
                missing_figures=item.missing_figures,
            )
        )
    return rows


def write_golden_period_report(
    path: Path,
    result: GoldenPeriodResult,
    *,
    run_id: str,
    reranker: str,
    nli_model: str,
    sample_mode: str,
) -> None:
    """Write one period JSON report under ``data/reports/golden/``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": result.period.ticker,
        "company_name": result.period.company_name,
        "fiscal_year": result.period.fiscal_year,
        "fiscal_quarter": result.period.fiscal_quarter,
        "reranker": reranker,
        "nli_model": nli_model,
        "reranker_backend": result.reranker_backend,
        "sample_mode": sample_mode,
        "retrieval_hit_rule": (
            ">50% of truth tokens (reference figures + claim keywords) in NLI reasoning"
        ),
        "llm_models_used": sorted(result.llm_models_used),
        "claims": [],
    }
    for item in result.claims:
        g = item.claim
        payload["claims"].append(
            format_claim_report(
                claim=g,
                predicted=item.predicted,
                nli_correct=item.nli_correct,
                retrieval_hit=item.retrieval_hit,
                bucket=item.bucket,
                matched_figures=item.matched_figures,
                missing_figures=item.missing_figures,
                nli_confidence=item.nli_confidence,
                nli_reasoning=item.nli_reasoning,
                nli_model=item.nli_model,
                matched_passage_indices=item.matched_passage_indices,
                retrieved=item.retrieved,
            )
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
