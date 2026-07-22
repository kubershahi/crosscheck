"""End-to-end retrieval + NLI orchestration for one company-period."""

from __future__ import annotations

from collections import Counter

from crosscheck.analysis.claims import load_saved_claims
from crosscheck.analysis.nli import classify_claim
from crosscheck.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    claims_path,
    get_llm_profile,
    resolve_llm_models,
)
from crosscheck.models import DocumentMeta, PipelineReport
from crosscheck.retrieval.embeddings import load_embedding_model, resolve_device
from crosscheck.retrieval.index import filings_index_path, load_filings_index, retrieve
from crosscheck.retrieval.rerank import load_reranker, rerank_claim_passages


def _step(n: int, total: int, msg: str) -> None:
    """Print a numbered pipeline step for debugging."""
    print(f"  [Step {n}/{total}] {msg}", flush=True)


def run_pipeline(
    period: DocumentMeta,
    *,
    top_k: int = 5,
    use_reranker: bool = True,
    rerank_pool_k: int | None = None,
) -> PipelineReport:
    """Load fixed claims, then run dense retrieval and NLI for one period."""
    total_steps = 5 if use_reranker else 4
    label = f"{period.ticker} FY{period.fiscal_year} Q{period.fiscal_quarter}"
    pool_k = rerank_pool_k if rerank_pool_k is not None else max(top_k * 10, 20)
    if pool_k < top_k:
        raise ValueError("rerank_pool_k must be greater than or equal to top_k")

    models = resolve_llm_models()
    print(f"  profile={get_llm_profile()}  llm_rank={', '.join(models)}", flush=True)
    print(
        f"  embedding={EMBEDDING_MODEL}  device={resolve_device()}  top_k={top_k}",
        flush=True,
    )
    if use_reranker:
        print(
            f"  reranker={RERANKER_MODEL}  rerank_pool_k={pool_k}",
            flush=True,
        )

    _step(1, total_steps, f"Load filings index from {filings_index_path()}")
    filings = load_filings_index()
    print(f"    total chunks in filings index={len(filings.chunks)}", flush=True)

    _step(2, total_steps, "Load embedding model")
    model = load_embedding_model()

    reranker = None
    if use_reranker:
        _step(3, total_steps, "Load cross-encoder reranker")
        reranker = load_reranker()

    saved_path = claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    claims_step = 4 if use_reranker else 3
    classify_step = 5 if use_reranker else 4
    _step(claims_step, total_steps, f"Load fixed claims from {saved_path}")
    saved_claims = load_saved_claims(period)
    print(
        f"    loaded {len(saved_claims.claims)} claims "
        f"(extracted via {saved_claims.llm_model_used})",
        flush=True,
    )

    _step(
        classify_step,
        total_steps,
        f"Dense retrieve{' + rerank' if use_reranker else ''} + NLI classify "
        f"({len(saved_claims.claims)} claims)",
    )
    findings = []
    models_used: set[str] = set()

    for i, claim in enumerate(saved_claims.claims, start=1):
        preview = claim.claim[:72] + ("…" if len(claim.claim) > 72 else "")
        dense_k = pool_k if use_reranker else top_k
        print(
            f"    [{i}/{len(saved_claims.claims)}] dense retrieve k={dense_k}: {preview}",
            flush=True,
        )

        dense_retrieved = retrieve(
            claim.claim,
            filings,
            model,
            k=dense_k,
            ticker=period.ticker,
            fiscal_year=period.fiscal_year,
        )
        retrieved = dense_retrieved
        if use_reranker and reranker is not None:
            retrieved = rerank_claim_passages(
                claim.claim,
                dense_retrieved,
                top_k=top_k,
                model=reranker,
            )
            print(
                f"    [{i}/{len(saved_claims.claims)}] reranked "
                f"{len(dense_retrieved)} → {len(retrieved)} passages",
                flush=True,
            )
        print(
            f"    [{i}/{len(saved_claims.claims)}] retrieved {len(retrieved)} passages → NLI …",
            flush=True,
        )

        finding, nli_model = classify_claim(claim, retrieved)
        findings.append(finding)
        models_used.add(nli_model)
        print(
            f"    [{i}/{len(saved_claims.claims)}] → {finding.classification} "
            f"(confidence={finding.confidence_score:.2f})",
            flush=True,
        )

    llm_model_used = ", ".join(sorted(m for m in models_used if m != "none")) or "none"

    counts = Counter(f.classification for f in findings)
    print(
        f"  done [{label}]: "
        f"Consistent={counts.get('Consistent', 0)}  "
        f"Contradictory={counts.get('Contradictory', 0)}  "
        f"Unverifiable={counts.get('Unverifiable', 0)}",
        flush=True,
    )

    return PipelineReport(
        ticker=period.ticker,
        company_name=period.company_name,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        findings=findings,
        llm_model_used=llm_model_used,
    )
