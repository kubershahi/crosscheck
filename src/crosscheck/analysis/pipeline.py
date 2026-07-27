"""End-to-end retrieval + NLI orchestration for one company-period."""

from __future__ import annotations

from collections import Counter

from crosscheck.analysis.claims import load_saved_claims
from crosscheck.analysis.nli import classify_claim
from crosscheck.models import DocumentMeta, PipelineReport
from crosscheck.retrieval.embeddings import load_embedding_model
from crosscheck.retrieval.index import (
    hybrid_retrieve,
    load_filings_index,
)
from crosscheck.retrieval.rerank import load_reranker, rerank_claim_passages


def _preview(text: str, width: int = 64) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def run_pipeline(
    period: DocumentMeta,
    *,
    top_k: int = 5,
    use_reranker: bool = True,
    rerank_pool_k: int | None = None,
) -> PipelineReport:
    """Load fixed claims, then run hybrid retrieval and NLI for one period."""
    label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
    pool_k = rerank_pool_k if rerank_pool_k is not None else max(top_k * 10, 20)
    if pool_k < top_k:
        raise ValueError("rerank_pool_k must be greater than or equal to top_k")

    print(f"  loading qdrant + models …", flush=True)
    filings = load_filings_index()
    model = load_embedding_model()
    reranker = load_reranker() if use_reranker else None
    saved_claims = load_saved_claims(period)
    n_claims = len(saved_claims.claims)
    print(
        f"  ready · {filings.backend}"
        f"{f' ({len(filings.chunks)} chunks)' if filings.chunks else ''}"
        f" · {n_claims} claims · top_k={top_k}"
        f"{f' · rerank_pool={pool_k}' if use_reranker else ''}",
        flush=True,
    )
    print(flush=True)

    findings = []
    models_used: set[str] = set()

    for i, claim in enumerate(saved_claims.claims, start=1):
        print(f"  claim {i}/{n_claims}", flush=True)
        print(f"    {_preview(claim.claim)}", flush=True)
        print(f"    retrieve …", end=" ", flush=True)

        retrieve_k = pool_k if use_reranker else top_k
        hybrid_retrieved = hybrid_retrieve(
            claim.claim,
            filings,
            model,
            k=retrieve_k,
            ticker=period.ticker,
            fiscal_year=period.fiscal_year,
            fiscal_quarter=period.fiscal_quarter,
        )
        retrieved = hybrid_retrieved
        if use_reranker and reranker is not None:
            print(f"rerank ({len(hybrid_retrieved)}) …", end=" ", flush=True)
            retrieved = rerank_claim_passages(
                claim.claim,
                hybrid_retrieved,
                top_k=top_k,
                model=reranker,
            )
        print(f"nli ({len(retrieved)} passages) …", flush=True)

        finding, nli_model, _matched = classify_claim(claim, retrieved, period=period)
        findings.append(finding)
        models_used.add(nli_model)
        print(
            f"    → {finding.classification}  "
            f"confidence={finding.confidence_score:.2f}  model={nli_model}",
            flush=True,
        )
        print(flush=True)

    llm_model_used = ", ".join(sorted(m for m in models_used if m != "none")) or "none"
    counts = Counter(f.classification for f in findings)
    print(
        f"  summary [{label}]: "
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
