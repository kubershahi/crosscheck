"""LLM-based NLI classification of claims against filing passages."""

from __future__ import annotations

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import NLI_SYSTEM, nli_user
from crosscheck.models import (
    Chunk,
    ContradictionFinding,
    DocumentMeta,
    FinancialClaim,
    IndexedChunk,
    NLIJudgment,
)


def _passage_for_nli(chunk: Chunk) -> dict[str, str | None]:
    """Build passage payload for the NLI prompt (metadata + text)."""
    months = None
    if chunk.quarter_months:
        months = ", ".join(chunk.quarter_months)
    return {
        "text": chunk.text,
        "section": chunk.section,
        "ticker": chunk.ticker,
        "company_name": chunk.company_name,
        "quarter_period_label": chunk.quarter_period_label,
        "quarter_months": months,
    }


def classify_claim(
    claim: FinancialClaim,
    retrieved: list[tuple[Chunk, float]],
    *,
    period: DocumentMeta,
) -> tuple[ContradictionFinding, str, int]:
    """Classify one claim against retrieved filing chunks.

    Returns ``(finding, model_used, matched_passage_index)`` where
    ``matched_passage_index`` is 1-based into ``retrieved`` (0 = none).
    Passages in the finding are in retrieval/rerank order (best first).
    """
    passages = [_passage_for_nli(c) for c, _ in retrieved]
    messages = [
        {"role": "system", "content": NLI_SYSTEM},
        {
            "role": "user",
            "content": nli_user(
                claim=claim.claim,
                speaker=claim.speaker,
                ticker=period.ticker,
                company_name=period.company_name,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                passages=passages,
            ),
        },
    ]
    judgment, model = complete_structured(
        response_model=NLIJudgment,
        messages=messages,
    )

    chunk_ids: list[str] = []
    global_ids: list[int] = []
    for chunk, _ in retrieved:
        chunk_ids.append(chunk.chunk_id)
        if isinstance(chunk, IndexedChunk):
            global_ids.append(int(chunk.global_id))
        else:
            global_ids.append(-1)

    matched = int(judgment.matched_passage_index)
    if matched < 0 or matched > len(retrieved):
        matched = 0

    # Citations always come from retrieval — never from the LLM.
    finding = ContradictionFinding(
        transcript_claim=claim.claim,
        source_speaker=claim.speaker,
        retrieved_filing_passages=[c.text for c, _ in retrieved],
        chunk_ids=chunk_ids,
        global_ids=global_ids,
        classification=judgment.classification,
        confidence_score=judgment.confidence_score,
        reasoning=judgment.reasoning,
    )
    return finding, model, matched
