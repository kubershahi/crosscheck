"""LLM-based NLI classification of claims against filing passages."""

from __future__ import annotations

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import NLI_SYSTEM, nli_user
from crosscheck.models import (
    Chunk,
    ContradictionFinding,
    DocumentMeta,
    FinancialClaim,
    NLIJudgment,
)


def _passage_section_label(chunk: Chunk) -> str:
    """Item/PART section label used in report ``source_sections``."""
    return chunk.section or "Unknown"


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
) -> tuple[ContradictionFinding, str]:
    """Classify one claim against retrieved filing chunks; return (finding, model_used)."""
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

    # Citations always come from retrieval — never from the LLM.
    finding = ContradictionFinding(
        transcript_claim=claim.claim,
        source_speaker=claim.speaker,
        retrieved_filing_passages=[c.text for c, _ in retrieved],
        source_sections=[_passage_section_label(c) for c, _ in retrieved],
        classification=judgment.classification,
        confidence_score=judgment.confidence_score,
        reasoning=judgment.reasoning,
    )
    return finding, model
