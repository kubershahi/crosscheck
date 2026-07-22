"""LLM-based NLI classification of claims against filing passages."""

from __future__ import annotations

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import NLI_SYSTEM, nli_user
from crosscheck.models import Chunk, ContradictionFinding, FinancialClaim


def classify_claim(
    claim: FinancialClaim,
    retrieved: list[tuple[Chunk, float]],
) -> tuple[ContradictionFinding, str]:
    """Classify one claim against retrieved filing chunks; return (finding, model_used)."""
    passages = [(c.text, c.section) for c, _ in retrieved]
    messages = [
        {"role": "system", "content": NLI_SYSTEM},
        {"role": "user", "content": nli_user(claim.claim, claim.speaker, passages)},
    ]
    result, model = complete_structured(
        response_model=ContradictionFinding,
        messages=messages,
    )

    # Ensure retrieval metadata is populated even if the model omits it
    if not result.retrieved_filing_passages and retrieved:
        result.retrieved_filing_passages = [c.text for c, _ in retrieved]
    if not result.source_sections and retrieved:
        result.source_sections = [c.section or "Unknown" for c, _ in retrieved]
    if not result.transcript_claim:
        result.transcript_claim = claim.claim
    if not result.source_speaker:
        result.source_speaker = claim.speaker

    return result, model
