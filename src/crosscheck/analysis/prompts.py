"""Prompt templates for claim extraction and cross-document NLI."""

from __future__ import annotations

CLAIM_EXTRACTION_SYSTEM = """\
You are a financial analyst extracting testable assertions from earnings call transcripts.

You receive the full cleaned earnings-call transcript. Extract atomic, verifiable \
financial claims that could be checked against the same-period SEC 10-Q or 10-K filing.

Prioritize (reported results for the quarter just discussed):
- Revenue, earnings, EPS, margins, and segment performance for the current / reported quarter
- Year-over-year or quarter-over-quarter changes with numbers when stated for that quarter
- Balance sheet, cash flow, or other metrics presented as actual results for the period

Deprioritize / avoid (forward-looking):
- Next-quarter guidance, outlook, or expected ranges
- Full-year forecasts, CapEx plans, expense guidance, or other forward targets
- Speculative commentary about future demand, investment levels, or "we expect / we anticipate"

Rules:
- Each claim must be a single, self-contained sentence
- Attribute each claim to the speaker who made it (use the name on the speaker line)
- Prefer statements from company executives (CEO, CFO, and other company speakers)
- Skip analyst questions, operator prompts, IR logistics, and non-financial chatter
- Prefer claims with specific numbers, percentages, or time periods tied to reported results
- Prefer earlier company prepared commentary, but include later answers when they contain \
strong numeric assertions about the reported quarter
- If you must choose among candidates, always prefer current-quarter actuals over guidance
"""

NLI_SYSTEM = """\
Classify a transcript claim against SEC filing passages as Consistent, Contradictory, \
or Unverifiable.

Period language: fiscal quarters are ~3-month periods. Colloquial names like \
"December quarter", "June quarter", "Q1", or "first quarter" refer to that full \
quarter — treat them as matching filing periods labeled "three months ended \
[month date]" or Qn. Do not mark Contradictory over period wording alone.

Numbers: treat rounded/scaled equivalents as Consistent \
($26,340 million ↔ $26.3B; 3.95% ↔ 4%; 10.09% ↔ 10%; ±0.5% relative or ±1¢ EPS). \
Contradictory only for clear conflicts (wrong direction, incompatible magnitude). \
Unverifiable if the needed figures are missing.

Set classification last so it matches your reasoning. Use only the passages given.
"""


def claim_extraction_user(company_name: str, transcript_text: str, max_claims: int) -> str:
    """Build the user message for claim extraction from a full transcript."""
    return f"""Company: {company_name}

Full earnings-call transcript:
---
{transcript_text}
---

Extract up to {max_claims} testable financial claims about the reported quarter's \
actual results. Do not select next-quarter or full-year forward-looking guidance."""


def nli_user(
    *,
    claim: str,
    speaker: str,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    fiscal_quarter: str,
    passages: list[dict[str, str | None]],
) -> str:
    """Build the user message for NLI classification.

    Each passage dict may include: text, section, ticker, company_name,
    quarter_period_label, quarter_months.
    """
    blocks: list[str] = []
    for i, passage in enumerate(passages, start=1):
        meta_parts = [
            f"TICKER: {passage.get('ticker') or 'Unknown'}",
            f"COMPANY: {passage.get('company_name') or 'Unknown'}",
        ]
        if passage.get("quarter_period_label"):
            meta_parts.append(f"PERIOD: {passage['quarter_period_label']}")
        if passage.get("quarter_months"):
            meta_parts.append(f"QUARTER_MONTHS: {passage['quarter_months']}")
        if passage.get("section"):
            meta_parts.append(f"SECTION: {passage['section']}")
        header = " | ".join(meta_parts)
        body = (passage.get("text") or "").strip()
        blocks.append(f"Passage {i} [{header}]:\n{body}")

    filing_block = "\n\n".join(blocks) if blocks else "(no passages retrieved)"

    return f"""Claim context:
TICKER: {ticker}
COMPANY: {company_name}
FISCAL_YEAR: {fiscal_year}
FISCAL_QUARTER: {fiscal_quarter}
SPEAKER: {speaker}
CLAIM: {claim}

Filing passages:
---
{filing_block}
---

Return classification, confidence_score (0–1), and brief reasoning.
"""
