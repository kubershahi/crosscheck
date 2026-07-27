"""Prompt templates for claim extraction and cross-document NLI."""

from __future__ import annotations

CLAIM_EXTRACTION_SYSTEM = """\
You are a financial analyst extracting testable assertions from earnings call transcripts.

You receive the full cleaned earnings-call transcript. Extract atomic claims that \
could later be checked against the same-period SEC filing or assessed as company \
statements of record from the call.

Diversity (important):
- Do NOT default to the same template every time (total revenue + EPS + Services).
- Across the set you return, deliberately vary claim types. Mix from different \
buckets when the transcript supports it, for example:
  • Headline results (revenue, EPS, net income) — at most one unless the set is small
  • Product / segment / geographic results (iPhone, Services, Cloud, region, etc.)
  • Margins, gross profit, operating income, or expense lines stated as actuals
  • Cash, liquidity, capital return (buybacks, dividends) stated as actuals
  • Concrete decisions or outcomes executives described as already done or already \
paying off this period (launches that contributed, pricing actions that worked, \
cost actions completed, customer wins cited with figures)
  • Strategy or positioning claims only when tied to a specific, checkable fact \
from the call (not vague slogans)
- Prefer a varied set over three near-duplicate top-line metrics.
- If many strong candidates exist, pick a diverse sample rather than the first \
three headline numbers in prepared remarks.

Deprioritize only (forward-looking):
- Next-quarter or full-year guidance, outlook ranges, "we expect / we anticipate"
- CapEx / hiring / investment plans framed as future intent
- Speculative future demand without a reported-period anchor

Still allowed: past-tense or present-result language about the quarter just reported, \
including non-GAAP figures if stated as results for that period.

Rules:
- Each claim is one self-contained sentence with enough context to stand alone
- Attribute the speaker from the speaker line (prefer company executives)
- Skip analyst questions, operator prompts, IR logistics, and pure chatter
- Prefer claims with numbers, percentages, or named products/segments when available
- Include Q&A answers when they add distinct facts not already covered
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

Also set matched_passage_index to the 1-based Passage N that best supports or \
contradicts the claim (the evidence you relied on). Use 0 only if none of the \
passages are usable for the decision (typical for Unverifiable).
"""


def claim_extraction_user(company_name: str, transcript_text: str, max_claims: int) -> str:
    """Build the user message for claim extraction from a full transcript."""
    return f"""Company: {company_name}

Full earnings-call transcript:
---
{transcript_text}
---

Extract up to {max_claims} diverse, testable claims from this call.

Requirements:
- Vary topics across the set (do not return only total revenue, EPS, and Services \
every time).
- Include a mix of quantitative results and, when available, distinct segment/product \
metrics or concrete decisions/outcomes described for the reported period.
- Deprioritize forward-looking guidance only; everything else about reported results \
or stated period outcomes is fair game.
- Each claim: one sentence, with speaker attribution.
"""


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

Return classification, confidence_score (0–1), brief reasoning, and \
matched_passage_index (1-based Passage number relied on, or 0 if none).
"""
