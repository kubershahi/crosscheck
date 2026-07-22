"""Prompt templates for claim extraction and cross-document NLI."""

from __future__ import annotations

CLAIM_EXTRACTION_SYSTEM = """\
You are a financial analyst extracting testable assertions from earnings call transcripts.

Source text is C-suite speech (CEO, CFO, president, chairman) in call order — usually \
opening remarks first, then answers later in the call. Extract atomic, verifiable \
financial claims that could be checked against an SEC 10-Q or 10-K filing. Focus on:
- Revenue, earnings, margins, and segment performance
- Year-over-year or quarter-over-quarter changes with numbers when stated
- Guidance, outlook, or forward-looking financial targets
- Balance sheet or cash flow metrics

Rules:
- Each claim must be a single, self-contained sentence
- Attribute each claim to the speaker who made it
- Skip generic boilerplate, thanks, and non-financial commentary
- Prefer claims with specific numbers, percentages, or time periods
- Prefer earlier prepared commentary, but include later answers when they contain \
strong numeric or testable assertions
- Ignore analyst questions, operator prompts, and investor-relations transitions
"""

NLI_SYSTEM = """\
You are a financial natural language inference (NLI) analyst.

Given a transcript claim (hypothesis) and retrieved passages from an SEC filing (premise), \
classify the relationship:

- Consistent: the filing supports or entails the claim (same facts, compatible figures)
- Contradictory: the filing directly conflicts with the claim (opposite direction, \
incompatible numbers, or explicit denial)
- Unverifiable: the filing lacks sufficient detail to confirm or deny the claim

Rules:
- Base your judgment ONLY on the provided filing passages — do not use outside knowledge
- Prefer Unverifiable when retrieval is weak, off-topic, or ambiguous
- Cite specific filing content in your reasoning
- confidence_score: 0.0–1.0 reflecting how strongly the filing passages support your label
"""


def claim_extraction_user(company_name: str, exec_text: str, max_claims: int) -> str:
    """Build the user message for claim extraction."""
    return f"""Company: {company_name}

Executive speech from the earnings call (CEO/CFO and similar roles, in call order):
---
{exec_text}
---

Extract up to {max_claims} testable financial claims from the text above."""


def nli_user(
    claim: str,
    speaker: str,
    passages: list[tuple[str, str | None]],
) -> str:
    """Build the user message for NLI classification."""
    blocks: list[str] = []
    for i, (text, section) in enumerate(passages, start=1):
        section_label = section or "Unknown section"
        blocks.append(f"Passage {i} [{section_label}]:\n{text.strip()}")

    filing_block = "\n\n".join(blocks) if blocks else "(no passages retrieved)"

    return f"""Transcript claim (hypothesis):
Speaker: {speaker}
Claim: {claim}

Retrieved SEC filing passages (premise):
---
{filing_block}
---

Classify whether the filing passages are Consistent, Contradictory, or Unverifiable \
relative to the transcript claim. Fill in transcript_claim and source_speaker from the \
claim above. Populate retrieved_filing_passages and source_sections from the passages above.
"""
