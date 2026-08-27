"""Prompt templates for claim extraction and cross-document NLI.

Each section is one LLM task: the ``*_SYSTEM`` string is the system message,
and the matching ``*_user`` function builds the user message.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Claim extraction
# Transcript → atomic GAAP claims for later filing verification.
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM = """\
You are a financial analyst extracting testable assertions from earnings call transcripts.

You receive the full cleaned earnings-call transcript. Extract atomic claims that \
could later be checked against the same-period SEC 10-Q / 10-K filing.

Performance metrics only (strict):
- Extract ONLY financial performance claims that map to GAAP P&L, balance-sheet, \
cash-flow, or segment financial results (revenue, EPS, net income, gross/operating \
margin, operating income, cost lines, cash, buybacks/dividends with dollar amounts, \
segment/product revenue dollars, etc.).
- Do NOT extract non-performance / product-usage / operational KPIs (MAU/DAU, \
subscriber or seat counts by country, engagement, installs, traffic, headcount \
as a usage metric, NPS, etc.) unless they are stated as a GAAP financial line.

GAAP-only (strict):
- Retain ONLY GAAP figures as reported for the period.
- Filter out non-GAAP / adjusted metrics (non-GAAP gross margin, adjusted EBITDA, \
constant-currency-only figures when presented as non-GAAP, "ex-items" earnings, etc.).
- 10-Q / 10-K disclosures emphasize GAAP line items — prefer claims that can be \
checked there.

Prioritize quantitative metrics (strict):
- Prefer claims that state a concrete dollar amount, percentage, basis-point change, \
EPS figure, or other numeric GAAP metric.
- Deprioritize qualitative / decision / narrative claims (strategy commentary, \
product color, management judgment, “we decided…”, “we remain focused…”) even \
when spoken by an executive, unless they include a checkable GAAP number.
- When choosing among candidates, rank: (1) numbered GAAP metrics, \
(2) directional GAAP metrics with a %, (3) everything else last.

Prefer clean, single-metric claims:
- Prefer claims with one self-contained sentence over multi-period or multi-metric compounds sentences.

Diversity (within the filters above):
- Do NOT default to the same template every time (total revenue + EPS + Services).
- Across the set, vary among allowed GAAP performance buckets when the transcript \
supports it (headline results; segment/geo revenue; margins/operating income; \
cash / capital return with dollars; balance-sheet / liquidity lines).
- Prefer a varied set of clean GAAP claims over near-duplicate top-line metrics.
- Speaker diversity: do NOT attribute nearly all claims to one person. When \
multiple company executives state distinct GAAP figures (CEO, CFO, segment leads, \
etc.), spread speakers across the set. Still prefer company executives over \
analysts; never use the operator.

Deprioritize (forward-looking):
- Deprioritize claims that mix several periods, several metrics, or several \
conditions in one sentence when a simpler atomic claim is available.
- Next-quarter or full-year guidance, outlook ranges, "we expect / we anticipate"
- CapEx / hiring / investment plans framed as future intent
- Speculative future demand without a reported-period GAAP anchor

Rules:
- Put the speaker name in the speaker field only (prefer company executives)
- Claim text must be the assertion alone — do NOT start with the speaker name \
or lead-ins like "X stated/said/noted that…", "According to X…", or \
"The speaker stated…"
- Skip analyst questions, operator prompts, IR logistics, and pure chatter
- Prefer claims with numbers or percentages tied to GAAP performance
- Include Q&A answers when they add distinct GAAP facts not already covered
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
- Financial performance / GAAP-only: no MAU/DAU-style usage metrics; no non-GAAP \
adjusted figures.
- Prioritize key numbered metrics ($, %, EPS, margins, segment dollars) over \
qualitative decisions or narrative facts.
- Prefer clean single-metric, single-period claims over heavy compounds.
- Vary topics across allowed GAAP buckets (do not return only total revenue + EPS \
+ one segment every time).
- Spread speakers across company executives when multiple executives state \
distinct GAAP figures; do not put almost every claim under one speaker.
- Deprioritize forward-looking guidance.
- Each claim: one sentence of assertion text only (no speaker lead-in); \
speaker goes in the separate speaker field.
"""


# ---------------------------------------------------------------------------
# Claim regeneration
# Replace one claim with a unique GAAP claim from the same transcript.
# ---------------------------------------------------------------------------

CLAIM_REGENERATE_SYSTEM = """\
You extract ONE replacement financial claim from an earnings-call transcript.

Follow the same performance / GAAP-only rules as ordinary claim extraction:
- ONLY GAAP P&L, balance-sheet, cash-flow, or segment financial results.
- No non-GAAP / adjusted metrics; no MAU/DAU-style usage KPIs.
- Prefer one clean single-metric sentence with a concrete number or percentage.
- Prioritize numbered GAAP metrics over qualitative decisions or narrative facts.
- Deprioritize forward-looking guidance.
- Claim text is the assertion only — no "X stated/said that…" lead-in; \
speaker belongs in the speaker field.
- Prefer a company executive speaker; when possible choose a speaker different \
from those already dominating FORBIDDEN_CLAIMS, if another executive stated a \
distinct GAAP figure.

Hard uniqueness rule:
- The new claim MUST NOT duplicate or near-paraphrase any FORBIDDEN_CLAIMS.
- Choose a different metric, period framing, or figure than the forbidden set.
"""


def claim_regenerate_user(
    *,
    company_name: str,
    transcript_text: str,
    forbidden_claims: list[str],
    previous_claim: str | None = None,
) -> str:
    """Build the user message for regenerating a single claim."""
    blocked = "\n".join(f"- {text}" for text in forbidden_claims) or "(none)"
    previous_block = ""
    if previous_claim:
        previous_block = f"""
PREVIOUS_CLAIM (being replaced; do not return the same claim):
---
{previous_claim}
---
"""
    return f"""Company: {company_name}
{previous_block}
FORBIDDEN_CLAIMS (do not duplicate or near-paraphrase these):
---
{blocked}
---

Full earnings-call transcript:
---
{transcript_text}
---

Return exactly one new claim (assertion-only claim text + speaker field) \
that is not in FORBIDDEN_CLAIMS.
"""


# ---------------------------------------------------------------------------
# NLI classification
# Claim + retrieved filing passages → Consistent / Contradictory / Unverifiable.
# Q4 composite arithmetic lives in query_processor._Q4_NLI_SUFFIX (appended
# to the user message in nli.classify_claim), not in this shared prompt.
# ---------------------------------------------------------------------------

NLI_SYSTEM = """\
Classify a transcript claim against SEC filing passages as Consistent, \
Contradictory, or Unverifiable. Use only the passages given.

Period language: fiscal quarters are ~3-month periods. Names like "December \
quarter", "Q1", or "first quarter" match filing labels "three months ended …" \
/ Qn. Do not mark Contradictory over period wording alone.

Equivalence / math:
- Unit scales are Consistent when mathematically equivalent within ordinary \
rounding (e.g. $65.6B ↔ $65,585M; $26.3B ↔ $26,340M; 3.95% ↔ 4%; EPS ±$0.05). \
State the tolerance in reasoning.
- Aggregates: claim total may equal filing components for the same metric/period \
(e.g. Total Revenue = Product + Services).
- Operating margin = operating income ÷ net sales (revenue), as a % if claimed \
that way. Prefer explicit margin lines; else compute from the same-period \
income-statement lines.
- YoY % change: prefer an explicit YoY rate in the passages (for Q1–Q3 3-month \
periods it is often in the same table). If none, obtain the prior-year metric \
with the same method used for the current period (same line, aggregate, or \
formula), then derive rate = (current − prior) / prior. Do not accept a YoY % \
on narrative alone.

Labels:
- Contradictory: filing has an explicit conflicting figure for the same \
metric and period (wrong direction or incompatible magnitude). Not for \
missing data, different metrics, or equivalent unit scaling.
- Unverifiable: required metric, period, or growth inputs absent from passages.
- Consistent: ENTIRE claim supported (every figure, direction, period, metric). \
Partial match is not Consistent (e.g. revenue ok but YoY wrong → Contradictory \
if conflict, else Unverifiable if that part missing). Multi-metric claims need \
every metric verified.

Output order (required):
1. reasoning — compare claim to passages (cite Passage N); do not invent a \
label first
2. classification — Consistent / Contradictory / Unverifiable
3. confidence_score — 0–1
4. matched_passage_index — 1-based Passage N best supporting the label; 0 if \
none usable (typical for Unverifiable)
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

Return fields in this order: reasoning (brief), classification, \
confidence_score (0–1), matched_passage_index (1-based Passage relied on, \
or 0 if none). Classification must match the reasoning; matched_passage_index \
must be the passage that backs that classification. \
Apply mathematical equivalence / unit scaling. Use \
Contradictory only for an explicit same-metric conflict; Unverifiable when \
the needed figures are absent. Consistent requires the whole claim to be \
true — every metric/figure must check out; partial correctness is not \
Consistent.
"""


# ---------------------------------------------------------------------------
# Unverifiable rewrite (eval / golden set)
# Turn a GAAP claim into a transcript-grounded Unverifiable claim.
# ---------------------------------------------------------------------------

UNVERIFIABLE_REWRITE_SYSTEM = """\
You rewrite one earnings-call claim so it becomes Unverifiable against the \
same-period SEC 10-Q / 10-K filing.

Crosscheck only checks claims against GAAP filing text (P&L, balance sheet, \
cash flow, segment financials, MD&A tables). Content that appears on the call \
but is not a filing-checkable GAAP line should be treated as Unverifiable.

Goal:
- Start from the given base claim and speaker.
- Use the transcript to add or emphasize content that is typically \
NOT a filing-checkable GAAP line.
- Prioritize transcript-only facts such as: key management decisions, \
execution/operational commentary, product/content mentions (e.g. “movie”), \
and any user/customer growth or engagement details that are not presented \
as GAAP line items.
- Deprioritize GAAP-style financial claims that are easy to verify from filings \
(revenue, EPS, gross margin, operating expenses, operating cash flow, etc.). \
If the base claim contains GAAP dollar/EPS/margin figures, rewrite the claim \
so it no longer depends on those exact GAAP numbers; instead anchor the \
unverifiable portion in the transcript-only decision/color/KPI details.
- The rewritten claim must still sound like something that speaker said.
- Claim text is the assertion only — no "X stated/said that…" lead-in; \
speaker stays in the speaker field.
- Prefer one sentence (two short sentences max).
- Do NOT invent precise GAAP dollar / EPS / margin figures that could \
accidentally match the filing. Avoid retaining exact GAAP figures when \
possible; focus on filing-uncitable transcript facts.
- Do not turn the claim into pure forward guidance with no call grounding; \
anchor the added material in the transcript.
- If FORBIDDEN_CLAIMS are provided, the rewritten claim must NOT duplicate \
or be a near-paraphrase of any forbidden claim.
"""


def unverifiable_rewrite_user(
    *,
    company_name: str,
    base_claim: str,
    speaker: str,
    transcript_text: str,
    forbidden_claims: list[str] | None = None,
) -> str:
    """Build the user message for rewriting a claim into an Unverifiable golden claim."""
    forbidden = forbidden_claims or []
    if forbidden:
        blocked = "\n".join(f"- {text}" for text in forbidden)
        forbid_block = f"""
FORBIDDEN_CLAIMS (do not duplicate or near-paraphrase these):
---
{blocked}
---
"""
    else:
        forbid_block = ""

    return f"""Company: {company_name}
SPEAKER: {speaker}
BASE_CLAIM: {base_claim}
{forbid_block}
Full earnings-call transcript (for grounding unverifiable detail):
---
{transcript_text}
---

Rewrite BASE_CLAIM into one Unverifiable claim for the same speaker.
Ground any added material in the transcript. Return assertion-only claim \
text plus the speaker field (no speaker lead-in in the claim text).
"""


# ---------------------------------------------------------------------------
# Ground-truth filing reference (eval / golden set)
# Claim + NLI reasoning + passages → one-line filing location/metric.
# ---------------------------------------------------------------------------

GROUND_TRUTH_REFERENCE_SYSTEM = """\
You write a short categorical ground-truth filing reference for a golden-set claim.

Given a claim, its NLI label (Consistent or Contradictory), NLI reasoning, and \
retrieved SEC filing passages (especially the matched passages), produce ONE \
line that names the concrete filing location(s) and metric(s) that verify or \
contradict the claim.

Style examples (single source):
- Condensed Consolidated Statements of Operations - Total net sales \
($124,300M vs $119,575M)
- Item 1. Financial Statements — Gross margin table — Company gross margin 46.9%
- Condensed Consolidated Statements of Cash Flows - Operating cash flow $29,943M

Style examples (Q4 subtraction — two sources in one line):
- 10-K Statements of Operations Total revenue $130,497M (12-mo) − \
Q3 10-Q Total revenue $94,930M (9-mo) = $35,567M Q4 derived
- 10-K Revenue table Data Center $115,199M (FY) − \
Q3 10-Q Data Center $79,623M (9-mo) = $35,576M Q4

Rules:
- Use all MATCHED passages when multiple are marked; for Q4 subtraction claims, \
include both the 12-month (10-K) and 9-month (Q3 10-Q) figures in a single line \
showing the subtraction.
- Include the section/table title when available and the key figure(s).
- Keep it to one line. Do not invent figures absent from the passages.
- Do not return JSON keys or multi-paragraph explanations — only the reference \
string via the schema field.
"""


def ground_truth_reference_user(
    *,
    claim: str,
    speaker: str,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    fiscal_quarter: str,
    expected_nli_label: str,
    nli_reasoning: str,
    matched_passage_indices: list[int],
    passages: list[dict[str, str | bool | int | None]],
) -> str:
    """Build the user message for ground-truth reference generation.

    ``matched_passage_indices`` is a sorted list of 1-based passage positions
    that the NLI reasoning relied on.  All are labelled MATCHED in the prompt.
    """
    matched_set = set(matched_passage_indices)
    blocks: list[str] = []
    for i, passage in enumerate(passages, start=1):
        marker = "MATCHED" if i in matched_set else f"Passage {i}"
        meta = [
            f"{marker}",
            f"chunk_id={passage.get('chunk_id') or 'unknown'}",
            f"section={passage.get('section') or 'unknown'}",
            f"is_table={passage.get('is_table')}",
        ]
        body = str(passage.get("text") or "").strip()
        if len(body) > 4000:
            body = body[:4000] + "\n…[truncated]"
        blocks.append(f"{' | '.join(meta)}:\n{body}")
    passage_block = "\n\n".join(blocks) if blocks else "(no passages)"

    indices_str = ", ".join(str(i) for i in matched_passage_indices) or "0"

    return f"""Claim context:
TICKER: {ticker}
COMPANY: {company_name}
FISCAL_YEAR: {fiscal_year}
FISCAL_QUARTER: {fiscal_quarter}
SPEAKER: {speaker}
CLAIM: {claim}
EXPECTED_NLI_LABEL: {expected_nli_label}
MATCHED_PASSAGE_INDICES: {indices_str}

NLI reasoning:
---
{nli_reasoning}
---

Retrieved filing passages:
---
{passage_block}
---

Return ground_truth_reference as one categorical filing reference line.
"""
