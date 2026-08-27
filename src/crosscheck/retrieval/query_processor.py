"""Zero-LLM query preprocessing: financial unit expansion + temporal routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Billions → millions: capture optional $, number, optional B / billion [dollars].
# Examples: "$65.6 billion", "$24.5B", "24B", "24.5 billion dollars"
_BILLION_AMOUNT_RE = re.compile(
    r"""
    (?P<prefix>\$)?
    (?P<number>\d+(?:\.\d+)?)
    \s*
    (?:
        (?P<b_suffix>[Bb])(?!\w)                  # $24.5B or 24B
        |
        (?P<billion>billions?)                    # billion / billions
        (?:\s+dollars?)?                          # optional "dollars"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ALREADY_EXPANDED_RE = re.compile(
    r"^\s*\(\s*\d+(?:\.\d+)?\s+million\s*\)",
    re.IGNORECASE,
)

# Financial figures (not quarter/year tokens like Q4 / FY2025).
_HAS_NUMERIC_VALUE_RE = re.compile(
    r"""
    \$
    | \d+(?:,\d{3})+(?:\.\d+)?
    | \d+(?:\.\d+)?\s*%
    | \d+(?:\.\d+)?\s*(?:million|billion|[MB])\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REVENUE_RE = re.compile(
    r"\b(revenues?)\b(?!\s*\(\s*net sales\s*\))",
    re.IGNORECASE,
)

_Q4_RE = re.compile(
    r"""
    \b(?:
        fourth[\s\-]+quarter        # fourth quarter / fourth-quarter
        | 4th[\s\-]+quarter         # 4th quarter
        | final[\s\-]+quarter       # final quarter
        | q4                        # Q4
        | \w+\s+quarter             # month-named quarters: september quarter, december quarter, etc.
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FULL_YEAR_RE = re.compile(
    r"""
    \b(?:
        full[\s\-]+year
        | fiscal[\s\-]+year
        | annual
        | trailing[\s\-]+(?:twelve|12)[\s\-]+months?
        | fy20\d{2}
        | fy\d{2}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_Q4_NLI_SUFFIX = """\
[Q4 ARITHMETIC VERIFICATION RULE]
For Q4 claims, follow this protocol in your reasoning:
1. Locate the 12-month metric from 10-K passages (typically Passages 1–4).
2. Locate the 9-month metric from Q3 10-Q passages (typically Passages 5–8).
3. Write the subtraction: [10-K 12-month] − [Q3 10-Q 9-month] = [Derived Q4].
4. Compare [Derived Q4] to the claimed amount (allow rounding/unit scaling):
   - Match → Consistent; direct conflict → Contradictory; either input missing → Unverifiable."""


class TemporalScope(str, Enum):
    """How a claim should be routed to SEC filing documents."""

    FULL_YEAR_ONLY = "FULL_YEAR_ONLY"
    Q4_COMPOSITE = "Q4_COMPOSITE"
    STANDARD_QUARTER = "STANDARD_QUARTER"


@dataclass(frozen=True)
class QueryPlan:
    """Normalized query + temporal retrieval plan for hybrid search."""

    processed_query: str
    temporal_scope: TemporalScope
    required_doc_types: list[str]
    nli_instruction_suffix: str


def expand_revenue_net_sales(query: str) -> str:
    """Append ``(net sales)`` after ``revenue`` when the claim has a numeric figure.

    SEC tables often label the line ``Net sales`` rather than ``Revenue``.
    Skips claims with no dollar / percent / million-billion figure so wording
    like ``Q4 revenue rose`` is left alone.
    """
    if not query:
        return query
    if not _HAS_NUMERIC_VALUE_RE.search(query):
        return query
    return _REVENUE_RE.sub(r"\1 (net sales)", query)


def expand_financial_units(query: str) -> str:
    """Append ``(N million)`` after billion-denominated amounts for table match.

    SEC tables often report ``In millions``; expanding ``$65.6 billion`` to
    include ``(65600 million)`` helps dense + BM25 retrieval. Also expands
    ``revenue`` to ``revenue (net sales)`` when a numeric figure is present.
    """
    if not query:
        return query

    parts: list[str] = []
    last = 0
    for match in _BILLION_AMOUNT_RE.finditer(query):
        start, end = match.start(), match.end()
        parts.append(query[last:start])
        matched = query[start:end]
        # Skip if already followed by an expansion.
        after = query[end:]
        if _ALREADY_EXPANDED_RE.match(after):
            parts.append(matched)
            last = end
            continue
        try:
            billions = float(match.group("number"))
        except (TypeError, ValueError):
            parts.append(matched)
            last = end
            continue
        millions = billions * 1000.0
        if millions.is_integer():
            millions_str = str(int(millions))
        else:
            millions_str = f"{millions:g}"
        parts.append(f"{matched} ({millions_str} million)")
        last = end
    parts.append(query[last:])
    return expand_revenue_net_sales("".join(parts))


def classify_claim_temporal_scope(
    claim_text: str,
    *,
    fiscal_quarter: str | int | None = None,
) -> TemporalScope:
    """Classify claim text into a filing-retrieval temporal scope (regex only).

    Temporal routing (full-year vs Q4-composite) runs **only** for Q4 claim
    periods. Q1–Q3 always return ``STANDARD_QUARTER`` regardless of wording.
    """
    from crosscheck.models import quarter_number

    if fiscal_quarter is not None:
        try:
            if quarter_number(fiscal_quarter) != 4:
                return TemporalScope.STANDARD_QUARTER
        except ValueError:
            # Unknown quarter token — do not activate Q4/full-year routing.
            return TemporalScope.STANDARD_QUARTER
    else:
        # No period context: do not activate temporal routing.
        return TemporalScope.STANDARD_QUARTER

    # Q4 period only from here.
    text = claim_text or ""
    has_q4 = bool(_Q4_RE.search(text))
    has_full_year = bool(_FULL_YEAR_RE.search(text))
    if has_q4:
        return TemporalScope.Q4_COMPOSITE
    if has_full_year:
        return TemporalScope.FULL_YEAR_ONLY
    # Q4 claim file with no explicit Q4/full-year wording: still composite
    # so retrieval can use 10-K + Q3 10-Q for subtraction.
    return TemporalScope.Q4_COMPOSITE


FY_PERIOD_PHRASE = "12 month (year ended) period"
Q3_PERIOD_PHRASE = "nine months ended period"


def _rewrite_q4_period(expanded_claim: str, replacement: str) -> str:
    """Replace Q4 temporal wording, or prefix when none is present.

    When the claim has no Q4/month-quarter phrase to substitute, prefix with
    ``for {replacement}`` so retrieval still targets the correct period.
    """
    text = expanded_claim or ""
    rewritten = _Q4_RE.sub(replacement, text)
    if rewritten == text:
        rewritten = f"for {replacement} {text}"
    return re.sub(r"\s+", " ", rewritten).strip()


def fy_annual_retrieval_query(expanded_claim: str) -> str:
    """Build Path A: Q4 → ``12 month (year ended) period``, numbers kept."""
    return _rewrite_q4_period(expanded_claim, FY_PERIOD_PHRASE)


def q3_ytd_retrieval_query(expanded_claim: str) -> str:
    """Build Path B: Q4 → ``nine months ended period``, numbers kept."""
    return _rewrite_q4_period(expanded_claim, Q3_PERIOD_PHRASE)


def _doc_types_for_scope(scope: TemporalScope) -> list[str]:
    if scope == TemporalScope.FULL_YEAR_ONLY:
        return ["10-K"]
    if scope == TemporalScope.Q4_COMPOSITE:
        return ["10-K", "10-Q"]
    return ["10-Q"]


def prepare_claim_query(
    claim_text: str,
    *,
    fiscal_quarter: str | int | None = None,
) -> QueryPlan:
    """Expand units and classify temporal scope for one claim query.

    ``fiscal_quarter`` gates temporal classification: only ``Q4`` activates
    full-year / Q4-composite routing; Q1–Q3 stay on standard 10-Q retrieval.
    Unit expansion always runs.
    """
    scope = classify_claim_temporal_scope(
        claim_text, fiscal_quarter=fiscal_quarter
    )
    processed = expand_financial_units(claim_text)
    suffix = _Q4_NLI_SUFFIX if scope == TemporalScope.Q4_COMPOSITE else ""
    return QueryPlan(
        processed_query=processed,
        temporal_scope=scope,
        required_doc_types=_doc_types_for_scope(scope),
        nli_instruction_suffix=suffix,
    )


def retrieval_path_log(plan: QueryPlan) -> str:
    """Short progress label once temporal scope is decided."""
    if plan.temporal_scope == TemporalScope.Q4_COMPOSITE:
        return "dual-path retrieve (10-K FY + Q3 10-Q, 4+4) …"
    if plan.temporal_scope == TemporalScope.FULL_YEAR_ONLY:
        return "single-path retrieve (10-K FY) …"
    return "retrieve …"
