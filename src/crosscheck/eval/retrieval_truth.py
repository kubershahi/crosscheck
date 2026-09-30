"""Canonical retrieval-hit rule for golden eval.

``retrieval_hit`` is True when >50% of *truth tokens* appear in NLI reasoning.
Truth tokens = money/% figures from ``ground_truth_reference`` + keywords from
the claim text.  NLI reasoning is scored (not raw passages) because the model
already grounds its answer in the retrieved passages it cites.
"""

from __future__ import annotations

import re

from crosscheck.eval.reference_overlap import extract_reference_figures

TRUTH_MATCH_THRESHOLD = 0.5

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "we",
        "our",
        "us",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "than",
        "over",
        "year",
        "years",
        "quarter",
        "quarters",
        "during",
        "about",
        "also",
        "up",
        "down",
        "compared",
        "versus",
        "vs",
        "million",
        "millions",
        "billion",
        "billions",
        "percent",
        "growth",
        "increased",
        "decrease",
        "decreased",
        "strong",
        "record",
        "approximately",
    }
)

_WORD_RE = re.compile(r"[a-z][a-z0-9'-]{2,}", re.I)


def claim_keywords(claim: str) -> list[str]:
    """Significant lowercase keywords from the earnings-call claim."""
    words = [w.lower() for w in _WORD_RE.findall(claim or "")]
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def truth_tokens(reference: str, claim: str) -> tuple[list[str], list[str], list[str]]:
    """Return ``(all_tokens, figure_tokens, keyword_tokens)``."""
    figures = [
        token
        for token in extract_reference_figures(reference)
        if not token.startswith("num:")
    ]
    keywords = claim_keywords(claim)
    all_tokens: list[str] = []
    seen: set[str] = set()
    for token in figures + keywords:
        if token not in seen:
            seen.add(token)
            all_tokens.append(token)
    return all_tokens, figures, keywords


def _token_in_text(token: str, text: str) -> bool:
    if token.startswith("usd_m:"):
        number = float(token.split(":", 1)[1])
        variants = {f"{number:g}", f"{number:,.0f}", f"{number:,.2f}"}
        if number >= 1000:
            variants.add(f"{number / 1000:g}")
            variants.add(f"{number / 1000:,.1f}")
        if number >= 1_000_000:
            variants.add(f"{number / 1e6:g}")
        text_lower = text.lower()
        return any(variant.lower() in text_lower for variant in variants)
    if token.startswith("pct:"):
        value = token.split(":", 1)[1]
        return value in text or f"{value}%" in text
    return token.lower() in text.lower()


def match_rate(tokens: list[str], text: str) -> tuple[float, list[str], list[str]]:
    if not tokens:
        return 0.0, [], []
    matched = [token for token in tokens if _token_in_text(token, text)]
    missing = [token for token in tokens if token not in matched]
    return len(matched) / len(tokens), matched, missing


def score_retrieval_hit(
    reference: str,
    claim: str,
    reasoning: str,
    *,
    threshold: float = TRUTH_MATCH_THRESHOLD,
) -> tuple[bool, float, list[str], list[str], list[str], list[str]]:
    """Return ``(hit, rate, matched, missing, figure_tokens, keyword_tokens)``."""
    all_tokens, figures, keywords = truth_tokens(reference, claim)
    if not all_tokens:
        return False, 0.0, [], [], figures, keywords
    rate, matched, missing = match_rate(all_tokens, reasoning or "")
    return rate > threshold, rate, matched, missing, figures, keywords
