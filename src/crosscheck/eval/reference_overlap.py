"""Deterministic recall scoring via reference figure overlap."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

RecallMode = Literal[
    "strict",
    "ignore_num",
    "half_money",
    "any_money",
    "reasoning_half",
    "reasoning_embed",
    "reasoning_sim",
    "hybrid",
]

# Default: ≥50% of money/% tokens from the reference appear in retrieved passages.
DEFAULT_RECALL_MODE: RecallMode = "half_money"

# ``reasoning_sim`` / ``reasoning_embed``: BGE-M3 cosine similarity threshold.
DEFAULT_EMBED_THRESHOLD: float = 0.60

# Legacy text SequenceMatcher fallback threshold (rarely used).
DEFAULT_SIM_THRESHOLD: float = 0.12

RECALL_MODES: tuple[RecallMode, ...] = (
    "strict",
    "ignore_num",
    "half_money",
    "any_money",
    "reasoning_half",
    "reasoning_embed",
    "reasoning_sim",
    "hybrid",
)

# Dollar / percent / scaled amounts (same spirit as golden_claims corrupt regex).
_FIGURE_RE = re.compile(
    r"""
    (?P<prefix>\$)?
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?:
        (?P<b_suffix>[Bb])(?!\w)
        | (?P<m_suffix>[Mm])(?!\w)
        | (?P<billion>billions?)
        | (?P<million>millions?)
        | (?P<pct>%)
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PCT_ONLY_RE = re.compile(r"(?<!\d)(?P<num>\d+(?:\.\d+)?)\s*%")

# Skip bare years that appear in filing references.
_YEAR_RE = re.compile(r"^\d{4}$")


def _parse_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _normalize_token(
    number: float,
    *,
    prefix: str | None,
    b_suffix: str | None,
    m_suffix: str | None,
    billion: str | None,
    million: str | None,
    pct: str | None,
) -> str | None:
    if pct:
        return f"pct:{number:g}"
    if prefix or billion or b_suffix:
        billions = number * (1000.0 if billion or b_suffix else 1.0)
        if billion or b_suffix:
            return f"usd_m:{billions * 1000:g}"
        return f"usd_m:{number:g}"
    if million or m_suffix:
        return f"usd_m:{number:g}"
    if _YEAR_RE.match(f"{number:g}".split(".")[0]):
        return None
    # Plain large numbers — treat as millions-scale filing table values.
    if number >= 100:
        return f"usd_m:{number:g}"
    return f"num:{number:g}"


def extract_reference_figures(reference: str) -> list[str]:
    """Pull comparable numeric tokens from a categorical reference line."""
    text = reference or ""
    tokens: set[str] = set()

    for match in _FIGURE_RE.finditer(text):
        try:
            number = _parse_number(match.group("number"))
        except ValueError:
            continue
        tok = _normalize_token(
            number,
            prefix=match.group("prefix"),
            b_suffix=match.group("b_suffix"),
            m_suffix=match.group("m_suffix"),
            billion=match.group("billion"),
            million=match.group("million"),
            pct=match.group("pct"),
        )
        if tok:
            tokens.add(tok)

    for match in _PCT_ONLY_RE.finditer(text):
        try:
            number = float(match.group("num"))
        except ValueError:
            continue
        tokens.add(f"pct:{number:g}")

    return sorted(tokens)


def succinct_reasoning(reasoning: str, *, max_chars: int = 400) -> str:
    """Trim NLI reasoning to the lead sentences for text-similarity compare."""
    text = " ".join((reasoning or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (". ", "; "):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + 1].strip()
    return cut.strip()


def _normalize_compare_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s%$.,]", " ", text)
    return " ".join(text.split())


def reference_reasoning_text_similarity(reference: str, reasoning: str) -> float:
    """SequenceMatcher ratio on normalized reference vs succinct reasoning."""
    a = _normalize_compare_text(reference)
    b = _normalize_compare_text(succinct_reasoning(reasoning))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def reference_reasoning_embedding_similarity(reference: str, reasoning: str) -> float:
    """Cosine similarity (BGE-M3) between reference and succinct NLI reasoning."""
    import numpy as np

    from crosscheck.retrieval.embeddings import embed_texts, load_embedding_model

    ref_text = " ".join((reference or "").split())
    reas_text = succinct_reasoning(reasoning)
    if not ref_text or not reas_text:
        return 0.0
    model = load_embedding_model()
    vectors = embed_texts(model, [ref_text, reas_text], batch_size=2)
    return float(np.dot(vectors[0], vectors[1]))


def _passage_tokens(text: str) -> set[str]:
    return set(extract_reference_figures(text))


def _money_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not t.startswith("num:")]


def score_overlap(
    required: list[str],
    found: set[str],
    *,
    mode: RecallMode = DEFAULT_RECALL_MODE,
) -> tuple[bool, list[str], list[str]]:
    """Apply a figure-overlap recall mode to required vs found tokens.

    Modes:
      - ``strict``: every extracted token must match (includes Item/Note nums).
      - ``ignore_num``: require all ``usd_m:`` / ``pct:`` tokens; drop bare ``num:*``.
      - ``half_money``: ≥50% of money/% tokens match.
      - ``any_money``: at least one money/% token matches.
    """
    if mode == "strict":
        tokens = required
        matched = [t for t in tokens if t in found]
        missing = [t for t in tokens if t not in found]
        return bool(tokens) and not missing, matched, missing

    money = _money_tokens(required)
    matched = [t for t in money if t in found]
    missing = [t for t in money if t not in found]
    if mode == "ignore_num":
        return bool(money) and not missing, matched, missing
    if mode == "any_money":
        return bool(money) and bool(matched), matched, missing
    if mode == "half_money":
        ok = bool(money) and (len(matched) / len(money) >= 0.5)
        return ok, matched, missing
    raise ValueError(f"mode {mode!r} is not a figure-overlap mode")


def _figure_hit(
    reference: str,
    text: str,
    *,
    mode: RecallMode,
) -> tuple[bool, list[str], list[str]]:
    required = extract_reference_figures(reference)
    if not required:
        return False, [], []
    found = _passage_tokens(text)
    return score_overlap(required, found, mode=mode)


def _reasoning_figure_hit(
    reference: str,
    reasoning: str,
) -> tuple[bool, list[str], list[str]]:
    """≥50% of reference money/% tokens appear in NLI reasoning."""
    return _figure_hit(reference, reasoning, mode="half_money")


def _reasoning_embed_hit(
    reference: str,
    reasoning: str,
    *,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> tuple[bool, list[str], list[str]]:
    """Hit when BGE-M3 cosine similarity clears threshold."""
    cosine = reference_reasoning_embedding_similarity(reference, reasoning)
    tag = f"embed_cos:{cosine:.3f}"
    hit = cosine >= embed_threshold
    return hit, ([tag] if hit else []), ([] if hit else [tag])


def _reasoning_sim_hit(
    reference: str,
    reasoning: str,
    *,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> tuple[bool, list[str], list[str]]:
    """Hit when embedding cosine OR figure-half vs reasoning clears bar."""
    embed_hit, embed_matched, embed_missing = _reasoning_embed_hit(
        reference, reasoning, embed_threshold=embed_threshold
    )
    if embed_hit:
        return True, embed_matched, embed_missing
    fig_hit, fig_matched, fig_missing = _reasoning_figure_hit(reference, reasoning)
    if fig_hit:
        return True, ["reasoning:half"] + fig_matched, fig_missing
    text_sim = reference_reasoning_text_similarity(reference, reasoning)
    if text_sim >= sim_threshold:
        return True, [f"text_sim:{text_sim:.3f}"], embed_missing
    return False, fig_matched, embed_missing + [f"text_sim:{text_sim:.3f}"]


def score_recall_hit(
    reference: str,
    passage_texts: list[str],
    *,
    reasoning: str | None = None,
    mode: RecallMode = DEFAULT_RECALL_MODE,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> tuple[bool, list[str], list[str]]:
    """Score retrieval recall for one C/X claim.

    Passage modes (``strict`` … ``any_money``): compare reference figures to
    retrieved passage text.

    ``reasoning_half``: ≥50% of reference money/% tokens appear in NLI reasoning.

    ``reasoning_embed``: BGE-M3 cosine similarity (reference vs succinct reasoning)
    ≥ ``embed_threshold`` (default 0.60).

    ``reasoning_sim``: embedding cosine **or** reasoning figure-half **or** text sim.

    ``hybrid``: passage ``half_money`` **or** ``reasoning_half``.
    """
    if mode in {"strict", "ignore_num", "half_money", "any_money"}:
        required = extract_reference_figures(reference)
        if not required:
            return False, [], []
        found: set[str] = set()
        for passage in passage_texts:
            found |= _passage_tokens(passage)
        return score_overlap(required, found, mode=mode)

    if mode in {"reasoning_half", "reasoning_embed", "reasoning_sim"}:
        if not reasoning:
            return False, [], ["reasoning:empty"]
        if mode == "reasoning_half":
            return _reasoning_figure_hit(reference, reasoning)
        if mode == "reasoning_embed":
            return _reasoning_embed_hit(
                reference, reasoning, embed_threshold=embed_threshold
            )
        return _reasoning_sim_hit(
            reference,
            reasoning,
            embed_threshold=embed_threshold,
            sim_threshold=sim_threshold,
        )

    if mode == "hybrid":
        required = extract_reference_figures(reference)
        if not required:
            return False, [], []
        found = set()
        for passage in passage_texts:
            found |= _passage_tokens(passage)
        pass_hit, pass_matched, pass_missing = score_overlap(
            required, found, mode="half_money"
        )
        if pass_hit:
            return True, ["passage:half"] + pass_matched, pass_missing
        if not reasoning:
            return False, pass_matched, pass_missing + ["reasoning:empty"]
        reas_hit, reas_matched, reas_missing = _reasoning_figure_hit(reference, reasoning)
        if reas_hit:
            return True, ["reasoning:half"] + reas_matched, reas_missing
        return False, pass_matched + reas_matched, pass_missing + reas_missing

    raise ValueError(f"unknown recall mode={mode!r}")


def reference_overlap_hit(
    reference: str,
    passage_texts: list[str],
    *,
    mode: RecallMode = DEFAULT_RECALL_MODE,
    reasoning: str | None = None,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> tuple[bool, list[str], list[str]]:
    """Return (hit, matched_tokens, missing_tokens) for recall scoring."""
    return score_recall_hit(
        reference,
        passage_texts,
        reasoning=reasoning,
        mode=mode,
        sim_threshold=sim_threshold,
        embed_threshold=embed_threshold,
    )
