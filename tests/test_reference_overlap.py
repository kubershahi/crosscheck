"""Tests for reference-overlap recall scoring."""

from __future__ import annotations

from crosscheck.eval.reference_overlap import (
    DEFAULT_RECALL_MODE,
    extract_reference_figures,
    reference_overlap_hit,
    score_recall_hit,
)


def test_default_mode_is_half_money() -> None:
    assert DEFAULT_RECALL_MODE == "half_money"


def test_extract_dollar_billions() -> None:
    tokens = extract_reference_figures("Net sales $124.3 billion in the quarter.")
    assert any(t.startswith("usd_m:") for t in tokens)


def test_extract_percent() -> None:
    tokens = extract_reference_figures("Operating margin was 25.4% for the period.")
    assert "pct:25.4" in tokens


def test_overlap_hit_all_figures_present() -> None:
    ref = "Revenue $58.9 billion; margin 25%"
    passages = [
        "Net sales were $58.9 billion for the quarter.",
        "Operating margin expanded to 25% year over year.",
    ]
    hit, matched, missing = reference_overlap_hit(ref, passages, mode="ignore_num")
    assert hit is True
    assert matched
    assert missing == []


def test_half_money_allows_partial() -> None:
    ref = "Revenue $58.9 billion; margin 25%"
    passages = ["Net sales were $58.9 billion."]
    hit, matched, missing = reference_overlap_hit(ref, passages, mode="half_money")
    assert hit is True
    assert any(t.startswith("usd_m:") for t in matched)
    assert "pct:25" in missing


def test_overlap_miss_when_figure_absent() -> None:
    ref = "Revenue $58.9 billion"
    hit, _matched, missing = reference_overlap_hit(ref, ["Other segment data only."])
    assert hit is False
    assert missing


def test_ignore_num_drops_item_section() -> None:
    ref = "Item 2. MD&A - Total operating expenses $15,278M"
    passages = ["Total operating expenses were $15,278 million."]
    hit_strict, _, missing_strict = reference_overlap_hit(ref, passages, mode="strict")
    hit_ignore, matched, missing = reference_overlap_hit(ref, passages, mode="ignore_num")
    assert hit_strict is False
    assert "num:2" in missing_strict
    assert hit_ignore is True
    assert "usd_m:15278" in matched
    assert missing == []


def test_reasoning_half_uses_nli_text() -> None:
    ref = "Item 2. MD&A - iPhone revenue $44,582M vs $39,296M (13% increase)"
    passages = ["Unrelated segment discussion only."]
    reasoning = (
        "Passage 1 shows iPhone net sales of $44,582 million compared to "
        "$39,296 million, a 13% increase."
    )
    hit, matched, _ = score_recall_hit(
        ref,
        passages,
        reasoning=reasoning,
        mode="reasoning_half",
    )
    assert hit is True
    assert "usd_m:44582" in matched


def test_hybrid_passage_or_reasoning() -> None:
    ref = "Services revenue $27,423M"
    passages = ["Services generated revenue of $27,423 million."]
    reasoning = "No figures cited."
    hit, _, _ = score_recall_hit(ref, passages, reasoning=reasoning, mode="hybrid")
    assert hit is True
