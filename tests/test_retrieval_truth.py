"""Tests for canonical retrieval-hit scoring."""

from __future__ import annotations

from crosscheck.eval.retrieval_truth import (
    claim_keywords,
    score_retrieval_hit,
    truth_tokens,
)


def test_truth_tokens_combine_figures_and_keywords() -> None:
    ref = "Item 2. MD&A - iPhone revenue $44,582M (13% increase)"
    claim = "iPhone revenue grew strongly in the quarter."
    all_t, figs, kws = truth_tokens(ref, claim)
    assert any(t.startswith("usd_m:") for t in figs)
    assert "iphone" in kws
    assert "iphone" in all_t


def test_retrieval_hit_reasoning_above_half() -> None:
    ref = "Services revenue $27,423M"
    claim = "Services revenue increased year over year."
    reasoning = (
        "Passage 2 shows Services net sales of $27,423 million, up from the prior year."
    )
    hit, rate, matched, missing, _, _ = score_retrieval_hit(ref, claim, reasoning)
    assert hit is True
    assert rate > 0.5
    assert any(t.startswith("usd_m:") for t in matched)
    assert "services" in matched


def test_retrieval_hit_miss_when_reasoning_lacks_tokens() -> None:
    ref = "Revenue $58.9 billion"
    claim = "Revenue was strong this quarter."
    hit, rate, _, missing, _, _ = score_retrieval_hit(ref, claim, "Demand remained robust.")
    assert hit is False
    assert rate < 0.5
    assert missing


def test_claim_keywords_drop_stopwords() -> None:
    kws = claim_keywords("We saw strong growth in the Services segment.")
    assert "we" not in kws
    assert "services" in kws
