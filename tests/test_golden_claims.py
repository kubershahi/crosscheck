"""Tests for golden-claim number corruption."""

from __future__ import annotations

import pytest

from crosscheck.analysis.golden_claims import corrupt_financial_number


def test_corrupt_changes_dollar_billions() -> None:
    text = "Apple reported revenue of $124.3 billion for the December quarter."
    out = corrupt_financial_number(text, seed="aapl-q1")
    assert out != text
    assert "$" in out
    assert "billion" in out
    assert "124.3" not in out


def test_corrupt_changes_margin_percent() -> None:
    text = "Company gross margin was 46.9% for the December quarter."
    out = corrupt_financial_number(text, seed="margin")
    assert out != text
    assert "46.9" not in out
    assert "%" in out


def test_corrupt_is_seeded_stable() -> None:
    text = "Mac revenue was $9 billion for the December quarter."
    a = corrupt_financial_number(text, seed="stable")
    b = corrupt_financial_number(text, seed="stable")
    c = corrupt_financial_number(text, seed="other")
    assert a == b
    # Different seeds may collide rarely; at least same seed is stable.
    assert isinstance(c, str)


def test_corrupt_raises_without_number() -> None:
    with pytest.raises(ValueError, match="no financial number"):
        corrupt_financial_number("Demand remained strong across regions.", seed="x")
