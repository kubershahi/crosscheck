"""Tests for golden mini sampler."""

from __future__ import annotations

from crosscheck.eval.sampler import MINI_PER_LABEL, sample_mini_golden
from crosscheck.models import GoldenClaim


def _claim(claim_id: str, label: str) -> GoldenClaim:
    return GoldenClaim(
        claim_id=claim_id,
        ticker="AAPL",
        company_name="Apple Inc.",
        fiscal_year=2025,
        fiscal_quarter="Q1",
        speaker="CEO",
        claim=f"claim {claim_id}",
        expected_nli_label=label,  # type: ignore[arg-type]
        is_in_filing=True,
        is_in_table=True,
        ground_truth_reference="$1 billion",
    )


def test_sample_mini_quota() -> None:
    claims = [
        _claim("c1", "Consistent"),
        _claim("c2", "Consistent"),
        _claim("c3", "Consistent"),
        _claim("x1", "Contradictory"),
        _claim("x2", "Contradictory"),
        _claim("u1", "Unverifiable"),
        _claim("u2", "Unverifiable"),
    ]
    picked = sample_mini_golden(claims, seed="test")
    assert len(picked) == sum(MINI_PER_LABEL.values())
    labels = [c.expected_nli_label for c in picked]
    assert labels.count("Consistent") == 2
    assert labels.count("Contradictory") == 1
    assert labels.count("Unverifiable") == 1


def test_sample_mini_stable() -> None:
    claims = [_claim(f"c{i}", "Consistent") for i in range(4)]
    claims += [_claim("x1", "Contradictory")]
    claims += [_claim("u1", "Unverifiable")]
    a = sample_mini_golden(claims, seed="fixed")
    b = sample_mini_golden(claims, seed="fixed")
    assert [c.claim_id for c in a] == [c.claim_id for c in b]
