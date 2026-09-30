"""Tests for golden eval metric aggregation."""

from __future__ import annotations

from crosscheck.eval.metrics import (
    ClaimScoreRow,
    assign_bucket,
    summarize,
)


def _row(
    *,
    claim_id: str,
    expected: str,
    predicted: str,
    retrieval_hit: bool | None,
) -> ClaimScoreRow:
    return ClaimScoreRow(
        claim_id=claim_id,
        ticker="AAPL",
        fiscal_year=2025,
        fiscal_quarter="Q1",
        expected=expected,  # type: ignore[arg-type]
        predicted=predicted,  # type: ignore[arg-type]
        nli_correct=expected == predicted,
        retrieval_hit=retrieval_hit,
        bucket=assign_bucket(
            expected=expected,  # type: ignore[arg-type]
            predicted=predicted,  # type: ignore[arg-type]
            retrieval_hit=retrieval_hit,
        ),
        nli_confidence=0.9,
    )


def test_assign_bucket_matrix() -> None:
    assert assign_bucket(expected="Consistent", predicted="Consistent", retrieval_hit=True) == "both_ok"
    assert (
        assign_bucket(expected="Consistent", predicted="Contradictory", retrieval_hit=True)
        == "nli_hallucination"
    )
    assert assign_bucket(expected="Contradictory", predicted="Contradictory", retrieval_hit=False) == "lucky_nli"


    assert assign_bucket(expected="Unverifiable", predicted="Consistent", retrieval_hit=None) == "unverifiable_nli"


def test_summarize_headline_metrics() -> None:
    rows = [
        _row(claim_id="c1", expected="Consistent", predicted="Consistent", retrieval_hit=True),
        _row(claim_id="c2", expected="Contradictory", predicted="Contradictory", retrieval_hit=False),
        _row(claim_id="u1", expected="Unverifiable", predicted="Unverifiable", retrieval_hit=None),
    ]
    summary = summarize(rows)
    assert summary["n_claims"] == 3
    assert summary["recall_n"] == 2
    assert summary["recall_hits"] == 1
    assert summary["nli_accuracy"] == 1.0
    assert summary["nli_accuracy_cx"] == 1.0
    assert summary["nli_cx_n"] == 2
    assert summary["error_buckets"]["both_ok"] == 1
    assert summary["error_buckets"]["lucky_nli"] == 1


def test_summarize_nli_accuracy_cx_excludes_u() -> None:
    rows = [
        _row(claim_id="c1", expected="Consistent", predicted="Consistent", retrieval_hit=True),
        _row(claim_id="x1", expected="Contradictory", predicted="Consistent", retrieval_hit=True),
        _row(claim_id="u1", expected="Unverifiable", predicted="Unverifiable", retrieval_hit=None),
    ]
    summary = summarize(rows)
    # all-3: 2/3; C/X only: 1/2
    assert summary["nli_accuracy"] == 2 / 3
    assert summary["nli_accuracy_cx"] == 0.5
    assert summary["nli_cx_correct"] == 1
    assert summary["unverifiable_nli_accuracy"] == 1.0
