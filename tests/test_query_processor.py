"""Unit tests for financial query preprocessing (no LLM / network)."""

from __future__ import annotations

import pytest

from crosscheck.retrieval.query_processor import (
    TemporalScope,
    classify_claim_temporal_scope,
    expand_financial_units,
    fy_annual_retrieval_query,
    prepare_claim_query,
    q3_ytd_retrieval_query,
)


class TestExpandFinancialUnits:
    def test_dollar_billions_float(self) -> None:
        q = "Microsoft reported total revenue of $65.6 billion in the third quarter"
        out = expand_financial_units(q)
        assert out == (
            "Microsoft reported total revenue (net sales) of $65.6 billion "
            "(65600 million) in the third quarter"
        )

    def test_dollar_billions_b_suffix(self) -> None:
        assert expand_financial_units("Cloud revenue hit $24.5B last year") == (
            "Cloud revenue (net sales) hit $24.5B (24500 million) last year"
        )

    def test_plain_b_suffix(self) -> None:
        assert expand_financial_units("Guidance of 24B for the year") == (
            "Guidance of 24B (24000 million) for the year"
        )

    def test_billion_dollars(self) -> None:
        assert "24500 million" in expand_financial_units(
            "about 24.5 billion dollars of cash"
        )

    def test_integer_billions(self) -> None:
        out = expand_financial_units("$24 billion in revenue")
        assert out == "$24 billion (24000 million) in revenue (net sales)"

    def test_multiple_amounts(self) -> None:
        out = expand_financial_units("$10 billion and $2.5B sequentially")
        assert "(10000 million)" in out
        assert "(2500 million)" in out

    def test_no_billions_noop(self) -> None:
        q = "Gross margin was 46.9% in Q2"
        assert expand_financial_units(q) == q

    def test_revenue_with_numeric_expands_net_sales(self) -> None:
        out = expand_financial_units("Revenue was $500 million in Q2")
        assert out == "Revenue (net sales) was $500 million in Q2"

    def test_revenue_with_percent_expands_net_sales(self) -> None:
        out = expand_financial_units("Revenue grew 16% year over year")
        assert out == "Revenue (net sales) grew 16% year over year"

    def test_revenue_without_numeric_not_expanded(self) -> None:
        q = "Q4 revenue rose sequentially"
        assert expand_financial_units(q) == q

    def test_revenue_already_expanded_skipped(self) -> None:
        q = "Revenue (net sales) was $500 million"
        assert expand_financial_units(q) == q

    def test_already_expanded_skipped(self) -> None:
        q = "$65.6 billion (65600 million) already done"
        assert expand_financial_units(q) == q

    def test_empty_and_none_safe(self) -> None:
        assert expand_financial_units("") == ""


class TestTemporalScope:
    """Temporal routing is gated on fiscal_quarter == Q4."""

    @pytest.mark.parametrize("quarter", ["Q1", "Q2", "Q3", 1, 2, 3])
    def test_q1_q3_always_standard_even_with_full_year_or_q4_words(
        self, quarter: str | int
    ) -> None:
        text = "Full year and Q4 fourth quarter revenue both rose in fiscal year 2025"
        assert (
            classify_claim_temporal_scope(text, fiscal_quarter=quarter)
            == TemporalScope.STANDARD_QUARTER
        )

    def test_no_quarter_context_is_standard(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "Full year revenue reached a record for fiscal year 2025"
            )
            == TemporalScope.STANDARD_QUARTER
        )

    def test_q4_full_year_only(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "Full year revenue reached a record for fiscal year 2025",
                fiscal_quarter="Q4",
            )
            == TemporalScope.FULL_YEAR_ONLY
        )

    def test_q4_trailing_twelve_months(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "Trailing twelve month period revenue rose 10%",
                fiscal_quarter="Q4",
            )
            == TemporalScope.FULL_YEAR_ONLY
        )
        assert (
            classify_claim_temporal_scope(
                "On a trailing 12 months basis, operating income grew",
                fiscal_quarter="Q4",
            )
            == TemporalScope.FULL_YEAR_ONLY
        )

    def test_q4_fy_token(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "FY2025 operating income rose", fiscal_quarter="Q4"
            )
            == TemporalScope.FULL_YEAR_ONLY
        )

    def test_q4_annual(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "Annual EPS grew double digits", fiscal_quarter=4
            )
            == TemporalScope.FULL_YEAR_ONLY
        )

    @pytest.mark.parametrize(
        "text",
        [
            "In the fourth quarter, revenue was up",
            "Q4 cloud revenue surpassed $40B",
            "The final quarter saw strong growth",
            "4th quarter operating income",
            "fourth-quarter demand was solid",
            "Revenue in the September quarter was $94.9 billion",
            "December quarter results exceeded expectations",
            "The January quarter was our best ever",
        ],
    )
    def test_q4_composite(self, text: str) -> None:
        assert (
            classify_claim_temporal_scope(text, fiscal_quarter="Q4")
            == TemporalScope.Q4_COMPOSITE
        )

    def test_q4_beats_full_year(self) -> None:
        text = "Fourth quarter and full year revenue both set records"
        assert (
            classify_claim_temporal_scope(text, fiscal_quarter="Q4")
            == TemporalScope.Q4_COMPOSITE
        )

    def test_q4_period_neutral_text_defaults_to_q4_composite(self) -> None:
        assert (
            classify_claim_temporal_scope(
                "Microsoft Cloud revenue increased 22% to $38.9 billion",
                fiscal_quarter="Q4",
            )
            == TemporalScope.Q4_COMPOSITE
        )


class TestPrepareClaimQuery:
    def test_plan_standard_q3(self) -> None:
        plan = prepare_claim_query(
            "Microsoft reported total revenue of $65.6 billion in the third quarter",
            fiscal_quarter="Q3",
        )
        assert plan.temporal_scope == TemporalScope.STANDARD_QUARTER
        assert plan.required_doc_types == ["10-Q"]
        assert plan.nli_instruction_suffix == ""
        assert "(65600 million)" in plan.processed_query
        assert "revenue (net sales)" in plan.processed_query

    def test_q1_ignores_full_year_words(self) -> None:
        plan = prepare_claim_query(
            "Full year fiscal year revenue was $245 billion",
            fiscal_quarter="Q1",
        )
        assert plan.temporal_scope == TemporalScope.STANDARD_QUARTER
        assert plan.required_doc_types == ["10-Q"]
        assert plan.nli_instruction_suffix == ""
        assert "(245000 million)" in plan.processed_query

    def test_plan_full_year_q4(self) -> None:
        plan = prepare_claim_query(
            "Full year fiscal year revenue was $245 billion",
            fiscal_quarter="Q4",
        )
        assert plan.temporal_scope == TemporalScope.FULL_YEAR_ONLY
        assert plan.required_doc_types == ["10-K"]
        assert plan.nli_instruction_suffix == ""
        assert "(245000 million)" in plan.processed_query

    def test_plan_q4_composite(self) -> None:
        plan = prepare_claim_query(
            "In Q4, revenue was $70 billion for the full year period",
            fiscal_quarter="Q4",
        )
        assert plan.temporal_scope == TemporalScope.Q4_COMPOSITE
        assert plan.required_doc_types == ["10-K", "10-Q"]
        assert "Q4 ARITHMETIC VERIFICATION RULE" in plan.nli_instruction_suffix
        assert "Write the subtraction" in plan.nli_instruction_suffix
        assert "(70000 million)" in plan.processed_query


class TestFyAnnualRetrievalQuery:
    def test_replaces_month_named_quarter(self) -> None:
        out = fy_annual_retrieval_query(
            "Revenue in the September quarter was $94.9 billion (94900 million)"
        )
        assert "September quarter" not in out
        assert "12 month (year ended) period" in out

    def test_replaces_q4_and_keeps_value(self) -> None:
        expanded = "In Q4, revenue was $70 billion (70000 million)"
        out = fy_annual_retrieval_query(expanded)
        assert "Q4" not in out
        assert out == (
            "In 12 month (year ended) period, revenue was "
            "$70 billion (70000 million)"
        )

    def test_replaces_fourth_quarter_and_keeps_amounts(self) -> None:
        expanded = (
            "For the fourth quarter of fiscal 2025, revenue was "
            "$39.3 billion (39300 million)."
        )
        out = fy_annual_retrieval_query(expanded)
        assert out == (
            "For the 12 month (year ended) period of fiscal 2025, revenue was "
            "$39.3 billion (39300 million)."
        )

    def test_prefixes_when_no_period_wording(self) -> None:
        expanded = (
            "International segment operating income was $1 billion (1000 million) "
            "with an operating margin of 2.1%"
        )
        out = fy_annual_retrieval_query(expanded)
        assert out == (
            "for 12 month (year ended) period International segment operating "
            "income was $1 billion (1000 million) with an operating margin of 2.1%"
        )


class TestQ3YtdRetrievalQuery:
    def test_replaces_month_named_quarter(self) -> None:
        out = q3_ytd_retrieval_query(
            "Revenue in the September quarter was $94.9 billion (94900 million)"
        )
        assert "September quarter" not in out
        assert "nine months ended period" in out

    def test_replaces_q4_and_keeps_value(self) -> None:
        expanded = "In Q4, revenue was $70 billion (70000 million)"
        out = q3_ytd_retrieval_query(expanded)
        assert "Q4" not in out
        assert out == (
            "In nine months ended period, revenue was $70 billion (70000 million)"
        )

    def test_replaces_fourth_quarter_and_keeps_amounts(self) -> None:
        expanded = (
            "For the fourth quarter of fiscal 2025, Data Center revenue was "
            "$35.6 billion (35600 million), up 16%."
        )
        out = q3_ytd_retrieval_query(expanded)
        assert out == (
            "For the nine months ended period of fiscal 2025, Data Center revenue was "
            "$35.6 billion (35600 million), up 16%."
        )

    def test_prefixes_when_no_period_wording(self) -> None:
        expanded = (
            "International segment operating income was $1 billion (1000 million) "
            "with an operating margin of 2.1%"
        )
        out = q3_ytd_retrieval_query(expanded)
        assert out == (
            "for nine months ended period International segment operating income "
            "was $1 billion (1000 million) with an operating margin of 2.1%"
        )


class TestRetrievalPathLog:
    def test_q4_composite_dual(self) -> None:
        from crosscheck.retrieval.query_processor import retrieval_path_log

        plan = prepare_claim_query("Q4 revenue rose", fiscal_quarter="Q4")
        assert "dual-path" in retrieval_path_log(plan)

    def test_q4_full_year_single(self) -> None:
        from crosscheck.retrieval.query_processor import retrieval_path_log

        plan = prepare_claim_query(
            "Full year fiscal year revenue rose", fiscal_quarter="Q4"
        )
        assert "single-path" in retrieval_path_log(plan)
        assert "10-K FY" in retrieval_path_log(plan)

    def test_q3_unchanged(self) -> None:
        from crosscheck.retrieval.query_processor import retrieval_path_log

        plan = prepare_claim_query("Q3 revenue rose", fiscal_quarter="Q3")
        assert retrieval_path_log(plan) == "retrieve …"
