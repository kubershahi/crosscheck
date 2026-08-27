"""Unit tests for Q4 composite dual-path retrieval (mocked, no Qdrant)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crosscheck.models import IndexedChunk
from crosscheck.retrieval.index import (
    CorpusIndex,
    Q4_COMPOSITE_PATH_K,
    _dedupe_chunks,
    retrieve_claim_passages,
)
from crosscheck.retrieval.query_processor import TemporalScope


def _chunk(chunk_id: str, doc_type: str, fiscal_period: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        global_id=hash(chunk_id) % 10000,
        text=f"text for {chunk_id}",
        ticker="NVDA",
        company_name="NVIDIA",
        fiscal_year=2025,
        fiscal_period=fiscal_period,  # type: ignore[arg-type]
        doc_type=doc_type,  # type: ignore[arg-type]
        section="ITEM 1",
        is_table=True,
    )


class TestDedupeChunks:
    def test_drops_duplicate_chunk_ids(self) -> None:
        c = _chunk("a", "10-K", "FY")
        out = _dedupe_chunks([(c, 1.0), (c, 0.5), (_chunk("b", "10-Q", "Q3"), 0.3)])
        assert len(out) == 2
        assert out[0][0].chunk_id == "a"
        assert out[1][0].chunk_id == "b"


class TestRetrieveClaimPassages:
    @pytest.fixture
    def filings_index(self) -> CorpusIndex:
        return CorpusIndex(
            corpus="filings",
            chunks=[],
            embedding_model="test",
            backend="qdrant",
        )

    def test_q3_standard_uses_single_hybrid_path(
        self, filings_index: CorpusIndex
    ) -> None:
        model = MagicMock()
        reranker = MagicMock()
        fake_hits = [(_chunk("q3_1", "10-Q", "Q3"), 0.9)]

        with patch(
            "crosscheck.retrieval.index.hybrid_retrieve",
            return_value=fake_hits,
        ) as mock_hybrid:
            with patch(
                "crosscheck.retrieval.rerank.rerank_claim_passages",
                return_value=fake_hits,
            ) as mock_rerank:
                out = retrieve_claim_passages(
                    "Revenue was $10 billion in Q3",
                    filings_index,
                    model,
                    k=5,
                    ticker="NVDA",
                    fiscal_year=2025,
                    fiscal_quarter="Q3",
                    rerank_pool_k=20,
                    reranker=reranker,
                    use_reranker=True,
                )

        mock_hybrid.assert_called_once()
        mock_rerank.assert_called_once()
        assert out == fake_hits

    def test_q4_composite_dual_path_no_post_merge_rerank(
        self, filings_index: CorpusIndex, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = MagicMock()
        reranker = MagicMock()
        path_a = [(_chunk(f"10k_{i}", "10-K", "FY"), 1.0 - i * 0.1) for i in range(4)]
        path_b = [(_chunk(f"q3_{i}", "10-Q", "Q3"), 0.5 - i * 0.1) for i in range(4)]

        call_count = {"n": 0}

        def fake_path(*_args, **_kwargs):
            call_count["n"] += 1
            return path_a if call_count["n"] == 1 else path_b

        with patch(
            "crosscheck.retrieval.index._qdrant_hybrid_path",
            side_effect=fake_path,
        ) as mock_path:
            with patch(
                "crosscheck.retrieval.index.hybrid_retrieve",
            ) as mock_hybrid:
                out = retrieve_claim_passages(
                    "Q4 revenue was $50 billion",
                    filings_index,
                    model,
                    k=5,
                    ticker="NVDA",
                    fiscal_year=2025,
                    fiscal_quarter="Q4",
                    rerank_pool_k=30,
                    reranker=reranker,
                    use_reranker=True,
                )

        mock_hybrid.assert_not_called()
        assert mock_path.call_count == 2
        assert len(out) == Q4_COMPOSITE_PATH_K * 2
        doc_types = {c.doc_type for c, _ in out}
        assert doc_types == {"10-K", "10-Q"}
        output = capsys.readouterr().out
        assert "path A [10-K FY] query:" in output
        assert "path A hybrid+rerank → 4/4" in output
        assert "path B [Q3 10-Q] query:" in output
        assert "12 month (year ended) period" in output
        assert "nine months ended period" in output
        assert "$50 billion (50000 million)" in output
        assert "path B hybrid+rerank → 4/4" in output

    def test_q4_composite_not_invoked_for_q1(
        self, filings_index: CorpusIndex
    ) -> None:
        from crosscheck.retrieval.query_processor import prepare_claim_query

        plan = prepare_claim_query("Q1 revenue rose", fiscal_quarter="Q1")
        assert plan.temporal_scope == TemporalScope.STANDARD_QUARTER
        assert plan.nli_instruction_suffix == ""

        model = MagicMock()
        with patch(
            "crosscheck.retrieval.index._retrieve_q4_composite",
        ) as mock_composite:
            with patch(
                "crosscheck.retrieval.index.hybrid_retrieve",
                return_value=[],
            ):
                retrieve_claim_passages(
                    "Q1 revenue rose",
                    filings_index,
                    model,
                    k=5,
                    fiscal_quarter="Q1",
                    use_reranker=False,
                )
        mock_composite.assert_not_called()
