#!/usr/bin/env python3
"""Crosscheck Streamlit demo — golden eval reports + architecture overview.

Run locally::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.config import GOLDEN_REPORTS_DIR, GOLDEN_RUNS_DIR  # noqa: E402
from crosscheck.models import as_fiscal_quarter  # noqa: E402

CLASS_COLORS = {
    "Consistent": "#1b7f4e",
    "Contradictory": "#b42318",
    "Unverifiable": "#8a6d3b",
}

BUCKET_LABELS = {
    "both_ok": "Retrieval hit + NLI correct",
    "nli_hallucination": "Retrieval hit + NLI wrong",
    "lucky_nli": "Retrieval miss + NLI correct",
    "both_fail": "Retrieval miss + NLI wrong",
    "unverifiable_nli": "Unverifiable (NLI only)",
}


def _badge(classification: str) -> str:
    color = CLASS_COLORS.get(classification, "#444")
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.55rem;'
        f"border-radius:999px;background:{color};color:#fff;"
        f'font-size:0.85rem;font-weight:600;">{classification}</span>'
    )


def _discover_golden_periods() -> list[tuple[str, int, str]]:
    found: set[tuple[str, int, str]] = set()
    if not GOLDEN_REPORTS_DIR.exists():
        return []
    for path in GOLDEN_REPORTS_DIR.rglob("*_golden_eval.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ticker = str(data["ticker"]).upper()
            year = int(data["fiscal_year"])
            quarter = as_fiscal_quarter(data["fiscal_quarter"])
            found.add((ticker, year, quarter))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda row: (row[1], row[0], row[2]))


def _golden_report_path(ticker: str, year: int, quarter: str) -> Path:
    from crosscheck.config import golden_report_path

    return golden_report_path(ticker, year, quarter)


def _load_golden_report(ticker: str, year: int, quarter: str) -> dict[str, Any] | None:
    path = _golden_report_path(ticker, year, quarter)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_run_dirs() -> list[Path]:
    if not GOLDEN_RUNS_DIR.exists():
        return []
    runs = [p for p in GOLDEN_RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("eval_")]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _load_run_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_demo_tab() -> None:
    st.subheader("Golden evaluation")
    st.caption(
        "Curated earnings-call claims scored with live hybrid retrieval + Gemini NLI. "
        "Run `python scripts/eval/run_golden_eval.py` to refresh."
    )

    runs = _discover_run_dirs()
    if runs:
        run_names = [p.name for p in runs]
        selected_run = st.selectbox("Latest run metrics", run_names, index=0)
        metrics = _load_run_metrics(GOLDEN_RUNS_DIR / selected_run)
        if metrics:
            meta = metrics.get("meta", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Claims", metrics.get("n_claims", 0))
            c2.metric("NLI (all)", f"{metrics.get('nli_accuracy', 0):.1%}")
            c3.metric(
                "NLI (C/X)",
                f"{metrics.get('nli_accuracy_cx', 0):.1%}",
                help="NLI accuracy on Consistent + Contradictory only",
            )
            c4.metric(
                "Recall@k",
                f"{metrics.get('recall_at_k', 0):.1%}",
                help="Reference figure overlap on Consistent + Contradictory claims",
            )
            c5.metric(
                "U NLI",
                f"{metrics.get('unverifiable_nli_accuracy', 0):.1%}",
            )
            st.caption(
                f"Run `{selected_run}` · sample={meta.get('sample_mode', '?')} · "
                f"reranker={meta.get('reranker', '?')} · nli={meta.get('nli_model', '?')}"
            )

            matrix = metrics.get("confusion_matrix", {})
            if matrix:
                st.markdown("**NLI confusion matrix (counts)**")
                rows = []
                for exp in ("Consistent", "Contradictory", "Unverifiable"):
                    row = {"expected": exp}
                    for pred in ("Consistent", "Contradictory", "Unverifiable"):
                        row[pred] = matrix.get(exp, {}).get(pred, 0)
                    rows.append(row)
                st.dataframe(rows, use_container_width=True, hide_index=True)

            buckets = metrics.get("error_buckets", {})
            if buckets:
                st.markdown("**Retrieval × NLI buckets**")
                bucket_rows = [
                    {"bucket": BUCKET_LABELS.get(k, k), "count": v}
                    for k, v in buckets.items()
                ]
                st.dataframe(bucket_rows, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No golden run metrics yet. Run `scripts/eval/run_golden_eval.py` "
            "after promoting golden claims."
        )

    st.divider()
    periods = _discover_golden_periods()
    if not periods:
        st.warning("No golden period reports under `data/reports/golden/`.")
        return

    tickers = sorted({t for t, _, _ in periods})
    years = sorted({y for _, y, _ in periods}, reverse=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.selectbox("Ticker", tickers, key="demo_ticker")
    year_options = [y for y in years if any(t == ticker and y == yy for t, yy, _ in periods)]
    with c2:
        year = st.selectbox("Fiscal year", year_options, key="demo_year")
    quarter_options = [q for t, y, q in periods if t == ticker and y == year]
    with c3:
        quarter = st.selectbox("Fiscal quarter", quarter_options, key="demo_quarter")

    report = _load_golden_report(ticker, year, quarter)
    if report is None:
        st.warning("Report missing for this period.")
        return

    st.markdown(
        f"**{report.get('company_name', ticker)}** ({ticker}) · "
        f"FY{year} {quarter} · run `{report.get('run_id', 'n/a')}`"
    )
    st.caption(
        f"Sample: {report.get('sample_mode', '?')} · "
        f"reranker={report.get('reranker', '?')} ({report.get('reranker_backend', '?')}) · "
        f"models: {', '.join(report.get('llm_models_used', []))}"
    )

    claims = report.get("claims", [])
    correct = sum(1 for c in claims if c.get("nli_correct"))
    hits = sum(
        1
        for c in claims
        if c.get("expected_nli_label") in {"Consistent", "Contradictory"}
        and c.get("retrieval_hit")
    )
    cx = sum(
        1 for c in claims if c.get("expected_nli_label") in {"Consistent", "Contradictory"}
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Claims in report", len(claims))
    m2.metric("NLI correct", f"{correct}/{len(claims)}" if claims else "0")
    m3.metric("Recall hits", f"{hits}/{cx}" if cx else "n/a")

    for i, row in enumerate(claims, start=1):
        expected = row.get("expected_nli_label", "?")
        predicted = row.get("predicted_nli_label", "?")
        with st.container(border=True):
            st.markdown(
                f"**Claim {i}** expected {_badge(expected)} → "
                f"predicted {_badge(predicted)} · bucket `{row.get('bucket', '?')}`",
                unsafe_allow_html=True,
            )
            st.markdown(f"*Speaker:* {row.get('speaker', 'n/a')}")
            st.write(row.get("claim", ""))
            if row.get("ground_truth_reference"):
                st.caption(f"Reference: {row['ground_truth_reference']}")
            indices = row.get("matched_passage_indices")
            if indices:
                st.caption(f"NLI passages cited: {indices}")
            recall = row.get("retrieval_hit")
            if recall is not None:
                st.caption(
                    f"Retrieval hit: {'yes' if recall else 'no'} · "
                    f"matched={row.get('matched_figures', [])} · "
                    f"missing={row.get('missing_figures', [])}"
                )
            st.markdown(f"**Reasoning.** {row.get('nli_reasoning', '')}")
            with st.expander("Retrieved passages"):
                for passage in row.get("retrieved", []):
                    st.markdown(
                        f"**Rank {passage.get('rank')}** · "
                        f"`{passage.get('chunk_id')}` · score={passage.get('score', 0):.4f}"
                    )
                    text = passage.get("text", "")
                    st.code(text[:4000] + ("…" if len(text) > 4000 else ""), language=None)


def _render_about_tab() -> None:
    st.subheader("Architecture")
    st.markdown(
        """
Crosscheck verifies earnings-call claims against same-period SEC filings:

1. **Ingest** — 10-Q / 10-K HTML from EDGAR; earnings-call transcripts.
2. **Chunk + index** — table-aware chunks; Qdrant hybrid (BGE-M3 dense + BM25 + RRF).
3. **Retrieve** — query expansion by fiscal quarter; Q4 uses dual-path 10-K FY + Q3 10-Q (4+4 passages).
4. **Rerank** — Pinecone Inference `bge-reranker-v2-m3` (default) or local MPS Torch fallback.
5. **NLI** — Gemini structured classification: Consistent / Contradictory / Unverifiable.

**Golden eval** uses 160 curated claims (5 tickers × 4 quarters × 8 labels: 4C/2X/2U).
Default runs sample **2C + 1X + 1U** per period (~80 claims) for cost control.
        """
    )

    st.subheader("Metrics")
    st.markdown(
        """
| Metric | Definition |
|--------|------------|
| **Recall@k** | >50% of truth tokens (reference $/% + claim keywords) found in **NLI reasoning** |
| **NLI accuracy (C/X)** | NLI correct on Consistent + Contradictory only |
| **NLI accuracy (all)** | NLI correct on all labels |
| **U NLI accuracy** | NLI correct on Unverifiable only |
| **Buckets** | `both_ok` · `nli_hallucination` · `lucky_nli` · `both_fail` · `unverifiable_nli` |

Outputs live under `data/reports/golden/` (per period) and `data/runs/golden/eval_*` (aggregate CSV/JSON).
        """
    )

    st.subheader("CLI")
    st.code(
        """python scripts/eval/run_golden_eval.py
python scripts/eval/run_golden_eval.py --ticker AAPL --year 2025 --full
python scripts/run_nli.py --reranker pinecone --nli-model gemini""",
        language="bash",
    )


def main() -> None:
    st.set_page_config(page_title="Crosscheck", page_icon="⌕", layout="wide")
    st.title("Crosscheck")
    st.caption("Earnings-call claims verified against SEC filings.")

    tab_demo, tab_about = st.tabs(["Demo", "About"])
    with tab_demo:
        _render_demo_tab()
    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
