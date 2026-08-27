#!/usr/bin/env python3
"""Crosscheck Streamlit demo — view (and optionally run) claim vs filing NLI reports.

Run locally::

    streamlit run streamlit_app.py

Selectors are prefixed ticker / year / quarter. Loads saved reports from
``data/reports/`` by default; can re-run the Qdrant hybrid + NLI pipeline when
claims exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.pipeline import run_pipeline  # noqa: E402
from crosscheck.config import CLAIMS_DIR, REPORTS_DIR, claims_path, report_path  # noqa: E402
from crosscheck.models import DocumentMeta, PipelineReport, as_fiscal_quarter  # noqa: E402

CLASS_COLORS = {
    "Consistent": "#1b7f4e",
    "Contradictory": "#b42318",
    "Unverifiable": "#8a6d3b",
}


def _discover_periods() -> list[tuple[str, int, str]]:
    """Union of periods that have claims or reports on disk."""
    found: set[tuple[str, int, str]] = set()
    for root in (CLAIMS_DIR, REPORTS_DIR):
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ticker = str(data["ticker"]).upper()
                year = int(data["fiscal_year"])
                quarter = as_fiscal_quarter(data["fiscal_quarter"])
                found.add((ticker, year, quarter))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return sorted(found, key=lambda row: (row[1], row[0], row[2]))


def _load_report(ticker: str, year: int, quarter: str) -> PipelineReport | None:
    path = report_path(ticker, year, quarter)
    if not path.exists():
        return None
    return PipelineReport.model_validate_json(path.read_text(encoding="utf-8"))


def _company_name(ticker: str, year: int, quarter: str) -> str:
    for loader in (_load_report,):
        report = loader(ticker, year, quarter)
        if report is not None:
            return report.company_name
    claims = claims_path(ticker, year, quarter)
    legacy = claims.with_suffix(".json")
    if claims.exists():
        from crosscheck.io.jsonl import iter_json_objects

        for row in iter_json_objects(claims):
            name = row.get("company_name")
            if name:
                return str(name)
            break
    elif legacy.exists():
        data = json.loads(legacy.read_text(encoding="utf-8"))
        return str(data.get("company_name") or ticker)
    return ticker


def _badge(classification: str) -> str:
    color = CLASS_COLORS.get(classification, "#444")
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.55rem;'
        f"border-radius:999px;background:{color};color:#fff;"
        f'font-size:0.85rem;font-weight:600;">{classification}</span>'
    )


def main() -> None:
    """Streamlit entry: period selectors + report viewer / runner."""
    st.set_page_config(
        page_title="Crosscheck",
        page_icon="⌕",
        layout="wide",
    )
    st.title("Crosscheck")
    st.caption(
        "Earnings-call claims verified against the same-period 10-Q via "
        "Qdrant hybrid retrieval (dense + BM25 + RRF) and Gemini NLI."
    )

    periods = _discover_periods()
    if not periods:
        st.warning(
            "No claims or reports under `data/claims` / `data/reports`. "
            "Run `extract_claims.py` and `run_nli.py` first."
        )
        return

    tickers = sorted({t for t, _, _ in periods})
    years = sorted({y for _, y, _ in periods}, reverse=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.selectbox("Ticker", tickers, index=0)
    year_options = [y for y in years if any(t == ticker and y == yy for t, yy, _ in periods)]
    with c2:
        year = st.selectbox("Fiscal year", year_options, index=0)
    quarter_options = [
        q for t, y, q in periods if t == ticker and y == year
    ]
    with c3:
        quarter = st.selectbox("Fiscal quarter", quarter_options, index=0)

    company = _company_name(ticker, year, quarter)
    st.subheader(f"{company} ({ticker}) · FY{year} {quarter}")

    claims_file = claims_path(ticker, year, quarter)
    has_claims = claims_file.exists() or claims_file.with_suffix(".json").exists()
    has_report = report_path(ticker, year, quarter).exists()
    st.caption(
        f"Claims on disk: {'yes' if has_claims else 'no'} · "
        f"Report on disk: {'yes' if has_report else 'no'}"
    )

    run_col, load_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "Run analysis",
            type="primary",
            disabled=not has_claims,
            help="Requires claims JSON + Qdrant filings index + GOOGLE_API_KEY",
        )
    with load_col:
        if not has_claims:
            st.info("No claims for this period — extract with `scripts/extract_claims.py`.")

    report: PipelineReport | None = None
    if run_clicked:
        with st.spinner("Qdrant hybrid retrieve + rerank + NLI …"):
            period = DocumentMeta(
                ticker=ticker,
                company_name=company,
                fiscal_year=year,
                fiscal_quarter=quarter,
            )
            report = run_pipeline(period)
            out = report_path(ticker, year, quarter)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                report.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            st.success(f"Wrote {out}")
    else:
        report = _load_report(ticker, year, quarter)

    if report is None:
        st.warning("No report yet. Click **Run analysis** or generate via `run_nli.py`.")
        return

    counts = {"Consistent": 0, "Contradictory": 0, "Unverifiable": 0}
    for finding in report.findings:
        counts[finding.classification] = counts.get(finding.classification, 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Findings", len(report.findings))
    m2.metric("Consistent", counts["Consistent"])
    m3.metric("Contradictory", counts["Contradictory"])
    m4.metric("Unverifiable", counts["Unverifiable"])
    st.caption(f"NLI model: `{report.llm_model_used}` · generated {report.generated_at}")

    for i, finding in enumerate(report.findings, start=1):
        with st.container(border=True):
            st.markdown(
                f"**Claim {i}** {_badge(finding.classification)} · "
                f"confidence {finding.confidence_score:.2f}",
                unsafe_allow_html=True,
            )
            st.markdown(f"*Speaker:* {finding.source_speaker}")
            st.write(finding.transcript_claim)
            st.markdown(f"**Reasoning.** {finding.reasoning}")
            with st.expander("Retrieved filing passages"):
                ids = getattr(finding, "chunk_ids", None) or []
                gids = getattr(finding, "global_ids", None) or []
                for j, passage in enumerate(finding.retrieved_filing_passages, start=1):
                    cid = ids[j - 1] if j - 1 < len(ids) else "n/a"
                    gid = gids[j - 1] if j - 1 < len(gids) else "n/a"
                    st.markdown(
                        f"**Passage {j}** — `chunk_id={cid}` · `global_id={gid}`"
                    )
                    st.code(
                        passage[:4000] + ("…" if len(passage) > 4000 else ""),
                        language=None,
                    )


if __name__ == "__main__":
    main()
