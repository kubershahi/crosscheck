"""Sidecar .meta.json refreshes on fetch even when content already exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crosscheck.ingest import edgar, transcript
from crosscheck.manifest import CompanyPeriod


def test_fetch_transcript_refreshes_meta_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_dir(ticker: str, year: int) -> Path:
        d = tmp_path / str(year) / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(transcript, "transcript_dir", fake_dir)

    period = CompanyPeriod(
        ticker="AMZN",
        name="Amazon",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_url="https://www.roic.ai/quote/AMZN/transcripts/2025-year/2-quarter",
    )
    paths = transcript.transcript_paths(period)
    txt = "Operator\n\nWelcome.\n\nAndrew R. Jassy\n\nThanks everyone.\n"
    paths["txt"].write_text(txt, encoding="utf-8")
    paths["meta"].write_text(
        json.dumps(
            {
                "source": "roic_ai",
                "url": period.transcript_url_str(),
                "fetched_at": "2020-01-01T00:00:00+00:00",
                "ticker": "AMZN",
                "company_name": "Old Name",
                "fiscal_year": 2025,
                "fiscal_quarter": "Q2",
                "chars": 1,
                "call_date": None,
                "call_participants": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = transcript.fetch_transcript(period, force=False)
    assert out == paths["txt"]
    assert paths["txt"].read_text(encoding="utf-8") == txt

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["company_name"] == "Amazon"
    assert meta["fetched_at"] != "2020-01-01T00:00:00+00:00"
    assert meta["chars"] == len(txt)
    names = {p["name"] for p in meta["call_participants"]}
    assert "Andrew R. Jassy" in names


def test_fetch_filing_refreshes_meta_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edgar, "filing_dir", lambda ticker, year: tmp_path)

    def _no_edgar(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("skip path must not call find_filing")

    monkeypatch.setattr(edgar, "find_filing", _no_edgar)

    dest = tmp_path / "AMZN_FY2025_Q2_10Q.html"
    dest.write_text("<html>filing</html>", encoding="utf-8")
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "accessionNumber": "0001018724-25-000086",
                "company_name": "Old",
                "source_url": "https://www.sec.gov/example",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = edgar.fetch_filing(
        "AMZN",
        form="10-Q",
        fiscal_year=2025,
        fiscal_quarter=2,
        company_name="Amazon",
        force=False,
    )
    assert out == dest

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["company_name"] == "Amazon"
    assert meta["ticker"] == "AMZN"
    assert meta["fiscal_year"] == "2025"
    assert meta["fiscal_quarter"] == "Q2"
    assert meta["accessionNumber"] == "0001018724-25-000086"
    assert meta["source_url"] == "https://www.sec.gov/example"
    assert "fetched_at" in meta
