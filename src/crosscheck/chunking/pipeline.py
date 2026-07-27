"""Discover raw filing/transcript pairs and orchestrate chunking.

Raw files under ``data/raw/`` are the source of truth for this stage. The
ingest manifest is intentionally not read after fetching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crosscheck.chunking.filings import chunk_filing_path
from crosscheck.chunking.store import chunk_output_path, write_chunks_jsonl
from crosscheck.chunking.transcripts import chunk_transcript_path
from crosscheck.config import FILINGS_DIR, TRANSCRIPTS_DIR, filing_dir, transcript_dir
from crosscheck.ingest.edgar import filing_filename
from crosscheck.models import Chunk, DocumentMeta


def resolve_filing_path(
    ticker: str,
    *,
    form: str,
    fiscal_year: int,
    fiscal_quarter: int | str,
    path: Path | None = None,
) -> Path:
    """Return the expected raw filing HTML path (or an explicit override)."""
    from crosscheck.models import quarter_number

    if path is not None:
        return path
    return filing_dir(ticker, fiscal_year) / filing_filename(
        ticker, form, fiscal_year, fiscal_quarter=quarter_number(fiscal_quarter)
    )


def resolve_transcript_path(
    ticker: str,
    *,
    fiscal_year: int,
    fiscal_quarter: int | str,
    path: Path | None = None,
) -> Path:
    """Locate a cleaned transcript ``.txt`` (prefer exact FY/Qn)."""
    from crosscheck.models import as_fiscal_quarter

    if path is not None:
        return path
    tdir = transcript_dir(ticker, fiscal_year)
    q = as_fiscal_quarter(fiscal_quarter)
    preferred = tdir / f"FY{fiscal_year}_{q}.txt"
    if preferred.exists():
        return preferred
    others = sorted(tdir.glob("*.txt"))
    if others:
        return others[0]
    return preferred


def _read_sidecar(path: Path) -> dict[str, Any]:
    """Load a JSON sidecar if present; return {} on missing/invalid."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def filing_meta_path(filing_html: Path) -> Path:
    """Sidecar next to EDGAR HTML (``*.html.meta.json``)."""
    return filing_html.with_suffix(filing_html.suffix + ".meta.json")


def transcript_meta_path(transcript_txt: Path) -> Path:
    """Sidecar next to cleaned transcript (``*.meta.json``)."""
    return transcript_txt.with_suffix(".meta.json")


def _as_str_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out = [str(x) for x in value if x is not None and str(x).strip()]
    return out or None


def enrich_filing_meta(meta: DocumentMeta, filing_path: Path) -> DocumentMeta:
    """Attach filingDate / reportDate / quarter_period fields from the sidecar."""
    raw = _read_sidecar(filing_meta_path(filing_path))
    if not raw:
        return meta
    qp = raw.get("quarter_period")
    qp_dict = qp if isinstance(qp, dict) else {}
    company = str(raw.get("company_name") or meta.company_name)
    return meta.model_copy(
        update={
            "company_name": company,
            "filing_date": (str(raw["filingDate"]) if raw.get("filingDate") else None),
            "report_date": (str(raw["reportDate"]) if raw.get("reportDate") else None),
            "quarter_period_label": (
                str(qp_dict["label"]) if qp_dict.get("label") else None
            ),
            "quarter_months": _as_str_list(qp_dict.get("months")),
        }
    )


def enrich_transcript_meta(
    meta: DocumentMeta, transcript_path: Path
) -> DocumentMeta:
    """Attach ``call_date`` (and company_name if present) from the sidecar."""
    raw = _read_sidecar(transcript_meta_path(transcript_path))
    if not raw:
        return meta
    company = str(raw.get("company_name") or meta.company_name)
    call_date = raw.get("call_date")
    return meta.model_copy(
        update={
            "company_name": company,
            "call_date": str(call_date) if call_date else None,
        }
    )


def build_chunks_for_period(
    period: DocumentMeta,
    *,
    filing_path: Path | None = None,
    transcript_path: Path | None = None,
    write: bool = True,
    force: bool = False,
) -> dict[str, list[Chunk] | Path | bool | None]:
    """Chunk filing + transcript for one discovered period.

    By default, skips sides whose JSONL already exists. Pass ``force=True`` to
    re-chunk and overwrite.

    Returns chunk lists plus source/output paths and skip flags.
    """
    form = period.form or ("10-K" if period.fiscal_quarter == "Q4" else "10-Q")
    fy = period.fiscal_year
    fq = period.fiscal_quarter_num
    ticker = period.ticker

    filing = resolve_filing_path(
        ticker,
        form=form,
        fiscal_year=fy,
        fiscal_quarter=fq,
        path=filing_path,
    )
    transcript = resolve_transcript_path(
        ticker,
        fiscal_year=fy,
        fiscal_quarter=fq,
        path=transcript_path,
    )

    if not filing.exists():
        raise FileNotFoundError(
            f"Missing filing: {filing}. Run: python scripts/fetch_corpus.py --ticker {ticker}"
        )
    if not transcript.exists():
        raise FileNotFoundError(
            f"Missing transcript: {transcript}. "
            f"Run fetch_corpus or drop a .txt under {transcript_dir(ticker, fy)}"
        )

    filing_out_path = chunk_output_path(
        ticker, doc_type=form, fiscal_year=fy, fiscal_quarter=fq
    )
    transcript_out_path = chunk_output_path(
        ticker, doc_type="transcript", fiscal_year=fy, fiscal_quarter=fq
    )
    skip_filing = write and filing_out_path.exists() and not force
    skip_transcript = write and transcript_out_path.exists() and not force

    base = DocumentMeta(
        ticker=ticker,
        company_name=period.company_name,
        fiscal_year=fy,
        fiscal_quarter=fq,
        form=form,
    )

    filing_chunks: list[Chunk] = []
    transcript_chunks: list[Chunk] = []
    filing_out: Path | None = None
    transcript_out: Path | None = None

    if skip_filing:
        filing_out = filing_out_path
    else:
        filing_meta = enrich_filing_meta(base, filing)
        filing_chunks = chunk_filing_path(filing, filing_meta)
        if write:
            filing_doc = filing_chunks[0].doc_type if filing_chunks else form
            filing_out = write_chunks_jsonl(
                filing_chunks,
                chunk_output_path(
                    ticker,
                    doc_type=filing_doc,
                    fiscal_year=fy,
                    fiscal_quarter=fq,
                ),
            )

    if skip_transcript:
        transcript_out = transcript_out_path
    else:
        transcript_meta = enrich_transcript_meta(base, transcript)
        transcript_chunks = chunk_transcript_path(transcript, transcript_meta)
        if write:
            transcript_out = write_chunks_jsonl(
                transcript_chunks,
                transcript_out_path,
            )

    return {
        "filing_chunks": filing_chunks,
        "transcript_chunks": transcript_chunks,
        "filing_path": filing,
        "transcript_path": transcript,
        "filing_out": filing_out,
        "transcript_out": transcript_out,
        "skipped_filing": skip_filing,
        "skipped_transcript": skip_transcript,
    }


def discover_raw_periods(
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    quarters: list[int] | None = None,
    filings_root: Path = FILINGS_DIR,
    transcripts_root: Path = TRANSCRIPTS_DIR,
) -> list[DocumentMeta]:
    """Discover complete raw periods, optionally limited by ticker / year / quarter."""
    periods: list[DocumentMeta] = []
    missing: list[str] = []
    requested = tickers is not None
    year_set = set(years) if years else None
    quarter_set = set(quarters) if quarters else None
    if tickers is None:
        tickers = sorted(
            {
                ticker_dir.name.upper()
                for year_dir in transcripts_root.glob("*")
                if year_dir.is_dir()
                for ticker_dir in year_dir.iterdir()
                if ticker_dir.is_dir()
            }
        )

    for ticker in dict.fromkeys(t.strip().upper() for t in tickers if t.strip()):
        ticker_periods: list[DocumentMeta] = []
        for ticker_dir in sorted(transcripts_root.glob(f"*/{ticker}")):
            try:
                directory_year = int(ticker_dir.parent.name)
            except ValueError:
                continue
            if year_set is not None and directory_year not in year_set:
                continue
            for transcript in sorted(ticker_dir.glob("FY*_Q*.txt")):
                stem_parts = transcript.stem.split("_")
                if len(stem_parts) != 2:
                    continue
                year_text, quarter_text = stem_parts
                if not year_text.startswith("FY") or not quarter_text.startswith("Q"):
                    continue
                try:
                    fiscal_year = int(year_text[2:])
                    fiscal_quarter = int(quarter_text[1:])
                except ValueError:
                    continue
                if fiscal_year != directory_year or fiscal_quarter not in range(1, 5):
                    continue
                if year_set is not None and fiscal_year not in year_set:
                    continue
                if quarter_set is not None and fiscal_quarter not in quarter_set:
                    continue

                form = "10-K" if fiscal_quarter == 4 else "10-Q"
                filing = (
                    filings_root
                    / str(fiscal_year)
                    / ticker
                    / filing_filename(
                        ticker,
                        form,
                        fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                    )
                )
                # Legacy 10-Q name without quarter (pre Day-3 multi-quarter).
                if not filing.exists() and form == "10-Q":
                    legacy = (
                        filings_root
                        / str(fiscal_year)
                        / ticker
                        / f"{ticker}_FY{fiscal_year}_10Q.html"
                    )
                    if legacy.exists():
                        filing = legacy
                if not filing.exists():
                    continue
                company_name = ticker
                transcript_meta = transcript_meta_path(transcript)
                raw_meta = _read_sidecar(transcript_meta)
                if raw_meta.get("company_name"):
                    company_name = str(raw_meta["company_name"])
                ticker_periods.append(
                    DocumentMeta(
                        ticker=ticker,
                        company_name=company_name,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                        form=form,
                    )
                )

        if requested and not ticker_periods:
            missing.append(ticker)
        periods.extend(ticker_periods)

    if missing:
        raise ValueError(
            "No complete raw filing/transcript pair found for: "
            f"{', '.join(missing)}. Run fetch_corpus.py first."
        )
    if not periods:
        raise ValueError(
            "No complete raw filing/transcript pairs found under data/raw. "
            "Run fetch_corpus.py first."
        )
    return periods
