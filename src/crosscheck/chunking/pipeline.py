"""Discover raw filing/transcript pairs and orchestrate chunking.

Raw files under ``data/raw/`` are the source of truth for this stage. The
ingest manifest is intentionally not read after fetching.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    path: Path | None = None,
) -> Path:
    """Return the expected raw filing HTML path (or an explicit override)."""
    if path is not None:
        return path
    return filing_dir(ticker, fiscal_year) / filing_filename(ticker, form, fiscal_year)


def resolve_transcript_path(
    ticker: str,
    *,
    fiscal_year: int,
    fiscal_quarter: int,
    path: Path | None = None,
) -> Path:
    """Locate a cleaned transcript ``.txt`` (prefer exact FY/Qn)."""
    if path is not None:
        return path
    tdir = transcript_dir(ticker, fiscal_year)
    preferred = tdir / f"FY{fiscal_year}_Q{fiscal_quarter}.txt"
    if preferred.exists():
        return preferred
    others = sorted(tdir.glob("*.txt"))
    if others:
        return others[0]
    return preferred


def build_chunks_for_period(
    period: DocumentMeta,
    *,
    filing_path: Path | None = None,
    transcript_path: Path | None = None,
    write: bool = True,
) -> dict[str, list[Chunk] | Path | None]:
    """Chunk filing + transcript for one discovered period.

    Returns chunk lists plus source/output paths.
    """
    form = period.form or ("10-K" if period.fiscal_quarter == 4 else "10-Q")
    fy = period.fiscal_year
    fq = period.fiscal_quarter
    ticker = period.ticker

    filing = resolve_filing_path(ticker, form=form, fiscal_year=fy, path=filing_path)
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

    meta = DocumentMeta(
        ticker=ticker,
        company_name=period.company_name,
        fiscal_year=fy,
        fiscal_quarter=fq,
        form=form,
    )
    filing_chunks = chunk_filing_path(filing, meta)
    transcript_chunks = chunk_transcript_path(transcript, meta)

    filing_out: Path | None = None
    transcript_out: Path | None = None
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
        transcript_out = write_chunks_jsonl(
            transcript_chunks,
            chunk_output_path(
                ticker,
                doc_type="transcript",
                fiscal_year=fy,
                fiscal_quarter=fq,
            ),
        )

    return {
        "filing_chunks": filing_chunks,
        "transcript_chunks": transcript_chunks,
        "filing_path": filing,
        "transcript_path": transcript,
        "filing_out": filing_out,
        "transcript_out": transcript_out,
    }


def discover_raw_periods(
    *,
    tickers: list[str] | None = None,
    filings_root: Path = FILINGS_DIR,
    transcripts_root: Path = TRANSCRIPTS_DIR,
) -> list[DocumentMeta]:
    """Discover complete raw periods, optionally limited to requested tickers."""
    periods: list[DocumentMeta] = []
    missing: list[str] = []
    requested = tickers is not None
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

                form = "10-K" if fiscal_quarter == 4 else "10-Q"
                filing = (
                    filings_root
                    / str(fiscal_year)
                    / ticker
                    / filing_filename(ticker, form, fiscal_year)
                )
                if not filing.exists():
                    continue
                company_name = ticker
                transcript_meta = transcript.with_suffix(".meta.json")
                if transcript_meta.exists():
                    try:
                        raw_meta = json.loads(
                            transcript_meta.read_text(encoding="utf-8")
                        )
                        company_name = str(raw_meta.get("company_name") or ticker)
                    except (json.JSONDecodeError, OSError):
                        pass
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
