"""Orchestrate filing + transcript chunking from a manifest company-period.

Resolves raw files under ``data/raw/.../{year}/{TICKER}/``, runs the filing
and transcript chunkers, and optionally writes JSONL under
``data/chunks/{year}/{TICKER}/``.

Periods must appear in ``data/manifests/companies.yml`` — there is no
hardcoded ticker fallback.
"""

from __future__ import annotations

from pathlib import Path

from crosscheck.chunking.filings import chunk_filing_path
from crosscheck.chunking.store import chunk_output_path, write_chunks_jsonl
from crosscheck.chunking.transcripts import chunk_transcript_path
from crosscheck.config import filing_dir, transcript_dir
from crosscheck.ingest.edgar import filing_filename
from crosscheck.manifest import (
    CompanyPeriod,
    filter_manifest,
    load_manifest,
)
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
    period: CompanyPeriod,
    *,
    filing_path: Path | None = None,
    transcript_path: Path | None = None,
    write: bool = True,
) -> dict[str, list[Chunk] | Path | None]:
    """Chunk filing + transcript for one manifest ``CompanyPeriod``.

    Returns chunk lists plus source/output paths.
    """
    form = period.resolved_form()
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
        company_name=period.name,
        fiscal_year=fy,
        fiscal_quarter=fq,
        form=form,
    )
    filing_chunks = chunk_filing_path(filing, meta)
    transcript_chunks = chunk_transcript_path(transcript, meta)

    filing_out: Path | None = None
    transcript_out: Path | None = None
    if write:
        filing_out = write_chunks_jsonl(
            filing_chunks,
            chunk_output_path(ticker, doc_type="filing", fiscal_year=fy, fiscal_quarter=fq),
        )
        transcript_out = write_chunks_jsonl(
            transcript_chunks,
            chunk_output_path(
                ticker, doc_type="transcript", fiscal_year=fy, fiscal_quarter=fq
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


def periods_from_manifest(
    *,
    tickers: list[str] | None = None,
    manifest_path: Path | str | None = None,
) -> list[CompanyPeriod]:
    """Load manifest rows; raise if a requested ticker is missing."""
    manifest = load_manifest(manifest_path)
    if not tickers:
        rows = list(manifest.companies)
    else:
        rows = filter_manifest(manifest, tickers=tickers)
        wanted = {t.upper() for t in tickers}
        found = {r.ticker for r in rows}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(
                f"Ticker(s) not in manifest: {', '.join(missing)}. "
                f"Add them to data/manifests/companies.yml"
            )
    if not rows:
        raise ValueError("Manifest has no companies to process.")
    return rows
