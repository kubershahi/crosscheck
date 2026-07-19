"""Project paths, optional CIK cache, SEC User-Agent, and public EDGAR URLs.

Layout (year-first)::

    data/raw/filings/{fiscal_year}/{TICKER}/
    data/raw/transcripts/{fiscal_year}/{TICKER}/
    data/chunks/{fiscal_year}/{TICKER}/

Company periods to fetch/chunk come from ``data/manifests/companies.yml``
(see :mod:`crosscheck.manifest`), not from hardcoded defaults here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FILINGS_DIR = RAW_DIR / "filings"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Official SEC JSON map: ticker → CIK (used when a ticker is not in KNOWN_CIKS).
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

load_dotenv(ROOT / ".env")

# Optional local cache of CIKs for common tickers (avoids an extra SEC round-trip
# and survives occasional 403s on company_tickers.json). Not a allowlist —
# unknown tickers are resolved via COMPANY_TICKERS_URL in edgar.resolve_cik.
KNOWN_CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "AVGO": "0001730168",
    "ORCL": "0001341439",
    "AMD": "0000002488",
    "JPM": "0000019617",
    "CRM": "0001108524",
}


def filing_dir(ticker: str, fiscal_year: int) -> Path:
    """Return ``data/raw/filings/{year}/{TICKER}/``."""
    return FILINGS_DIR / str(fiscal_year) / ticker.upper()


def transcript_dir(ticker: str, fiscal_year: int) -> Path:
    """Return ``data/raw/transcripts/{year}/{TICKER}/``."""
    return TRANSCRIPTS_DIR / str(fiscal_year) / ticker.upper()


def chunks_dir(ticker: str, fiscal_year: int) -> Path:
    """Return ``data/chunks/{year}/{TICKER}/``."""
    return CHUNKS_DIR / str(fiscal_year) / ticker.upper()


def get_sec_user_agent() -> str:
    """Load and validate ``SEC_USER_AGENT`` from the environment / ``.env``."""
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "you@email.com" in ua.lower() or "you@domain.com" in ua.lower():
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env to 'ProjectName Your Name you@domain.com' "
            "(required by SEC EDGAR fair-access policy). "
            "Use a real email domain — GitHub noreply addresses often get 403."
        )
    return ua
