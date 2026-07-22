"""Project paths, optional CIK cache, SEC User-Agent, and public EDGAR URLs.

Layout (year-first)::

    data/raw/filings/{fiscal_year}/{TICKER}/
    data/raw/transcripts/{fiscal_year}/{TICKER}/
    data/chunks/{fiscal_year}/{TICKER}/          # per-company JSONL (stateless)
    data/claims/{fiscal_year}/{TICKER}/
    data/indices/filings/                        # 10-K / 10-Q only (NLI retrieval)
      all_chunks.jsonl
      embeddings.npy
      index.faiss
      manifest.json
    data/indices/transcripts/                    # transcript corpus (separate)
      all_chunks.jsonl
      embeddings.npy
      index.faiss
      manifest.json
    data/reports/{fiscal_year}/{TICKER}/

Company periods to fetch come from ``data/manifests/companies.yml``. Downstream
stages discover inputs from ``data/raw``, ``data/chunks``, and ``data/claims``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FILINGS_DIR = RAW_DIR / "filings"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
CLAIMS_DIR = DATA_DIR / "claims"
INDICES_DIR = DATA_DIR / "indices"
REPORTS_DIR = DATA_DIR / "reports"
MANIFESTS_DIR = DATA_DIR / "manifests"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Gemini model fallback order (highest free-tier RPM/RPD first).
# Both development and production profiles use this same rank via resolve_llm_models().
LLM_MODEL_RANK: list[str] = [
    "gemini-3.1-flash-lite",  # ~15 RPM / 500 RPD
    "gemini-2.5-flash-lite",  # ~10 RPM / 20 RPD
    "gemini-3-flash-preview",  # Flash tier (~5 RPM / 20 RPD)
    "gemini-2.5-flash",
    "gemini-3.5-flash",
]

# Backward-compatible alias.
DEVELOPMENT_LLM_RANK = LLM_MODEL_RANK

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


def claims_path(ticker: str, fiscal_year: int, fiscal_quarter: int) -> Path:
    """Return the persisted claims JSON path for one company-period."""
    ticker = ticker.upper()
    name = f"{ticker}_FY{fiscal_year}_Q{fiscal_quarter}_claims.json"
    return CLAIMS_DIR / str(fiscal_year) / ticker / name


def indices_dir(ticker: str, fiscal_year: int) -> Path:
    """Deprecated per-ticker path; prefer corpus dirs under ``data/indices/``."""
    return INDICES_DIR / str(fiscal_year) / ticker.upper()


def filings_index_path() -> Path:
    """Return ``data/indices/filings/index.faiss``."""
    return INDICES_DIR / "filings" / "index.faiss"


def transcripts_index_path() -> Path:
    """Return ``data/indices/transcripts/index.faiss``."""
    return INDICES_DIR / "transcripts" / "index.faiss"


def unified_master_index_path() -> Path:
    """Deprecated alias for :func:`filings_index_path`."""
    return filings_index_path()


def report_path(ticker: str, fiscal_year: int, fiscal_quarter: int) -> Path:
    """Return ``data/reports/{year}/{TICKER}/{TICKER}_FY{y}_Q{n}.json``."""
    ticker = ticker.upper()
    name = f"{ticker}_FY{fiscal_year}_Q{fiscal_quarter}.json"
    return REPORTS_DIR / str(fiscal_year) / ticker / name


def get_embedding_device_pref() -> str:
    """Return preferred embedding device: ``mps``, ``cuda``, or ``cpu``."""
    return os.getenv("CROSSCHECK_EMBEDDING_DEVICE", "mps").strip().lower()


def get_llm_profile() -> Literal["development", "production"]:
    """Return ``development`` (default) or ``production`` (logging label only).

    Both profiles use the same :data:`LLM_MODEL_RANK` via :func:`resolve_llm_models`.
    """
    raw = os.getenv("CROSSCHECK_LLM_PROFILE", "development").strip().lower()
    # Accept legacy "test" as an alias for development.
    if raw in {"development", "test", "dev"}:
        return "development"
    if raw == "production":
        return "production"
    return "development"


def resolve_llm_models() -> list[str]:
    """Return ranked Gemini model ids (same order for development and production)."""
    return list(LLM_MODEL_RANK)


def get_google_api_key() -> str:
    """Load Google AI Studio / GenAI API key from the environment / ``.env``."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        key = os.getenv(name, "").strip()
        if key:
            return key
    raise RuntimeError(
        "Set GOOGLE_API_KEY in .env (https://aistudio.google.com/apikey). "
        "Required for claim extraction and NLI classification (Google GenAI)."
    )


def resolve_llm_primary_backup() -> tuple[str, str]:
    """Convenience: first two models from :func:`resolve_llm_models`."""
    models = resolve_llm_models()
    if not models:
        raise RuntimeError("No LLM models configured")
    if len(models) == 1:
        return models[0], models[0]
    return models[0], models[1]


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
