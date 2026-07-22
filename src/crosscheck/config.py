"""Project paths, optional CIK cache, SEC User-Agent, and public EDGAR URLs.

Layout (year-first)::

    data/raw/filings/{fiscal_year}/{TICKER}/
    data/raw/transcripts/{fiscal_year}/{TICKER}/
    data/chunks/{fiscal_year}/{TICKER}/          # per-company JSONL (stateless)
    data/claims/{fiscal_year}/{TICKER}/
    data/indices/all_chunks.jsonl                # master merge (+ global_id)
    data/indices/embeddings.npy                  # disk-backed float32 memmap
    data/indices/unified_master.faiss            # single FAISS IndexFlatIP
    data/indices/unified_master.manifest.json
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

# Both profiles use Google GenAI SDK (Gemini). Development prefers Flash, Lite backup.
# Use currently available Gemini 3 ids (2.5 flash/lite are closed to many new API keys).
DEVELOPMENT_LLM_PRIMARY = "gemini-3-flash-preview"
DEVELOPMENT_LLM_BACKUP = "gemini-3.1-flash-lite"

# Production presets are native Gemini model ids (not OpenRouter slugs).
PRODUCTION_MODEL_PRESETS: dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "gemini-lite": "gemini-3.1-flash-lite",
    "gemini-pro": "gemini-2.5-pro",
}
DEFAULT_PRODUCTION_PRESET = "gemini"

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
    """Deprecated per-ticker path; prefer ``unified_master.faiss``."""
    return INDICES_DIR / str(fiscal_year) / ticker.upper()


def unified_master_index_path() -> Path:
    """Return ``data/indices/unified_master.faiss``."""
    return INDICES_DIR / "unified_master.faiss"


def report_path(ticker: str, fiscal_year: int, fiscal_quarter: int) -> Path:
    """Return ``data/reports/{year}/{TICKER}/{TICKER}_FY{y}_Q{n}.json``."""
    ticker = ticker.upper()
    name = f"{ticker}_FY{fiscal_year}_Q{fiscal_quarter}.json"
    return REPORTS_DIR / str(fiscal_year) / ticker / name


def get_embedding_device_pref() -> str:
    """Return preferred embedding device: ``mps``, ``cuda``, or ``cpu``."""
    return os.getenv("CROSSCHECK_EMBEDDING_DEVICE", "mps").strip().lower()


def get_llm_profile() -> Literal["development", "production"]:
    """Return ``development`` (default) or ``production`` Gemini tier."""
    raw = os.getenv("CROSSCHECK_LLM_PROFILE", "development").strip().lower()
    # Accept legacy "test" as an alias for development.
    if raw in {"development", "test", "dev"}:
        return "development"
    if raw == "production":
        return "production"
    return "development"


def get_production_preset() -> str:
    """Return production preset key (``gemini``, ``gemini-lite``, ``gemini-pro``)."""
    return os.getenv("CROSSCHECK_PRODUCTION_MODEL", DEFAULT_PRODUCTION_PRESET).strip().lower()


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


def resolve_llm_models() -> tuple[str, str]:
    """Return (primary_model, backup_model) for the active LLM profile."""
    if get_llm_profile() == "development":
        return DEVELOPMENT_LLM_PRIMARY, DEVELOPMENT_LLM_BACKUP

    preset = get_production_preset()
    primary = PRODUCTION_MODEL_PRESETS.get(preset)
    if primary is None:
        valid = ", ".join(sorted(PRODUCTION_MODEL_PRESETS))
        raise RuntimeError(
            f"Unknown CROSSCHECK_PRODUCTION_MODEL={preset!r}. "
            f"Choose one of: {valid}"
        )

    # Backup: alternate Gemini tier within the same Google GenAI backend
    if preset == "gemini":
        backup = PRODUCTION_MODEL_PRESETS["gemini-lite"]
    elif preset == "gemini-lite":
        backup = PRODUCTION_MODEL_PRESETS["gemini"]
    else:
        backup = PRODUCTION_MODEL_PRESETS["gemini"]
    return primary, backup


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
