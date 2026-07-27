"""Project paths, optional CIK cache, SEC User-Agent, and public EDGAR URLs.

Layout (year-first)::

    data/raw/filings/{fiscal_year}/{TICKER}/
    data/raw/transcripts/{fiscal_year}/{TICKER}/
    data/chunks/{fiscal_year}/{TICKER}/          # per-company JSONL (stateless)
    data/claims/{fiscal_year}/{TICKER}/
    data/indices/filings/                        # merge + dense embed scratch for Qdrant
      all_chunks.jsonl
      embeddings.npy
      manifest.json
    data/indices/transcripts/                    # merge + dense embed scratch for Qdrant
      all_chunks.jsonl
      embeddings.npy
      manifest.json
    data/indices/qdrant/                         # local Qdrant path fallback
    data/reports/{fiscal_year}/{TICKER}/
    data/runs/                                   # NLI run summary CSVs (run_<timestamp>.csv)
    data/eval/                                   # golden-set candidates (candidates.jsonl)

Filings and transcripts NLI/retrieval use Qdrant Cloud (or local path) hybrid
dense+BM25+RRF. Set ``QDRANT_ENDPOINT`` + ``QDRANT_API_KEY`` in ``.env``.

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
RUNS_DIR = DATA_DIR / "runs"
EVAL_DIR = DATA_DIR / "eval"
MANIFESTS_DIR = DATA_DIR / "manifests"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Gemini model fallback order (highest free-tier RPM/RPD first).
# Both development and production profiles use this same rank via resolve_llm_models().
LLM_MODEL_RANK: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
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


def claims_path(ticker: str, fiscal_year: int, fiscal_quarter: int | str) -> Path:
    """Return the persisted claims JSON path for one company-period."""
    from crosscheck.models import as_fiscal_quarter

    ticker = ticker.upper()
    q = as_fiscal_quarter(fiscal_quarter)
    name = f"{ticker}_FY{fiscal_year}_{q}_claims.json"
    return CLAIMS_DIR / str(fiscal_year) / ticker / name


def indices_dir(ticker: str, fiscal_year: int) -> Path:
    """Deprecated per-ticker path; prefer corpus dirs under ``data/indices/``."""
    return INDICES_DIR / str(fiscal_year) / ticker.upper()


def filings_index_path() -> Path:
    """Deprecated FAISS path; filings dense search uses Qdrant (see qdrant helpers)."""
    return INDICES_DIR / "filings" / "index.faiss"


def transcripts_index_path() -> Path:
    """Return ``data/indices/transcripts/index.faiss``."""
    return INDICES_DIR / "transcripts" / "index.faiss"


def get_qdrant_endpoint() -> str | None:
    """Return Qdrant Cloud cluster URL, or None for local path mode."""
    for name in ("QDRANT_ENDPOINT", "QDRANT_URL"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def get_qdrant_api_key() -> str | None:
    """Return Qdrant API key when using Cloud / secured server."""
    key = os.getenv("QDRANT_API_KEY", "").strip()
    return key or None


def get_qdrant_path() -> Path:
    """Local embedded Qdrant path when ``QDRANT_ENDPOINT`` is unset."""
    raw = os.getenv("QDRANT_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return INDICES_DIR / "qdrant"


def get_qdrant_filings_collection() -> str:
    """Qdrant collection name for filing chunks."""
    return os.getenv("QDRANT_FILINGS_COLLECTION", "filings").strip() or "filings"


def get_qdrant_transcripts_collection() -> str:
    """Qdrant collection name for transcript chunks."""
    return (
        os.getenv("QDRANT_TRANSCRIPTS_COLLECTION", "transcripts").strip()
        or "transcripts"
    )


def unified_master_index_path() -> Path:
    """Deprecated alias for :func:`filings_index_path`."""
    return filings_index_path()


def report_path(ticker: str, fiscal_year: int, fiscal_quarter: int | str) -> Path:
    """Return ``data/reports/{year}/{TICKER}/{TICKER}_FY{y}_Q{n}_reports.json``."""
    from crosscheck.models import as_fiscal_quarter

    ticker = ticker.upper()
    q = as_fiscal_quarter(fiscal_quarter)
    name = f"{ticker}_FY{fiscal_year}_{q}_reports.json"
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
