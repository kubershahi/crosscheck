"""Load and validate the company-period ingest manifest (YAML).

The manifest is the source of truth for *what* to fetch: ticker, fiscal period,
optional SEC form override, and Motley Fool transcript URL.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator

from crosscheck.config import MANIFESTS_DIR


class CompanyPeriod(BaseModel):
    """One company + fiscal period to ingest (filing + optional transcript URL)."""

    ticker: str
    name: str  # colloquial name people search by (e.g. "Apple", not legal entity name)
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    form: str | None = None  # null → auto: Q4→10-K, else 10-Q
    transcript_url: HttpUrl | str | None = None
    include: bool = False  # opt-in: fetch/chunk only when true (unless --ticker overrides)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, value: str) -> str:
        """Normalize tickers to uppercase."""
        return value.strip().upper()

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        """Require a non-empty colloquial company name."""
        name = value.strip()
        if not name:
            raise ValueError("name must be a non-empty colloquial company name")
        return name

    @field_validator("form", mode="before")
    @classmethod
    def _normalize_form(cls, value: object) -> str | None:
        """Treat empty strings as unset; uppercase form types."""
        if value is None or value == "":
            return None
        return str(value).strip().upper()

    def resolved_form(self) -> str:
        """Return explicit form, or Q4→``10-K`` / else ``10-Q``."""
        if self.form:
            return self.form
        return "10-K" if self.fiscal_quarter == 4 else "10-Q"

    def transcript_url_str(self) -> str | None:
        """Return the transcript URL as a plain string (or None)."""
        if self.transcript_url is None:
            return None
        return str(self.transcript_url)


class Manifest(BaseModel):
    """Top-level YAML schema: a list of :class:`CompanyPeriod` rows."""

    companies: list[CompanyPeriod]


def default_manifest_path() -> Path:
    """Return the default path ``data/manifests/companies.yml``."""
    return MANIFESTS_DIR / "companies.yml"


def load_manifest(path: Path | str | None = None) -> Manifest:
    """Parse and validate a YAML manifest file."""
    path = Path(path) if path else default_manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Manifest.model_validate(raw)


def filter_manifest(
    manifest: Manifest,
    *,
    tickers: list[str] | None = None,
    include_only: bool = True,
) -> list[CompanyPeriod]:
    """Filter manifest rows by ticker and/or ``include`` flag.

    When ``tickers`` is unset and ``include_only`` is True (default), only rows
    with ``include: true`` are returned. An explicit ``--ticker`` list always
    wins and ignores the include flag so you can force a one-off fetch.
    """
    rows = list(manifest.companies)
    if tickers:
        wanted = {t.upper() for t in tickers}
        return [c for c in rows if c.ticker in wanted]
    if include_only:
        return [c for c in rows if c.include]
    return rows
