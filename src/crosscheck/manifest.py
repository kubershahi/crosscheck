"""Load and validate the company-year ingest manifest (YAML).

The manifest is the source of truth for *what* to fetch: one entry per
company per fiscal year, with per-quarter ``fetch`` flags and transcript URLs.
Load expands enabled quarters into :class:`CompanyPeriod` rows for ingest.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from crosscheck.config import MANIFESTS_DIR

_QUARTER_KEY = re.compile(r"^Q?([1-4])$", re.IGNORECASE)


def _parse_quarter_key(key: str | int) -> int:
    """Normalize YAML keys ``1`` / ``Q1`` / ``q1`` to int 1–4."""
    if isinstance(key, int):
        if 1 <= key <= 4:
            return key
        raise ValueError(f"quarter must be 1–4, got {key}")
    match = _QUARTER_KEY.match(str(key).strip())
    if not match:
        raise ValueError(f"Invalid quarter key {key!r}; use Q1–Q4 or 1–4")
    return int(match.group(1))


def _empty_url_to_none(value: object) -> object:
    """Treat blank URLs as unset."""
    if value is None or value == "":
        return None
    return value


class QuarterSpec(BaseModel):
    """One fiscal quarter under a company-year entry.

    Prefer Motley Fool ``transcript_url`` links; ROIC.ai or other hosts are
    fine when Fool is unavailable. The fetcher picks a parser from the host.
    """

    fetch: bool = False
    transcript_url: HttpUrl | str | None = None
    form: str | None = None  # null → auto: Q4→10-K, Q1–Q3→10-Q

    @field_validator("form", mode="before")
    @classmethod
    def _normalize_form(cls, value: object) -> str | None:
        """Treat empty strings as unset; uppercase form types."""
        if value is None or value == "":
            return None
        return str(value).strip().upper()

    @field_validator("transcript_url", mode="before")
    @classmethod
    def _normalize_urls(cls, value: object) -> object:
        return _empty_url_to_none(value)


class CompanyYear(BaseModel):
    """One company + fiscal year with optional quarters to fetch."""

    ticker: str
    name: str
    fiscal_year: int
    include: bool = False  # opt-in company for default fetch (unless --ticker)
    quarters: dict[int, QuarterSpec] = Field(default_factory=dict)

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

    @field_validator("quarters", mode="before")
    @classmethod
    def _normalize_quarter_keys(cls, value: object) -> dict[int, object]:
        """Accept Q1/Q4 or 1/4 keys in YAML."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("quarters must be a mapping of Q1–Q4 → spec")
        out: dict[int, object] = {}
        for key, spec in value.items():
            q = _parse_quarter_key(key)
            if q in out:
                raise ValueError(f"Duplicate quarter Q{q} in quarters map")
            out[q] = spec
        return out

    def to_periods(self, *, fetch_only: bool = True) -> list[CompanyPeriod]:
        """Expand this company-year into :class:`CompanyPeriod` rows.

        When ``fetch_only`` is True (default), only quarters with ``fetch: true``
        are expanded.
        """
        periods: list[CompanyPeriod] = []
        for quarter in sorted(self.quarters):
            spec = self.quarters[quarter]
            if fetch_only and not spec.fetch:
                continue
            periods.append(
                CompanyPeriod(
                    ticker=self.ticker,
                    name=self.name,
                    fiscal_year=self.fiscal_year,
                    fiscal_quarter=quarter,
                    form=spec.form,
                    transcript_url=spec.transcript_url,
                    include=self.include and spec.fetch,
                )
            )
        return periods


class CompanyPeriod(BaseModel):
    """One company + fiscal period to ingest (filing + optional transcript URL).

    Runtime shape after expanding a :class:`CompanyYear` row. Ingest CLIs and
    fetchers consume this, not the YAML company-year schema directly.
    """

    ticker: str
    name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    form: str | None = None
    transcript_url: HttpUrl | str | None = None
    include: bool = False

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("form", mode="before")
    @classmethod
    def _normalize_form(cls, value: object) -> str | None:
        if value is None or value == "":
            return None
        return str(value).strip().upper()

    def resolved_form(self) -> str:
        """Return explicit form, or Q4→``10-K`` / else ``10-Q``."""
        if self.form:
            return self.form
        return "10-K" if self.fiscal_quarter == 4 else "10-Q"

    def transcript_url_str(self) -> str | None:
        """Return the transcript URL string, or None if unset."""
        return str(self.transcript_url) if self.transcript_url else None


class Manifest(BaseModel):
    """Top-level YAML schema: a list of :class:`CompanyYear` rows."""

    companies: list[CompanyYear]

    @model_validator(mode="after")
    def _unique_ticker_year(self) -> Manifest:
        """Reject duplicate ticker+fiscal_year pairs."""
        seen: set[tuple[str, int]] = set()
        for company in self.companies:
            key = (company.ticker, company.fiscal_year)
            if key in seen:
                raise ValueError(
                    f"Duplicate company-year entry: {company.ticker} "
                    f"FY{company.fiscal_year}"
                )
            seen.add(key)
        return self


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
    """Select company-years, then expand ``fetch: true`` quarters to periods.

    When ``tickers`` is unset and ``include_only`` is True (default), only
    company-years with ``include: true`` are considered. An explicit
    ``--ticker`` list ignores company ``include`` so you can force a one-off
    fetch. In all cases, only quarters with ``fetch: true`` are expanded.
    """
    rows = list(manifest.companies)
    if tickers:
        wanted = {t.upper() for t in tickers}
        rows = [c for c in rows if c.ticker in wanted]
    elif include_only:
        rows = [c for c in rows if c.include]

    periods: list[CompanyPeriod] = []
    for company in rows:
        periods.extend(company.to_periods(fetch_only=True))
    return periods
