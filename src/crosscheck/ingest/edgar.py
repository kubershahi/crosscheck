"""SEC EDGAR filing download helpers.

Resolves ticker → CIK, finds a matching 10-K/10-Q, and downloads the primary
HTML document into ``data/raw/filings/{fiscal_year}/{TICKER}/``.

Uses the public data.sec.gov / www.sec.gov APIs with a compliant User-Agent
and a polite inter-request delay (~5 req/s).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from crosscheck.config import (
    ARCHIVES_BASE,
    COMPANY_TICKERS_URL,
    KNOWN_CIKS,
    SUBMISSIONS_URL_TEMPLATE,
    filing_dir,
    get_sec_user_agent,
)

TICKERS_URL = COMPANY_TICKERS_URL
SUBMISSIONS_URL = SUBMISSIONS_URL_TEMPLATE


# Be polite: SEC asks for max ~10 req/s; we stay well under that.
_MIN_INTERVAL_S = 0.2
_last_request_at = 0.0


def _headers() -> dict[str, str]:
    """Build SEC-compliant request headers (must include contact User-Agent)."""
    return {
        "User-Agent": get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html, */*",
    }


def _get(url: str, *, timeout: float = 60) -> requests.Response:
    """GET with rate limiting; raise a clear error on SEC 403s."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    _last_request_at = time.monotonic()
    if resp.status_code == 403:
        raise requests.HTTPError(
            f"403 Forbidden from SEC for {url}. "
            "Confirm SEC_USER_AGENT in .env looks like "
            "'Your Name your.real.email@domain.com' (GitHub noreply addresses "
            "are often blocked).",
            response=resp,
        )
    resp.raise_for_status()
    return resp


def resolve_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for a ticker.

    Lookup order:
    1. ``KNOWN_CIKS`` cache in :mod:`crosscheck.config` (optional fast path)
    2. SEC ``company_tickers.json`` at :data:`crosscheck.config.COMPANY_TICKERS_URL`

    Raises ``ValueError`` if the ticker is unknown or the SEC map request fails.
    """
    ticker = ticker.upper().strip()
    if ticker in KNOWN_CIKS:
        return KNOWN_CIKS[ticker]
    try:
        data = _get(TICKERS_URL).json()
    except requests.HTTPError as exc:
        raise ValueError(
            f"Ticker {ticker} not in KNOWN_CIKS and SEC company_tickers.json "
            f"request failed ({exc}). Add the CIK to KNOWN_CIKS or retry later."
        ) from exc
    for row in data.values():
        if str(row.get("ticker", "")).upper() == ticker:
            return f"{int(row['cik_str']):010d}"
    raise ValueError(
        f"Ticker {ticker} not found in SEC company_tickers.json "
        f"({TICKERS_URL}). Check the symbol or add a CIK to KNOWN_CIKS."
    )


def _filing_list(submissions: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the nested EDGAR submissions ``recent`` filing arrays."""
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ("form", "accessionNumber", "primaryDocument", "reportDate", "filingDate")
    if not recent or not all(k in recent for k in keys):
        return []
    n = len(recent["form"])
    return [
        {
            "form": recent["form"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
            "reportDate": recent["reportDate"][i] or "",
            "filingDate": recent["filingDate"][i] or "",
        }
        for i in range(n)
    ]


def _year_from_report_date(report_date: str) -> int | None:
    """Parse YYYY from an EDGAR ISO-ish date string."""
    if not report_date or len(report_date) < 4:
        return None
    try:
        return int(report_date[:4])
    except ValueError:
        return None


def find_filing(
    ticker: str,
    *,
    form: str,
    fiscal_year: int,
) -> dict[str, str]:
    """Find the best matching filing for ticker/form/fiscal_year.

    Matching heuristic: reportDate year == fiscal_year (works for calendar
    and near-calendar fiscal years; NVDA FY2025 10-K has reportDate in early 2025).
    Falls back to the most recent filing of that form if year filter misses.
    """
    cik = resolve_cik(ticker)
    submissions = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    filings = _filing_list(submissions)
    form_u = form.upper()

    candidates = [f for f in filings if f["form"].upper() == form_u]
    if not candidates:
        raise FileNotFoundError(f"No {form} filings found for {ticker}")

    year_hits = [
        f
        for f in candidates
        if _year_from_report_date(f["reportDate"]) == fiscal_year
    ]
    if not year_hits:
        year_hits = [
            f
            for f in candidates
            if _year_from_report_date(f["filingDate"]) == fiscal_year
            or _year_from_report_date(f["filingDate"]) == fiscal_year + 1
        ]

    chosen = year_hits[0] if year_hits else candidates[0]
    chosen = {**chosen, "cik": cik, "ticker": ticker.upper()}
    return chosen


def download_primary_document(filing: dict[str, str], dest: Path) -> Path:
    """Download the primary HTML/TXT document and write a sidecar ``.meta.json``."""
    cik_int = str(int(filing["cik"]))
    accession_nodash = filing["accessionNumber"].replace("-", "")
    primary = filing["primaryDocument"]
    url = f"{ARCHIVES_BASE}/{cik_int}/{accession_nodash}/{primary}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    content = _get(url).content
    dest.write_bytes(content)

    meta_path = dest.with_suffix(dest.suffix + ".meta.json")
    meta_path.write_text(json.dumps({**filing, "source_url": url}, indent=2), encoding="utf-8")
    return dest


def filing_filename(ticker: str, form: str, fiscal_year: int) -> str:
    """Build a stable on-disk filename, e.g. ``AAPL_FY2024_10K.html``."""
    form_slug = re.sub(r"[^A-Za-z0-9]+", "", form.upper())
    return f"{ticker.upper()}_FY{fiscal_year}_{form_slug}.html"


def fetch_filing(
    ticker: str,
    *,
    form: str = "10-K",
    fiscal_year: int = 2024,
    fiscal_quarter: int = 4,
    company_name: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch a filing into ``data/raw/filings/{year}/{TICKER}/`` and return the path."""
    out_dir = filing_dir(ticker, fiscal_year)
    out_path = out_dir / filing_filename(ticker, form, fiscal_year)
    if out_path.exists() and not force:
        return out_path

    filing = find_filing(ticker, form=form, fiscal_year=fiscal_year)
    filing["fiscal_year"] = str(fiscal_year)
    filing["fiscal_quarter"] = str(fiscal_quarter)
    if company_name:
        filing["company_name"] = company_name
    return download_primary_document(filing, out_path)
