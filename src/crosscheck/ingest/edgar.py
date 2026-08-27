"""SEC EDGAR filing download helpers.

Resolves ticker → CIK, finds a matching 10-K/10-Q for a fiscal year **and**
quarter (using the company's SEC ``fiscalYearEnd``), and downloads the primary
HTML document into ``data/raw/filings/{fiscal_year}/{TICKER}/``.

Uses the public data.sec.gov / www.sec.gov APIs with a compliant User-Agent
and a polite inter-request delay (~5 req/s).
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timezone
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


def _parse_fye_month(fiscal_year_end: str | None) -> int:
    """Parse SEC ``fiscalYearEnd`` (``MMDD``) to month 1–12; default December."""
    if not fiscal_year_end or len(fiscal_year_end) < 2:
        return 12
    try:
        month = int(fiscal_year_end[:2])
    except ValueError:
        return 12
    if 1 <= month <= 12:
        return month
    return 12


def _parse_iso_date(value: str) -> date | None:
    """Parse ``YYYY-MM-DD`` (or longer ISO) into a date."""
    if not value or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def quarter_end_month(fye_month: int, fiscal_quarter: int) -> int:
    """Return the calendar month when ``fiscal_quarter`` typically ends.

    Quarters advance 3 months from the fiscal year-end month:
    Q1 = FYE+3, Q2 = FYE+6, Q3 = FYE+9, Q4 = FYE.
    """
    if fiscal_quarter not in range(1, 5):
        raise ValueError(f"fiscal_quarter must be 1–4, got {fiscal_quarter}")
    if fye_month not in range(1, 13):
        raise ValueError(f"fye_month must be 1–12, got {fye_month}")
    offset = fiscal_quarter * 3  # Q1→3 … Q4→12
    return ((fye_month - 1 + offset) % 12) + 1


def expected_quarter_end_year(
    *,
    fiscal_year: int,
    fye_month: int,
    expected_end_month: int,
) -> int:
    """Calendar year of the expected quarter-end month for a fiscal year label.

    Example (Apple FYE September, FY2025 Q1 → December): end year is 2024.
    """
    if expected_end_month > fye_month:
        return fiscal_year - 1
    return fiscal_year


def quarter_period_months(
    *,
    fiscal_year: int,
    fiscal_quarter: int,
    fye_month: int,
) -> dict[str, object]:
    """Describe the calculated calendar span for a fiscal quarter.

    Returns expected end month/year plus the three ``YYYY-MM`` months in the
    quarter (inclusive of the end month).
    """
    end_month = quarter_end_month(fye_month, fiscal_quarter)
    end_year = expected_quarter_end_year(
        fiscal_year=fiscal_year,
        fye_month=fye_month,
        expected_end_month=end_month,
    )
    months: list[str] = []
    year, month = end_year, end_month
    for _ in range(3):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    months.reverse()
    return {
        "label": f"FY{fiscal_year} Q{fiscal_quarter}",
        "expected_end_month": end_month,
        "expected_end_year": end_year,
        "months": months,
    }


def fiscal_year_for_report(report: date, fye_month: int) -> int:
    """Map a period-end ``reportDate`` to the issuer's fiscal year label.

    SEC fiscal years are named by the calendar year of the fiscal year-end.
    Example (Apple FYE September): Dec 2024 → FY2025; Jun 2025 → FY2025.
    """
    if report.month > fye_month:
        return report.year + 1
    return report.year


def _score_report_match(
    report: date,
    *,
    fye_month: int,
    fiscal_year: int,
    fiscal_quarter: int,
) -> int | None:
    """Return a lower-is-better score when ``report`` matches year+quarter.

    Allows ±1 calendar month around the expected quarter-end month (Apple
    often ends late December rather than exactly month-end).
    """
    if fiscal_year_for_report(report, fye_month) != fiscal_year:
        return None
    expected_month = quarter_end_month(fye_month, fiscal_quarter)
    # Circular month distance (e.g. Dec↔Jan = 1)
    delta = min(
        abs(report.month - expected_month),
        12 - abs(report.month - expected_month),
    )
    if delta > 1:
        return None
    # Prefer exact month, then closer day-of-month to month end (~28–31)
    return delta * 100 + abs(report.day - 28)


def find_filing(
    ticker: str,
    *,
    form: str,
    fiscal_year: int,
    fiscal_quarter: int = 4,
) -> dict[str, Any]:
    """Find the filing for ticker/form/fiscal year **and** quarter.

    Uses SEC submissions ``fiscalYearEnd`` to map ``reportDate`` → fiscal
    year and expected quarter-end month. Picks the best-scoring match among
    recent filings of the requested form. Also returns ``fiscal_year_end`` and
    a calculated ``quarter_period`` for downstream metadata.
    """
    cik = resolve_cik(ticker)
    submissions = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    filings = _filing_list(submissions)
    form_u = form.upper()
    fye_month = _parse_fye_month(submissions.get("fiscalYearEnd"))

    candidates = [f for f in filings if f["form"].upper() == form_u]
    if not candidates:
        raise FileNotFoundError(f"No {form} filings found for {ticker}")

    scored: list[tuple[int, dict[str, str]]] = []
    for filing in candidates:
        report = _parse_iso_date(filing["reportDate"])
        if report is None:
            continue
        score = _score_report_match(
            report,
            fye_month=fye_month,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        if score is not None:
            scored.append((score, filing))

    if not scored:
        raise FileNotFoundError(
            f"No {form} for {ticker} FY{fiscal_year} Q{fiscal_quarter} "
            f"(fiscalYearEnd={submissions.get('fiscalYearEnd')!r}). "
            "Check the fiscal year/quarter or SEC submissions history."
        )

    scored.sort(key=lambda item: item[0])
    fye_raw = submissions.get("fiscalYearEnd")
    period = quarter_period_months(
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fye_month=fye_month,
    )
    chosen = {
        **scored[0][1],
        "cik": cik,
        "ticker": ticker.upper(),
        "fiscal_year_end": fye_raw,  # SEC MMDD, e.g. "0926"
        "fiscal_year_end_month": fye_month,
        "quarter_period": period,
    }
    return chosen


def download_primary_document(filing: dict[str, Any], dest: Path) -> Path:
    """Download the primary HTML/TXT document and write a sidecar ``.meta.json``."""
    cik_int = str(int(filing["cik"]))
    accession_nodash = str(filing["accessionNumber"]).replace("-", "")
    primary = str(filing["primaryDocument"])
    url = f"{ARCHIVES_BASE}/{cik_int}/{accession_nodash}/{primary}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    content = _get(url).content
    dest.write_bytes(content)

    write_filing_meta(dest, {**filing, "source_url": url})
    return dest


def filing_meta_path(dest: Path) -> Path:
    """Sidecar path for a downloaded filing (``*.html.meta.json``)."""
    return dest.with_suffix(dest.suffix + ".meta.json")


def write_filing_meta(dest: Path, filing: dict[str, Any]) -> Path:
    """Write / overwrite the filing sidecar ``.meta.json``."""
    payload = {
        **filing,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = filing_meta_path(dest)
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return meta_path


def refresh_filing_meta(
    dest: Path,
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int,
    company_name: str | None = None,
) -> Path:
    """Rewrite sidecar meta from the on-disk filing + current period fields.

    Does not hit EDGAR. Accession / source_url stay as previously downloaded.
    """
    meta_path = filing_meta_path(dest)
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["ticker"] = ticker.upper()
    existing["fiscal_year"] = str(fiscal_year)
    existing["fiscal_quarter"] = f"Q{fiscal_quarter}"
    if company_name:
        existing["company_name"] = company_name
    return write_filing_meta(dest, existing)


def filing_filename(
    ticker: str,
    form: str,
    fiscal_year: int,
    fiscal_quarter: int | None = None,
) -> str:
    """Build a stable on-disk filename.

    - 10-K: ``AAPL_FY2025_10K.html``
    - 10-Q: ``AAPL_FY2025_Q1_10Q.html`` (quarter required so Q1–Q3 do not collide)
    """
    form_slug = re.sub(r"[^A-Za-z0-9]+", "", form.upper())
    ticker_u = ticker.upper()
    if form_slug == "10Q":
        if fiscal_quarter is None:
            raise ValueError("fiscal_quarter is required for 10-Q filenames")
        return f"{ticker_u}_FY{fiscal_year}_Q{fiscal_quarter}_{form_slug}.html"
    return f"{ticker_u}_FY{fiscal_year}_{form_slug}.html"


def fetch_filing(
    ticker: str,
    *,
    form: str = "10-K",
    fiscal_year: int = 2024,
    fiscal_quarter: int = 4,
    company_name: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch a filing into ``data/raw/filings/{year}/{TICKER}/`` and return the path.

    Sidecar ``.meta.json`` is rewritten on every call (including skips).
    ``force`` re-downloads the primary document from EDGAR.
    """
    out_dir = filing_dir(ticker, fiscal_year)
    out_path = out_dir / filing_filename(
        ticker, form, fiscal_year, fiscal_quarter=fiscal_quarter
    )
    if out_path.exists() and not force:
        refresh_filing_meta(
            out_path,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            company_name=company_name,
        )
        return out_path

    filing = find_filing(
        ticker,
        form=form,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )
    filing["fiscal_year"] = str(fiscal_year)
    filing["fiscal_quarter"] = f"Q{fiscal_quarter}"
    if company_name:
        filing["company_name"] = company_name
    return download_primary_document(filing, out_path)
