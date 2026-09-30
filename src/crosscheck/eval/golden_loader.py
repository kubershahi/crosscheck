"""Load golden claim JSONL from ``data/eval/golden/``."""

from __future__ import annotations

from pathlib import Path

from crosscheck.config import EVAL_GOLDEN_DIR
from crosscheck.io.jsonl import iter_json_objects
from crosscheck.models import GoldenClaim, as_fiscal_quarter


def golden_claims_path(ticker: str, fiscal_year: int, fiscal_quarter: str | int) -> Path:
    ticker = ticker.upper()
    q = as_fiscal_quarter(fiscal_quarter)
    name = f"{ticker}_FY{fiscal_year}_{q}_golden_claims.jsonl"
    return EVAL_GOLDEN_DIR / str(fiscal_year) / ticker / name


def load_golden_claims(path: Path) -> list[GoldenClaim]:
    claims: list[GoldenClaim] = []
    for row in iter_json_objects(path):
        claims.append(GoldenClaim.model_validate(row))
    return claims


def discover_golden_files(
    *,
    ticker: str | None = None,
    year: int | None = None,
    quarter: str | None = None,
) -> list[Path]:
    root = EVAL_GOLDEN_DIR
    if not root.exists():
        return []

    if year is not None:
        year_dirs = [root / str(year)]
    else:
        year_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())

    files: list[Path] = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            continue
        if ticker:
            ticker_dirs = [year_dir / ticker.upper()]
        else:
            ticker_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())
        for ticker_dir in ticker_dirs:
            if not ticker_dir.is_dir():
                continue
            files.extend(sorted(ticker_dir.glob("*_golden_claims.jsonl")))

    if quarter is not None:
        wanted = as_fiscal_quarter(quarter)
        filtered: list[Path] = []
        for path in files:
            try:
                parts = path.stem.split("_")
                q = as_fiscal_quarter(parts[2])
            except (ValueError, IndexError):
                continue
            if q == wanted:
                filtered.append(path)
        files = filtered
    return files
