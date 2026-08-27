#!/usr/bin/env python3
"""Load saved claims, then run hybrid retrieval and NLI.

By default skips periods that already have a ``*_reports.json``. Pass
``--force`` to re-run and overwrite.

After the run, writes a timestamped summary CSV under ``data/runs/`` with:

- company summary (one row per ticker)
- period detail (one row per company-quarter)
- TOTAL rows at the bottom of both tables

Examples::

    python scripts/run_nli.py
    python scripts/run_nli.py --year 2025
    python scripts/run_nli.py --ticker AAPL --year 2025 --quarter Q1
    python scripts/run_nli.py --year 2025 --force

Output::

    data/reports/{year}/{TICKER}/{TICKER}_FY{y}_Q{n}_reports.json
    data/runs/run_YYYYMMDD_HHMMSS.csv

Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    python scripts/extract_claims.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.claims import load_saved_claims  # noqa: E402
from crosscheck.analysis.pipeline import run_pipeline  # noqa: E402
from crosscheck.config import CLAIMS_DIR, RUNS_DIR, report_path  # noqa: E402
from crosscheck.models import (  # noqa: E402
    DocumentMeta,
    PipelineReport,
    SavedTranscriptClaims,
    as_fiscal_quarter,
    quarter_number,
)


def _period_from_claims_path(path: Path) -> DocumentMeta:
    parts = path.stem.split("_")
    ticker = parts[0]
    year_s = parts[1]
    if year_s.upper().startswith("FY"):
        year_s = year_s[2:]
    return DocumentMeta(
        ticker=ticker,
        company_name=ticker,
        fiscal_year=int(year_s),
        fiscal_quarter=as_fiscal_quarter(parts[2]),
    )


def _discover_claim_files(
    *,
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> list[Path]:
    """Return claim JSON paths under ``data/claims`` matching optional filters."""
    if year is not None:
        year_roots = [CLAIMS_DIR / str(year)]
    else:
        year_roots = sorted(
            p for p in CLAIMS_DIR.iterdir() if p.is_dir() and p.name.isdigit()
        )

    files: list[Path] = []
    for year_dir in year_roots:
        if not year_dir.is_dir():
            continue
        if ticker:
            ticker_dirs = [year_dir / ticker.upper()]
        else:
            ticker_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())

        for ticker_dir in ticker_dirs:
            if not ticker_dir.is_dir():
                continue
            jsonl_files = sorted(ticker_dir.glob("*_claims.jsonl"))
            if jsonl_files:
                files.extend(jsonl_files)
            else:
                files.extend(sorted(ticker_dir.glob("*_claims.json")))

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


def _count_labels(report: PipelineReport) -> dict[str, int]:
    """Count Consistent / Contradictory / Unverifiable findings."""
    counts = {"Consistent": 0, "Contradictory": 0, "Unverifiable": 0}
    for finding in report.findings:
        key = finding.classification
        if key in counts:
            counts[key] += 1
    return counts


def _load_reports_for_summary(
    claim_files: list[Path],
) -> list[PipelineReport]:
    """Load existing reports for every claim file that has one on disk."""
    reports: list[PipelineReport] = []
    for claim_file in claim_files:
        try:
            saved = load_saved_claims(_period_from_claims_path(claim_file))
        except Exception:
            continue
        path = report_path(saved.ticker, saved.fiscal_year, saved.fiscal_quarter)
        if not path.exists():
            continue
        try:
            reports.append(
                PipelineReport.model_validate_json(path.read_text(encoding="utf-8"))
            )
        except Exception:
            continue
    return reports


def write_run_summary_csv(
    reports: list[PipelineReport],
    *,
    run_timestamp: str,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Write company + period tables into one CSV; return path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"run_{run_timestamp}.csv"

    period_rows: list[dict[str, object]] = []
    for report in sorted(
        reports,
        key=lambda r: (r.ticker, r.fiscal_year, r.fiscal_quarter),
    ):
        counts = _count_labels(report)
        period_rows.append(
            {
                "ticker": report.ticker,
                "company_name": report.company_name,
                "fiscal_year": report.fiscal_year,
                "fiscal_quarter": report.fiscal_quarter,
                "claims_tested": len(report.findings),
                "Consistent": counts["Consistent"],
                "Contradictory": counts["Contradictory"],
                "Unverifiable": counts["Unverifiable"],
            }
        )

    by_ticker: dict[str, dict[str, object]] = {}
    for row in period_rows:
        ticker = str(row["ticker"])
        bucket = by_ticker.setdefault(
            ticker,
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "years": set(),
                "quarters": set(),
                "claims_tested": 0,
                "Consistent": 0,
                "Contradictory": 0,
                "Unverifiable": 0,
            },
        )
        years = bucket["years"]
        quarters = bucket["quarters"]
        assert isinstance(years, set) and isinstance(quarters, set)
        years.add(int(row["fiscal_year"]))  # type: ignore[arg-type]
        quarters.add(str(row["fiscal_quarter"]))
        bucket["claims_tested"] = int(bucket["claims_tested"]) + int(
            row["claims_tested"]  # type: ignore[arg-type]
        )
        for label in ("Consistent", "Contradictory", "Unverifiable"):
            bucket[label] = int(bucket[label]) + int(row[label])  # type: ignore[arg-type]

    company_rows: list[dict[str, object]] = []
    for ticker in sorted(by_ticker):
        bucket = by_ticker[ticker]
        years = sorted(bucket["years"])  # type: ignore[arg-type]
        quarters = sorted(bucket["quarters"])  # type: ignore[arg-type]
        company_rows.append(
            {
                "ticker": ticker,
                "company_name": bucket["company_name"],
                "years": ";".join(str(y) for y in years),
                "quarters": ";".join(quarters),
                "claims_tested": bucket["claims_tested"],
                "Consistent": bucket["Consistent"],
                "Contradictory": bucket["Contradictory"],
                "Unverifiable": bucket["Unverifiable"],
            }
        )

    def _totals(rows: list[dict[str, object]], extra: dict[str, object]) -> dict[str, object]:
        return {
            **extra,
            "claims_tested": sum(int(r["claims_tested"]) for r in rows),  # type: ignore[arg-type]
            "Consistent": sum(int(r["Consistent"]) for r in rows),  # type: ignore[arg-type]
            "Contradictory": sum(int(r["Contradictory"]) for r in rows),  # type: ignore[arg-type]
            "Unverifiable": sum(int(r["Unverifiable"]) for r in rows),  # type: ignore[arg-type]
        }

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)

        writer.writerow(["# COMPANY_SUMMARY"])
        company_fields = [
            "ticker",
            "company_name",
            "years",
            "quarters",
            "claims_tested",
            "Consistent",
            "Contradictory",
            "Unverifiable",
        ]
        writer.writerow(company_fields)
        for row in company_rows:
            writer.writerow([row[f] for f in company_fields])
        if company_rows:
            total = _totals(
                company_rows,
                {
                    "ticker": "TOTAL",
                    "company_name": "",
                    "years": "",
                    "quarters": "",
                },
            )
            writer.writerow([total[f] for f in company_fields])

        writer.writerow([])
        writer.writerow(["# PERIOD_DETAIL"])
        period_fields = [
            "ticker",
            "company_name",
            "fiscal_year",
            "fiscal_quarter",
            "claims_tested",
            "Consistent",
            "Contradictory",
            "Unverifiable",
        ]
        writer.writerow(period_fields)
        for row in period_rows:
            writer.writerow([row[f] for f in period_fields])
        if period_rows:
            total = _totals(
                period_rows,
                {
                    "ticker": "TOTAL",
                    "company_name": "",
                    "fiscal_year": "",
                    "fiscal_quarter": "",
                },
            )
            writer.writerow([total[f] for f in period_fields])

    return out_path


def main() -> None:
    """CLI entry: run retrieval + NLI and write ``*_reports.json`` + run CSV."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", help="Optional ticker filter.")
    parser.add_argument("--year", type=int, help="Optional fiscal year filter.")
    parser.add_argument("--quarter", help="Fiscal quarter: Q1–Q4 or 1–4.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run NLI and overwrite existing reports.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Passages per claim.")
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Candidate pool before rerank (default: max(top_k*10, 20)).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking.",
    )
    parser.add_argument(
        "--profile",
        choices=("development", "production", "test", "dev"),
        help="Override CROSSCHECK_LLM_PROFILE.",
    )
    args = parser.parse_args()

    if args.profile:
        profile = "development" if args.profile in {"test", "dev"} else args.profile
        os.environ["CROSSCHECK_LLM_PROFILE"] = profile

    if args.quarter is not None:
        quarter_number(args.quarter)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    ticker = args.ticker.strip().upper() if args.ticker else None
    claim_files = _discover_claim_files(
        ticker=ticker,
        year=args.year,
        quarter=args.quarter,
    )
    if not claim_files:
        scope = []
        if ticker:
            scope.append(f"ticker={ticker}")
        if args.year is not None:
            scope.append(f"year={args.year}")
        if args.quarter is not None:
            scope.append(f"quarter={as_fiscal_quarter(args.quarter)}")
        scope_s = ", ".join(scope) if scope else "no filters"
        print(f"No claims under {CLAIMS_DIR} ({scope_s}).")
        sys.exit(1)

    work: list[tuple[Path, SavedTranscriptClaims, Path]] = []
    skipped = 0
    for claim_file in claim_files:
        saved = load_saved_claims(_period_from_claims_path(claim_file))
        out = report_path(saved.ticker, saved.fiscal_year, saved.fiscal_quarter)
        if out.exists() and not args.force:
            skipped += 1
            continue
        work.append((claim_file, saved, out))

    print(
        f"claims={len(claim_files)}  to_run={len(work)}  skipped={skipped}"
        + ("  (--force to overwrite)" if skipped and not args.force else "")
    )
    if work:
        print()

        for index, (claim_file, saved, out) in enumerate(work):
            period = DocumentMeta(
                ticker=saved.ticker,
                company_name=saved.company_name,
                fiscal_year=saved.fiscal_year,
                fiscal_quarter=saved.fiscal_quarter,
            )
            label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
            print("─" * 60)
            print(f"{label}  ({index + 1}/{len(work)})")
            print(f"  claims:  {claim_file.name}")
            print(f"  report:  {out}")
            print()

            try:
                report = run_pipeline(
                    period,
                    top_k=args.top_k,
                    use_reranker=not args.no_rerank,
                    rerank_pool_k=args.rerank_pool_k,
                )
            except FileNotFoundError as exc:
                print(exc)
                sys.exit(1)

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"  wrote {out}")
            print()
    else:
        print("Nothing new to process; summarizing existing reports.")
        print()

    reports = _load_reports_for_summary(claim_files)
    if not reports:
        print("No reports on disk for this selection; skipped run CSV.")
        print(f"done  processed={len(work)}  skipped={skipped}")
        return

    summary_path = write_run_summary_csv(reports, run_timestamp=run_timestamp)
    print("─" * 60)
    print(f"done  processed={len(work)}  skipped={skipped}")
    print(f"run summary → {summary_path}")


if __name__ == "__main__":
    main()
