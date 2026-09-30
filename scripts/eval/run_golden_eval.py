#!/usr/bin/env python3
"""Run live retrieve + NLI against curated golden claims and score metrics.

Default: **mini sample** (2 Consistent + 1 Contradictory + 1 Unverifiable per
period) for cost control. Pass ``--full`` for all 8 claims per period.

Writes:

- Period reports: ``data/reports/golden/{year}/{TICKER}/…``
- Run metrics: ``data/runs/golden/eval_<timestamp>/`` (CSV + JSON)

Examples::

    python scripts/eval/run_golden_eval.py
    python scripts/eval/run_golden_eval.py --ticker AAPL --year 2025 --quarter Q1
    python scripts/eval/run_golden_eval.py --full --reranker mps
    python scripts/eval/run_golden_eval.py --force

Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    Golden JSONL under ``data/eval/golden/`` (from promote_eval_candidates)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.pipeline_options import (  # noqa: E402
    add_retrieve_nli_arguments,
    apply_retrieve_nli_options,
)
from crosscheck.config import (  # noqa: E402
    EVAL_GOLDEN_DIR,
    GOLDEN_RUNS_DIR,
    golden_report_path,
)
from crosscheck.eval.golden_eval import (  # noqa: E402
    period_result_to_score_rows,
    run_golden_period,
    write_golden_period_report,
)
from crosscheck.eval.golden_loader import discover_golden_files, load_golden_claims  # noqa: E402
from crosscheck.eval.metrics import print_summary, write_metrics_run_dir  # noqa: E402
from crosscheck.eval.sampler import sample_mini_golden  # noqa: E402
from crosscheck.models import DocumentMeta, as_fiscal_quarter, quarter_number  # noqa: E402
from crosscheck.retrieval.rerank import active_rerank_backend  # noqa: E402


def _period_from_golden_path(path: Path) -> DocumentMeta:
    parts = path.stem.split("_")
    ticker = parts[0]
    year_s = parts[1]
    if year_s.upper().startswith("FY"):
        year_s = year_s[2:]
    claims = load_golden_claims(path)
    company = claims[0].company_name if claims else ticker
    return DocumentMeta(
        ticker=ticker,
        company_name=company,
        fiscal_year=int(year_s),
        fiscal_quarter=as_fiscal_quarter(parts[2]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", help="Optional ticker filter.")
    parser.add_argument("--year", type=int, help="Optional fiscal year filter.")
    parser.add_argument("--quarter", help="Fiscal quarter: Q1–Q4 or 1–4.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all golden claims (default: 2C+1X+1U mini sample).",
    )
    parser.add_argument(
        "--seed",
        default="golden",
        help="Sampler seed for mini runs (default: golden).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden period reports.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Passages per claim (Q4 uses 4+4).")
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Candidate pool before rerank (default: max(top_k*10, 20)).",
    )
    add_retrieve_nli_arguments(parser)
    args = parser.parse_args()

    apply_retrieve_nli_options(reranker=args.reranker, nli_model=args.nli_model)

    if args.quarter is not None:
        quarter_number(args.quarter)

    run_id = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    sample_mode = "full" if args.full else "mini"

    ticker = args.ticker.strip().upper() if args.ticker else None
    golden_files = discover_golden_files(
        ticker=ticker,
        year=args.year,
        quarter=args.quarter,
    )
    if not golden_files:
        scope = []
        if ticker:
            scope.append(f"ticker={ticker}")
        if args.year is not None:
            scope.append(f"year={args.year}")
        if args.quarter is not None:
            scope.append(f"quarter={as_fiscal_quarter(args.quarter)}")
        scope_s = ", ".join(scope) if scope else "no filters"
        print(f"No golden claims under {EVAL_GOLDEN_DIR} ({scope_s}).")
        sys.exit(1)

    work: list[tuple[Path, DocumentMeta, list]] = []
    skipped = 0
    for path in golden_files:
        period = _period_from_golden_path(path)
        out = golden_report_path(period.ticker, period.fiscal_year, period.fiscal_quarter)
        if out.exists() and not args.force:
            skipped += 1
            continue
        all_claims = load_golden_claims(path)
        if args.full:
            claims = sorted(all_claims, key=lambda c: c.claim_id)
        else:
            period_key = f"{period.ticker}|{period.fiscal_year}|{period.fiscal_quarter}"
            claims = sample_mini_golden(all_claims, seed=f"{args.seed}|{period_key}")
        if not claims:
            print(f"skip empty {path}")
            continue
        work.append((path, period, claims))

    print(
        f"golden_files={len(golden_files)}  to_run={len(work)}  skipped={skipped}"
        f"  sample={sample_mode}"
        + ("  (--force to overwrite)" if skipped and not args.force else "")
    )
    if not work:
        print("Nothing new to process.")
        return

    print()
    all_rows = []
    period_results = []

    for index, (path, period, claims) in enumerate(work):
        label = f"{period.ticker} FY{period.fiscal_year} {period.fiscal_quarter}"
        out = golden_report_path(period.ticker, period.fiscal_year, period.fiscal_quarter)
        print("─" * 60)
        print(f"{label}  ({index + 1}/{len(work)})  claims={len(claims)}")
        print(f"  golden:  {path.name}")
        print(f"  report:  {out}")
        print()

        try:
            result = run_golden_period(
                period,
                claims,
                top_k=args.top_k,
                use_reranker=not args.no_rerank,
                rerank_pool_k=args.rerank_pool_k,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        write_golden_period_report(
            out,
            result,
            run_id=run_id,
            reranker=args.reranker,
            nli_model=args.nli_model,
            sample_mode=sample_mode,
        )
        period_results.append(result)
        all_rows.extend(period_result_to_score_rows(result))
        print(f"  wrote {out}")
        print()

    run_dir = GOLDEN_RUNS_DIR / run_id
    meta = {
        "run_id": run_id,
        "sample_mode": sample_mode,
        "seed": args.seed,
        "reranker": args.reranker,
        "reranker_backend": active_rerank_backend() if not args.no_rerank else "none",
        "nli_model": args.nli_model,
        "retrieval_hit_rule": (
            ">50% truth tokens (reference figures + claim keywords) in NLI reasoning"
        ),
        "top_k": args.top_k,
        "periods": len(period_results),
        "claims": len(all_rows),
    }
    write_metrics_run_dir(run_dir, all_rows, meta)
    print_summary(all_rows)
    print()
    print("─" * 60)
    print(f"done  processed={len(work)}  skipped={skipped}")
    print(f"run metrics → {run_dir}")


if __name__ == "__main__":
    main()
