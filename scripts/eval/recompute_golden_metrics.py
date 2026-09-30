#!/usr/bin/env python3
"""Recompute retrieval_hit / buckets from existing golden period reports.

Uses the canonical rule: >50% of truth tokens (reference figures + claim
keywords) found in NLI reasoning. Optional ``--flips`` CSV overrides per claim.

Examples::

    python scripts/eval/recompute_golden_metrics.py
    python scripts/eval/recompute_golden_metrics.py \\
        --flips data/runs/golden/eval_20260827_074408/threeway_truth_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.config import GOLDEN_REPORTS_DIR, GOLDEN_RUNS_DIR  # noqa: E402
from crosscheck.eval.metrics import (  # noqa: E402
    ClaimScoreRow,
    assign_bucket,
    print_summary,
    summarize,
    write_metrics_run_dir,
)
from crosscheck.eval.retrieval_truth import score_retrieval_hit  # noqa: E402


def _load_flips(path: Path) -> dict[str, bool]:
    """Map claim_id → force retrieval_hit (Y/N columns: flip_to_hit, manual_truth_hit)."""
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            claim_id = (row.get("claim_id") or "").strip()
            if not claim_id:
                continue
            for col in ("manual_truth_hit", "flip_to_hit"):
                raw = (row.get(col) or "").strip().upper()
                if raw in {"Y", "YES", "TRUE", "1"}:
                    out[claim_id] = True
                    break
                if raw in {"N", "NO", "FALSE", "0"}:
                    out[claim_id] = False
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--flips",
        type=Path,
        help="Optional CSV with claim_id + flip_to_hit or manual_truth_hit (Y/N).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=GOLDEN_REPORTS_DIR,
        help="Golden period reports root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Write metrics here instead of a new timestamped run dir.",
    )
    args = parser.parse_args()

    flips = _load_flips(args.flips) if args.flips else {}
    report_paths = sorted(args.reports_dir.rglob("*_golden_eval.json"))
    if not report_paths:
        print(f"No golden reports under {args.reports_dir}")
        sys.exit(1)

    rows: list[ClaimScoreRow] = []
    flip_applied = 0
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for claim in data.get("claims", []):
            expected = claim["expected_nli_label"]
            predicted = claim["predicted_nli_label"]
            nli_correct = bool(claim.get("nli_correct", expected == predicted))
            claim_id = claim["claim_id"]

            if expected == "Unverifiable":
                hit = None
                matched: list[str] = []
                missing: list[str] = []
            else:
                hit, _rate, matched, missing, _figs, _kws = score_retrieval_hit(
                    claim.get("ground_truth_reference") or "",
                    claim.get("claim") or "",
                    claim.get("nli_reasoning") or "",
                )
                if claim_id in flips:
                    hit = flips[claim_id]
                    flip_applied += 1

            bucket = assign_bucket(
                expected=expected,
                predicted=predicted,
                retrieval_hit=hit,
            )
            rows.append(
                ClaimScoreRow(
                    claim_id=claim_id,
                    ticker=claim_id.split("_")[0],
                    fiscal_year=int(data["fiscal_year"]),
                    fiscal_quarter=str(data["fiscal_quarter"]),
                    expected=expected,
                    predicted=predicted,
                    nli_correct=nli_correct,
                    retrieval_hit=hit,
                    bucket=bucket,
                    nli_confidence=float(claim.get("nli_confidence") or 0.0),
                    matched_figures=matched,
                    missing_figures=missing,
                )
            )

    run_id = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S_recomputed")
    run_dir = args.out_dir or (GOLDEN_RUNS_DIR / run_id)
    meta = {
        "run_id": run_id,
        "source": "recompute_from_reports",
        "retrieval_hit_rule": (
            ">50% truth tokens (reference figures + claim keywords) in NLI reasoning"
        ),
        "flips_path": str(args.flips) if args.flips else None,
        "flips_applied": flip_applied,
        "reports": len(report_paths),
        "claims": len(rows),
    }
    write_metrics_run_dir(run_dir, rows, meta)
    print_summary(rows)
    print()
    print(f"wrote {run_dir}")
    if flips:
        print(f"flip overrides applied: {flip_applied}/{len(flips)}")


if __name__ == "__main__":
    main()
