#!/usr/bin/env python3
"""Backfill golden period reports without re-running retrieve + NLI.

Updates each ``*_golden_eval.json`` in place:

- ``matched_passage_indices`` — scraped from ``Passage N`` in ``nli_reasoning``
- ``matched_figures`` / ``missing_figures`` / ``retrieval_hit`` / ``bucket``
- stable field order (``matched_passage_indices`` immediately before ``retrieved``)

Examples::

    python scripts/eval/backfill_golden_reports.py --dry-run
    python scripts/eval/backfill_golden_reports.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.nli import extract_passage_indices  # noqa: E402
from crosscheck.config import GOLDEN_REPORTS_DIR  # noqa: E402
from crosscheck.eval.golden_eval import format_claim_report  # noqa: E402
from crosscheck.eval.metrics import assign_bucket  # noqa: E402
from crosscheck.eval.retrieval_truth import score_retrieval_hit  # noqa: E402
from crosscheck.models import GoldenClaim  # noqa: E402


def _backfill_claim_row(claim: dict) -> dict[str, object]:
    retrieved = claim.get("retrieved") or []
    reasoning = claim.get("nli_reasoning") or ""
    n_passages = len(retrieved)

    matched_passage_indices = extract_passage_indices(
        reasoning,
        primary_index=0,
        n_passages=n_passages,
    )

    expected = claim.get("expected_nli_label")
    predicted = claim.get("predicted_nli_label")
    nli_correct = bool(claim.get("nli_correct", expected == predicted))

    if expected == "Unverifiable":
        retrieval_hit = None
        matched_figures: list[str] = []
        missing_figures: list[str] = []
    else:
        retrieval_hit, _rate, matched_figures, missing_figures, _, _ = score_retrieval_hit(
            claim.get("ground_truth_reference") or "",
            claim.get("claim") or "",
            reasoning,
        )

    bucket = assign_bucket(
        expected=expected,
        predicted=predicted,
        retrieval_hit=retrieval_hit,
    )

    golden = GoldenClaim(
        claim_id=claim["claim_id"],
        ticker=claim["claim_id"].split("_")[0],
        company_name=claim.get("company_name") or claim["claim_id"].split("_")[0],
        fiscal_year=int(claim.get("fiscal_year") or claim["claim_id"].split("_")[1]),
        fiscal_quarter=claim.get("fiscal_quarter") or claim["claim_id"].split("_")[2],
        speaker=claim.get("speaker") or "",
        claim=claim.get("claim") or "",
        expected_nli_label=expected,
        is_in_filing=bool(claim.get("is_in_filing", True)),
        is_in_table=bool(claim.get("is_in_table", False)),
        ground_truth_reference=claim.get("ground_truth_reference") or "",
    )

    return format_claim_report(
        claim=golden,
        predicted=predicted,
        nli_correct=nli_correct,
        retrieval_hit=retrieval_hit,
        bucket=bucket,
        matched_figures=matched_figures,
        missing_figures=missing_figures,
        nli_confidence=float(claim.get("nli_confidence") or 0.0),
        nli_reasoning=reasoning,
        nli_model=str(claim.get("nli_model") or ""),
        matched_passage_indices=matched_passage_indices,
        retrieved=retrieved,
    )


def backfill_report(path: Path, *, dry_run: bool) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = data.get("claims") or []
    new_claims: list[dict[str, object]] = []
    changed = 0
    for claim in claims:
        before = json.dumps(claim, sort_keys=True)
        row = _backfill_claim_row(claim)
        new_claims.append(row)
        if json.dumps(row, sort_keys=True) != before:
            changed += 1

    if not dry_run:
        data["claims"] = new_claims
        data["retrieval_hit_rule"] = (
            ">50% of truth tokens (reference figures + claim keywords) in NLI reasoning"
        )
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return len(claims), changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=GOLDEN_REPORTS_DIR,
        help="Root containing golden period reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write files.",
    )
    args = parser.parse_args()

    paths = sorted(args.reports_dir.rglob("*_golden_eval.json"))
    if not paths:
        print(f"No reports under {args.reports_dir}")
        sys.exit(1)

    total_claims = 0
    total_changed = 0
    for path in paths:
        n_claims, n_changed = backfill_report(path, dry_run=args.dry_run)
        total_claims += n_claims
        total_changed += n_changed
        if n_changed or not args.dry_run:
            print(f"{'would rewrite' if args.dry_run else 'rewrote'} {path.name}: {n_changed}/{n_claims} claims changed")

    print()
    print(
        f"{'dry-run: ' if args.dry_run else ''}"
        f"{len(paths)} reports  {total_claims} claims  {total_changed} changed"
    )


if __name__ == "__main__":
    main()
