"""Aggregate golden-eval metrics and write CSV/JSON artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Classification = Literal["Consistent", "Contradictory", "Unverifiable"]
Bucket = Literal[
    "both_ok",
    "nli_hallucination",
    "lucky_nli",
    "both_fail",
    "unverifiable_nli",
]

LABELS: tuple[Classification, ...] = (
    "Consistent",
    "Contradictory",
    "Unverifiable",
)


@dataclass
class ClaimScoreRow:
    claim_id: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: str
    expected: Classification
    predicted: Classification
    nli_correct: bool
    retrieval_hit: bool | None
    bucket: Bucket
    nli_confidence: float
    matched_figures: list[str] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)


def assign_bucket(
    *,
    expected: Classification,
    predicted: Classification,
    retrieval_hit: bool | None,
) -> Bucket:
    nli_correct = expected == predicted
    if expected == "Unverifiable":
        return "unverifiable_nli"
    assert retrieval_hit is not None
    if retrieval_hit and nli_correct:
        return "both_ok"
    if retrieval_hit and not nli_correct:
        return "nli_hallucination"
    if not retrieval_hit and nli_correct:
        return "lucky_nli"
    return "both_fail"


def confusion_matrix(rows: list[ClaimScoreRow]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        exp: {pred: 0 for pred in LABELS} for exp in LABELS
    }
    for row in rows:
        matrix[row.expected][row.predicted] += 1
    return matrix


def confusion_matrix_pct(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for exp in LABELS:
        total = sum(matrix[exp].values())
        out[exp] = {
            pred: (matrix[exp][pred] / total if total else 0.0) for pred in LABELS
        }
    return out


def bucket_counts(rows: list[ClaimScoreRow]) -> Counter[str]:
    return Counter(row.bucket for row in rows)


def summarize(rows: list[ClaimScoreRow]) -> dict[str, Any]:
    n = len(rows)
    nli_correct = sum(1 for r in rows if r.nli_correct)
    cx_rows = [r for r in rows if r.expected in {"Consistent", "Contradictory"}]
    cx_nli_correct = sum(1 for r in cx_rows if r.nli_correct)
    recall_hits = sum(1 for r in cx_rows if r.retrieval_hit)
    u_rows = [r for r in rows if r.expected == "Unverifiable"]
    u_correct = sum(1 for r in u_rows if r.nli_correct)

    matrix = confusion_matrix(rows)
    return {
        "n_claims": n,
        "nli_accuracy": (nli_correct / n if n else 0.0),
        "nli_accuracy_cx": (cx_nli_correct / len(cx_rows) if cx_rows else 0.0),
        "nli_cx_n": len(cx_rows),
        "nli_cx_correct": cx_nli_correct,
        "recall_at_k": (recall_hits / len(cx_rows) if cx_rows else 0.0),
        "recall_n": len(cx_rows),
        "recall_hits": recall_hits,
        "unverifiable_n": len(u_rows),
        "unverifiable_nli_accuracy": (u_correct / len(u_rows) if u_rows else 0.0),
        "confusion_matrix": matrix,
        "confusion_matrix_pct": confusion_matrix_pct(matrix),
        "error_buckets": dict(bucket_counts(rows)),
    }


def write_metrics_run_dir(run_dir: Path, rows: list[ClaimScoreRow], meta: dict[str, Any]) -> None:
    """Write summary.csv, confusion_matrix.csv, error_buckets.csv, per_claim.csv, metrics.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)

    # summary.csv
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in (
            "n_claims",
            "nli_accuracy",
            "nli_accuracy_cx",
            "nli_cx_n",
            "nli_cx_correct",
            "recall_at_k",
            "recall_n",
            "recall_hits",
            "unverifiable_n",
            "unverifiable_nli_accuracy",
        ):
            writer.writerow([key, summary[key]])

    # confusion_matrix.csv
    matrix = summary["confusion_matrix"]
    with (run_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["expected"] + list(LABELS))
        for exp in LABELS:
            writer.writerow([exp] + [matrix[exp][pred] for pred in LABELS])

    pct = summary["confusion_matrix_pct"]
    with (run_dir / "confusion_matrix_pct.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["expected"] + list(LABELS))
        for exp in LABELS:
            writer.writerow([exp] + [f"{pct[exp][pred]:.4f}" for pred in LABELS])

    # error_buckets.csv
    buckets = summary["error_buckets"]
    with (run_dir / "error_buckets.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bucket", "count"])
        for name in (
            "both_ok",
            "nli_hallucination",
            "lucky_nli",
            "both_fail",
            "unverifiable_nli",
        ):
            writer.writerow([name, buckets.get(name, 0)])

    # per_claim.csv
    with (run_dir / "per_claim.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "claim_id",
                "ticker",
                "fiscal_year",
                "fiscal_quarter",
                "expected",
                "predicted",
                "nli_correct",
                "retrieval_hit",
                "bucket",
                "nli_confidence",
                "matched_figures",
                "missing_figures",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "claim_id": row.claim_id,
                    "ticker": row.ticker,
                    "fiscal_year": row.fiscal_year,
                    "fiscal_quarter": row.fiscal_quarter,
                    "expected": row.expected,
                    "predicted": row.predicted,
                    "nli_correct": row.nli_correct,
                    "retrieval_hit": row.retrieval_hit,
                    "bucket": row.bucket,
                    "nli_confidence": row.nli_confidence,
                    "matched_figures": "|".join(row.matched_figures),
                    "missing_figures": "|".join(row.missing_figures),
                }
            )

    payload = {"meta": meta, **summary}
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(rows: list[ClaimScoreRow]) -> None:
    """Human-readable terminal summary."""
    summary = summarize(rows)
    matrix = summary["confusion_matrix"]
    buckets = summary["error_buckets"]

    print("\n=== NLI confusion matrix (counts) ===")
    header = "expected\\pred".ljust(16) + "".join(l.rjust(14) for l in LABELS)
    print(header)
    for exp in LABELS:
        line = exp.ljust(16) + "".join(str(matrix[exp][pred]).rjust(14) for pred in LABELS)
        print(line)

    print("\n=== Retrieval × NLI buckets (C/X only) ===")
    for name in ("both_ok", "nli_hallucination", "lucky_nli", "both_fail"):
        print(f"  {name}: {buckets.get(name, 0)}")

    print("\n=== Headline metrics ===")
    print(
        f"  NLI accuracy (all): {summary['nli_accuracy']:.1%} "
        f"({summary['n_claims']} claims)"
    )
    print(
        f"  NLI accuracy (C/X): {summary['nli_accuracy_cx']:.1%} "
        f"({summary['nli_cx_correct']}/{summary['nli_cx_n']})"
    )
    print(
        f"  Recall@k:           {summary['recall_at_k']:.1%} "
        f"({summary['recall_hits']}/{summary['recall_n']} C/X)"
    )
    print(
        f"  U NLI accuracy:     {summary['unverifiable_nli_accuracy']:.1%} "
        f"({summary['unverifiable_n']})"
    )
