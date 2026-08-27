#!/usr/bin/env python3
"""First-pass wrapper: get → verify → promote eval candidates.

Runs the three eval scripts sequentially and stops on the first non-zero exit.
Intended for the initial period pass. After that, iterate on each script
individually (``--mode modify``, re-verify, re-promote).

Does not fetch, chunk, index, or extract source claims.

Examples::

    python scripts/eval/run_eval_candidates.py --ticker AAPL --year 2025 --quarter Q1
    python scripts/eval/run_eval_candidates.py --ticker AAPL --year 2025 --force

Prerequisites::

    python scripts/build_indices.py --corpus filings --force
    python scripts/extract_claims.py --n 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = (
    ("get", HERE / "get_eval_candidates.py"),
    ("verify", HERE / "verify_eval_candidates.py"),
    ("promote", HERE / "promote_eval_candidates.py"),
)


def _forward_args(args: argparse.Namespace, *, include_profile: bool) -> list[str]:
    forwarded: list[str] = []
    if args.ticker:
        forwarded.extend(["--ticker", args.ticker])
    if args.year is not None:
        forwarded.extend(["--year", str(args.year)])
    if args.quarter:
        forwarded.extend(["--quarter", args.quarter])
    if args.force:
        forwarded.append("--force")
    if include_profile and args.profile:
        forwarded.extend(["--profile", args.profile])
    return forwarded


def main() -> None:
    """Run get → verify → promote with shared ticker/year/quarter filters."""
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
        help="Forward --force to each step (replace existing outputs).",
    )
    parser.add_argument(
        "--profile",
        choices=("development", "production", "test", "dev"),
        help="Forward CROSSCHECK_LLM_PROFILE override (get + promote).",
    )
    args = parser.parse_args()

    for index, (name, script) in enumerate(STEPS, start=1):
        # verify_eval_candidates.py hardcodes the development profile and has
        # no --profile flag.
        forwarded = _forward_args(args, include_profile=name != "verify")
        cmd = [sys.executable, str(script), *forwarded]
        print("═" * 60, flush=True)
        print(f"[{index}/{len(STEPS)}] {name}: {' '.join(cmd)}", flush=True)
        print("═" * 60, flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"\n{name} failed with exit {result.returncode}; stopping.",
                flush=True,
            )
            sys.exit(result.returncode)
        print(flush=True)

    print("done  get → verify → promote", flush=True)


if __name__ == "__main__":
    main()
