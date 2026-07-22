#!/usr/bin/env python3
"""Build the unified master FAISS index from per-company chunk JSONL.

Sequence::

    A) Merge company JSONL → data/indices/all_chunks.jsonl (+ global_id)
    B) Stream contextual BGE-M3 vectors → data/indices/embeddings.npy
    C) Add read-only memmap → data/indices/unified_master.faiss

Examples::

    python scripts/build_indices.py
    python scripts/build_indices.py --force --batch-size 24

Prerequisite: ``python scripts/build_chunks.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.retrieval.embeddings import load_embedding_model  # noqa: E402
from crosscheck.retrieval.index import (  # noqa: E402
    build_unified_index,
    master_index_exists,
    unified_index_path,
)


def main() -> None:
    """CLI entry: merge chunks and write the unified master index."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all master JSONL, embeddings, and FAISS artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Streaming embedding batch size (default: 24).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if master_index_exists() and not args.force:
        print(f"skip: master index exists at {unified_index_path()} (use --force)")
        return

    print("[build_indices] loading embedding model …", flush=True)
    model = load_embedding_model()
    print("[build_indices] assembling master index …", flush=True)
    try:
        out = build_unified_index(
            model=model,
            force=True,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    print(f"done: {out}")


if __name__ == "__main__":
    main()
