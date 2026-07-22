#!/usr/bin/env python3
"""Build separate filings and transcripts FAISS indices from chunk JSONL.

Sequence per corpus::

    A) Merge matching company JSONL → data/indices/{corpus}/all_chunks.jsonl
    B) Stream contextual BGE-M3 vectors → data/indices/{corpus}/embeddings.npy
    C) Add read-only memmap → data/indices/{corpus}/index.faiss

Examples::

    python scripts/build_indices.py --force --batch-size 24
    python scripts/build_indices.py --corpus filings --force
    python scripts/build_indices.py --corpus transcripts --force

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
    CorpusKind,
    build_all_indices,
    corpus_index_exists,
    corpus_index_path,
)


def main() -> None:
    """CLI entry: build filings and/or transcripts corpus indices."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        choices=("filings", "transcripts", "both"),
        default="both",
        help="Which corpus to build (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected corpus JSONL, embeddings, and FAISS artifacts.",
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

    corpora: list[CorpusKind]
    if args.corpus == "both":
        corpora = ["filings", "transcripts"]
    else:
        corpora = [args.corpus]  # type: ignore[list-item]

    if not args.force and all(corpus_index_exists(c) for c in corpora):
        for corpus in corpora:
            print(f"skip: {corpus} index exists at {corpus_index_path(corpus)} (use --force)")
        return

    print("[build_indices] loading embedding model …", flush=True)
    model = load_embedding_model()
    try:
        outs = build_all_indices(
            model=model,
            force=args.force,
            batch_size=args.batch_size,
            corpora=corpora,
        )
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    for corpus, path in outs.items():
        print(f"done [{corpus}]: {path}")


if __name__ == "__main__":
    main()
