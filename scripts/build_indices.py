#!/usr/bin/env python3
"""Build filings and/or transcripts Qdrant hybrid indices from chunk JSONL.

Sequence (per corpus)::

    A) Merge company JSONL → data/indices/{corpus}/all_chunks.jsonl
    B) Stream BGE-M3 dense vectors → data/indices/{corpus}/embeddings.npy
    C) Upsert dense + BM25 sparse + full payload → Qdrant
       (filings → QDRANT_FILINGS_COLLECTION, transcripts → QDRANT_TRANSCRIPTS_COLLECTION)

Examples::

    python scripts/build_indices.py --corpus filings --force --batch-size 8
    python scripts/build_indices.py --corpus transcripts --force
    python scripts/build_indices.py --corpus both --force --batch-size 8

Requires ``QDRANT_ENDPOINT`` + ``QDRANT_API_KEY`` in ``.env`` for Cloud.
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
    corpus_manifest_path,
)


def main() -> None:
    """CLI entry: build filings and/or transcripts Qdrant indices."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        choices=("filings", "transcripts", "both"),
        default="filings",
        help="Which corpus to build (default: filings).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild merge, embeddings, and Qdrant artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Streaming embedding batch size (default: 8).",
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
            print(
                f"skip: {corpus} index exists at {corpus_manifest_path(corpus)} "
                "(use --force)"
            )
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
