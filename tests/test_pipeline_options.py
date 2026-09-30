"""Tests for shared retrieve/NLI CLI options."""

from __future__ import annotations

import argparse
import os

from crosscheck.analysis.pipeline_options import (
    add_retrieve_nli_arguments,
    apply_retrieve_nli_options,
)


def test_apply_reranker_mps_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("CROSSCHECK_RERANK_BACKEND", raising=False)
    monkeypatch.delenv("CROSSCHECK_EMBEDDING_DEVICE", raising=False)
    apply_retrieve_nli_options(reranker="mps", nli_model="gemini")
    assert os.environ["CROSSCHECK_RERANK_BACKEND"] == "torch"
    assert os.environ["CROSSCHECK_EMBEDDING_DEVICE"] == "mps"


def test_apply_reranker_pinecone_sets_env(monkeypatch) -> None:
    monkeypatch.setenv("CROSSCHECK_RERANK_BACKEND", "torch")
    apply_retrieve_nli_options(reranker="pinecone", nli_model="gemini")
    assert os.environ["CROSSCHECK_RERANK_BACKEND"] == "pinecone"


def test_argparse_registers_flags() -> None:
    parser = argparse.ArgumentParser()
    add_retrieve_nli_arguments(parser)
    args = parser.parse_args(["--reranker", "mps", "--nli-model", "gemini"])
    assert args.reranker == "mps"
    assert args.nli_model == "gemini"
