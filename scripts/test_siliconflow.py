#!/usr/bin/env python3
"""Smoke-test SiliconFlow for BGE-M3 embed + bge-reranker-v2-m3.

Reads ``SILICONFLOW_API_KEY`` and optional ``SILICONFLOW_ENDPOINT`` from ``.env``.
Also lists account-visible models that match ``bge`` / ``embed`` / ``rerank``.

Override model ids with ``SILICONFLOW_EMBED_MODEL`` / ``SILICONFLOW_RERANK_MODEL``.

Examples::

    python scripts/test_siliconflow.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EMBED_MODEL = os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3").strip()
RERANK_MODEL = os.getenv(
    "SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
).strip()
DEFAULT_BASE = "https://api.siliconflow.com"


def _base_url() -> str:
    raw = os.getenv("SILICONFLOW_ENDPOINT", DEFAULT_BASE).strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw[: -len("/v1")]
    return raw or DEFAULT_BASE


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def list_relevant_models(base: str, api_key: str) -> list[str]:
    """Print models whose ids look like embed / bge / rerank."""
    url = f"{base}/v1/models"
    print(f"[models] GET {url}")
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    print(f"  status={resp.status_code}")
    resp.raise_for_status()
    names = sorted(
        m.get("id") for m in (resp.json().get("data") or []) if m.get("id")
    )
    hits = [
        n
        for n in names
        if any(k in n.lower() for k in ("bge", "embed", "rerank", "e5", "gte"))
    ]
    print(f"  total={len(names)} · relevant={len(hits)}")
    for n in hits:
        print(f"    {n}")
    return names


def test_embed(base: str, api_key: str, model: str) -> None:
    url = f"{base}/v1/embeddings"
    payload = {
        "model": model,
        "input": ["hello world", "flag embedding"],
        "encoding_format": "float",
    }
    print(f"[embed] POST {url} model={model}")
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
    print(f"  status={resp.status_code}")
    if resp.status_code != 200:
        print(f"  body={resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    vectors = data.get("data") or []
    if not vectors:
        raise RuntimeError(f"no embedding data: {json.dumps(data)[:400]}")
    dim = len(vectors[0].get("embedding") or [])
    print(f"  ok · {len(vectors)} vectors · dim={dim}")
    if model.lower().endswith("bge-m3") and dim != 1024:
        print(f"  warn: expected dim 1024 for BGE-M3, got {dim}")


def test_rerank(base: str, api_key: str, model: str) -> None:
    url = f"{base}/v1/rerank"
    payload = {
        "model": model,
        "query": "Apple revenue grew in Q4",
        "documents": [
            "Apple reported higher net sales in the December quarter.",
            "Banana prices fell in South America.",
            "Fruit exports increased year over year.",
        ],
        "top_n": 2,
        "return_documents": True,
    }
    print(f"[rerank] POST {url} model={model}")
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
    print(f"  status={resp.status_code}")
    if resp.status_code != 200:
        print(f"  body={resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"no rerank results: {json.dumps(data)[:400]}")
    top = results[0]
    score = top.get("relevance_score", top.get("score"))
    idx = top.get("index")
    print(f"  ok · {len(results)} results · top index={idx} score={score}")


def main() -> None:
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("Set SILICONFLOW_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    base = _base_url()
    print(f"base={base}")
    print(f"key=…{api_key[-4:]}" if len(api_key) >= 4 else "key=(short)")

    failures = 0
    try:
        list_relevant_models(base, api_key)
    except Exception as exc:
        failures += 1
        print(f"  FAIL [models]: {type(exc).__name__}: {exc}")

    for name, fn, model in (
        ("embed", test_embed, EMBED_MODEL),
        ("rerank", test_rerank, RERANK_MODEL),
    ):
        try:
            fn(base, api_key, model)
        except Exception as exc:
            failures += 1
            print(f"  FAIL [{name}]: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{failures} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
