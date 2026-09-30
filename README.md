# Crosscheck

Cross-document RAG that checks earnings-call claims against the same company-quarter’s SEC filing.

Executives describe results on a call; the 10-Q or 10-K states them in audited tables. Crosscheck retrieves filing passages for each claim and classifies the pair as **Consistent**, **Contradictory**, or **Unverifiable**, with citations, confidence, and reasoning. Retrieval and the NLI label are scored separately on a 160-claim golden set.

## Overview

```text
Transcript claim  →  hybrid retrieve 10-Q / 10-K  →  rerank  →  NLI
                     (same ticker + year + quarter)
```

Q1–Q3 claims search that quarter’s 10-Q. Q4 claims use two stacks: the FY 10-K and the Q3 10-Q, four passages from each. Dense embeddings stay local (`BAAI/bge-m3`). Reranking defaults to Pinecone Inference (`bge-reranker-v2-m3`), with a local Torch fallback on MPS.

Evaluated on FY2025 Q1–Q4 for AAPL, AMZN, GOOGL, META, and NVDA.

![Crosscheck overview: corpus build, per-claim hybrid retrieval and Gemini NLI on an AAPL Mac revenue claim, and golden eval metrics](assets/project-overview.png)

## Project structure

```text
.
├── src/crosscheck/
│   ├── ingest/            # EDGAR HTML + transcript scrape
│   ├── chunking/          # section/table filings, speaker-turn transcripts
│   ├── retrieval/         # BGE-M3, Qdrant hybrid, query expansion, rerank
│   ├── analysis/          # claim extraction, NLI, prompts, LLM client
│   ├── eval/              # golden loader, sampler, retrieval-hit rule, metrics
│   ├── config.py
│   └── models.py
├── scripts/
│   ├── fetch_corpus.py
│   ├── build_chunks.py
│   ├── build_indices.py
│   ├── extract_claims.py
│   ├── run_nli.py
│   └── eval/              # candidate loop, golden eval, backfill, recompute
├── tests/
├── streamlit_app.py       # Demo (golden metrics) + About
├── data/manifests/companies.yml
└── README.md
```

Generated corpora, indices, reports, and eval JSONL live under `data/` and are gitignored. Qdrant Cloud holds the filings and transcripts collections.

## Pipeline

| Stage              | What it does                                                                                                            | Script                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **Fetch**          | 10-Q / 10-K HTML from EDGAR; earnings-call transcript from the manifest URL                                             | `scripts/fetch_corpus.py`             |
| **Chunk**          | Filings split on Item/PART headers; each HTML table is one Markdown chunk. Transcripts split on speaker turns           | `scripts/build_chunks.py`             |
| **Index**          | Local BGE-M3 vectors plus Qdrant BM25, upserted with full chunk payload                                                 | `scripts/build_indices.py`            |
| **Claims**         | One Gemini call per transcript; numeric, current-quarter assertions                                                     | `scripts/extract_claims.py`           |
| **Retrieve + NLI** | Period filter, hybrid search, rerank, then a structured NLI judgment. Citations come from retrieval, not from the model | `scripts/run_nli.py`                  |
| **Golden set**     | Draft labels → live verify → promote matches into `data/eval/golden/`                                                   | `scripts/eval/run_eval_candidates.py` |
| **Golden eval**    | Live retrieve + NLI on curated claims; write period reports and metric CSV/JSON                                         | `scripts/eval/run_golden_eval.py`     |

Shared flags on the retrieve + NLI scripts: `--reranker {pinecone,mps}` and `--nli-model gemini`.

## Labels

| Label             | Meaning                                               |
| ----------------- | ----------------------------------------------------- |
| **Consistent**    | The retrieved filing supports the claim               |
| **Contradictory** | The filing states a conflicting figure or fact        |
| **Unverifiable**  | The passages do not contain enough evidence to decide |

## Evaluation metrics

- **Recall@k** — on Consistent and Contradictory claims, a hit when more than half of the truth tokens appear in the NLI reasoning. Truth tokens are dollar and percent figures from the ground-truth reference (section numbers such as “Item 2” are ignored) plus keywords from the claim. Unverifiable claims are excluded.
- **NLI accuracy (C/X)** — predicted label matches the golden label on Consistent and Contradictory claims.
- **NLI accuracy (all)** — same, including Unverifiable.
- **Buckets** — `both_ok`, `nli_hallucination` (retrieval hit, NLI wrong), `lucky_nli`, `both_fail`.

`matched_passage_indices` records which retrieved passages the reasoning cites (`Passage N`, plus the model’s chosen index on a live run).

## Quick start

### Install

```bash
conda create -n crosscheck python=3.11 -y
conda activate crosscheck
pip install -e ".[dev]"
cp .env.example .env
```

Required in `.env`:

```bash
SEC_USER_AGENT="Crosscheck Your Name you@domain.com"
GOOGLE_API_KEY=...
QDRANT_ENDPOINT=...
QDRANT_API_KEY=...
PINECONE_API_KEY=...
```

Optional: `CROSSCHECK_EMBEDDING_DEVICE=mps`, `CROSSCHECK_RERANK_BACKEND=pinecone`.

### Run the pipeline

Companies and quarters to fetch are listed in `data/manifests/companies.yml`. Later stages discover files already on disk.

```bash
python scripts/fetch_corpus.py --year 2025
python scripts/build_chunks.py --year 2025
python scripts/build_indices.py --corpus filings --force --batch-size 32
python scripts/extract_claims.py --year 2025 --n 6
python scripts/run_nli.py --ticker AAPL --year 2025 --quarter Q1
```

### Golden eval

```bash
python scripts/eval/run_golden_eval.py --full
python scripts/eval/backfill_golden_reports.py
python scripts/eval/recompute_golden_metrics.py \
  --out-dir data/runs/golden/eval_20260827_074408
```

`--full` scores every golden claim (8 per period). The default mini sample is 2 Consistent + 1 Contradictory + 1 Unverifiable per period.

### Demo and tests

```bash
streamlit run streamlit_app.py
python -m pytest tests/ -q
```

The Demo tab reads `data/reports/golden/` and `data/runs/golden/`. About describes the pipeline and metrics.

## Key findings

1. **NLI is strong on this set.** 91.9% label accuracy overall (147/160), 89.2% on Consistent and Contradictory claims (107/120), and 100% on Unverifiable (40/40).
2. **Retrieval and NLI have to be scored apart.** A passage-token rule that required every extracted number, including “Item 2”, reported 40.8% recall while NLI was already ~92% correct. The gap was the scorer, not the retriever.
3. **The canonical recall rule agrees with the model’s evidence.** A hit is >50% of reference figures plus claim keywords found in the NLI reasoning, which already cites the passages it used. Recall@k is 95.8% (115/120).
4. **Most errors are NLI, not retrieval.** Of 120 Consistent/Contradictory claims, 104 are both correct, 11 retrieved the evidence but mislabeled it, 3 were labeled correctly without a retrieval hit, and 2 failed both.

## Golden eval results

Run `eval_20260827_074408`. Live retrieve + Gemini NLI on 160 FY2025 claims (5 companies × 4 quarters × 8: 4 Consistent, 2 Contradictory, 2 Unverifiable). Reranker: Pinecone `bge-reranker-v2-m3`. Metrics recomputed with the canonical retrieval rule.

| Metric             | Result              |
| ------------------ | ------------------- |
| Recall@k (C/X)     | **95.8%** (115/120) |
| NLI accuracy (C/X) | **89.2%** (107/120) |
| NLI accuracy (all) | **91.9%** (147/160) |
| Unverifiable NLI   | **100%** (40/40)    |

### Retrieval × NLI (C/X only)

| Bucket            | Count |
| ----------------- | ----: |
| both_ok           |   104 |
| nli_hallucination |    11 |
| lucky_nli         |     3 |
| both_fail         |     2 |

### Confusion matrix (counts)

| Expected \ predicted | Consistent | Contradictory | Unverifiable |
| -------------------- | ---------- | ------------- | ------------ |
| Consistent           | 69         | 5             | 6            |
| Contradictory        | 2          | 38            | 0            |
| Unverifiable         | 0          | 0             | 40           |

Period reports: `data/reports/golden/{year}/{TICKER}/`. Aggregates: `data/runs/golden/eval_20260827_074408/` (`summary.csv`, `confusion_matrix.csv`, `error_buckets.csv`, `per_claim.csv`, `metrics.json`).
