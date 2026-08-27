# Crosscheck

Cross-document RAG that verifies earnings-call claims against the same
company-quarter’s SEC **10-Q / 10-K** filing.

For each claim the system retrieves filing evidence and returns a structured
finding: **Consistent**, **Contradictory**, or **Unverifiable**, with cited
passages, confidence, and reasoning.

Design / roadmap: [`earnings-rag-blueprint.md`](earnings-rag-blueprint.md).

---

## What it does

Companies speak to investors on earnings calls (verbal, colloquial) and in SEC
filings (written, audited). Crosscheck checks whether numeric claims from the
call are supported by the matching-period filing.

```text
Transcript claim  →  hybrid retrieve 10-Q/10-K passages  →  NLI classify
                     (same ticker + year + quarter)
```

**Validated so far:** FY2025 **Q1** transcript ↔ **10-Q** retrieval for five
companies (AAPL, MSFT, GOOGL, META, NVDA) with correct period-scoped hits.

---

## Pipeline overview

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Fetch     │ → │ 2. Chunk     │ → │ 3. Index     │
│ EDGAR + call │   │ JSONL        │   │ FAISS+BM25   │
└──────────────┘   └──────────────┘   └──────────────┘
                                              │
┌──────────────┐   ┌──────────────┐   ┌───────▼──────┐
│ 6. Report    │ ← │ 5. NLI       │ ← │ 4. Claims    │
│ *_reports.js │   │ retrieve+LLM │   │ extract LLM  │
└──────────────┘   └──────────────┘   └──────────────┘
```

| Stage | Script | LLM? | Output |
|-------|--------|------|--------|
| Fetch | `scripts/fetch_corpus.py` | No | `data/raw/...` |
| Chunk | `scripts/build_chunks.py` | No | `data/chunks/.../*.jsonl` |
| Index | `scripts/build_indices.py` | No | `data/indices/{filings\|transcripts}/` |
| Claims | `scripts/extract_claims.py` | Yes (1/period) | `data/claims/.../*_claims.jsonl` |
| NLI | `scripts/run_nli.py` | Yes (1/claim) | `data/reports/...` + `data/runs/run_<ts>.csv` |
| Get eval candidates | `scripts/eval/get_eval_candidates.py` | Yes | `data/eval/claims/.../*_claims.jsonl` |
| Verify eval candidates | `scripts/eval/verify_eval_candidates.py` | Yes (1/claim) | `data/eval/candidates/{year}/{TICKER}/` |
| Promote eval candidates | `scripts/eval/promote_eval_candidates.py` | Yes (matches) | `data/eval/golden/{year}/{TICKER}/` |

---

## Status

| Stage | Status |
|-------|--------|
| Fetch + chunk | **Done** |
| Embed + hybrid retrieve + rerank + NLI | **Done** |
| Q1 2025 multi-ticker validation | **Done** (5 cos) |
| Next (ship path) | Review `data/eval/golden/` → retrieval/NLI eval harness |

---

## Setup

```bash
conda create -n crosscheck python=3.11 -y
conda activate crosscheck
pip install -e .

cp .env.example .env
```

Required in `.env`:

```bash
SEC_USER_AGENT="Crosscheck Your Name you@domain.com"   # real email (SEC policy)
GOOGLE_API_KEY=...                                    # Google AI Studio
```

Optional:

```bash
CROSSCHECK_LLM_PROFILE=development    # development | production (same model rank)
CROSSCHECK_EMBEDDING_DEVICE=mps       # local BGE-M3 embed (+ rerank fallback): mps|cpu|cuda
CROSSCHECK_RERANK_BACKEND=pinecone    # pinecone (default) | local
PINECONE_API_KEY=...                  # required for default Pinecone Inference rerank
```

Dense embed (`BAAI/bge-m3`) stays **local** via sentence-transformers.
Rerank (`bge-reranker-v2-m3`) defaults to **Pinecone Inference**, with local
Torch/MPS fallback if the API call fails or `CROSSCHECK_RERANK_BACKEND=local`.

```bash
python scripts/build_indices.py --corpus filings --force --batch-size 32
```
---

## Demo (Streamlit)

```bash
pip install -e .
streamlit run streamlit_app.py
```

Prefixed **ticker / year / quarter** selectors. Loads `*_reports.json` when
present; **Run analysis** re-runs Qdrant hybrid + rerank + NLI when claims
exist (needs `.env` keys).

---

## End-to-end workflow

### 1 — Configure what to fetch

Edit [`data/manifests/companies.yml`](data/manifests/companies.yml). Manifest
controls **fetch only**; later stages discover files on disk.

```yaml
companies:
  - ticker: AAPL
    name: Apple
    fiscal_year: 2025
    include: true
    quarters:
      Q1:
        fetch: true
        transcript_url: "https://www.fool.com/earnings/call-transcripts/..."
```

`form` omitted → Q1–Q3 = 10-Q, Q4 = 10-K.

### 2 — Fetch corpus

```bash
python scripts/fetch_corpus.py                              # include: true rows
python scripts/fetch_corpus.py --ticker AAPL
python scripts/fetch_corpus.py --year 2025 --quarter Q2
python scripts/fetch_corpus.py --ticker AAPL --year 2025 --quarter Q3
# --filings-only | --transcripts-only | --force | --all
```

**Implementation:** `ingest/edgar.py` resolves CIK (`KNOWN_CIKS` then SEC map),
downloads HTML + sidecar `.meta.json`. `ingest/transcript.py` fetches the call
URL, saves raw HTML, writes cleaned speaker-turn `.txt` + meta (Fool / ROIC /
generic host detection).

### 3 — Chunk

```bash
python scripts/build_chunks.py                              # only missing JSONL
python scripts/build_chunks.py --ticker AAPL
python scripts/build_chunks.py --year 2025 --quarter Q2
python scripts/build_chunks.py --ticker AAPL --year 2025 --quarter Q3 --force
python scripts/sanity_chunks.py --ticker AAPL  # spot-check
```

**Filings** (`chunking/filings.py`):

- `section` = Item / PART headers
- Sticky `subsection` / `subsubsection` (right-aligned or bold+justify titles)
- Tables: centered preceding titles (or short caption fallback) + full Markdown
  table + footers; whole `<table>` = one chunk
- Prose: subsection titles prepended into `text` (metadata kept; not re-injected
  at embed time)

**Transcripts** (`chunking/transcripts.py`): speaker-turn chunks with
`speaker_name` / `speaker_role` / `call_date`.

### 4 — Build indices (filings → Qdrant Cloud)

```bash
# Requires QDRANT_ENDPOINT + QDRANT_API_KEY in .env
python scripts/build_indices.py --corpus filings --force --batch-size 32
python scripts/build_indices.py --corpus transcripts --force --batch-size 32
# or both:
python scripts/build_indices.py --corpus both --force
```
**Filings** (`retrieval/index.py` + `retrieval/qdrant_store.py`):

1. Merge company JSONL → `all_chunks.jsonl` with sequential `global_id`
2. Stream BGE-M3 dense vectors → `embeddings.npy` (local scratch)
3. Upsert each chunk to Qdrant Cloud as one point:
   - named vector `dense` (BGE-M3)
   - named sparse `sparse` (Qdrant BM25 over chunk text)
   - **full** `IndexedChunk` payload (all metadata + text)

**Query-time hybrid (on Cloud):** dense + BM25 prefetches → RRF + payload
filters (`ticker`, `fiscal_year`, `fiscal_period`). Local machine only embeds
the claim and runs the cross-encoder reranker afterward.

**Contextual injection (index time only):** JSONL stores raw `text`. Before
encode, `chunk_embedding_text()` prepends a metadata header. Query-time claims
use **raw claim text** (asymmetric by design).

Transcripts use the same Qdrant hybrid path under collection
`QDRANT_TRANSCRIPTS_COLLECTION` (default `transcripts`).

### 5 — Extract claims

```bash
python scripts/extract_claims.py --n 5
python scripts/extract_claims.py --ticker AAPL --n 5
python scripts/extract_claims.py --year 2025 --quarter Q2 --n 5
python scripts/extract_claims.py --ticker AAPL --year 2025 --quarter Q3 --force --n 5
```

**Implementation:** one Gemini structured call over the full cleaned transcript
`.txt`. Prompt prefers current-quarter actuals; skips analyst/operator noise.
Persists `*_claims.json` with header `{ticker, company_name, fiscal_year,
fiscal_quarter}` + `claims[{claim, speaker}]`.

### 6 — Retrieve + NLI

```bash
python scripts/run_nli.py --ticker AAPL --year 2025 --quarter Q1
python scripts/run_nli.py --year 2025            # all tickers that year
python scripts/run_nli.py                        # all claims on disk
python scripts/run_nli.py --ticker AAPL --no-rerank
# On HTTP 429 the LLM client sleeps 62s and retries (no proactive period gap)
```

**Per claim:**

1. **Query preprocess** (deterministic, no LLM) — expand `$N billion` →
   `(N000 million)` for table match; for **Q4 claim periods only**, classify
   temporal scope (full-year → 10-K / Q4-composite → dual 10-K+Q3 10-Q). Q1–Q3 stay
   on standard 10-Q filters.
2. **Filters** from claims **file header** (ticker, year) plus temporal plan;
   claim-quarter used for standard Q1–Q3 10-Q retrieval
3. **Hybrid retrieve** — Qdrant Cloud dense + BM25 + RRF → `pool_k`
   (embed/BM25 use the *processed* query; rerank still uses raw claim text)
4. **Rerank** — `BAAI/bge-reranker-v2-m3` → `top_k` (default 5)
5. **NLI** — Gemini judges Consistent / Contradictory / Unverifiable  
   Citations (`retrieved_filing_passages`, `chunk_ids`, `global_ids`) filled from
   retrieval in code, not by the LLM. Passages are ranked (rerank score, best first).

**Quarter match:** standard claims use chunk `fiscal_period == Qn`. Full-year
claims target `10-K`/`FY`. Q4-composite claims run separate retrieval paths
for `(10-K, FY)` and `(10-Q, Q3)`, preserving three passages from each.

**NLI prompt context:** claim side gets ticker/company/year/quarter + speaker +
claim; each passage gets ticker/company/period label/quarter months/section +
text.

Output: `data/reports/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_reports.json`  
plus a run summary CSV: `data/runs/run_YYYYMMDD_HHMMSS.csv` (company +
period tables with TOTAL rows).

### 7 — Eval candidates → golden set

Labeled drafts live under `data/eval/claims/` (4 Consistent + 2 Contradictory +
2 Unverifiable). Verify scores them with retrieve + NLI into
`data/eval/candidates/`. Promote copies label-matching rows into
`data/eval/golden/`.

`run_nli.py` is unchanged — it still reads pure claims from `data/claims/`.

```bash
# first pass (get → verify → promote)
python scripts/eval/run_eval_candidates.py --ticker AAPL --year 2025 --quarter Q1

# later: iterate one stage at a time
python scripts/eval/get_eval_candidates.py --mode modify --ticker AAPL --year 2025 --quarter Q1
python scripts/eval/verify_eval_candidates.py --mode modify --ticker AAPL --year 2025 --quarter Q1
python scripts/eval/promote_eval_candidates.py --ticker AAPL --year 2025 --quarter Q1
```

---

## Data layout

```text
data/
  manifests/companies.yml
  raw/
    filings/{year}/{TICKER}/     # *.html + *.meta.json
    transcripts/{year}/{TICKER}/ # *.html, *.txt, *.meta.json
  chunks/{year}/{TICKER}/        # per-doc JSONL (stateless)
  claims/{year}/{TICKER}/        # *_claims.json (extract_claims / run_nli)
  indices/
    filings/                     # all_chunks.jsonl, embeddings.npy, manifest.json
    transcripts/                 # merge + embeddings scratch for Qdrant
    qdrant/                      # local path fallback if no QDRANT_ENDPOINT
  reports/{year}/{TICKER}/       # *_reports.json
  runs/                          # run_<timestamp>.csv NLI summaries
  eval/
    claims/{year}/{TICKER}/      # labeled drafts (get_eval_candidates)
    candidates/{year}/{TICKER}/  # retrieve+NLI scores (verify_eval_candidates)
    golden/{year}/{TICKER}/      # curated golden set (promote_eval_candidates)
```

Qdrant Cloud holds filings points (dense + BM25 + full payload). Set
`QDRANT_ENDPOINT` / `QDRANT_API_KEY` in `.env`.
Raw corpora, chunks, indices, and reports are gitignored; the manifest is
committed.

---

## Major design choices

| Topic | Choice |
|-------|--------|
| Dual corpus | Filings + transcripts on **Qdrant Cloud** hybrid (separate collections) |
| Hybrid search | Qdrant dense + BM25 → RRF (+ local BGE rerank) |
| Period safety | Qdrant payload filter `ticker` + `fiscal_year` + `fiscal_period` |
| Tables | Atomic Markdown chunks; centered/caption titles kept with the table |
| Claims | Fixed JSON per period (reproducible NLI runs) |
| Embeddings | `BAAI/bge-m3` (1024-d); headers at index time only |
| LLM | Google GenAI + `instructor`; ranked Flash fallback |
| Cost | 1 LLM call / transcript period (claims) + 1 / claim (NLI) |

### LLM model rank

1. `gemini-3.5-flash-lite`
2. `gemini-3.1-flash-lite`
3. `gemini-2.5-flash-lite`
4. `gemini-3.5-flash`

Per model, modes: `STRUCTURED_OUTPUTS` → `JSON` → `TOOLS`. HTTP timeout 60s.

---

## Library map

| Path | Role |
|------|------|
| `src/crosscheck/config.py` | Paths, CIKs, LLM rank, env |
| `src/crosscheck/manifest.py` | YAML → fetch periods |
| `src/crosscheck/models.py` | Chunk, claims, NLI schemas |
| `src/crosscheck/ingest/edgar.py` | SEC filing download |
| `src/crosscheck/ingest/transcript.py` | Call scrape + clean |
| `src/crosscheck/chunking/filings.py` | Section/table filing chunks |
| `src/crosscheck/chunking/transcripts.py` | Speaker-turn chunks |
| `src/crosscheck/chunking/pipeline.py` | Chunk orchestration |
| `src/crosscheck/retrieval/embeddings.py` | BGE-M3 + contextual headers |
| `src/crosscheck/retrieval/query_processor.py` | Unit expansion + temporal scope for retrieve |
| `src/crosscheck/retrieval/index.py` | Merge, embed, filings/transcripts → Qdrant |
| `src/crosscheck/retrieval/qdrant_store.py` | Cloud client, upsert, hybrid query + filters |
| `src/crosscheck/retrieval/hybrid.py` | Local BM25 helpers (legacy / transcripts path) |
| `src/crosscheck/retrieval/rerank.py` | BGE cross-encoder |
| `src/crosscheck/analysis/claims.py` | Claim extraction I/O |
| `src/crosscheck/analysis/nli.py` | NLI call + citation fill |
| `src/crosscheck/analysis/pipeline.py` | Retrieve → rerank → NLI |
| `src/crosscheck/analysis/llm.py` | GenAI client + fallbacks |
| `src/crosscheck/analysis/prompts.py` | Claim + NLI prompts |

---

## What’s next

Remaining MVP vs full-product work is tracked at the end of
[`earnings-rag-blueprint.md`](earnings-rag-blueprint.md)
(**Remaining steps**).
