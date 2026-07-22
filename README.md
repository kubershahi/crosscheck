# Crosscheck

Cross-document RAG that checks earnings-call claims against the same
company-quarter’s 10-Q/10-K filing, and returns structured findings:
**Consistent**, **Contradictory**, or **Unverifiable**.

> **Note:** Day 2 was built on FY2025 Q4 transcripts vs 10-K — a scope mismatch.
> Day 3 switches eval to Q1–Q3 (transcript ↔ 10-Q) before tackling Q4/10-K.

Design notes: [`earnings-rag-blueprint.md`](earnings-rag-blueprint.md).

---

## Step 1 — Fetch + chunk (done)

Fetch company-periods from the YAML manifest, then build the downstream corpus
from files present on disk:

1. **Filings** from SEC EDGAR (raw HTML kept for structure-aware chunking)
2. **Transcripts** from manifest URLs — Motley Fool primary; other hosts (e.g. TickerTrends) via generic extract
3. **Chunks** as JSONL under year-first folders

Day 1 does **not** embed or call an LLM.

### Data layout (year → ticker)

```text
data/
  manifests/companies.yml
  raw/
    filings/{fiscal_year}/{TICKER}/
      AAPL_FY2025_10K.html
      AAPL_FY2025_10K.html.meta.json
    transcripts/{fiscal_year}/{TICKER}/
      FY2025_Q4.fool.html      # raw page HTML
      FY2025_Q4.txt            # cleaned transcript
      FY2025_Q4.meta.json      # url, call_date, call_participants, …
  chunks/{fiscal_year}/{TICKER}/
    AAPL_FY2025_Q4_10-K.jsonl
    AAPL_FY2025_Q4_transcript.jsonl
  claims/{fiscal_year}/{TICKER}/
    AAPL_FY2025_Q4_claims.json
  indices/
    filings/
      all_chunks.jsonl
      embeddings.npy
      index.faiss
      manifest.json
    transcripts/
      all_chunks.jsonl
      embeddings.npy
      index.faiss
      manifest.json
  reports/{fiscal_year}/{TICKER}/
    AAPL_FY2025_Q4.json
```

Raw filings/transcripts, chunk JSONL, indices, and reports are gitignored; the manifest is committed.

### Setup

```bash
conda create -n crosscheck python=3.11 -y
conda activate crosscheck
pip install -e .

cp .env.example .env
# SEC_USER_AGENT="Crosscheck Your Name you@domain.com"   # real email (not GitHub noreply)
# GOOGLE_API_KEY=...                                     # Google GenAI (dev + production)
```

Day 2 optional env (see `.env.example`):

```bash
CROSSCHECK_LLM_PROFILE=development   # development | production (same model rank)
CROSSCHECK_EMBEDDING_DEVICE=mps      # mps | cpu | cuda
```

### Manifest (fetch source of truth)

[`data/manifests/companies.yml`](data/manifests/companies.yml) controls only
what ingest fetches. Chunking and later stages discover files in their input
data directories.

```yaml
companies:
  - ticker: AAPL
    name: Apple              # colloquial; stored on chunks as company_name
    fiscal_year: 2025
    fiscal_quarter: 4      # Q4 → 10-K; Q1–Q3 → 10-Q when form is null
    form: null
    transcript_url: "https://www.fool.com/earnings/call-transcripts/..."
```

Current Day 1 corpus: 12 mega-caps (the `KNOWN_CIKS` set), FY2025 Q4, with Motley Fool links (Oracle uses TickerTrends).

### CIK lookup

EDGAR needs a **CIK**, not a ticker. Order in `edgar.resolve_cik`:

1. `KNOWN_CIKS` in [`config.py`](src/crosscheck/config.py)
2. Else SEC `company_tickers.json`

No hardcoded “default companies” outside the manifest.

### Commands

```bash
# 1) Fetch EDGAR filing + transcript (skips files already on disk)
python scripts/fetch_corpus.py                 # all rows in manifest
python scripts/fetch_corpus.py --ticker AAPL
# python scripts/fetch_corpus.py --filings-only
# python scripts/fetch_corpus.py --transcripts-only --force

# 2) Chunk → JSONL (discovers complete pairs under data/raw)
python scripts/build_chunks.py                 # every raw ticker
python scripts/build_chunks.py --ticker AAPL
python scripts/build_chunks.py --ticker AAPL,MSFT

# 3) Spot-check random chunks + metadata
python scripts/sanity_chunks.py --ticker AAPL
```

### Transcript `.txt` layout

Cleaned Fool (and generic) transcripts always look like:

```text
Date
…

Call participants
…                    # or: section not found

Industry glossary
…                    # or: section not found

Full Conference Call Transcript

Operator / Name: …
```

Call body start is flexible (full-transcript header → prepared remarks → first speaker). Takeaways / Summary / Risks from Fool are dropped.

### Chunking

| Corpus | Strategy | Module |
|--------|----------|--------|
| Filings | Section-aware prose + **atomic** HTML tables (TSV) | `src/crosscheck/chunking/filings.py` |
| Transcripts | Speaker-turn (`Name:`, `Name -- Title`, `Operator`); `speaker_title` from Call participants when available | `src/crosscheck/chunking/transcripts.py` |

Every chunk carries `ticker` + `company_name` (colloquial) so later retrieval can filter by either.

### Library map

| Path | Role |
|------|------|
| `src/crosscheck/config.py` | Paths, `KNOWN_CIKS`, SEC URLs, `SEC_USER_AGENT` |
| `src/crosscheck/manifest.py` | YAML → `CompanyPeriod` |
| `src/crosscheck/models.py` | `Chunk`, `DocumentMeta` |
| `src/crosscheck/ingest/edgar.py` | EDGAR fetch (raw `requests`, not `sec-edgar-api`) |
| `src/crosscheck/ingest/motley_fool.py` | Transcript scrape + clean |
| `src/crosscheck/chunking/pipeline.py` | Orchestrate chunk + write |
| `src/crosscheck/chunking/store.py` | JSONL read/write |
| `scripts/fetch_corpus.py` | Dual-ingest CLI |
| `scripts/build_chunks.py` | Chunk CLI (`--ticker` optional) |
| `scripts/sanity_chunks.py` | Inspect CLI |

---

## Step 2 — Embed, retrieve, rerank, and NLI (done)

Day 2 turns Day 1 chunks into searchable indices, extracts fixed transcript claims,
retrieves filing evidence per claim, and classifies each claim as **Consistent**,
**Contradictory**, or **Unverifiable**. Full design notes:
[`earnings-rag-blueprint.md`](earnings-rag-blueprint.md) (Day 2 section).

### Key decisions

| Decision | Choice |
|----------|--------|
| Manifest scope | Fetch only; chunking+ downstream discover files in `data/raw`, `data/chunks`, `data/claims` |
| Chunk vs index state | Stateless per-company JSONL → merged corpus rows with `global_id` aligned to FAISS row `i` |
| Vector corpora | **Separate** filings and transcripts indices; NLI searches **filings only** |
| Embeddings | `BAAI/bge-m3` (1024-d), contextual headers at index time; raw claim text at query time |
| Claim source | C-suite speaker turns (CEO/CFO/etc.) stitched in call order — not section labels |
| Claim persistence | One JSON per company-quarter; skipped on re-run unless `--force` |
| Retrieval | Dense FAISS → filter `ticker` + `fiscal_year` → cross-encoder rerank → top-k |
| Reranker | `BAAI/bge-reranker-v2-m3` (pairs with BGE-M3 bi-encoder) |
| LLM | Google GenAI + `instructor`; ranked Gemini fallback by free-tier RPM/RPD |
| NLI cost | **One LLM request per claim** (3 claims → 3 NLI calls) |

### Pipeline at a glance

```text
chunks → build_indices (filings + transcripts)
claims ← extract_claims (1 LLM call per transcript period)
report ← run_pipeline:  claim → FAISS → rerank → NLI (1 LLM call per claim)
```

### Commands

```bash
# 4) Corpus indices (disk-stream embeddings → FAISS)
python scripts/build_indices.py --force --batch-size 24
# python scripts/build_indices.py --corpus filings --force   # NLI minimum

# 5) Fixed claims (optional --ticker; skip existing unless --force)
python scripts/extract_claims.py --ticker AAPL --n 3
python scripts/extract_claims.py --n 3

# 6) Retrieve + rerank + NLI → report JSON
python scripts/run_pipeline.py --ticker AAPL
python scripts/run_pipeline.py --ticker AAPL --top-k 5 --rerank-pool-k 50
python scripts/run_pipeline.py --ticker AAPL --no-rerank   # dense-only ablation
```

Reports: `data/reports/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}.json`  
Claims: `data/claims/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_claims.json`

Set `GOOGLE_API_KEY`. LLM profile: `CROSSCHECK_LLM_PROFILE=development|production`.

### Contextual injection (index time only)

Chunk JSONL stores **raw text**. Headers are prepended in memory before BGE-M3 encoding only (`chunk_embedding_text` in `embeddings.py`). Query-time claims use raw claim text with no header.

| Corpus | Injected header (then `\n` + raw `chunk.text`) |
|--------|--------------------------------------------------|
| Filing (prose/table) | `[COMPANY: {ticker} \| PERIOD: {FY\|Qn}{year} \| TYPE: 10-K\|10-Q \| SECTION: {section} \| TABLE: {true\|false}]` |
| Transcript | `[COMPANY: {ticker} \| PERIOD: {Qn}{year} \| TYPE: transcript \| SPEAKER: {name} ({role})]` |

Reranker input uses a lighter header (`rerank.py`): `[TYPE: … \| PERIOD: … \| SECTION: … \| TABLE: …]` + raw text. Table chunks today are **TSV** (tab-separated rows from HTML); Day 3 may switch to markdown or key-value if retrieval stays weak.

### LLM configuration (Google GenAI + instructor)

Env: `GOOGLE_API_KEY`, `CROSSCHECK_LLM_PROFILE` (optional label; **both profiles use the same model rank**).

**Development and production** — identical try order (highest free-tier RPM/RPD first):

1. `gemini-3.1-flash-lite` (~15 RPM / 500 RPD)
2. `gemini-2.5-flash-lite` (~10 RPM / 20 RPD)
3. `gemini-3-flash-preview`
4. `gemini-2.5-flash`
5. `gemini-3.5-flash`

Per model, `instructor` tries modes in order: `GENAI_STRUCTURED_OUTPUTS` → `GENAI_JSON` → `GENAI_TOOLS`. On 404/unavailable, skip to next model; on rate limit, retry once then fall back.

**API cost:** claim extraction = 1 call per transcript period; NLI = **1 call per claim**.

### Library map (Day 2)

| Path | Role |
|------|------|
| `src/crosscheck/retrieval/embeddings.py` | BGE-M3, bracketed context, MPS device |
| `src/crosscheck/retrieval/index.py` | Corpus merge, FAISS build/load, `retrieve()` |
| `src/crosscheck/retrieval/rerank.py` | BGE cross-encoder reranking |
| `src/crosscheck/analysis/llm.py` | Google GenAI + instructor, model fallback |
| `src/crosscheck/analysis/executives.py` | C-suite speaker filter + text stitching |
| `src/crosscheck/analysis/claims.py` | Claim extraction + saved-claims I/O |
| `src/crosscheck/analysis/nli.py` | NLI classification |
| `src/crosscheck/analysis/pipeline.py` | Retrieve → rerank → NLI orchestrator |
| `scripts/build_indices.py` | Index CLI (`--corpus filings\|transcripts\|both`) |
| `scripts/extract_claims.py` | Claim extraction CLI |
| `scripts/run_pipeline.py` | Pipeline CLI |

---

## Step 3 — Retrieval quality + scale (planned)

Day 2 proved the loop on FY2025 Q4, but **Q4 transcript vs 10-K is a hard pairing** — the annual filing covers the full year while the call discusses one quarter; retrieval and NLI quality suffer. Day 3 prioritizes easier pairs first, then scales infra.

| Priority | Work |
|----------|------|
| 1 | **Switch eval corpus to Q1–Q3** — transcript ↔ matching **10-Q** (same quarter scope); re-fetch, chunk, claims, pipeline |
| 2 | **Hybrid retrieval** — BM25 + dense FAISS, merged with reciprocal rank fusion (Non-GAAP / exact-term gap) |
| 3 | **Table chunk formats** — if hybrid still weak, re-embed tables as markdown or key-value pairs instead of TSV |
| 4 | **Qdrant** — once retrieval is good on Q1–Q3, migrate from local FAISS to managed Qdrant (metadata filter, incremental updates) |
| 5 | **Q4 vs 10-K** — after Q1–Q3 works: temporal tagging (`historical` / `current_period` / `forward_guidance`), horizon-biased retrieval, prompt engineering |
| + | Eval harness (Context Recall@K vs label precision), Streamlit demo |

Full Day 3 spec: [`earnings-rag-blueprint.md`](earnings-rag-blueprint.md) (Day 3 section).

---

## Status

| Step | Status |
|------|--------|
| **1 — Fetch + chunk** | **Done** — manifest dual ingest, year-first storage, section/speaker chunking, JSONL |
| **2 — Embed + retrieve + rerank + NLI** | **Done** — BGE-M3/MPS, dual FAISS corpora, fixed claims, BGE reranker, Gemini NLI, JSON reports |
| **3 — Q1–Q3 corpus + hybrid + Qdrant + Q4/10-K** | Planned — see Step 3 table above |
