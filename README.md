# Crosscheck

Cross-document RAG that checks earnings-call claims against the same
company-quarter’s 10-Q/10-K filing, and returns structured findings:
**Consistent**, **Contradictory**, or **Unverifiable**.

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
    AAPL_FY2025_Q4_filing.jsonl
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
CROSSCHECK_LLM_PROFILE=development   # development | production
CROSSCHECK_PRODUCTION_MODEL=gemini   # gemini | gemini-lite | gemini-pro
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

## Step 2 — Embed, retrieve, and NLI (done)

Day 2 embeds chunk JSONL with **BGE-M3** (`BAAI/bge-m3`) on local **MPS** (Apple Silicon), builds dual FAISS indices (vectors) joined to Day 1 chunk JSONL (metadata + text), persists fixed executive claims via Google GenAI, dense-retrieves filing passages, and classifies each claim.

### Embeddings and context injection

Chunks are embedded with bracketed structural headers before encoding:

- **Narrative filing:** `[COMPANY: Apple | TYPE: 10-K | SECTION: Item 7 - MD&A]` + text
- **Table filing:** `[COMPANY: Apple | TYPE: 10-K | SECTION: … | TABLE: …]` + `Row: col1 | col2 | …` lines
- **Transcript:** adds `| SPEAKER: Name, Title`

Query-time claim embeddings use the raw claim text only.

### Vector store layout (FAISS + chunk metadata)

There is **no separate metadata vector DB** (no Chroma/Weaviate). Day 2 keeps
**separate** filings and transcripts corpora:

| On disk | Role |
|---------|------|
| `data/indices/filings/all_chunks.jsonl` | Filing text + metadata with sequential `global_id` |
| `data/indices/filings/embeddings.npy` | Disk-backed `(N, 1024)` float32 embedding memmap |
| `data/indices/filings/index.faiss` | Filings `IndexFlatIP` used by NLI |
| `data/indices/filings/manifest.json` | Model, dimensions, paths, counts |
| `data/indices/transcripts/*` | Same layout for transcript chunks (not used by NLI) |
| `data/chunks/{year}/{TICKER}/*.jsonl` | Stateless source chunks; no `global_id` |

At load time, FAISS row `i` is joined to `global_id=i` in the matching corpus
`all_chunks.jsonl`. NLI loads the **filings** index only.

### LLM profiles (Google GenAI for both)

Development tries models in **rate-limit rank order** (highest free-tier RPM/RPD first):

1. `gemini-3.1-flash-lite`
2. `gemini-2.5-flash-lite`
3. `gemini-3-flash-preview` → `gemini-2.5-flash` → `gemini-3.5-flash`

| Profile | Env | Model selection |
|---------|-----|-----------------|
| **development** (default) | `CROSSCHECK_LLM_PROFILE=development` | ranked list above |
| **production** | `CROSSCHECK_LLM_PROFILE=production` | preset first, then same rank |

Set `GOOGLE_API_KEY` for both profiles.

Production presets: `gemini-lite` (default), `gemini`, `gemini-pro`.

### Commands

```bash
# 4) Separate filings + transcripts indices (disk-stream embeddings → FAISS)
python scripts/build_indices.py --force --batch-size 24
# Or one corpus only:
# python scripts/build_indices.py --corpus filings --force
# python scripts/build_indices.py --corpus transcripts --force

# 5) Extract fixed claims (one ticker, or omit --ticker for all missing)
python scripts/extract_claims.py --ticker AAPL --n 10
python scripts/extract_claims.py --n 10

# 6) Load saved claims + retrieval + NLI → JSON report
python scripts/run_pipeline.py --ticker AAPL
python scripts/run_pipeline.py --ticker AAPL --profile production --top-k 5
```

Output: terminal JSON + `data/reports/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}.json`.

Claims are stored at `data/claims/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_claims.json`. Existing files are skipped unless `extract_claims.py --force` is used, giving RAG runs a fixed, repeatable debugging input.

**Note on free models:** `tencent/hy3:free` works for structured calls but 10× NLI can be slow. Use `--profile production` for reliable runs.

### Library map (Day 2)

| Path | Role |
|------|------|
| `src/crosscheck/retrieval/embeddings.py` | BGE-M3, bracketed context, MPS device |
| `src/crosscheck/retrieval/index.py` | FAISS build/load, `retrieve(k=5)` |
| `src/crosscheck/analysis/llm.py` | Google GenAI + instructor, model fallback |
| `src/crosscheck/analysis/executives.py` | CEO/CFO prepared-remarks filter |
| `src/crosscheck/analysis/claims.py` | LLM claim extraction + saved-claims storage |
| `src/crosscheck/analysis/nli.py` | LLM NLI classification |
| `src/crosscheck/analysis/pipeline.py` | End-to-end orchestrator |
| `src/crosscheck/models.py` | `FinancialClaim`, `ContradictionFinding`, `PipelineReport` |
| `scripts/build_indices.py` | Index CLI |
| `scripts/extract_claims.py` | Extract fixed claim sets for one/all missing companies |
| `scripts/run_pipeline.py` | Pipeline CLI |

---

## Status

| Step | Status |
|------|--------|
| **1 — Fetch + chunk** | **Done** — manifest dual ingest, year-first storage, section/speaker chunking, JSONL |
| **2 — Embed + retrieve + NLI** | **Done** — BGE-M3/MPS, dual FAISS + chunk JSONL metadata, Google GenAI, structured JSON reports |
| **3 — Hybrid + Streamlit** | Planned — BM25+RRF, demo UI |
