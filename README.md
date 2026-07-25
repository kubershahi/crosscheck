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
| Claims | `scripts/extract_claims.py` | Yes (1/period) | `data/claims/.../*_claims.json` |
| NLI | `scripts/run_nli.py` | Yes (1/claim) | `data/reports/.../*_reports.json` |

---

## Status

| Stage | Status |
|-------|--------|
| Fetch + chunk | **Done** |
| Embed + hybrid retrieve + rerank + NLI | **Done** |
| Q1 2025 multi-ticker validation | **Done** (5 cos) |
| Next (ship path) | **Q2+Q3 → Qdrant → Streamlit deploy → ~45 golden + eval** — see blueprint |

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
CROSSCHECK_EMBEDDING_DEVICE=mps       # mps | cpu | cuda
```

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
python scripts/fetch_corpus.py                 # all opted-in manifest rows
python scripts/fetch_corpus.py --ticker AAPL
# --filings-only | --transcripts-only | --force
```

**Implementation:** `ingest/edgar.py` resolves CIK (`KNOWN_CIKS` then SEC map),
downloads HTML + sidecar `.meta.json`. `ingest/transcript.py` fetches the call
URL, saves raw HTML, writes cleaned speaker-turn `.txt` + meta (Fool / ROIC /
generic host detection).

### 3 — Chunk

```bash
python scripts/build_chunks.py                 # all complete raw pairs
python scripts/build_chunks.py --ticker AAPL
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

### 4 — Build indices

```bash
python scripts/build_indices.py --force --batch-size 24
python scripts/build_indices.py --corpus filings --force   # NLI minimum
```

**Implementation** (`retrieval/index.py`):

1. Merge company JSONL → `all_chunks.jsonl` with sequential `global_id`
2. Stream BGE-M3 embeddings to `embeddings.npy` (memmap)
3. Build FAISS `IndexFlatIP` (normalized vectors → cosine via IP)
4. On **filings** load: build in-memory BM25 from the same chunk texts (no
   separate BM25 artifact)

**Contextual injection (index time only):** JSONL stores raw `text`. Before
encode, `chunk_embedding_text()` prepends a metadata header
(`[COMPANY|NAME|PERIOD|TYPE|SECTION|TABLE|…]`). Query-time claims use **raw
claim text** (asymmetric by design).

Separate corpora: `data/indices/filings/` (NLI retrieval) and
`data/indices/transcripts/`.

### 5 — Extract claims

```bash
python scripts/extract_claims.py --ticker AAPL --n 3
python scripts/extract_claims.py --n 3          # all periods on disk
# --force to overwrite existing claims JSON
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
# --gap-seconds 62 between claim files (default; rate-limit spacing)
```

**Per claim:**

1. **Filters** from claims **file header** (not the claim sentence):
   `ticker`, `fiscal_year`, `fiscal_quarter`
2. **Hybrid retrieve** — dense FAISS + BM25, each filtered, RRF (`k=60`) →
   `pool_k`
3. **Rerank** — `BAAI/bge-reranker-v2-m3` → `top_k` (default 5)
4. **NLI** — Gemini judges Consistent / Contradictory / Unverifiable  
   Citations (`retrieved_filing_passages`, `source_sections`) filled from
   retrieval in code, not by the LLM

**Quarter match:** chunk `fiscal_period == Qn`; annual `FY` matches only when
claim quarter is `Q4`.

**NLI prompt context:** claim side gets ticker/company/year/quarter + speaker +
claim; each passage gets ticker/company/period label/quarter months/section +
text.

Output: `data/reports/{year}/{TICKER}/{TICKER}_FY{year}_Q{n}_reports.json`

---

## Data layout

```text
data/
  manifests/companies.yml
  raw/
    filings/{year}/{TICKER}/     # *.html + *.meta.json
    transcripts/{year}/{TICKER}/ # *.html, *.txt, *.meta.json
  chunks/{year}/{TICKER}/        # per-doc JSONL (stateless)
  claims/{year}/{TICKER}/        # *_claims.json
  indices/
    filings/                     # all_chunks.jsonl, embeddings.npy, index.faiss
    transcripts/
  reports/{year}/{TICKER}/       # *_reports.json
```

Raw corpora, chunks, indices, and reports are gitignored; the manifest is
committed.

---

## Major design choices

| Topic | Choice |
|-------|--------|
| Dual corpus | Separate filings vs transcripts FAISS; NLI queries **filings only** |
| Hybrid search | Dense + BM25 → RRF → BGE rerank |
| Period safety | Hard filter `ticker` + `fiscal_year` + `fiscal_quarter` on both channels |
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
| `src/crosscheck/retrieval/index.py` | Merge, FAISS, `hybrid_retrieve` |
| `src/crosscheck/retrieval/hybrid.py` | BM25 + RRF + period match |
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
