# Crosscheck

Cross-document RAG that checks earnings-call claims against the same
company-quarter’s 10-Q/10-K filing, and returns structured findings:
**Consistent**, **Contradictory**, or **Unverifiable**.

Design notes: [`earnings-rag-blueprint.md`](earnings-rag-blueprint.md).

---

## Step 1 — Fetch + chunk (implemented)

Build a dual corpus for one or more company-periods:

1. **Filings** from SEC EDGAR (raw HTML kept for structure-aware chunking)
2. **Transcripts** from URLs in a YAML manifest — usually Motley Fool; other hosts (e.g. TickerTrends) work via a generic extractor (raw HTML + cleaned `.txt`)
3. **Chunks** as JSONL under year-first folders

### Data layout (year → ticker)

```text
data/
  manifests/companies.yml
  raw/
    filings/{fiscal_year}/{TICKER}/
      AAPL_FY2024_10K.html
      AAPL_FY2024_10K.html.meta.json
    transcripts/{fiscal_year}/{TICKER}/
      FY2024_Q4.fool.html      # raw Motley Fool page
      FY2024_Q4.txt            # cleaned speaker-turn text
      FY2024_Q4.meta.json
  chunks/{fiscal_year}/{TICKER}/
    AAPL_FY2024_Q4_filing.jsonl
    AAPL_FY2024_Q4_transcript.jsonl
```

### Setup

```bash
conda create -n crosscheck python=3.11 -y
conda activate crosscheck
pip install -e .

cp .env.example .env
# SEC_USER_AGENT="Crosscheck Your Name you@domain.com"   # real email domain
```

### Manifest

Edit [`data/manifests/companies.yml`](data/manifests/companies.yml):

```yaml
companies:
  - ticker: AAPL
    name: Apple              # colloquial; stored on chunks as company_name
    fiscal_year: 2025
    fiscal_quarter: 4      # Q4 → 10-K; Q1–Q3 → 10-Q
    form: null
    transcript_url: "https://www.fool.com/earnings/call-transcripts/..."
```

### CIK lookup (ticker → SEC company id)

EDGAR URLs need a **CIK**, not a ticker. Resolution order in `edgar.resolve_cik`:

1. Optional cache ``KNOWN_CIKS`` in [`config.py`](src/crosscheck/config.py) (common mega-caps)
2. Else SEC’s public map: `https://www.sec.gov/files/company_tickers.json` (`COMPANY_TICKERS_URL` in config)

If a ticker is missing from both, fetch fails with a clear error — add the CIK to `KNOWN_CIKS` or fix the symbol. There is **no** `DEFAULT_TARGETS` fallback; companies must be listed in the manifest.

### Commands

```bash
# 1) Fetch EDGAR filing + Motley Fool transcript
python scripts/fetch_corpus.py --ticker AAPL
# python scripts/fetch_corpus.py --filings-only
# python scripts/fetch_corpus.py --transcripts-only --force

# 2) Chunk → JSONL under data/chunks/{year}/{TICKER}/
python scripts/build_chunks.py --ticker AAPL

# 3) Spot-check random chunks + metadata
python scripts/sanity_chunks.py --ticker AAPL
```

### Chunking strategies

| Corpus | Strategy | Module |
|--------|----------|--------|
| Filings | Section-aware prose + **atomic** HTML tables (TSV) | `src/crosscheck/chunking/filings.py` |
| Transcripts | **Speaker-turn** splits (`Name:`, `Name -- Title`, …); optional ``speaker_title`` from Call participants | `src/crosscheck/chunking/transcripts.py` |

No embedding model in Step 1. Day 2 adds `all-mpnet-base-v2` + dual FAISS indices.

### Library map

| Path | Role |
|------|------|
| `src/crosscheck/config.py` | Paths, year/ticker helpers, `KNOWN_CIKS` cache, SEC URLs, `SEC_USER_AGENT` |
| `src/crosscheck/manifest.py` | YAML `CompanyPeriod` loader (**source of truth for what to fetch/chunk**) |
| `src/crosscheck/models.py` | `Chunk`, `DocumentMeta` |
| `src/crosscheck/ingest/edgar.py` | EDGAR fetch (raw HTML) |
| `src/crosscheck/ingest/motley_fool.py` | Fool scrape + clean `.txt` |
| `src/crosscheck/chunking/pipeline.py` | Orchestrate chunk + write |
| `src/crosscheck/chunking/store.py` | JSONL read/write |
| `scripts/fetch_corpus.py` | Dual-ingest CLI |
| `scripts/build_chunks.py` | Chunk CLI |
| `scripts/sanity_chunks.py` | Inspect CLI |

---

## Status

- **Step 1 done:** manifest dual ingest, year-first storage, section/speaker chunking, JSONL persistence  
- **Next (Step 2 / Day 2):** embeddings, dual FAISS, structured NLI contradiction findings
