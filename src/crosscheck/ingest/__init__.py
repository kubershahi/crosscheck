"""Ingest adapters: SEC EDGAR filings and Motley Fool transcripts."""

from crosscheck.ingest.edgar import fetch_filing, resolve_cik
from crosscheck.ingest.motley_fool import fetch_transcript

__all__ = ["fetch_filing", "resolve_cik", "fetch_transcript"]
