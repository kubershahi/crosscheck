"""Shared Pydantic models for documents, retrievable chunks, and NLI output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["10-K", "10-Q", "transcript"]
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class Chunk(BaseModel):
    """Stateless chunk unit — source metadata only (no global index ids)."""

    chunk_id: str
    ticker: str
    company_name: str | None = None
    doc_type: DocType
    fiscal_year: int
    fiscal_period: FiscalPeriod
    is_table: bool = False
    text: str

    # Document-specific
    section: str | None = None
    speaker_name: str | None = None
    speaker_role: str | None = None

    def metadata_dict(self) -> dict:
        """Return all fields except ``text`` (handy for sanity printing)."""
        return self.model_dump(exclude={"text"})


class IndexedChunk(Chunk):
    """Merged master-corpus row with a sequential FAISS-aligned ``global_id``."""

    global_id: int = Field(ge=0)


class DocumentMeta(BaseModel):
    """Period identity used while chunking a source document."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    form: str | None = None  # 10-K / 10-Q for filings
    source_path: str | None = None


def fiscal_period_from_quarter(quarter: int) -> FiscalPeriod:
    """Map fiscal quarter 1–4 to ``Q1``…``Q4``."""
    return f"Q{quarter}"  # type: ignore[return-value]


def filing_doc_type(form: str | None, fiscal_quarter: int) -> DocType:
    """Resolve SEC form to ``10-K`` or ``10-Q``."""
    if form:
        form = form.strip().upper()
        if form in {"10-K", "10-Q"}:
            return form  # type: ignore[return-value]
    return "10-K" if fiscal_quarter == 4 else "10-Q"


def filing_fiscal_period(form: str | None, fiscal_quarter: int) -> FiscalPeriod:
    """Annual filings use ``FY``; quarterly filings use ``Q1``…``Q4``."""
    doc = filing_doc_type(form, fiscal_quarter)
    if doc == "10-K":
        return "FY"
    return fiscal_period_from_quarter(fiscal_quarter)


def make_chunk_id(
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    doc_type: DocType,
    index: int,
) -> str:
    """Build a stable per-document chunk id (not a global FAISS id)."""
    # e.g. AAPL_2025_Q4_transcript_chunk_12  or  AAPL_2025_FY_10-K_chunk_3
    return f"{ticker.upper()}_{fiscal_year}_{fiscal_period}_{doc_type}_chunk_{index}"


class FinancialClaim(BaseModel):
    """One testable financial assertion from an earnings call."""

    claim: str
    speaker: str


class TranscriptClaimsList(BaseModel):
    """Up to 10 atomic claims extracted from executive prepared remarks."""

    claims: list[FinancialClaim] = Field(max_length=10)


class SavedTranscriptClaims(BaseModel):
    """Persisted, repeatable claim set for one company-period."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int
    claims: list[FinancialClaim] = Field(max_length=10)
    llm_model_used: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


Classification = Literal["Consistent", "Contradictory", "Unverifiable"]


class ContradictionFinding(BaseModel):
    """NLI result comparing a transcript claim to retrieved filing passages."""

    transcript_claim: str
    source_speaker: str
    retrieved_filing_passages: list[str]
    source_sections: list[str]
    classification: Classification
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class PipelineReport(BaseModel):
    """Full structured output for one company-period analysis run."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int
    findings: list[ContradictionFinding]
    llm_model_used: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
