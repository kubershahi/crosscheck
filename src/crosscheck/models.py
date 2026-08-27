"""Shared Pydantic models for documents, retrievable chunks, and NLI output."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocType = Literal["10-K", "10-Q", "transcript"]
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]
FiscalQuarter = Literal["Q1", "Q2", "Q3", "Q4"]

_QUARTER_RE = re.compile(r"^Q?([1-4])$", re.IGNORECASE)


def quarter_number(value: object) -> int:
    """Normalize ``1`` / ``\"1\"`` / ``\"Q1\"`` to int 1–4."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid fiscal quarter: {value!r}")
    if isinstance(value, int):
        if 1 <= value <= 4:
            return value
        raise ValueError(f"fiscal_quarter must be 1–4, got {value}")
    text = str(value).strip().upper()
    match = _QUARTER_RE.match(text)
    if not match:
        raise ValueError(f"Invalid fiscal quarter: {value!r}; use Q1–Q4 or 1–4")
    return int(match.group(1))


def as_fiscal_quarter(value: object) -> FiscalQuarter:
    """Normalize any quarter input to stored form ``Q1``…``Q4``."""
    return f"Q{quarter_number(value)}"  # type: ignore[return-value]


def make_claim_id(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int | str,
    index: int,
) -> str:
    """Stable claim id matching candidates.jsonl, e.g. ``AAPL_2025_Q1_01``."""
    if index < 1:
        raise ValueError("claim index must be 1-based")
    q = as_fiscal_quarter(fiscal_quarter)
    return f"{ticker.upper()}_{int(fiscal_year)}_{q}_{index:02d}"


class Chunk(BaseModel):
    """Stateless chunk unit — shared identity fields + corpus-specific optionals.

    JSONL field order (``model_dump`` / ``model_dump_json``)::

        1. Common identity (always present)
        2. Corpus-specific optionals (omitted when null via ``exclude_none``)
        3. ``text`` last

    Common::

        chunk_id, ticker, company_name, doc_type, fiscal_year, fiscal_period,
        is_table[, section]

    ``section`` is set for filings (Item / MD&A); omitted for transcripts.

    Filing-only optionals::

        filing_date, report_date, quarter_period_label, quarter_months,
        subsection, subsubsection

    Transcript-only optionals::

        speaker_name, speaker_role, call_date
    """

    # Common identity (front of each JSONL row)
    chunk_id: str
    ticker: str
    company_name: str | None = None
    doc_type: DocType
    fiscal_year: int
    fiscal_period: FiscalPeriod
    is_table: bool = False
    section: str | None = None

    # Filing sidecar (from EDGAR .meta.json)
    filing_date: str | None = None
    report_date: str | None = None
    quarter_period_label: str | None = None
    quarter_months: list[str] | None = None

    # Sticky subsection labels (filings). Also prepended into prose ``text``;
    # tables carry titles/captions in ``text`` via the table preface path.
    subsection: str | None = None
    subsubsection: str | None = None

    # Transcript fields
    speaker_name: str | None = None
    speaker_role: str | None = None
    call_date: str | None = None

    # Body last so metadata is easy to scan in JSONL
    text: str

    def metadata_dict(self) -> dict:
        """Return all fields except ``text`` (handy for sanity printing)."""
        return self.model_dump(exclude={"text"}, exclude_none=True)


class IndexedChunk(Chunk):
    """Merged master-corpus row with a sequential FAISS-aligned ``global_id``."""

    global_id: int = Field(ge=0)


class DocumentMeta(BaseModel):
    """Period identity + optional sidecar fields used while chunking."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: FiscalQuarter
    form: str | None = None  # 10-K / 10-Q for filings
    source_path: str | None = None

    # Filing sidecar
    filing_date: str | None = None
    report_date: str | None = None
    quarter_period_label: str | None = None
    quarter_months: list[str] | None = None

    # Transcript sidecar
    call_date: str | None = None

    @field_validator("fiscal_quarter", mode="before")
    @classmethod
    def _normalize_fiscal_quarter(cls, value: object) -> FiscalQuarter:
        return as_fiscal_quarter(value)

    @property
    def fiscal_quarter_num(self) -> int:
        """Integer 1–4 for path math / EDGAR helpers."""
        return quarter_number(self.fiscal_quarter)


def fiscal_period_from_quarter(quarter: object) -> FiscalPeriod:
    """Map fiscal quarter 1–4 / ``Q1``…``Q4`` to ``Q1``…``Q4``."""
    return as_fiscal_quarter(quarter)


def filing_doc_type(form: str | None, fiscal_quarter: object) -> DocType:
    """Resolve SEC form to ``10-K`` or ``10-Q``."""
    if form:
        form = form.strip().upper()
        if form in {"10-K", "10-Q"}:
            return form  # type: ignore[return-value]
    return "10-K" if quarter_number(fiscal_quarter) == 4 else "10-Q"


def filing_fiscal_period(form: str | None, fiscal_quarter: object) -> FiscalPeriod:
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


Classification = Literal["Consistent", "Contradictory", "Unverifiable"]


class FinancialClaim(BaseModel):
    """One testable financial assertion from an earnings call."""

    claim: str
    speaker: str
    claim_id: str | None = Field(
        default=None,
        description="Stable id e.g. AAPL_2025_Q1_01; set on extract/eval/golden claims.",
    )
    intended_label: Classification | None = Field(
        default=None,
        description=(
            "Optional golden-set intent: Consistent / Contradictory / Unverifiable. "
            "Unset for ordinary extract_claims output."
        ),
    )
    is_golden_claim: bool | None = Field(
        default=None,
        description=(
            "True once promoted into the curated golden set; "
            "False for eval/candidate claims; unset for ordinary extracts."
        ),
    )
    regenerate: bool = Field(
        default=False,
        description=(
            "When true, ``extract_claims.py --mode modify`` regenerates this claim "
            "without duplicating other claims in the same period file."
        ),
    )


class ClaimRewrite(BaseModel):
    """LLM rewrite of a single claim (text + speaker only)."""

    claim: str
    speaker: str


class GroundTruthReference(BaseModel):
    """Short categorical filing reference for a golden claim."""

    ground_truth_reference: str = Field(
        description=(
            "One-line categorical reference to the filing table/section/line "
            "that verifies or contradicts the claim, including key figures."
        ),
    )


class GoldenClaim(BaseModel):
    """Curated golden-set claim promoted from matching candidates."""

    claim_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: FiscalQuarter
    speaker: str
    claim: str
    expected_nli_label: Classification
    is_in_filing: bool
    is_in_table: bool
    ground_truth_reference: str

    @field_validator("fiscal_quarter", mode="before")
    @classmethod
    def _normalize_fiscal_quarter(cls, value: object) -> FiscalQuarter:
        return as_fiscal_quarter(value)


class TranscriptClaimsList(BaseModel):
    """Up to 10 atomic claims extracted from a full earnings-call transcript."""

    claims: list[FinancialClaim] = Field(max_length=10)


class SavedTranscriptClaims(BaseModel):
    """Persisted, repeatable claim set for one company-period."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: FiscalQuarter
    claims: list[FinancialClaim] = Field(max_length=10)
    llm_model_used: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @field_validator("fiscal_quarter", mode="before")
    @classmethod
    def _normalize_fiscal_quarter(cls, value: object) -> FiscalQuarter:
        return as_fiscal_quarter(value)


class NLIJudgment(BaseModel):
    """LLM NLI decision. Field order matters for structured output: the model
    fills fields in schema order — reason over the passages, then label, then
    cite which passage the decision rested on.
    """

    reasoning: str = Field(
        description=(
            "Step-by-step comparison of the claim against the passages; "
            "write this before choosing the label."
        ),
    )
    classification: Classification = Field(
        description=(
            "Final label after reasoning: Consistent, Contradictory, or "
            "Unverifiable. Must match the reasoning above."
        ),
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    matched_passage_index: int = Field(
        ge=0,
        le=20,
        description=(
            "1-based index of the filing passage that best supports the "
            "classification above (the evidence cited in reasoning); "
            "0 if none of the passages are usable."
        ),
    )


class ContradictionFinding(BaseModel):
    """NLI result comparing a transcript claim to retrieved filing passages."""

    transcript_claim: str
    source_speaker: str
    retrieved_filing_passages: list[str]
    chunk_ids: list[str] = Field(default_factory=list)
    global_ids: list[int] = Field(default_factory=list)
    classification: Classification
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class PipelineReport(BaseModel):
    """Full structured output for one company-period analysis run."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: FiscalQuarter
    findings: list[ContradictionFinding]
    llm_model_used: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @field_validator("fiscal_quarter", mode="before")
    @classmethod
    def _normalize_fiscal_quarter(cls, value: object) -> FiscalQuarter:
        return as_fiscal_quarter(value)
