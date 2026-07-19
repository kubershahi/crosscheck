"""Shared Pydantic models for documents and retrievable chunks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit with dual-corpus metadata (filing or transcript)."""

    text: str
    doc_type: Literal["filing", "transcript"]
    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int
    chunk_index: int

    # Filing-specific
    section_header: str | None = None
    is_table: bool = False

    # Transcript-specific
    speaker_name: str | None = None
    speaker_title: str | None = None
    section_type: Literal["prepared_remarks", "qa", "unknown"] | None = None

    source_path: str | None = None

    def metadata_dict(self) -> dict:
        """Return all fields except ``text`` (handy for sanity printing)."""
        return self.model_dump(exclude={"text"})


class DocumentMeta(BaseModel):
    """Period identity attached to every chunk produced from a source document."""

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    form: str | None = None
    source_path: str | None = None
