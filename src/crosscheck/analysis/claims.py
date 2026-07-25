"""Extract and persist fixed claim sets from full earnings-call transcripts."""

from __future__ import annotations

import json

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import CLAIM_EXTRACTION_SYSTEM, claim_extraction_user
from crosscheck.config import claims_path
from crosscheck.models import DocumentMeta, SavedTranscriptClaims, TranscriptClaimsList


def extract_claims(
    company_name: str,
    transcript_text: str,
    *,
    max_claims: int = 10,
) -> tuple[TranscriptClaimsList, str]:
    """Extract up to ``max_claims`` financial claims from a full transcript."""
    if not 1 <= max_claims <= 10:
        raise ValueError("max_claims must be between 1 and 10")
    if not transcript_text.strip():
        return TranscriptClaimsList(claims=[]), "none"

    messages = [
        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": claim_extraction_user(
                company_name, transcript_text, max_claims
            ),
        },
    ]
    result, model = complete_structured(
        response_model=TranscriptClaimsList,
        messages=messages,
    )
    result.claims = result.claims[:max_claims]
    print(
        f"  [claims] parsed {len(result.claims)} claim(s) from {model}",
        flush=True,
    )
    for i, claim in enumerate(result.claims, start=1):
        preview = claim.claim[:90] + ("…" if len(claim.claim) > 90 else "")
        print(f"  [claims]   {i}. [{claim.speaker}] {preview}", flush=True)
    return result, model


def save_claims(
    period: DocumentMeta,
    claims: TranscriptClaimsList,
    *,
    model_used: str,
) -> SavedTranscriptClaims:
    """Write a fixed claim set for one company-period and return it."""
    saved = SavedTranscriptClaims(
        ticker=period.ticker,
        company_name=period.company_name,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        claims=claims.claims,
        llm_model_used=model_used,
    )
    path = claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(saved.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return saved


def load_saved_claims(period: DocumentMeta) -> SavedTranscriptClaims:
    """Load the required fixed claim set for one company-period."""
    path = claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Saved claims missing: {path}. Run: "
            f"python scripts/extract_claims.py --ticker {period.ticker}"
        )
    return SavedTranscriptClaims.model_validate_json(
        path.read_text(encoding="utf-8")
    )
