"""Extract and persist fixed claim sets from full earnings-call transcripts."""

from __future__ import annotations

import re
from pathlib import Path

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_REGENERATE_SYSTEM,
    claim_extraction_user,
    claim_regenerate_user,
)
from crosscheck.config import claims_path
from crosscheck.io.jsonl import iter_json_objects, write_json_objects
from crosscheck.models import (
    ClaimRewrite,
    DocumentMeta,
    FinancialClaim,
    SavedTranscriptClaims,
    TranscriptClaimsList,
    make_claim_id,
)

_WS_RE = re.compile(r"\s+")


def _norm_claim_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def extract_claims(
    company_name: str,
    transcript_text: str,
    *,
    max_claims: int = 10,
) -> tuple[TranscriptClaimsList, str]:
    """Extract up to ``max_claims`` financial claims from a full transcript."""
    if not 1 <= max_claims <= 15:
        raise ValueError("max_claims must be between 1 and 15")
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


def regenerate_one_claim(
    *,
    company_name: str,
    transcript_text: str,
    forbidden_claims: list[str],
    previous_claim: str | None = None,
    max_attempts: int = 3,
) -> tuple[FinancialClaim, str]:
    """LLM-extract one claim that does not duplicate ``forbidden_claims``."""
    if not transcript_text.strip():
        raise ValueError("empty transcript text")

    forbidden_norms = {_norm_claim_text(t) for t in forbidden_claims if t.strip()}
    last: FinancialClaim | None = None
    model = "none"
    for attempt in range(max_attempts):
        messages = [
            {"role": "system", "content": CLAIM_REGENERATE_SYSTEM},
            {
                "role": "user",
                "content": claim_regenerate_user(
                    company_name=company_name,
                    transcript_text=transcript_text,
                    forbidden_claims=forbidden_claims,
                    previous_claim=previous_claim,
                ),
            },
        ]
        rewritten, model = complete_structured(
            response_model=ClaimRewrite,
            messages=messages,
        )
        last = FinancialClaim(
            claim=rewritten.claim.strip(),
            speaker=rewritten.speaker.strip(),
            regenerate=False,
        )
        if _norm_claim_text(last.claim) not in forbidden_norms:
            return last, model
        print(
            f"  [claims] regenerate duplicated an existing claim; "
            f"retry {attempt + 1}/{max_attempts} …",
            flush=True,
        )
    assert last is not None
    raise ValueError("regenerated claim kept duplicating existing claims")


def assign_claim_ids(
    period: DocumentMeta,
    claims: list[FinancialClaim],
) -> list[FinancialClaim]:
    """Stamp stable ``claim_id`` values (preserving other fields)."""
    out: list[FinancialClaim] = []
    for i, claim in enumerate(claims, start=1):
        out.append(
            claim.model_copy(
                update={
                    "claim_id": make_claim_id(
                        period.ticker,
                        period.fiscal_year,
                        period.fiscal_quarter,
                        i,
                    ),
                }
            )
        )
    return out


def save_claims(
    period: DocumentMeta,
    claims: TranscriptClaimsList,
    *,
    model_used: str,
) -> tuple[SavedTranscriptClaims, Path]:
    """Write a flattened JSONL claim set (one claim object per line)."""
    stamped = assign_claim_ids(period, list(claims.claims))
    saved = SavedTranscriptClaims(
        ticker=period.ticker,
        company_name=period.company_name,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        claims=stamped,
        llm_model_used=model_used,
    )
    path = claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop legacy nested JSON if present so discovery doesn't see both.
    legacy = path.with_suffix(".json")
    if legacy.exists() and legacy != path:
        legacy.unlink()

    write_json_objects(
        path,
        [
            {
                "claim": c.claim,
                "speaker": c.speaker,
                "claim_id": c.claim_id,
                "regenerate": bool(c.regenerate),
                "ticker": period.ticker,
                "company_name": period.company_name,
                "fiscal_year": period.fiscal_year,
                "fiscal_quarter": period.fiscal_quarter,
            }
            for c in stamped
        ],
    )
    return saved, path


def _legacy_json_path(period: DocumentMeta) -> Path:
    return claims_path(
        period.ticker, period.fiscal_year, period.fiscal_quarter
    ).with_suffix(".json")


def load_saved_claims(period: DocumentMeta) -> SavedTranscriptClaims:
    """Load the fixed claim set for one company-period (JSONL or legacy JSON)."""
    path = claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    legacy = _legacy_json_path(period)

    if path.exists():
        rows = list(iter_json_objects(path))
        if not rows:
            raise ValueError(f"Empty claims file: {path}")
        claims = [FinancialClaim.model_validate(obj) for obj in rows]
        first = rows[0]
        company_name = str(first.get("company_name") or period.company_name or period.ticker)
        ticker = str(first.get("ticker") or period.ticker).upper()
        fiscal_year = int(first.get("fiscal_year") or period.fiscal_year)
        fiscal_quarter = str(first.get("fiscal_quarter") or period.fiscal_quarter)
        claims = assign_claim_ids(
            DocumentMeta(
                ticker=ticker,
                company_name=company_name,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
            ),
            claims,
        )
        return SavedTranscriptClaims(
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            claims=claims,
            llm_model_used="jsonl",
        )

    if legacy.exists():
        saved = SavedTranscriptClaims.model_validate_json(
            legacy.read_text(encoding="utf-8")
        )
        stamped = assign_claim_ids(
            DocumentMeta(
                ticker=saved.ticker,
                company_name=saved.company_name,
                fiscal_year=saved.fiscal_year,
                fiscal_quarter=saved.fiscal_quarter,
            ),
            list(saved.claims),
        )
        return saved.model_copy(update={"claims": stamped})

    raise FileNotFoundError(
        f"Saved claims missing: {path}. Run: "
        f"python scripts/extract_claims.py --ticker {period.ticker}"
    )


def modify_saved_claims(
    saved: SavedTranscriptClaims,
    *,
    transcript_text: str,
) -> tuple[TranscriptClaimsList, str]:
    """Regenerate only claims with ``regenerate=true``; keep others intact.

    New claims must not duplicate any kept (or already-regenerated) claim text.
    Successful regenerations are written back with ``regenerate=false``.
    """
    if not saved.claims:
        raise ValueError("claims file is empty")

    pending = [c for c in saved.claims if c.regenerate]
    if not pending:
        return TranscriptClaimsList(claims=list(saved.claims)), saved.llm_model_used

    out: list[FinancialClaim] = []
    model_used = saved.llm_model_used or "none"
    kept_texts = [c.claim for c in saved.claims if not c.regenerate]
    forbidden = list(kept_texts)

    for claim in saved.claims:
        if not claim.regenerate:
            out.append(claim)
            continue

        cid = claim.claim_id or "?"
        print(f"  [claims] regenerating {cid} …", flush=True)
        rewritten, model_used = regenerate_one_claim(
            company_name=saved.company_name,
            transcript_text=transcript_text,
            forbidden_claims=forbidden,
            previous_claim=claim.claim,
        )
        stamped = rewritten.model_copy(
            update={
                "claim_id": claim.claim_id,
                "regenerate": False,
            }
        )
        preview = stamped.claim[:90] + ("…" if len(stamped.claim) > 90 else "")
        print(f"  [claims]   → [{stamped.speaker}] {preview}", flush=True)
        out.append(stamped)
        forbidden.append(stamped.claim)

    return TranscriptClaimsList(claims=out), model_used


def top_up_claims(
    saved: SavedTranscriptClaims,
    *,
    transcript_text: str,
    target_count: int,
) -> tuple[TranscriptClaimsList, str]:
    """Append new claims until ``target_count``, avoiding duplicates of existing ones."""
    if not 1 <= target_count <= 15:
        raise ValueError("target_count must be between 1 and 15")
    existing = list(saved.claims)
    if len(existing) >= target_count:
        return TranscriptClaimsList(claims=existing[:target_count]), saved.llm_model_used

    need = target_count - len(existing)
    forbidden = [c.claim for c in existing]
    model_used = saved.llm_model_used or "none"
    out = list(existing)

    print(
        f"  [claims] top-up: have {len(existing)}, need {need} more "
        f"(target={target_count})",
        flush=True,
    )
    for i in range(need):
        print(f"  [claims] extracting additional claim {i + 1}/{need} …", flush=True)
        rewritten, model_used = regenerate_one_claim(
            company_name=saved.company_name,
            transcript_text=transcript_text,
            forbidden_claims=forbidden,
        )
        out.append(rewritten)
        forbidden.append(rewritten.claim)
        preview = rewritten.claim[:90] + ("…" if len(rewritten.claim) > 90 else "")
        print(f"  [claims]   → [{rewritten.speaker}] {preview}", flush=True)

    return TranscriptClaimsList(claims=out), model_used
