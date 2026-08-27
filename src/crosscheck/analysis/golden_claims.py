"""Build golden-eval claim sets: Consistent / Contradictory / Unverifiable."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from crosscheck.analysis.llm import complete_structured
from crosscheck.analysis.prompts import (
    UNVERIFIABLE_REWRITE_SYSTEM,
    unverifiable_rewrite_user,
)
from crosscheck.config import eval_claims_path
from crosscheck.io.jsonl import iter_json_objects, write_json_objects
from crosscheck.models import (
    ClaimRewrite,
    DocumentMeta,
    FinancialClaim,
    SavedTranscriptClaims,
    TranscriptClaimsList,
    make_claim_id,
)

# Prefer absolute money / EPS / margin figures over trailing YoY growth %.
_NUMBER_RE = re.compile(
    r"""
    (?P<head>\$\s*)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)
    (?P<unit>\s*(?:trillion|billion|million|bn|mm|B|M)\b)?
    (?P<pct>\s*%)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Factors clearly outside ordinary rounding (±0.5% relative / ±1¢ EPS).
_CORRUPT_FACTORS = (0.88, 0.92, 1.08, 1.12)


def _seed_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _norm_claim_text(text: str) -> str:
    """Normalize claim text for duplicate detection."""
    return " ".join((text or "").lower().split())


def _format_number(value: float, original: str) -> str:
    """Format ``value`` to roughly match the original numeric token style."""
    raw = original.replace(",", "")
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
        text = f"{value:.{decimals}f}"
    else:
        text = str(int(round(value)))
    if "," in original and abs(value) >= 1000:
        # Re-insert thousands separators for integer-looking originals.
        if "." not in text:
            return f"{int(text):,}"
        whole, frac = text.split(".", 1)
        return f"{int(whole):,}.{frac}"
    return text


def corrupt_financial_number(claim_text: str, *, seed: str = "") -> str:
    """Slightly alter one financial figure so the claim conflicts with filings.

    Picks a high-priority numeric span (currency / large units / % margins)
    and scales it by a seeded factor outside ordinary rounding. Raises
    ``ValueError`` if no suitable number is found.
    """
    matches = list(_NUMBER_RE.finditer(claim_text))
    if not matches:
        raise ValueError(f"no financial number to corrupt: {claim_text!r}")

    scored: list[tuple[int, re.Match[str]]] = []
    for m in matches:
        score = 0
        if m.group("head"):
            score += 40
        if m.group("unit"):
            score += 30
        if m.group("pct"):
            # Prefer standalone margins over trailing "up 4%" growth when tied.
            after = claim_text[m.end() : m.end() + 40].lower()
            before = claim_text[max(0, m.start() - 24) : m.start()].lower()
            if any(
                tok in before or tok in after
                for tok in ("up ", "down ", "higher", "increase", "grew")
            ):
                score += 5
            else:
                score += 25
        # Prefer earlier headline figures when scores tie.
        scored.append((score, m))

    scored.sort(key=lambda item: (-item[0], item[1].start()))
    target = scored[0][1]
    num_raw = target.group("num")
    value = float(num_raw.replace(",", ""))
    if value == 0:
        raise ValueError(f"cannot corrupt zero in claim: {claim_text!r}")

    factor = _CORRUPT_FACTORS[_seed_int(seed, claim_text, num_raw) % len(_CORRUPT_FACTORS)]
    new_value = value * factor
    # Keep EPS-like small absolutes visibly wrong (at least ~$0.05 / 0.1pp).
    if abs(new_value - value) < 0.05 and not target.group("unit"):
        new_value = value + (0.12 if factor > 1 else -0.12)
    if target.group("pct") and abs(new_value - value) < 0.3:
        new_value = value + (1.2 if factor > 1 else -1.2)

    new_num = _format_number(new_value, num_raw)
    return claim_text[: target.start("num")] + new_num + claim_text[target.end("num") :]


def rewrite_unverifiable_claim(
    *,
    company_name: str,
    base: FinancialClaim,
    transcript_text: str,
    forbidden_claims: list[str] | None = None,
) -> tuple[FinancialClaim, str]:
    """LLM-rewrite ``base`` into an Unverifiable golden claim; return claim + model."""
    messages = [
        {"role": "system", "content": UNVERIFIABLE_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": unverifiable_rewrite_user(
                company_name=company_name,
                base_claim=base.claim,
                speaker=base.speaker,
                transcript_text=transcript_text,
                forbidden_claims=forbidden_claims,
            ),
        },
    ]
    result, model = complete_structured(
        response_model=ClaimRewrite,
        messages=messages,
    )
    rewritten = FinancialClaim(
        claim=result.claim.strip(),
        speaker=(result.speaker or base.speaker).strip(),
        intended_label="Unverifiable",
    )
    return rewritten, model


def _slot_label(index: int) -> str:
    """Map 0-based claim index to intended label under the 4/2/2 layout."""
    if index < 4:
        return "Consistent"
    if index < 6:
        return "Contradictory"
    return "Unverifiable"


def _label_claim(
    claim: FinancialClaim,
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
    index: int,
    is_golden_claim: bool = False,
) -> FinancialClaim:
    return claim.model_copy(
        update={
            "claim_id": make_claim_id(ticker, fiscal_year, fiscal_quarter, index),
            "intended_label": claim.intended_label or _slot_label(index - 1),
            "is_golden_claim": is_golden_claim,
        }
    )


def build_golden_claim_set(
    source: SavedTranscriptClaims,
    *,
    transcript_text: str,
) -> tuple[TranscriptClaimsList, str]:
    """Return 8 labeled eval claims from a ≥6-claim extracted set (create mode).

    Layout (1-based positions):
    - claim 1–4 → Consistent (copied)
    - claim 5–6 → Contradictory (financial number corrupted)
    - claim 7–8 → Unverifiable (LLM + transcript rewrite; transcript-only focus)

    All claims start with ``is_golden_claim=False``.
    """
    if len(source.claims) < 6:
        raise ValueError(
            f"{source.ticker} FY{source.fiscal_year} {source.fiscal_quarter}: "
            f"need ≥6 extracted claims, found {len(source.claims)}. "
            "Re-run: python scripts/extract_claims.py --n 6"
        )

    out: list[FinancialClaim] = []
    # Claims 1–4: Consistent copies.
    for claim in source.claims[:4]:
        out.append(
            FinancialClaim(
                claim=claim.claim,
                speaker=claim.speaker,
                intended_label="Consistent",
            )
        )

    seed_base = f"{source.ticker}|{source.fiscal_year}|{source.fiscal_quarter}"
    # Claims 5–6: Contradictory by corrupting financial numbers.
    for i, claim in enumerate(source.claims[4:6]):
        corrupted = corrupt_financial_number(
            claim.claim,
            seed=f"{seed_base}|contradictory|{i}|{claim.claim}",
        )
        out.append(
            FinancialClaim(
                claim=corrupted,
                speaker=claim.speaker,
                intended_label="Contradictory",
            )
        )

    # Claims 7–8: Unverifiable rewrites, avoiding duplicates with earlier claims.
    forbidden_texts = [c.claim for c in out]
    unverifiable_1, model = rewrite_unverifiable_claim(
        company_name=source.company_name,
        base=source.claims[4],
        transcript_text=transcript_text,
        forbidden_claims=forbidden_texts,
    )
    out.append(unverifiable_1)

    forbidden_texts = [c.claim for c in out]
    unverifiable_2, _ = rewrite_unverifiable_claim(
        company_name=source.company_name,
        base=source.claims[5],
        transcript_text=transcript_text,
        forbidden_claims=forbidden_texts,
    )
    out.append(unverifiable_2)

    labeled: list[FinancialClaim] = []
    for i, claim in enumerate(out, start=1):
        labeled.append(
            _label_claim(
                claim,
                ticker=source.ticker,
                fiscal_year=source.fiscal_year,
                fiscal_quarter=source.fiscal_quarter,
                index=i,
                is_golden_claim=False,
            )
        )
    return TranscriptClaimsList(claims=labeled), model


def _pick_consistent_from_source(
    *,
    source: SavedTranscriptClaims,
    forbidden: set[str],
    preferred_index: int,
) -> FinancialClaim:
    """Copy a source claim for a Consistent slot, avoiding forbidden texts."""
    order = list(range(len(source.claims)))
    if preferred_index in order:
        order.remove(preferred_index)
        order.insert(0, preferred_index)
    for idx in order:
        claim = source.claims[idx]
        if _norm_claim_text(claim.claim) in forbidden:
            continue
        return FinancialClaim(
            claim=claim.claim,
            speaker=claim.speaker,
            intended_label="Consistent",
        )
    raise ValueError(
        f"{source.ticker} FY{source.fiscal_year} {source.fiscal_quarter}: "
        "no non-duplicate source claim available for Consistent rewrite"
    )


def _make_contradictory(
    *,
    source: SavedTranscriptClaims,
    preferred_index: int,
    forbidden: set[str],
    seed_base: str,
) -> FinancialClaim:
    """Corrupt a source claim into a Contradictory claim not in ``forbidden``."""
    candidates = []
    if preferred_index < len(source.claims):
        candidates.append(source.claims[preferred_index])
    candidates.extend(
        c for i, c in enumerate(source.claims) if i != preferred_index
    )
    last_err: Exception | None = None
    for attempt, base in enumerate(candidates):
        for twist in range(len(_CORRUPT_FACTORS) * 2):
            try:
                corrupted = corrupt_financial_number(
                    base.claim,
                    seed=f"{seed_base}|{preferred_index}|{attempt}|{twist}|{base.claim}",
                )
            except ValueError as exc:
                last_err = exc
                break
            if _norm_claim_text(corrupted) in forbidden:
                continue
            if _norm_claim_text(corrupted) == _norm_claim_text(base.claim):
                continue
            return FinancialClaim(
                claim=corrupted,
                speaker=base.speaker,
                intended_label="Contradictory",
            )
    raise ValueError(
        f"{source.ticker} FY{source.fiscal_year} {source.fiscal_quarter}: "
        f"could not build non-duplicate Contradictory claim ({last_err})"
    )


def _make_unverifiable(
    *,
    source: SavedTranscriptClaims,
    preferred_index: int,
    transcript_text: str,
    forbidden_texts: list[str],
    forbidden_norms: set[str],
) -> tuple[FinancialClaim, str]:
    """LLM-rewrite into Unverifiable, retrying if it duplicates forbidden claims."""
    base = (
        source.claims[preferred_index]
        if preferred_index < len(source.claims)
        else source.claims[-1]
    )
    last: FinancialClaim | None = None
    model = "none"
    for attempt in range(3):
        rewritten, model = rewrite_unverifiable_claim(
            company_name=source.company_name,
            base=base,
            transcript_text=transcript_text,
            forbidden_claims=forbidden_texts,
        )
        last = rewritten
        if _norm_claim_text(rewritten.claim) not in forbidden_norms:
            return rewritten, model
        print(
            f"  [modify] unverifiable rewrite duplicated a golden claim; "
            f"retry {attempt + 1}/3 …",
            flush=True,
        )
    assert last is not None
    raise ValueError(
        f"{source.ticker} FY{source.fiscal_year} {source.fiscal_quarter}: "
        "unverifiable rewrite kept duplicating golden claims"
    )


def modify_eval_claim_set(
    eval_claims: SavedTranscriptClaims,
    source: SavedTranscriptClaims,
    *,
    transcript_text: str,
) -> tuple[TranscriptClaimsList, str]:
    """Rewrite only ``is_golden_claim=False`` slots; keep golden claims intact.

    Slot rules match create mode (by index): Consistent / Contradictory /
    Unverifiable. Rewrites must not duplicate any ``is_golden_claim=True`` claim.
    """
    if len(eval_claims.claims) < 1:
        raise ValueError("eval claims file is empty")
    if len(source.claims) < 6:
        raise ValueError(
            f"{source.ticker} FY{source.fiscal_year} {source.fiscal_quarter}: "
            f"need ≥6 source claims for modify, found {len(source.claims)}"
        )

    golden_texts = [
        c.claim for c in eval_claims.claims if c.is_golden_claim is True
    ]
    forbidden = {_norm_claim_text(t) for t in golden_texts}
    seed_base = (
        f"modify|{eval_claims.ticker}|{eval_claims.fiscal_year}|"
        f"{eval_claims.fiscal_quarter}"
    )

    out: list[FinancialClaim] = []
    model_used = eval_claims.llm_model_used or "none"
    rewrote = 0

    for i, existing in enumerate(eval_claims.claims):
        if existing.is_golden_claim is True:
            out.append(existing)
            continue

        label = _slot_label(i)
        if label == "Consistent":
            rewritten = _pick_consistent_from_source(
                source=source,
                forbidden=forbidden,
                preferred_index=i,
            )
        elif label == "Contradictory":
            rewritten = _make_contradictory(
                source=source,
                preferred_index=i,
                forbidden=forbidden,
                seed_base=seed_base,
            )
        else:
            rewritten, model_used = _make_unverifiable(
                source=source,
                preferred_index=i,
                transcript_text=transcript_text,
                forbidden_texts=golden_texts,
                forbidden_norms=forbidden,
            )

        claim_id = existing.claim_id or make_claim_id(
            eval_claims.ticker,
            eval_claims.fiscal_year,
            eval_claims.fiscal_quarter,
            i + 1,
        )
        final = rewritten.model_copy(
            update={
                "claim_id": claim_id,
                "intended_label": label,
                "is_golden_claim": False,
            }
        )
        # Guard against colliding with golden + already-written non-golden.
        if _norm_claim_text(final.claim) in forbidden:
            raise ValueError(
                f"rewrite for {claim_id} still duplicates a golden/kept claim"
            )
        forbidden.add(_norm_claim_text(final.claim))
        out.append(final)
        rewrote += 1
        print(
            f"  [modify] rewrote {claim_id} → {label}",
            flush=True,
        )

    if rewrote == 0:
        print("  [modify] nothing to rewrite (all claims are golden)", flush=True)

    return TranscriptClaimsList(claims=out), model_used


def save_eval_claims(
    period: DocumentMeta,
    claims: TranscriptClaimsList,
    *,
    model_used: str,
) -> tuple[SavedTranscriptClaims, Path]:
    """Persist golden claims under ``data/eval/claims/``.

    Each line is an eval claim object with period metadata. Does **not** write
    extract-only fields such as ``regenerate``.
    """
    saved = SavedTranscriptClaims(
        ticker=period.ticker,
        company_name=period.company_name,
        fiscal_year=period.fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        claims=claims.claims,
        llm_model_used=model_used,
    )
    path = eval_claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_objects(
        path,
        [
            {
                "claim": c.claim,
                "speaker": c.speaker,
                "claim_id": c.claim_id,
                "intended_label": c.intended_label,
                "is_golden_claim": c.is_golden_claim,
                "ticker": period.ticker,
                "company_name": period.company_name,
                "fiscal_year": period.fiscal_year,
                "fiscal_quarter": period.fiscal_quarter,
            }
            for c in claims.claims
        ],
    )
    return saved, path


def load_eval_claims(period: DocumentMeta) -> SavedTranscriptClaims:
    """Load labeled golden claims from ``data/eval/claims/``."""
    path = eval_claims_path(
        period.ticker,
        period.fiscal_year,
        period.fiscal_quarter,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Eval claims missing: {path}. Run: "
            f"python scripts/eval/get_eval_candidates.py --mode create "
            f"--ticker {period.ticker}"
        )

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Empty eval claims file: {path}")

    # Backward-compat: single JSON object wrapping a ``claims`` array.
    try:
        return SavedTranscriptClaims.model_validate_json(raw)
    except Exception:
        pass

    rows = list(iter_json_objects(path))
    claims = [FinancialClaim.model_validate(obj) for obj in rows]

    if not claims:
        raise ValueError(f"No claims found in eval claims file: {path}")

    first = rows[0]
    company_name = str(first.get("company_name") or period.company_name or period.ticker)
    ticker = str(first.get("ticker") or period.ticker).upper()
    fiscal_year = int(first.get("fiscal_year") or period.fiscal_year)
    fiscal_quarter = str(first.get("fiscal_quarter") or period.fiscal_quarter)

    return SavedTranscriptClaims(
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        claims=claims,
        llm_model_used="jsonl",
    )
