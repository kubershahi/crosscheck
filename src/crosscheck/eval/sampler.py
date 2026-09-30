"""Golden-set sampling for cost-controlled eval runs."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from crosscheck.models import GoldenClaim

MINI_PER_LABEL = {
    "Consistent": 2,
    "Contradictory": 1,
    "Unverifiable": 1,
}


def _seed_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def sample_mini_golden(
    claims: list[GoldenClaim],
    *,
    seed: str = "",
) -> list[GoldenClaim]:
    """Return up to 2C + 1X + 1U from one period, stable across runs."""
    by_label: dict[str, list[GoldenClaim]] = defaultdict(list)
    for claim in claims:
        by_label[str(claim.expected_nli_label)].append(claim)

    picked: list[GoldenClaim] = []
    for label, quota in MINI_PER_LABEL.items():
        pool = sorted(by_label.get(label, []), key=lambda c: c.claim_id)
        if not pool:
            continue
        start = _seed_int(seed, label) % len(pool)
        rotated = pool[start:] + pool[:start]
        picked.extend(rotated[:quota])
    return sorted(picked, key=lambda c: c.claim_id)


def sample_all_periods(
    period_claims: dict[str, list[GoldenClaim]],
    *,
    seed: str = "",
    full: bool = False,
) -> list[GoldenClaim]:
    """Flatten sampled (or full) claims across periods."""
    out: list[GoldenClaim] = []
    for period_key in sorted(period_claims):
        claims = period_claims[period_key]
        if full:
            out.extend(sorted(claims, key=lambda c: c.claim_id))
        else:
            out.extend(sample_mini_golden(claims, seed=f"{seed}|{period_key}"))
    return out
