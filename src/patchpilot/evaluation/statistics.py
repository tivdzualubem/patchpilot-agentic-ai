"""Statistical analysis helpers for PatchPilot evaluations."""

from __future__ import annotations

from math import comb

from pydantic import BaseModel, ConfigDict, Field


class McNemarResult(BaseModel):
    """Exact McNemar result for paired binary outcomes."""

    model_config = ConfigDict(extra="forbid")

    both_success: int = Field(ge=0)
    full_only_success: int = Field(ge=0)
    baseline_only_success: int = Field(ge=0)
    both_failure: int = Field(ge=0)
    discordant: int = Field(ge=0)
    p_value: float = Field(ge=0, le=1)


def exact_mcnemar_test(
    *,
    full_success: list[bool],
    baseline_success: list[bool],
) -> McNemarResult:
    """Compute the exact two-sided McNemar test for paired binary outcomes."""
    if len(full_success) != len(baseline_success):
        raise ValueError("Paired outcome lists must have the same length.")

    both_success = 0
    full_only_success = 0
    baseline_only_success = 0
    both_failure = 0

    for full, baseline in zip(full_success, baseline_success, strict=True):
        if full and baseline:
            both_success += 1
        elif full and not baseline:
            full_only_success += 1
        elif baseline and not full:
            baseline_only_success += 1
        else:
            both_failure += 1

    discordant = full_only_success + baseline_only_success

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(full_only_success, baseline_only_success)
        lower_tail = sum(
            comb(discordant, k) * 0.5**discordant
            for k in range(smaller + 1)
        )
        p_value = min(1.0, 2 * lower_tail)

    return McNemarResult(
        both_success=both_success,
        full_only_success=full_only_success,
        baseline_only_success=baseline_only_success,
        both_failure=both_failure,
        discordant=discordant,
        p_value=p_value,
    )
