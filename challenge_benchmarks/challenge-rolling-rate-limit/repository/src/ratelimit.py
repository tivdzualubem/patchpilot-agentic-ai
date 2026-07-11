"""Rolling-window rate limiting."""

from __future__ import annotations

from collections import deque


def admit_events(
    timestamps: list[int],
    *,
    limit: int,
    window: int,
) -> list[bool]:
    """Return admission decisions for monotonic integer timestamps."""
    if limit < 0 or window < 0:
        raise ValueError("limit and window must be positive")

    accepted: deque[int] = deque()
    decisions: list[bool] = []

    for timestamp in timestamps:
        while accepted and accepted[0] < timestamp - window:
            accepted.popleft()

        allowed = len(accepted) <= limit
        decisions.append(allowed)
        accepted.append(timestamp)

    return decisions
