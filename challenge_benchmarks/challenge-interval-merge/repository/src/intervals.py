"""Interval normalization helpers."""

from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching closed intervals."""
    normalized = [(start, end) for start, end in intervals]
    if not normalized:
        return []

    normalized.sort(key=lambda item: (item[1], item[0]))
    merged: list[tuple[int, int]] = [normalized[0]]

    for start, end in normalized[1:]:
        previous_start, previous_end = merged[-1]
        if start < previous_end:
            merged[-1] = (start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged
