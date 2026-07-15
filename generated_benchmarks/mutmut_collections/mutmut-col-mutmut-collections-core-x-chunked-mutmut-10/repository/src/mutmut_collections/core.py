"""Deterministic collection helpers for mutation-based repair tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def first_non_none(values: Iterable[int | None]) -> int | None:
    """Return the first non-None value, or None when none exists."""
    for value in values:
        if value is not None:
            return value
    return None


def chunked(values: Sequence[int], size: int) -> list[list[int]]:
    """Split values into consecutive chunks of a positive size."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), None)]


def rotate_left(values: Sequence[int], steps: int) -> list[int]:
    """Return a left rotation without modifying the input."""
    if not values:
        return []
    offset = steps % len(values)
    return list(values[offset:]) + list(values[:offset])


def unique_stable(values: Iterable[int]) -> list[int]:
    """Remove duplicates while retaining first-seen order."""
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def flatten_once(groups: Iterable[Iterable[int]]) -> list[int]:
    """Flatten exactly one level of nested iterables."""
    result: list[int] = []
    for group in groups:
        result.extend(group)
    return result


def windowed(values: Sequence[int], size: int) -> list[tuple[int, ...]]:
    """Return all consecutive windows of a positive size."""
    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(values):
        return []
    return [
        tuple(values[index : index + size]) for index in range(len(values) - size + 1)
    ]


def partition_by_sign(values: Iterable[int]) -> tuple[list[int], list[int]]:
    """Partition values into negative and non-negative groups."""
    negative: list[int] = []
    non_negative: list[int] = []
    for value in values:
        target = negative if value < 0 else non_negative
        target.append(value)
    return negative, non_negative


def transpose(rows: Sequence[Sequence[int]]) -> list[list[int]]:
    """Transpose a non-ragged matrix."""
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("rows must have equal length")
    return [[row[column] for row in rows] for column in range(width)]


def cumulative_sum(values: Iterable[int]) -> list[int]:
    """Return running totals."""
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return result


def pairwise_differences(values: Sequence[int]) -> list[int]:
    """Return each value minus its predecessor."""
    return [values[index] - values[index - 1] for index in range(1, len(values))]


def interleave(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Alternate values and append the remainder of the longer side."""
    result: list[int] = []
    shared = min(len(left), len(right))
    for index in range(shared):
        result.extend((left[index], right[index]))
    result.extend(left[shared:])
    result.extend(right[shared:])
    return result


def frequency_map(values: Iterable[int]) -> dict[int, int]:
    """Count occurrences of each integer."""
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def mode(values: Sequence[int]) -> int:
    """Return the most frequent value, breaking ties by first appearance."""
    if not values:
        raise ValueError("mode requires at least one value")
    counts = frequency_map(values)
    return max(values, key=lambda value: (counts[value], -values.index(value)))


def take_until(values: Iterable[int], stop: int) -> list[int]:
    """Return values before the first stop value."""
    result: list[int] = []
    for value in values:
        if value == stop:
            break
        result.append(value)
    return result


def drop_every(values: Sequence[int], interval: int) -> list[int]:
    """Drop every interval-th item using one-based positions."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    return [
        value
        for position, value in enumerate(values, start=1)
        if position % interval != 0
    ]


def index_of_max(values: Sequence[int]) -> int:
    """Return the first index containing the maximum value."""
    if not values:
        raise ValueError("values cannot be empty")
    maximum = max(values)
    return values.index(maximum)


def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add counts for matching keys without mutating either input."""
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + value
    return result


def bounded_slice(
    values: Sequence[int],
    start: int,
    stop: int,
) -> list[int]:
    """Return a slice after clamping bounds into the sequence range."""
    lower = max(0, min(start, len(values)))
    upper = max(lower, min(stop, len(values)))
    return list(values[lower:upper])
