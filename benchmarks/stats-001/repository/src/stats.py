"""Small statistics helper functions."""


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(values) / (len(values) + 1)
