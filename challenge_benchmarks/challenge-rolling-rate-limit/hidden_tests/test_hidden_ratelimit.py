import pytest

from src.ratelimit import admit_events


def test_zero_limit_is_invalid() -> None:
    with pytest.raises(ValueError):
        admit_events([1], limit=0, window=10)


def test_multiple_events_at_same_timestamp() -> None:
    assert admit_events([3, 3, 3], limit=2, window=5) == [
        True,
        True,
        False,
    ]


def test_capacity_recovers_after_window() -> None:
    assert admit_events([1, 2, 3, 6, 7], limit=2, window=5) == [
        True,
        True,
        False,
        True,
        True,
    ]
