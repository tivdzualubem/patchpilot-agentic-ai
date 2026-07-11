import pytest

from src.ratelimit import admit_events


def test_enforces_capacity() -> None:
    assert admit_events([0, 1, 2], limit=2, window=10) == [
        True,
        True,
        False,
    ]


def test_boundary_event_expires() -> None:
    assert admit_events([0, 9, 10], limit=2, window=10) == [
        True,
        True,
        True,
    ]


def test_rejected_event_does_not_consume_capacity() -> None:
    assert admit_events([0, 1, 2, 10], limit=2, window=10) == [
        True,
        True,
        False,
        True,
    ]


def test_rejects_out_of_order_timestamps() -> None:
    with pytest.raises(ValueError):
        admit_events([5, 4], limit=2, window=10)
