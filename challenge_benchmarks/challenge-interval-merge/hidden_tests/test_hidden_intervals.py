from src.intervals import merge_intervals


def test_nested_intervals_keep_outer_start() -> None:
    assert merge_intervals([(1, 10), (3, 4), (5, 7)]) == [(1, 10)]


def test_touching_singletons_merge() -> None:
    assert merge_intervals([(2, 2), (2, 5), (5, 5)]) == [(2, 5)]


def test_empty_input() -> None:
    assert merge_intervals([]) == []
