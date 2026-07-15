from src.intervals import merge_intervals


def test_merges_overlaps_and_touching_ranges() -> None:
    assert merge_intervals([(5, 8), (1, 3), (3, 6)]) == [(1, 8)]


def test_keeps_disjoint_ranges_sorted() -> None:
    assert merge_intervals([(10, 12), (1, 2), (5, 6)]) == [
        (1, 2),
        (5, 6),
        (10, 12),
    ]


def test_normalizes_reversed_bounds() -> None:
    assert merge_intervals([(8, 5), (6, 9)]) == [(5, 9)]
