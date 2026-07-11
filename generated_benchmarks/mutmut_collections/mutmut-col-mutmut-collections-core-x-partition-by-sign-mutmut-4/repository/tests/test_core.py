import pytest
from mutmut_collections.core import (
    bounded_slice,
    chunked,
    cumulative_sum,
    drop_every,
    first_non_none,
    flatten_once,
    frequency_map,
    index_of_max,
    interleave,
    merge_counts,
    mode,
    pairwise_differences,
    partition_by_sign,
    rotate_left,
    take_until,
    transpose,
    unique_stable,
    windowed,
)


def test_first_non_none() -> None:
    assert first_non_none([None, None, 4, 5]) == 4
    assert first_non_none([None]) is None


def test_chunked() -> None:
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        chunked([1], 0)


def test_rotate_left() -> None:
    assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate_left([1, 2, 3], 4) == [2, 3, 1]
    assert rotate_left([], 7) == []


def test_unique_and_flatten() -> None:
    assert unique_stable([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert flatten_once([[1, 2], [], [3]]) == [1, 2, 3]


def test_windowed() -> None:
    assert windowed([1, 2, 3, 4], 3) == [(1, 2, 3), (2, 3, 4)]
    assert windowed([1, 2], 3) == []
    with pytest.raises(ValueError):
        windowed([1], -1)


def test_partition_by_sign() -> None:
    assert partition_by_sign([-2, 0, 3, -1]) == ([-2, -1], [0, 3])


def test_transpose() -> None:
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose([]) == []
    with pytest.raises(ValueError):
        transpose([[1], [2, 3]])


def test_cumulative_sum() -> None:
    assert cumulative_sum([3, -1, 4]) == [3, 2, 6]
    assert cumulative_sum([]) == []


def test_pairwise_differences() -> None:
    assert pairwise_differences([4, 9, 7, 10]) == [5, -2, 3]
    assert pairwise_differences([1]) == []


def test_interleave() -> None:
    assert interleave([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5]
    assert interleave([], [7, 8]) == [7, 8]


def test_frequency_map() -> None:
    assert frequency_map([2, 1, 2, 3, 2, 1]) == {2: 3, 1: 2, 3: 1}


def test_mode() -> None:
    assert mode([4, 2, 4, 2]) == 4
    assert mode([7]) == 7
    with pytest.raises(ValueError):
        mode([])


def test_take_until() -> None:
    assert take_until([1, 2, 9, 3], 9) == [1, 2]
    assert take_until([1, 2], 9) == [1, 2]


def test_drop_every() -> None:
    assert drop_every([1, 2, 3, 4, 5, 6], 3) == [1, 2, 4, 5]
    with pytest.raises(ValueError):
        drop_every([1], 0)


def test_index_of_max() -> None:
    assert index_of_max([3, 9, 9, 2]) == 1
    with pytest.raises(ValueError):
        index_of_max([])


def test_merge_counts() -> None:
    left = {"a": 2, "b": 1}
    right = {"b": 4, "c": 3}
    assert merge_counts(left, right) == {"a": 2, "b": 5, "c": 3}
    assert left == {"a": 2, "b": 1}


def test_bounded_slice() -> None:
    assert bounded_slice([0, 1, 2, 3, 4], -2, 3) == [0, 1, 2]
    assert bounded_slice([0, 1, 2], 2, 20) == [2]
    assert bounded_slice([0, 1, 2], 3, 1) == []
