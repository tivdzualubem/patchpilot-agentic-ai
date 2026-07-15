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


def test_hidden_first_non_none_and_chunked() -> None:
    assert first_non_none([None, -3, 0]) == -3
    assert first_non_none([]) is None
    assert chunked([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]
    assert chunked([], 2) == []


def test_hidden_rotation_and_uniqueness() -> None:
    assert rotate_left([1, 2, 3, 4], -1) == [4, 1, 2, 3]
    assert rotate_left([9], 99) == [9]
    assert unique_stable([0, 0, -1, 0, 2, -1]) == [0, -1, 2]


def test_hidden_flatten_and_windows() -> None:
    assert flatten_once([[], [5], [6, 7], []]) == [5, 6, 7]
    assert windowed([1, 2, 3], 1) == [(1,), (2,), (3,)]
    assert windowed([], 1) == []


def test_hidden_partition_and_transpose() -> None:
    assert partition_by_sign([0, -1, -2, 4]) == ([-1, -2], [0, 4])
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]]
    assert transpose([[], []]) == []


def test_hidden_running_and_pairwise_math() -> None:
    assert cumulative_sum([-2, -3, 10]) == [-2, -5, 5]
    assert pairwise_differences([10, 7, 7, -1]) == [-3, 0, -8]


def test_hidden_interleave_and_frequency() -> None:
    assert interleave([1], [2, 3, 4]) == [1, 2, 3, 4]
    assert frequency_map([]) == {}
    assert frequency_map([-1, -1, 0]) == {-1: 2, 0: 1}


def test_hidden_mode_and_take_until() -> None:
    assert mode([5, 6, 6, 5, 7]) == 5
    assert take_until([8, 1, 2], 8) == []
    assert take_until([], 4) == []


def test_hidden_drop_and_maximum() -> None:
    assert drop_every([1, 2, 3, 4, 5], 1) == []
    assert drop_every([1, 2, 3], 10) == [1, 2, 3]
    assert index_of_max([-9, -2, -2, -8]) == 1


def test_hidden_merge_and_slice() -> None:
    assert merge_counts({}, {"x": 2}) == {"x": 2}
    assert merge_counts({"x": -1}, {"x": 1}) == {"x": 0}
    assert bounded_slice([0, 1, 2, 3], 1, 1) == []
    assert bounded_slice([0, 1, 2, 3], -10, 99) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        chunked([1, 2], -3)
