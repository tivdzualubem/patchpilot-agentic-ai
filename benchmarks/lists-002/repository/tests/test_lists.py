from src.lists import total


def test_total_positive_values() -> None:
    assert total([1, 2, 3]) == 6


def test_total_empty_list() -> None:
    assert total([]) == 0


def test_total_mixed_values() -> None:
    assert total([-1, 5]) == 4
