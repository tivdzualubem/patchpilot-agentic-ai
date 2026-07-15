from src.lists import total


def test_total_larger_mixed_values() -> None:
    assert total([10, -5, 2, 8]) == 15


def test_total_repeated_values() -> None:
    assert total([4, 4, 4, 4]) == 16
