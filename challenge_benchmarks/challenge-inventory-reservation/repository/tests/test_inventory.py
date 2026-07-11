import pytest

from src.inventory import reserve_inventory


def test_combines_repeated_requests() -> None:
    stock = {"A": 10, "B": 4}
    assert reserve_inventory(stock, [("A", 3), ("A", 2)]) == {
        "A": 5,
        "B": 4,
    }
    assert stock == {"A": 10, "B": 4}


def test_allows_exact_depletion() -> None:
    assert reserve_inventory({"A": 5}, [("A", 5)]) == {"A": 0}


def test_failure_is_atomic() -> None:
    stock = {"A": 5, "B": 1}
    with pytest.raises(ValueError):
        reserve_inventory(stock, [("A", 2), ("B", 2)])
    assert stock == {"A": 5, "B": 1}
