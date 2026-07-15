import pytest

from src.inventory import reserve_inventory


def test_rejects_zero_quantity_without_mutation() -> None:
    stock = {"A": 3}
    with pytest.raises(ValueError):
        reserve_inventory(stock, [("A", 0)])
    assert stock == {"A": 3}


def test_duplicate_requests_are_validated_as_total() -> None:
    stock = {"A": 5}
    with pytest.raises(ValueError):
        reserve_inventory(stock, [("A", 3), ("A", 3)])
    assert stock == {"A": 5}


def test_missing_sku_is_atomic() -> None:
    stock = {"A": 5}
    with pytest.raises(KeyError):
        reserve_inventory(stock, [("A", 2), ("B", 1)])
    assert stock == {"A": 5}
