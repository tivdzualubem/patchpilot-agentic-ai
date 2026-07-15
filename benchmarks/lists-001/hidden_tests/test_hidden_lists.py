from src.lists import first_item


def test_first_item_when_first_is_zero() -> None:
    assert first_item([0, 5, 9]) == 0


def test_first_item_with_negative_values() -> None:
    assert first_item([-8, -3, -1]) == -8
