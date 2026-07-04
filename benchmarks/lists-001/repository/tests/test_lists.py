from src.lists import first_item


def test_first_item_multiple_values() -> None:
    assert first_item([1, 2, 3]) == 1


def test_first_item_two_values() -> None:
    assert first_item([8, 9]) == 8


def test_first_item_single_value() -> None:
    assert first_item([42]) == 42
