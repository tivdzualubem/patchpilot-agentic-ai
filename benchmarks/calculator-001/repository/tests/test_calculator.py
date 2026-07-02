from src.calculator import add


def test_add_positive_integers() -> None:
    assert add(2, 3) == 5


def test_add_negative_integers() -> None:
    assert add(-4, -3) == -7


def test_add_with_zero() -> None:
    assert add(9, 0) == 9
