from src.calculator import subtract


def test_subtract_positive_integers() -> None:
    assert subtract(9, 4) == 5


def test_subtract_negative_integers() -> None:
    assert subtract(-4, -3) == -1


def test_subtract_zero() -> None:
    assert subtract(9, 0) == 9
