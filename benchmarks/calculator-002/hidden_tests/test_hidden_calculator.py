from src.calculator import subtract


def test_subtract_from_zero() -> None:
    assert subtract(0, 7) == -7


def test_subtract_larger_negative_value() -> None:
    assert subtract(-10, 6) == -16
