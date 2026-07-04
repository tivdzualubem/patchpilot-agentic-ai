from src.calculator import negate


def test_negate_positive_integer() -> None:
    assert negate(5) == -5


def test_negate_negative_integer() -> None:
    assert negate(-4) == 4


def test_negate_zero() -> None:
    assert negate(0) == 0
