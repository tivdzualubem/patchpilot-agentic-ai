from src.calculator import multiply


def test_multiply_positive_integers() -> None:
    assert multiply(2, 3) == 6


def test_multiply_negative_integer() -> None:
    assert multiply(-4, 3) == -12


def test_multiply_by_zero() -> None:
    assert multiply(9, 0) == 0
