from src.calculator import negate


def test_negate_large_positive_value() -> None:
    assert negate(123) == -123


def test_negate_large_negative_value() -> None:
    assert negate(-91) == 91
