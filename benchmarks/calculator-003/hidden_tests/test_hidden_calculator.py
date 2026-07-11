from src.calculator import multiply


def test_multiply_larger_values() -> None:
    assert multiply(7, 8) == 56


def test_multiply_two_negative_values() -> None:
    assert multiply(-6, -9) == 54
