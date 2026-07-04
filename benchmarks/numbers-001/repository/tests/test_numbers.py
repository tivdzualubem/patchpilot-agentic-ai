from src.numbers import is_even


def test_positive_even_number() -> None:
    assert is_even(8) is True


def test_positive_odd_number() -> None:
    assert is_even(7) is False


def test_zero_is_even() -> None:
    assert is_even(0) is True
