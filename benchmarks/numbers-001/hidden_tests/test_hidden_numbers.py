from src.numbers import is_even


def test_negative_even_number() -> None:
    assert is_even(-4) is True


def test_negative_odd_number() -> None:
    assert is_even(-3) is False
