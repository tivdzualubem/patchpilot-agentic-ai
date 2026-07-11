from src.calculator import add


def test_add_large_mixed_sign_values() -> None:
    assert add(100, -40) == 60


def test_add_two_negative_values() -> None:
    assert add(-25, -17) == -42
