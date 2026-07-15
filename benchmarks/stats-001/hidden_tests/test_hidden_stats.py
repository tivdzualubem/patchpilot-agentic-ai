from src.stats import mean


def test_mean_fractional_values() -> None:
    assert mean([1.5, 2.5]) == 2.0


def test_mean_larger_collection() -> None:
    assert mean([1.0, 3.0, 5.0, 7.0]) == 4.0
