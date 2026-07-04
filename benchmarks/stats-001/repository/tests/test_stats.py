from src.stats import mean


def test_mean_multiple_values() -> None:
    assert mean([2.0, 4.0, 6.0]) == 4.0


def test_mean_single_value() -> None:
    assert mean([10.0]) == 10.0


def test_mean_symmetric_values() -> None:
    assert mean([-2.0, 2.0]) == 0.0
