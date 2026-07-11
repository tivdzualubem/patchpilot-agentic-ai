from src.geometry import rectangle_area


def test_rectangle_area_fractional_dimensions() -> None:
    assert rectangle_area(1.25, 8.0) == 10.0


def test_rectangle_area_square() -> None:
    assert rectangle_area(6.0, 6.0) == 36.0
