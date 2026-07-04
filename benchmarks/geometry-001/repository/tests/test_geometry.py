from src.geometry import rectangle_area


def test_rectangle_area_positive_dimensions() -> None:
    assert rectangle_area(3.0, 4.0) == 12.0


def test_rectangle_area_zero_height() -> None:
    assert rectangle_area(5.0, 0.0) == 0.0


def test_rectangle_area_float_width() -> None:
    assert rectangle_area(2.5, 4.0) == 10.0
