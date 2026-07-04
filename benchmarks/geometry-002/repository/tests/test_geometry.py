from src.geometry import rectangle_perimeter


def test_rectangle_perimeter_positive_dimensions() -> None:
    assert rectangle_perimeter(3.0, 4.0) == 14.0


def test_rectangle_perimeter_zero_height() -> None:
    assert rectangle_perimeter(5.0, 0.0) == 10.0


def test_rectangle_perimeter_float_width() -> None:
    assert rectangle_perimeter(2.5, 4.0) == 13.0
