from src.geometry import rectangle_perimeter


def test_rectangle_perimeter_fractional_dimensions() -> None:
    assert rectangle_perimeter(1.5, 2.5) == 8.0


def test_rectangle_perimeter_square() -> None:
    assert rectangle_perimeter(6.0, 6.0) == 24.0
