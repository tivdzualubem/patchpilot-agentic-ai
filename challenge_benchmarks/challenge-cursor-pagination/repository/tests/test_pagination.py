import pytest

from src.pagination import paginate


def test_walks_all_pages_without_duplicates() -> None:
    first, cursor = paginate([1, 2, 3, 4, 5], limit=2)
    second, cursor = paginate([1, 2, 3, 4, 5], limit=2, cursor=cursor)
    third, cursor = paginate([1, 2, 3, 4, 5], limit=2, cursor=cursor)
    assert first + second + third == [1, 2, 3, 4, 5]
    assert cursor is None


def test_exact_page_has_no_cursor() -> None:
    page, cursor = paginate(["a", "b"], limit=2)
    assert page == ["a", "b"]
    assert cursor is None


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        paginate([1], limit=0)
