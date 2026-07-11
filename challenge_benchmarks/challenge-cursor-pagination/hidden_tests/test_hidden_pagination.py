import pytest

from src.pagination import paginate


def test_invalid_cursor_is_rejected() -> None:
    with pytest.raises(ValueError):
        paginate([1, 2], limit=1, cursor="not-base64!")


def test_cursor_beyond_collection_is_rejected() -> None:
    _, cursor = paginate([1, 2, 3], limit=2)
    assert cursor is not None
    with pytest.raises(ValueError):
        paginate([1], limit=1, cursor=cursor)


def test_empty_collection() -> None:
    assert paginate([], limit=3) == ([], None)
