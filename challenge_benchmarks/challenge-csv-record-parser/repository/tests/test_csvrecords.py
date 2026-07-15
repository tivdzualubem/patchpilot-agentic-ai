import pytest

from src.csvrecords import parse_record


def test_quoted_comma() -> None:
    assert parse_record('alpha,"beta,gamma",delta') == [
        "alpha",
        "beta,gamma",
        "delta",
    ]


def test_escaped_quote() -> None:
    assert parse_record('"say ""hello""",world') == [
        'say "hello"',
        "world",
    ]


def test_preserves_empty_fields() -> None:
    assert parse_record(",middle,") == ["", "middle", ""]


def test_rejects_unterminated_quote() -> None:
    with pytest.raises(ValueError):
        parse_record('alpha,"broken')
