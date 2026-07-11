from src.csvrecords import parse_record


def test_empty_record_is_one_empty_field() -> None:
    assert parse_record("") == [""]


def test_quoted_empty_field() -> None:
    assert parse_record('a,"",c') == ["a", "", "c"]


def test_commas_inside_multiple_quoted_fields() -> None:
    assert parse_record('"a,b","c,d"') == ["a,b", "c,d"]
