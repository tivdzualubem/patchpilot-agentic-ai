import pytest
from mutmut_textdata.core import (
    camel_to_snake,
    count_substring,
    format_bytes,
    format_initials,
    longest_common_prefix,
    mask_email,
    normalize_newlines,
    normalize_spaces,
    parse_csv_line,
    parse_key_values,
    redact_terms,
    remove_duplicate_words,
    slugify,
    split_sentences,
    strip_prefixes,
    truncate_middle,
    word_frequencies,
    wrap_words,
)


def test_hidden_normalization_and_slugging() -> None:
    assert normalize_spaces("\talpha   beta\n") == "alpha beta"
    assert normalize_spaces("") == ""
    assert slugify("Already---Slugged") == "already-slugged"
    assert slugify(" *** ") == ""


def test_hidden_truncation_and_email_masking() -> None:
    assert truncate_middle("abcdefgh", 6) == "ab...h"
    assert truncate_middle("abc", 3) == "abc"
    assert mask_email("ab@host") == "a*@host"
    with pytest.raises(ValueError):
        mask_email("@host")


def test_hidden_word_analysis() -> None:
    assert word_frequencies("One two TWO one's") == {
        "one": 1,
        "two": 2,
        "one's": 1,
    }
    assert (
        longest_common_prefix(["interspecies", "interstellar", "interstate"])
        == "inters"
    )
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""


def test_hidden_csv_and_initials() -> None:
    assert parse_csv_line("one") == ["one"]
    assert parse_csv_line("a,,c") == ["a", "", "c"]
    assert format_initials("single") == "S."


def test_hidden_count_and_deduplication() -> None:
    assert count_substring("abababa", "aba") == 2
    assert remove_duplicate_words("One two ONE three Two") == "One two three"


def test_hidden_sentences_and_wrapping() -> None:
    assert split_sentences("Wait?! Really... Yes!") == ["Wait", "Really", "Yes"]
    assert wrap_words("", 5) == []
    assert wrap_words("oversized tiny", 4) == ["oversized", "tiny"]


def test_hidden_key_values_and_bytes() -> None:
    assert parse_key_values("\n x = 1=2 \n y= \n") == {"x": "1=2", "y": ""}
    assert format_bytes(1023) == "1023 B"
    assert format_bytes(1024 * 1024 * 2) == "2.0 MiB"
    with pytest.raises(ValueError):
        parse_key_values("=value")


def test_hidden_newlines_and_redaction() -> None:
    assert normalize_newlines("a\rb\r\nc") == "a\nb\nc"
    assert redact_terms("Alpha beta ALPHA", ["alpha", ""]) == "***** beta *****"
    assert redact_terms("unchanged", []) == "unchanged"


def test_hidden_identifier_and_prefix_helpers() -> None:
    assert camel_to_snake("XMLHttpRequest") == "xml_http_request"
    assert camel_to_snake("two-Words") == "two_words"
    assert strip_prefixes("foobar", ["foo", "f"]) == "bar"
    assert strip_prefixes("", ["x"]) == ""
