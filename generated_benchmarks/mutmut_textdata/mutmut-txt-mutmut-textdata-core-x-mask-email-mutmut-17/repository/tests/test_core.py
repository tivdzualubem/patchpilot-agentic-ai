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


def test_normalize_spaces_and_slugify() -> None:
    assert normalize_spaces("  agent\n  repair\t works ") == "agent repair works"
    assert slugify(" PatchPilot: Agentic AI ") == "patchpilot-agentic-ai"


def test_truncate_middle() -> None:
    assert truncate_middle("abcdefghij", 7) == "ab...ij"
    assert truncate_middle("short", 8) == "short"
    with pytest.raises(ValueError):
        truncate_middle("abc", 2)


def test_mask_email() -> None:
    assert mask_email("alice@example.com") == "a****@example.com"
    assert mask_email("x@example.com") == "x*@example.com"
    with pytest.raises(ValueError):
        mask_email("invalid")


def test_word_frequencies() -> None:
    assert word_frequencies("AI ai, repair's AI") == {"ai": 3, "repair's": 1}


def test_longest_common_prefix() -> None:
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix([]) == ""


def test_parse_csv_line() -> None:
    assert parse_csv_line(" alpha, beta ,gamma ") == ["alpha", "beta", "gamma"]
    assert parse_csv_line("") == []


def test_format_initials() -> None:
    assert format_initials("Ada Lovelace") == "A.L."
    assert format_initials("  grace   brewster murray hopper ") == "G.B.M.H."
    assert format_initials("") == ""


def test_count_substring() -> None:
    assert count_substring("aaaa", "aa") == 2
    with pytest.raises(ValueError):
        count_substring("abc", "")


def test_remove_duplicate_words() -> None:
    assert remove_duplicate_words("AI ai Repair repair Works") == "AI Repair Works"


def test_split_sentences() -> None:
    assert split_sentences("First. Second! Third?") == ["First", "Second", "Third"]
    assert split_sentences("...") == []


def test_wrap_words() -> None:
    assert wrap_words("one two three four", 7) == ["one two", "three", "four"]
    with pytest.raises(ValueError):
        wrap_words("text", 0)


def test_parse_key_values() -> None:
    assert parse_key_values("a=1\n b = two \n") == {"a": "1", "b": "two"}
    with pytest.raises(ValueError):
        parse_key_values("missing")


def test_format_bytes() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1_572_864) == "1.5 MiB"
    with pytest.raises(ValueError):
        format_bytes(-1)


def test_normalize_newlines() -> None:
    assert normalize_newlines("a\r\nb\rc\n") == "a\nb\nc\n"


def test_redact_terms() -> None:
    assert redact_terms("Secret token SECRET", ["secret"]) == "****** token ******"


def test_camel_to_snake() -> None:
    assert camel_to_snake("HTTPResponseCode") == "http_response_code"
    assert camel_to_snake("already_snake") == "already_snake"


def test_strip_prefixes() -> None:
    assert strip_prefixes("pre-value", ["x-", "pre-"]) == "value"
    assert strip_prefixes("value", ["pre-"]) == "value"
