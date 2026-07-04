from src.strings import reverse_text


def test_reverse_regular_word() -> None:
    assert reverse_text("patch") == "hctap"


def test_reverse_palindrome() -> None:
    assert reverse_text("level") == "level"


def test_reverse_with_punctuation() -> None:
    assert reverse_text("AI!") == "!IA"
