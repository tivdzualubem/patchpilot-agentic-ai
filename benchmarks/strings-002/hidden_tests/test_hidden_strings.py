from src.strings import lowercase


def test_lowercase_with_digits_and_punctuation() -> None:
    assert lowercase("AI-2026!") == "ai-2026!"


def test_lowercase_preserves_spaces() -> None:
    assert lowercase("  Mixed Case  ") == "  mixed case  "
