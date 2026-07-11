from src.strings import reverse_text


def test_reverse_longer_word() -> None:
    assert reverse_text("agentic") == "citnega"


def test_reverse_spaces() -> None:
    assert reverse_text("a b c") == "c b a"
