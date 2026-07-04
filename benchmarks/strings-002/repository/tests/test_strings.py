from src.strings import lowercase


def test_lowercase_mixed_case() -> None:
    assert lowercase("PatchPilot") == "patchpilot"


def test_lowercase_all_caps() -> None:
    assert lowercase("AI") == "ai"


def test_lowercase_empty_string() -> None:
    assert lowercase("") == ""
