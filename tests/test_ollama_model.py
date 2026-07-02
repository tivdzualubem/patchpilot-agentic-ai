import json
from urllib.error import URLError

import pytest

import patchpilot.models.ollama as ollama_module
from patchpilot.models import OllamaChatModel, OllamaModelError


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_generate_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(
            {"message": {"content": '{"tool":"list_files"}'}}
        )

    monkeypatch.setattr(ollama_module, "urlopen", fake_open)

    result = OllamaChatModel(timeout_seconds=30).generate(
        "system",
        "user",
    )

    assert result == '{"tool":"list_files"}'


def test_connection_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(ollama_module, "urlopen", fail)

    with pytest.raises(OllamaModelError, match="Could not connect"):
        OllamaChatModel().generate("system", "user")


@pytest.mark.parametrize("timeout", [0, 901])
def test_invalid_timeout_is_rejected(timeout: int) -> None:
    with pytest.raises(ValueError):
        OllamaChatModel(timeout_seconds=timeout)
