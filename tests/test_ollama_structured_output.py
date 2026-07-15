# Tests for Ollama-native JSON-schema structured outputs.

from __future__ import annotations

import json
from typing import Any

import pytest

from patchpilot.models import OllamaChatModel


class _FakeResponse:
    # Minimal context-managed HTTP response.

    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return self._body


def test_json_schema_is_sent_as_ollama_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: object, timeout: int) -> _FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "tool": "run_tests",
                            "arguments": {},
                        }
                    )
                }
            }
        )

    monkeypatch.setattr(
        "patchpilot.models.ollama.urlopen",
        fake_urlopen,
    )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["run_tests"],
            },
            "arguments": {
                "type": "object",
                "additionalProperties": False,
            },
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }

    result = OllamaChatModel(timeout_seconds=30).generate(
        "system prompt",
        "user prompt",
        response_schema=schema,
    )

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))

    assert captured["timeout"] == 30
    assert payload["format"] == schema
    assert payload["stream"] is False
    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-1]["content"] == "user prompt"
    assert "RESPONSE JSON SCHEMA:" not in payload["messages"][-1]["content"]
    assert json.loads(result)["tool"] == "run_tests"


def test_unstructured_request_omits_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: object, timeout: int) -> _FakeResponse:
        del timeout
        captured["request"] = request
        return _FakeResponse({"message": {"content": "plain response"}})

    monkeypatch.setattr(
        "patchpilot.models.ollama.urlopen",
        fake_urlopen,
    )

    result = OllamaChatModel().generate(
        "system prompt",
        "user prompt",
    )

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))

    assert "format" not in payload
    assert payload["messages"][-1]["content"] == "user prompt"
    assert result == "plain response"


def test_non_serializable_schema_is_rejected_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_urlopen(request: object, timeout: int) -> _FakeResponse:
        nonlocal called
        del request, timeout
        called = True
        return _FakeResponse({"message": {"content": "{}"}})

    monkeypatch.setattr(
        "patchpilot.models.ollama.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        ValueError,
        match="response_schema must be JSON-serializable",
    ):
        OllamaChatModel().generate(
            "system prompt",
            "user prompt",
            response_schema={"invalid": object()},
        )

    assert called is False
