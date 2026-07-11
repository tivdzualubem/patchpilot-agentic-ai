"""Local Ollama text-generation backend for PatchPilot."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaModelError(RuntimeError):
    """Raised when Ollama cannot return a valid model response."""


class OllamaChatModel:
    """Generate deterministic structured decisions through Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 180,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty.")

        if not 1 <= timeout_seconds <= 900:
            raise ValueError("timeout_seconds must be between 1 and 900.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096.")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Return one non-streaming JSON-mode chat response."""
        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.max_tokens,
            },
        }

        if response_schema is not None:
            try:
                schema_text = json.dumps(
                    response_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("response_schema must be JSON-serializable.") from exc

            payload["format"] = response_schema
            messages = payload["messages"]
            if not isinstance(messages, list):
                raise AssertionError("Ollama messages payload must be a list.")

            user_message = messages[-1]
            if not isinstance(user_message, dict):
                raise AssertionError("Ollama user message must be an object.")

            user_message["content"] = (
                f"{user_prompt}\n\nRESPONSE JSON SCHEMA:\n{schema_text}"
            )

        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaModelError(
                f"Ollama returned HTTP {exc.code}: {body[:500]}"
            ) from exc
        except URLError as exc:
            raise OllamaModelError("Could not connect to the Ollama service.") from exc
        except TimeoutError as exc:
            raise OllamaModelError("Ollama generation timed out.") from exc

        try:
            result = json.loads(body)
            content = result["message"]["content"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaModelError(
                "Ollama returned an invalid response payload."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise OllamaModelError("Ollama returned empty generated content.")

        return content.strip()
