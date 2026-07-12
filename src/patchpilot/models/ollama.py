"""Local Ollama text-generation backend for PatchPilot."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaModelError(RuntimeError):
    """Raised when Ollama cannot return a valid model response."""


class OllamaChatModel:
    """Generate deterministic structured decisions through Ollama."""

    _CONTEXT_WINDOW = 4096
    _KEEP_ALIVE = "30m"
    _STRUCTURED_TOKEN_CAP = 256
    _UNSTRUCTURED_TOKEN_CAP = 384

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

    def trace_metadata(self) -> dict[str, object]:
        """Return stable backend identity and generation configuration."""
        return {
            "backend": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "context_window": self._CONTEXT_WINDOW,
            "keep_alive": self._KEEP_ALIVE,
            "structured_token_cap": min(
                self.max_tokens,
                self._STRUCTURED_TOKEN_CAP,
            ),
            "unstructured_token_cap": min(
                self.max_tokens,
                self._UNSTRUCTURED_TOKEN_CAP,
            ),
        }

    def _send(self, payload: dict[str, object]) -> str:
        """Send one non-streaming Ollama chat payload."""
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

    def warmup(self) -> None:
        """Load and retain the model before timed evaluation runs."""
        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "keep_alive": self._KEEP_ALIVE,
            "messages": [
                {
                    "role": "system",
                    "content": "Reply with exactly OK.",
                },
                {
                    "role": "user",
                    "content": "OK",
                },
            ],
            "options": {
                "temperature": 0.0,
                "seed": self.seed,
                "num_predict": 1,
                "num_ctx": self._CONTEXT_WINDOW,
            },
        }
        self._send(payload)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Return one bounded non-streaming chat response."""
        cap = (
            self._STRUCTURED_TOKEN_CAP
            if response_schema is not None
            else self._UNSTRUCTURED_TOKEN_CAP
        )
        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "keep_alive": self._KEEP_ALIVE,
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
                "num_predict": min(self.max_tokens, cap),
                "num_ctx": self._CONTEXT_WINDOW,
            },
        }

        if response_schema is not None:
            try:
                json.dumps(response_schema)
            except (TypeError, ValueError) as exc:
                raise ValueError("response_schema must be JSON-serializable.") from exc
            payload["format"] = response_schema

        return self._send(payload)
