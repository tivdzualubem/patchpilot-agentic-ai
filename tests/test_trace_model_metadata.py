"""Tests for traceable model identity and reproducible configuration."""

from patchpilot.models import OllamaChatModel


def test_ollama_trace_metadata_reports_generation_configuration() -> None:
    model = OllamaChatModel(
        model="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/",
        timeout_seconds=240,
        temperature=0.2,
        seed=7,
        max_tokens=1024,
    )

    assert model.trace_metadata() == {
        "backend": "ollama",
        "model": "qwen2.5-coder:7b",
        "base_url": "http://127.0.0.1:11434",
        "timeout_seconds": 240,
        "temperature": 0.2,
        "seed": 7,
        "max_tokens": 1024,
        "context_window": 4096,
        "keep_alive": "30m",
        "structured_token_cap": 256,
        "unstructured_token_cap": 384,
    }
