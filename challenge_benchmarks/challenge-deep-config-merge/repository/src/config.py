"""Configuration merge helpers."""

from __future__ import annotations

from typing import Any


def merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge dictionaries without mutating either input."""
    result = base
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        elif isinstance(current, list) and isinstance(value, list):
            result[key] = current + value
        else:
            result[key] = value
    return result
