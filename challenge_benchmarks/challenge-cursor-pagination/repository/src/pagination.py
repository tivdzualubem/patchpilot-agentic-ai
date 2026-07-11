"""Opaque cursor pagination helpers."""

from __future__ import annotations

import base64
import json
from typing import Any


def _encode_cursor(offset: int) -> str:
    payload = json.dumps({"offset": offset}).encode()
    return base64.urlsafe_b64encode(payload).decode()


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    payload = base64.urlsafe_b64decode(cursor)
    data = json.loads(payload)
    return int(data.get("offset", 0))


def paginate(
    items: list[Any],
    *,
    limit: int,
    cursor: str | None = None,
) -> tuple[list[Any], str | None]:
    """Return one page and an opaque cursor for the next page."""
    if limit < 0:
        raise ValueError("limit must be positive")

    offset = _decode_cursor(cursor)
    page = items[offset : offset + limit]
    next_cursor = _encode_cursor(len(page)) if page else None
    return page, next_cursor
