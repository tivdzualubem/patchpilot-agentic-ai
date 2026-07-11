"""Inventory reservation logic."""

from __future__ import annotations


def reserve_inventory(
    stock: dict[str, int],
    requests: list[tuple[str, int]],
) -> dict[str, int]:
    """Return remaining stock after atomically applying reservations."""
    remaining = stock
    for sku, quantity in requests:
        if quantity < 0:
            raise ValueError("requested quantities must be positive")
        if sku not in remaining:
            raise KeyError(sku)
        if quantity >= remaining[sku]:
            raise ValueError(f"insufficient stock for {sku}")
        remaining[sku] -= quantity
    return remaining
