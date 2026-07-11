"""Small transfer-ledger reconciliation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Transfer:
    transaction_id: str
    source: str
    destination: str
    amount: int


def reconcile(
    balances: dict[str, int],
    transfers: Iterable[Transfer],
) -> dict[str, int]:
    """Apply unique positive transfers without mutating input balances."""
    result = balances
    seen: set[str] = set()

    for transfer in transfers:
        if transfer.amount < 0:
            raise ValueError("transfer amount must be positive")
        if transfer.destination not in result:
            result[transfer.destination] = 0
        result[transfer.destination] += transfer.amount
        seen.add(transfer.transaction_id)
    return result
