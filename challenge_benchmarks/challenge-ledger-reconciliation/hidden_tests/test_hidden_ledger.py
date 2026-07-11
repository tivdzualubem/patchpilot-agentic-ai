import pytest

from src.ledger import Transfer, reconcile


def test_unknown_account_is_rejected_atomically() -> None:
    balances = {"a": 5}
    with pytest.raises(KeyError):
        reconcile(balances, [Transfer("x", "a", "missing", 1)])
    assert balances == {"a": 5}


def test_duplicate_id_with_different_payload_is_still_ignored() -> None:
    result = reconcile(
        {"a": 10, "b": 0},
        [
            Transfer("dup", "a", "b", 3),
            Transfer("dup", "a", "b", 8),
        ],
    )
    assert result == {"a": 7, "b": 3}


def test_transfer_iterable_can_be_generator() -> None:
    transfers = (
        item
        for item in [
            Transfer("x", "a", "b", 1),
            Transfer("y", "b", "a", 2),
        ]
    )
    assert reconcile({"a": 5, "b": 5}, transfers) == {"a": 6, "b": 4}
