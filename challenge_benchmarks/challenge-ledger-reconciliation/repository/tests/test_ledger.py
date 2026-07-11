import pytest

from src.ledger import Transfer, reconcile


def test_balanced_transfer() -> None:
    balances = {"checking": 100, "savings": 20}
    result = reconcile(
        balances,
        [Transfer("t1", "checking", "savings", 30)],
    )
    assert result == {"checking": 70, "savings": 50}
    assert balances == {"checking": 100, "savings": 20}


def test_duplicate_transaction_is_idempotent() -> None:
    transfer = Transfer("same", "a", "b", 4)
    assert reconcile({"a": 10, "b": 0}, [transfer, transfer]) == {
        "a": 6,
        "b": 4,
    }


def test_invalid_later_transfer_is_atomic() -> None:
    balances = {"a": 10, "b": 0}
    with pytest.raises(ValueError):
        reconcile(
            balances,
            [
                Transfer("ok", "a", "b", 2),
                Transfer("bad", "a", "b", 0),
            ],
        )
    assert balances == {"a": 10, "b": 0}
