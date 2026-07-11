from src.config import merge_config


def test_merges_nested_mappings() -> None:
    base = {"db": {"host": "localhost", "port": 5432}}
    override = {"db": {"port": 6432}}
    assert merge_config(base, override) == {
        "db": {"host": "localhost", "port": 6432}
    }


def test_replaces_lists() -> None:
    assert merge_config({"plugins": ["a", "b"]}, {"plugins": ["c"]}) == {
        "plugins": ["c"]
    }


def test_does_not_mutate_inputs() -> None:
    base = {"nested": {"values": [1, 2]}}
    override = {"nested": {"enabled": True}}
    result = merge_config(base, override)
    result["nested"]["values"].append(3)
    assert base == {"nested": {"values": [1, 2]}}
    assert override == {"nested": {"enabled": True}}
