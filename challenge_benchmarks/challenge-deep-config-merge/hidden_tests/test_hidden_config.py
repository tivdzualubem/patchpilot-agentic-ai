from src.config import merge_config


def test_type_change_replaces_subtree() -> None:
    assert merge_config({"value": {"x": 1}}, {"value": [2, 3]}) == {
        "value": [2, 3]
    }


def test_override_is_deep_copied() -> None:
    override = {"nested": {"items": ["x"]}}
    result = merge_config({}, override)
    result["nested"]["items"].append("y")
    assert override == {"nested": {"items": ["x"]}}


def test_empty_override_returns_independent_copy() -> None:
    base = {"nested": {"count": 1}}
    result = merge_config(base, {})
    result["nested"]["count"] = 2
    assert base["nested"]["count"] == 1
