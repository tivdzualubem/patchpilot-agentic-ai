from src.dependency import dependency_order


def test_lexicographic_tie_breaking() -> None:
    assert dependency_order({"c": set(), "a": set(), "b": set()}) == [
        "a",
        "b",
        "c",
    ]


def test_diamond_dependency() -> None:
    graph = {
        "package": {"lint", "test"},
        "lint": {"source"},
        "test": {"source"},
    }
    assert dependency_order(graph) == ["source", "lint", "test", "package"]


def test_empty_graph() -> None:
    assert dependency_order({}) == []
