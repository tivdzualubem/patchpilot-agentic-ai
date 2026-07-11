import pytest

from src.dependency import dependency_order


def test_prerequisites_precede_dependents() -> None:
    graph = {"deploy": {"build"}, "build": {"test"}, "test": set()}
    assert dependency_order(graph) == ["test", "build", "deploy"]


def test_includes_prerequisites_not_declared_as_keys() -> None:
    assert dependency_order({"app": {"lib"}}) == ["lib", "app"]


def test_rejects_cycles() -> None:
    with pytest.raises(ValueError):
        dependency_order({"a": {"b"}, "b": {"a"}})
