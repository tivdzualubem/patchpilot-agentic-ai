"""Dependency graph helpers."""

from __future__ import annotations


def dependency_order(graph: dict[str, set[str]]) -> list[str]:
    """Return a deterministic topological order for node prerequisites."""
    nodes = set(graph)

    dependents: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for node, prerequisites in graph.items():
        for prerequisite in prerequisites:
            dependents.setdefault(node, set()).add(prerequisite)
            indegree[prerequisite] = indegree.get(prerequisite, 0) + 1

    ready = [node for node, count in indegree.items() if count == 0]
    ordered: list[str] = []

    while ready:
        node = ready.pop()
        ordered.append(node)
        for dependent in dependents.get(node, set()):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    return ordered
