# ruff: noqa: E501
"""Create the eight PatchPilot manual challenge benchmarks."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TASKS: list[dict[str, Any]] = [
    {
        "id": "challenge-interval-merge",
        "module": "intervals.py",
        "title": "Merge overlapping and touching intervals",
        "goal": "Repair the interval merge implementation so it sorts correctly, merges touching ranges, and preserves normalized interval bounds.",
        "category": "multi_line_interval_logic",
        "difficulty": "hard",
        "patterns": ["wrong_sort_key", "boundary_condition", "state_update"],
        "rationale": "Requires coordinated fixes to ordering, overlap semantics, and merged-state updates.",
        "fixed": '"""Interval normalization helpers."""\n\nfrom __future__ import annotations\n\n\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    """Merge overlapping or touching closed intervals."""\n    normalized = [(min(start, end), max(start, end)) for start, end in intervals]\n    if not normalized:\n        return []\n\n    normalized.sort(key=lambda item: (item[0], item[1]))\n    merged: list[tuple[int, int]] = [normalized[0]]\n\n    for start, end in normalized[1:]:\n        previous_start, previous_end = merged[-1]\n        if start <= previous_end:\n            merged[-1] = (previous_start, max(previous_end, end))\n        else:\n            merged.append((start, end))\n    return merged\n',
        "defective": '"""Interval normalization helpers."""\n\nfrom __future__ import annotations\n\n\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    """Merge overlapping or touching closed intervals."""\n    normalized = [(start, end) for start, end in intervals]\n    if not normalized:\n        return []\n\n    normalized.sort(key=lambda item: (item[1], item[0]))\n    merged: list[tuple[int, int]] = [normalized[0]]\n\n    for start, end in normalized[1:]:\n        previous_start, previous_end = merged[-1]\n        if start < previous_end:\n            merged[-1] = (start, max(previous_end, end))\n        else:\n            merged.append((start, end))\n    return merged\n',
        "visible": "from src.intervals import merge_intervals\n\n\ndef test_merges_overlaps_and_touching_ranges() -> None:\n    assert merge_intervals([(5, 8), (1, 3), (3, 6)]) == [(1, 8)]\n\n\ndef test_keeps_disjoint_ranges_sorted() -> None:\n    assert merge_intervals([(10, 12), (1, 2), (5, 6)]) == [\n        (1, 2),\n        (5, 6),\n        (10, 12),\n    ]\n\n\ndef test_normalizes_reversed_bounds() -> None:\n    assert merge_intervals([(8, 5), (6, 9)]) == [(5, 9)]\n",
        "hidden": "from src.intervals import merge_intervals\n\n\ndef test_nested_intervals_keep_outer_start() -> None:\n    assert merge_intervals([(1, 10), (3, 4), (5, 7)]) == [(1, 10)]\n\n\ndef test_touching_singletons_merge() -> None:\n    assert merge_intervals([(2, 2), (2, 5), (5, 5)]) == [(2, 5)]\n\n\ndef test_empty_input() -> None:\n    assert merge_intervals([]) == []\n",
    },
    {
        "id": "challenge-inventory-reservation",
        "module": "inventory.py",
        "title": "Atomic inventory reservation",
        "goal": "Repair inventory reservation so validation is atomic, exact depletion is allowed, and caller-owned stock is never mutated.",
        "category": "transactional_state_management",
        "difficulty": "hard",
        "patterns": ["partial_mutation", "wrong_boundary", "aliasing"],
        "rationale": "The repair must coordinate validation, copy semantics, and update ordering.",
        "fixed": '"""Inventory reservation logic."""\n\nfrom __future__ import annotations\n\n\ndef reserve_inventory(\n    stock: dict[str, int],\n    requests: list[tuple[str, int]],\n) -> dict[str, int]:\n    """Return remaining stock after atomically applying reservations."""\n    totals: dict[str, int] = {}\n    for sku, quantity in requests:\n        if quantity <= 0:\n            raise ValueError("requested quantities must be positive")\n        totals[sku] = totals.get(sku, 0) + quantity\n\n    for sku, quantity in totals.items():\n        if sku not in stock:\n            raise KeyError(sku)\n        if quantity > stock[sku]:\n            raise ValueError(f"insufficient stock for {sku}")\n\n    remaining = dict(stock)\n    for sku, quantity in totals.items():\n        remaining[sku] -= quantity\n    return remaining\n',
        "defective": '"""Inventory reservation logic."""\n\nfrom __future__ import annotations\n\n\ndef reserve_inventory(\n    stock: dict[str, int],\n    requests: list[tuple[str, int]],\n) -> dict[str, int]:\n    """Return remaining stock after atomically applying reservations."""\n    remaining = stock\n    for sku, quantity in requests:\n        if quantity < 0:\n            raise ValueError("requested quantities must be positive")\n        if sku not in remaining:\n            raise KeyError(sku)\n        if quantity >= remaining[sku]:\n            raise ValueError(f"insufficient stock for {sku}")\n        remaining[sku] -= quantity\n    return remaining\n',
        "visible": 'import pytest\n\nfrom src.inventory import reserve_inventory\n\n\ndef test_combines_repeated_requests() -> None:\n    stock = {"A": 10, "B": 4}\n    assert reserve_inventory(stock, [("A", 3), ("A", 2)]) == {\n        "A": 5,\n        "B": 4,\n    }\n    assert stock == {"A": 10, "B": 4}\n\n\ndef test_allows_exact_depletion() -> None:\n    assert reserve_inventory({"A": 5}, [("A", 5)]) == {"A": 0}\n\n\ndef test_failure_is_atomic() -> None:\n    stock = {"A": 5, "B": 1}\n    with pytest.raises(ValueError):\n        reserve_inventory(stock, [("A", 2), ("B", 2)])\n    assert stock == {"A": 5, "B": 1}\n',
        "hidden": 'import pytest\n\nfrom src.inventory import reserve_inventory\n\n\ndef test_rejects_zero_quantity_without_mutation() -> None:\n    stock = {"A": 3}\n    with pytest.raises(ValueError):\n        reserve_inventory(stock, [("A", 0)])\n    assert stock == {"A": 3}\n\n\ndef test_duplicate_requests_are_validated_as_total() -> None:\n    stock = {"A": 5}\n    with pytest.raises(ValueError):\n        reserve_inventory(stock, [("A", 3), ("A", 3)])\n    assert stock == {"A": 5}\n\n\ndef test_missing_sku_is_atomic() -> None:\n    stock = {"A": 5}\n    with pytest.raises(KeyError):\n        reserve_inventory(stock, [("A", 2), ("B", 1)])\n    assert stock == {"A": 5}\n',
    },
    {
        "id": "challenge-topological-order",
        "module": "dependency.py",
        "title": "Deterministic dependency ordering",
        "goal": "Repair dependency ordering so every node is included, prerequisites appear first, ties are deterministic, and cycles are rejected.",
        "category": "graph_algorithm_invariant",
        "difficulty": "hard",
        "patterns": ["wrong_indegree", "missing_nodes", "nondeterministic_order"],
        "rationale": "Requires preserving graph invariants across initialization, traversal, and cycle detection.",
        "fixed": '"""Dependency graph helpers."""\n\nfrom __future__ import annotations\n\nimport heapq\n\n\ndef dependency_order(graph: dict[str, set[str]]) -> list[str]:\n    """Return a deterministic topological order for node prerequisites."""\n    nodes = set(graph)\n    for prerequisites in graph.values():\n        nodes.update(prerequisites)\n\n    dependents: dict[str, set[str]] = {node: set() for node in nodes}\n    indegree = {node: 0 for node in nodes}\n    for node, prerequisites in graph.items():\n        indegree[node] += len(prerequisites)\n        for prerequisite in prerequisites:\n            dependents[prerequisite].add(node)\n\n    ready = [node for node, count in indegree.items() if count == 0]\n    heapq.heapify(ready)\n    ordered: list[str] = []\n\n    while ready:\n        node = heapq.heappop(ready)\n        ordered.append(node)\n        for dependent in sorted(dependents[node]):\n            indegree[dependent] -= 1\n            if indegree[dependent] == 0:\n                heapq.heappush(ready, dependent)\n\n    if len(ordered) != len(nodes):\n        raise ValueError("dependency cycle detected")\n    return ordered\n',
        "defective": '"""Dependency graph helpers."""\n\nfrom __future__ import annotations\n\n\ndef dependency_order(graph: dict[str, set[str]]) -> list[str]:\n    """Return a deterministic topological order for node prerequisites."""\n    nodes = set(graph)\n\n    dependents: dict[str, set[str]] = {node: set() for node in nodes}\n    indegree = {node: 0 for node in nodes}\n    for node, prerequisites in graph.items():\n        for prerequisite in prerequisites:\n            dependents.setdefault(node, set()).add(prerequisite)\n            indegree[prerequisite] = indegree.get(prerequisite, 0) + 1\n\n    ready = [node for node, count in indegree.items() if count == 0]\n    ordered: list[str] = []\n\n    while ready:\n        node = ready.pop()\n        ordered.append(node)\n        for dependent in dependents.get(node, set()):\n            indegree[dependent] -= 1\n            if indegree[dependent] == 0:\n                ready.append(dependent)\n\n    return ordered\n',
        "visible": 'import pytest\n\nfrom src.dependency import dependency_order\n\n\ndef test_prerequisites_precede_dependents() -> None:\n    graph = {"deploy": {"build"}, "build": {"test"}, "test": set()}\n    assert dependency_order(graph) == ["test", "build", "deploy"]\n\n\ndef test_includes_prerequisites_not_declared_as_keys() -> None:\n    assert dependency_order({"app": {"lib"}}) == ["lib", "app"]\n\n\ndef test_rejects_cycles() -> None:\n    with pytest.raises(ValueError):\n        dependency_order({"a": {"b"}, "b": {"a"}})\n',
        "hidden": 'from src.dependency import dependency_order\n\n\ndef test_lexicographic_tie_breaking() -> None:\n    assert dependency_order({"c": set(), "a": set(), "b": set()}) == [\n        "a",\n        "b",\n        "c",\n    ]\n\n\ndef test_diamond_dependency() -> None:\n    graph = {\n        "package": {"lint", "test"},\n        "lint": {"source"},\n        "test": {"source"},\n    }\n    assert dependency_order(graph) == ["source", "lint", "test", "package"]\n\n\ndef test_empty_graph() -> None:\n    assert dependency_order({}) == []\n',
    },
    {
        "id": "challenge-deep-config-merge",
        "module": "config.py",
        "title": "Non-mutating recursive configuration merge",
        "goal": "Repair configuration merging so nested dictionaries merge recursively, lists replace rather than concatenate, and inputs remain unchanged.",
        "category": "recursive_data_merge",
        "difficulty": "hard",
        "patterns": ["shallow_merge", "input_mutation", "wrong_list_semantics"],
        "rationale": "Requires recursive copy semantics and type-sensitive conflict resolution.",
        "fixed": '"""Configuration merge helpers."""\n\nfrom __future__ import annotations\n\nfrom copy import deepcopy\nfrom typing import Any\n\n\ndef merge_config(\n    base: dict[str, Any],\n    override: dict[str, Any],\n) -> dict[str, Any]:\n    """Deep-merge dictionaries without mutating either input."""\n    result = deepcopy(base)\n    for key, value in override.items():\n        current = result.get(key)\n        if isinstance(current, dict) and isinstance(value, dict):\n            result[key] = merge_config(current, value)\n        else:\n            result[key] = deepcopy(value)\n    return result\n',
        "defective": '"""Configuration merge helpers."""\n\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\ndef merge_config(\n    base: dict[str, Any],\n    override: dict[str, Any],\n) -> dict[str, Any]:\n    """Deep-merge dictionaries without mutating either input."""\n    result = base\n    for key, value in override.items():\n        current = result.get(key)\n        if isinstance(current, dict) and isinstance(value, dict):\n            current.update(value)\n        elif isinstance(current, list) and isinstance(value, list):\n            result[key] = current + value\n        else:\n            result[key] = value\n    return result\n',
        "visible": 'from src.config import merge_config\n\n\ndef test_merges_nested_mappings() -> None:\n    base = {"db": {"host": "localhost", "port": 5432}}\n    override = {"db": {"port": 6432}}\n    assert merge_config(base, override) == {\n        "db": {"host": "localhost", "port": 6432}\n    }\n\n\ndef test_replaces_lists() -> None:\n    assert merge_config({"plugins": ["a", "b"]}, {"plugins": ["c"]}) == {\n        "plugins": ["c"]\n    }\n\n\ndef test_does_not_mutate_inputs() -> None:\n    base = {"nested": {"values": [1, 2]}}\n    override = {"nested": {"enabled": True}}\n    result = merge_config(base, override)\n    result["nested"]["values"].append(3)\n    assert base == {"nested": {"values": [1, 2]}}\n    assert override == {"nested": {"enabled": True}}\n',
        "hidden": 'from src.config import merge_config\n\n\ndef test_type_change_replaces_subtree() -> None:\n    assert merge_config({"value": {"x": 1}}, {"value": [2, 3]}) == {\n        "value": [2, 3]\n    }\n\n\ndef test_override_is_deep_copied() -> None:\n    override = {"nested": {"items": ["x"]}}\n    result = merge_config({}, override)\n    result["nested"]["items"].append("y")\n    assert override == {"nested": {"items": ["x"]}}\n\n\ndef test_empty_override_returns_independent_copy() -> None:\n    base = {"nested": {"count": 1}}\n    result = merge_config(base, {})\n    result["nested"]["count"] = 2\n    assert base["nested"]["count"] == 1\n',
    },
    {
        "id": "challenge-rolling-rate-limit",
        "module": "ratelimit.py",
        "title": "Rolling-window admission decisions",
        "goal": "Repair rolling-window rate limiting so boundary events expire correctly, rejected events consume no capacity, and timestamps must be monotonic.",
        "category": "stateful_window_algorithm",
        "difficulty": "hard",
        "patterns": [
            "wrong_expiry_boundary",
            "rejected_state_leak",
            "missing_validation",
        ],
        "rationale": "Multiple state transitions must be corrected together to preserve rolling-window invariants.",
        "fixed": '"""Rolling-window rate limiting."""\n\nfrom __future__ import annotations\n\nfrom collections import deque\n\n\ndef admit_events(\n    timestamps: list[int],\n    *,\n    limit: int,\n    window: int,\n) -> list[bool]:\n    """Return admission decisions for monotonic integer timestamps."""\n    if limit <= 0 or window <= 0:\n        raise ValueError("limit and window must be positive")\n\n    accepted: deque[int] = deque()\n    decisions: list[bool] = []\n    previous: int | None = None\n\n    for timestamp in timestamps:\n        if previous is not None and timestamp < previous:\n            raise ValueError("timestamps must be monotonic")\n        previous = timestamp\n\n        while accepted and accepted[0] <= timestamp - window:\n            accepted.popleft()\n\n        allowed = len(accepted) < limit\n        decisions.append(allowed)\n        if allowed:\n            accepted.append(timestamp)\n\n    return decisions\n',
        "defective": '"""Rolling-window rate limiting."""\n\nfrom __future__ import annotations\n\nfrom collections import deque\n\n\ndef admit_events(\n    timestamps: list[int],\n    *,\n    limit: int,\n    window: int,\n) -> list[bool]:\n    """Return admission decisions for monotonic integer timestamps."""\n    if limit < 0 or window < 0:\n        raise ValueError("limit and window must be positive")\n\n    accepted: deque[int] = deque()\n    decisions: list[bool] = []\n\n    for timestamp in timestamps:\n        while accepted and accepted[0] < timestamp - window:\n            accepted.popleft()\n\n        allowed = len(accepted) <= limit\n        decisions.append(allowed)\n        accepted.append(timestamp)\n\n    return decisions\n',
        "visible": "import pytest\n\nfrom src.ratelimit import admit_events\n\n\ndef test_enforces_capacity() -> None:\n    assert admit_events([0, 1, 2], limit=2, window=10) == [\n        True,\n        True,\n        False,\n    ]\n\n\ndef test_boundary_event_expires() -> None:\n    assert admit_events([0, 9, 10], limit=2, window=10) == [\n        True,\n        True,\n        True,\n    ]\n\n\ndef test_rejected_event_does_not_consume_capacity() -> None:\n    assert admit_events([0, 1, 2, 10], limit=2, window=10) == [\n        True,\n        True,\n        False,\n        True,\n    ]\n\n\ndef test_rejects_out_of_order_timestamps() -> None:\n    with pytest.raises(ValueError):\n        admit_events([5, 4], limit=2, window=10)\n",
        "hidden": "import pytest\n\nfrom src.ratelimit import admit_events\n\n\ndef test_zero_limit_is_invalid() -> None:\n    with pytest.raises(ValueError):\n        admit_events([1], limit=0, window=10)\n\n\ndef test_multiple_events_at_same_timestamp() -> None:\n    assert admit_events([3, 3, 3], limit=2, window=5) == [\n        True,\n        True,\n        False,\n    ]\n\n\ndef test_capacity_recovers_after_window() -> None:\n    assert admit_events([1, 2, 3, 6, 7], limit=2, window=5) == [\n        True,\n        True,\n        False,\n        True,\n        True,\n    ]\n",
    },
    {
        "id": "challenge-csv-record-parser",
        "module": "csvrecords.py",
        "title": "Quoted CSV record parsing",
        "goal": "Repair the CSV parser so it handles quoted commas, escaped quotes, empty fields, and malformed unterminated records.",
        "category": "state_machine_parser",
        "difficulty": "hard",
        "patterns": ["missing_state_machine", "escaped_delimiter", "malformed_input"],
        "rationale": "Requires coordinated character-state handling rather than a one-line delimiter split.",
        "fixed": '"""A small RFC-4180-style single-record CSV parser."""\n\nfrom __future__ import annotations\n\n\ndef parse_record(record: str) -> list[str]:\n    """Parse one CSV record with commas and doubled-quote escapes."""\n    fields: list[str] = []\n    current: list[str] = []\n    quoted = False\n    index = 0\n\n    while index < len(record):\n        character = record[index]\n        if quoted:\n            if character == \'"\':\n                if index + 1 < len(record) and record[index + 1] == \'"\':\n                    current.append(\'"\')\n                    index += 1\n                else:\n                    quoted = False\n            else:\n                current.append(character)\n        elif character == ",":\n            fields.append("".join(current))\n            current = []\n        elif character == \'"\' and not current:\n            quoted = True\n        else:\n            current.append(character)\n        index += 1\n\n    if quoted:\n        raise ValueError("unterminated quoted field")\n    fields.append("".join(current))\n    return fields\n',
        "defective": '"""A small RFC-4180-style single-record CSV parser."""\n\nfrom __future__ import annotations\n\n\ndef parse_record(record: str) -> list[str]:\n    """Parse one CSV record with commas and doubled-quote escapes."""\n    fields: list[str] = []\n    for part in record.split(","):\n        value = part.strip()\n        if value.startswith(\'"\'):\n            value = value[1:]\n        if value.endswith(\'"\'):\n            value = value[:-1]\n        fields.append(value.replace(\'""\', \'"\'))\n    return fields\n',
        "visible": 'import pytest\n\nfrom src.csvrecords import parse_record\n\n\ndef test_quoted_comma() -> None:\n    assert parse_record(\'alpha,"beta,gamma",delta\') == [\n        "alpha",\n        "beta,gamma",\n        "delta",\n    ]\n\n\ndef test_escaped_quote() -> None:\n    assert parse_record(\'"say ""hello""",world\') == [\n        \'say "hello"\',\n        "world",\n    ]\n\n\ndef test_preserves_empty_fields() -> None:\n    assert parse_record(",middle,") == ["", "middle", ""]\n\n\ndef test_rejects_unterminated_quote() -> None:\n    with pytest.raises(ValueError):\n        parse_record(\'alpha,"broken\')\n',
        "hidden": 'from src.csvrecords import parse_record\n\n\ndef test_empty_record_is_one_empty_field() -> None:\n    assert parse_record("") == [""]\n\n\ndef test_quoted_empty_field() -> None:\n    assert parse_record(\'a,"",c\') == ["a", "", "c"]\n\n\ndef test_commas_inside_multiple_quoted_fields() -> None:\n    assert parse_record(\'"a,b","c,d"\') == ["a,b", "c,d"]\n',
    },
    {
        "id": "challenge-cursor-pagination",
        "module": "pagination.py",
        "title": "Stable opaque cursor pagination",
        "goal": "Repair cursor pagination so cursors encode the absolute next offset, invalid cursors are rejected, and page boundaries never duplicate or skip items.",
        "category": "cursor_state_protocol",
        "difficulty": "hard",
        "patterns": ["relative_offset", "cursor_validation", "boundary_protocol"],
        "rationale": "The fix spans cursor decoding, offset arithmetic, and terminal-page behavior.",
        "fixed": '"""Opaque cursor pagination helpers."""\n\nfrom __future__ import annotations\n\nimport base64\nimport json\nfrom typing import Any\n\n\ndef _encode_cursor(offset: int) -> str:\n    payload = json.dumps({"offset": offset}, separators=(",", ":")).encode()\n    return base64.urlsafe_b64encode(payload).decode().rstrip("=")\n\n\ndef _decode_cursor(cursor: str | None) -> int:\n    if cursor is None:\n        return 0\n    try:\n        padding = "=" * (-len(cursor) % 4)\n        payload = base64.urlsafe_b64decode(cursor + padding)\n        data = json.loads(payload)\n        offset = data["offset"]\n    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:\n        raise ValueError("invalid cursor") from error\n    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:\n        raise ValueError("invalid cursor")\n    return offset\n\n\ndef paginate(\n    items: list[Any],\n    *,\n    limit: int,\n    cursor: str | None = None,\n) -> tuple[list[Any], str | None]:\n    """Return one page and an opaque cursor for the next page."""\n    if limit <= 0:\n        raise ValueError("limit must be positive")\n\n    offset = _decode_cursor(cursor)\n    if offset > len(items):\n        raise ValueError("cursor is beyond the collection")\n\n    page = items[offset : offset + limit]\n    next_offset = offset + len(page)\n    next_cursor = (\n        _encode_cursor(next_offset)\n        if next_offset < len(items)\n        else None\n    )\n    return page, next_cursor\n',
        "defective": '"""Opaque cursor pagination helpers."""\n\nfrom __future__ import annotations\n\nimport base64\nimport json\nfrom typing import Any\n\n\ndef _encode_cursor(offset: int) -> str:\n    payload = json.dumps({"offset": offset}).encode()\n    return base64.urlsafe_b64encode(payload).decode()\n\n\ndef _decode_cursor(cursor: str | None) -> int:\n    if cursor is None:\n        return 0\n    payload = base64.urlsafe_b64decode(cursor)\n    data = json.loads(payload)\n    return int(data.get("offset", 0))\n\n\ndef paginate(\n    items: list[Any],\n    *,\n    limit: int,\n    cursor: str | None = None,\n) -> tuple[list[Any], str | None]:\n    """Return one page and an opaque cursor for the next page."""\n    if limit < 0:\n        raise ValueError("limit must be positive")\n\n    offset = _decode_cursor(cursor)\n    page = items[offset : offset + limit]\n    next_cursor = _encode_cursor(len(page)) if page else None\n    return page, next_cursor\n',
        "visible": 'import pytest\n\nfrom src.pagination import paginate\n\n\ndef test_walks_all_pages_without_duplicates() -> None:\n    first, cursor = paginate([1, 2, 3, 4, 5], limit=2)\n    second, cursor = paginate([1, 2, 3, 4, 5], limit=2, cursor=cursor)\n    third, cursor = paginate([1, 2, 3, 4, 5], limit=2, cursor=cursor)\n    assert first + second + third == [1, 2, 3, 4, 5]\n    assert cursor is None\n\n\ndef test_exact_page_has_no_cursor() -> None:\n    page, cursor = paginate(["a", "b"], limit=2)\n    assert page == ["a", "b"]\n    assert cursor is None\n\n\ndef test_limit_must_be_positive() -> None:\n    with pytest.raises(ValueError):\n        paginate([1], limit=0)\n',
        "hidden": 'import pytest\n\nfrom src.pagination import paginate\n\n\ndef test_invalid_cursor_is_rejected() -> None:\n    with pytest.raises(ValueError):\n        paginate([1, 2], limit=1, cursor="not-base64!")\n\n\ndef test_cursor_beyond_collection_is_rejected() -> None:\n    _, cursor = paginate([1, 2, 3], limit=2)\n    assert cursor is not None\n    with pytest.raises(ValueError):\n        paginate([1], limit=1, cursor=cursor)\n\n\ndef test_empty_collection() -> None:\n    assert paginate([], limit=3) == ([], None)\n',
    },
    {
        "id": "challenge-ledger-reconciliation",
        "module": "ledger.py",
        "title": "Idempotent transfer ledger reconciliation",
        "goal": "Repair ledger reconciliation so duplicate transaction IDs are ignored, transfers are balanced, invalid amounts fail atomically, and input balances are preserved.",
        "category": "idempotent_financial_state",
        "difficulty": "hard",
        "patterns": ["duplicate_processing", "one_sided_transfer", "partial_mutation"],
        "rationale": "Requires transaction validation, idempotency, two-sided updates, and copy-on-write semantics.",
        "fixed": '"""Small transfer-ledger reconciliation."""\n\nfrom __future__ import annotations\n\nfrom collections.abc import Iterable\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Transfer:\n    transaction_id: str\n    source: str\n    destination: str\n    amount: int\n\n\ndef reconcile(\n    balances: dict[str, int],\n    transfers: Iterable[Transfer],\n) -> dict[str, int]:\n    """Apply unique positive transfers without mutating input balances."""\n    pending = list(transfers)\n    seen: set[str] = set()\n\n    for transfer in pending:\n        if transfer.amount <= 0:\n            raise ValueError("transfer amount must be positive")\n        if transfer.source not in balances or transfer.destination not in balances:\n            raise KeyError("unknown account")\n        if transfer.transaction_id in seen:\n            continue\n        seen.add(transfer.transaction_id)\n\n    result = dict(balances)\n    seen.clear()\n    for transfer in pending:\n        if transfer.transaction_id in seen:\n            continue\n        seen.add(transfer.transaction_id)\n        result[transfer.source] -= transfer.amount\n        result[transfer.destination] += transfer.amount\n    return result\n',
        "defective": '"""Small transfer-ledger reconciliation."""\n\nfrom __future__ import annotations\n\nfrom collections.abc import Iterable\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Transfer:\n    transaction_id: str\n    source: str\n    destination: str\n    amount: int\n\n\ndef reconcile(\n    balances: dict[str, int],\n    transfers: Iterable[Transfer],\n) -> dict[str, int]:\n    """Apply unique positive transfers without mutating input balances."""\n    result = balances\n    seen: set[str] = set()\n\n    for transfer in transfers:\n        if transfer.amount < 0:\n            raise ValueError("transfer amount must be positive")\n        if transfer.destination not in result:\n            result[transfer.destination] = 0\n        result[transfer.destination] += transfer.amount\n        seen.add(transfer.transaction_id)\n    return result\n',
        "visible": 'import pytest\n\nfrom src.ledger import Transfer, reconcile\n\n\ndef test_balanced_transfer() -> None:\n    balances = {"checking": 100, "savings": 20}\n    result = reconcile(\n        balances,\n        [Transfer("t1", "checking", "savings", 30)],\n    )\n    assert result == {"checking": 70, "savings": 50}\n    assert balances == {"checking": 100, "savings": 20}\n\n\ndef test_duplicate_transaction_is_idempotent() -> None:\n    transfer = Transfer("same", "a", "b", 4)\n    assert reconcile({"a": 10, "b": 0}, [transfer, transfer]) == {\n        "a": 6,\n        "b": 4,\n    }\n\n\ndef test_invalid_later_transfer_is_atomic() -> None:\n    balances = {"a": 10, "b": 0}\n    with pytest.raises(ValueError):\n        reconcile(\n            balances,\n            [\n                Transfer("ok", "a", "b", 2),\n                Transfer("bad", "a", "b", 0),\n            ],\n        )\n    assert balances == {"a": 10, "b": 0}\n',
        "hidden": 'import pytest\n\nfrom src.ledger import Transfer, reconcile\n\n\ndef test_unknown_account_is_rejected_atomically() -> None:\n    balances = {"a": 5}\n    with pytest.raises(KeyError):\n        reconcile(balances, [Transfer("x", "a", "missing", 1)])\n    assert balances == {"a": 5}\n\n\ndef test_duplicate_id_with_different_payload_is_still_ignored() -> None:\n    result = reconcile(\n        {"a": 10, "b": 0},\n        [\n            Transfer("dup", "a", "b", 3),\n            Transfer("dup", "a", "b", 8),\n        ],\n    )\n    assert result == {"a": 7, "b": 3}\n\n\ndef test_transfer_iterable_can_be_generator() -> None:\n    transfers = (\n        item\n        for item in [\n            Transfer("x", "a", "b", 1),\n            Transfer("y", "b", "a", 2),\n        ]\n    )\n    assert reconcile({"a": 5, "b": 5}, transfers) == {"a": 6, "b": 4}\n',
    },
]


def write_text(path: Path, content: str) -> None:
    """Write one UTF-8 text file with one final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    """Write deterministic human-readable JSON."""
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a directory tree deterministically."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def pytest_environment() -> dict[str, str]:
    """Return a deterministic isolated test environment."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return environment


def run_pytest(
    repository: Path,
    target: Path,
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run one visible or hidden pytest suite."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(target),
        ],
        cwd=repository,
        env=pytest_environment(),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def failure_count(output: str) -> int:
    """Extract pytest's failing-test count."""
    match = re.search(r"(\d+) failed", output)
    if match is not None:
        return int(match.group(1))
    return 0


def count_tests(source: str) -> int:
    """Count top-level pytest test functions."""
    module = ast.parse(source)
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in module.body
    )


def source_diff(
    module: str,
    fixed_source: str,
    defective_source: str,
) -> str:
    """Build a clean-to-defective unified diff."""
    return "".join(
        difflib.unified_diff(
            fixed_source.splitlines(keepends=True),
            defective_source.splitlines(keepends=True),
            fromfile=f"a/src/{module}",
            tofile=f"b/src/{module}",
        )
    )


def changed_line_count(diff: str) -> int:
    """Count substantive changed source lines."""
    return sum(
        1
        for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )


def write_repository(
    task_root: Path,
    task: dict[str, Any],
    *,
    fixed: bool,
) -> Path:
    """Write one challenge repository and its external hidden suite."""
    repository = task_root / "repository"
    source_dir = repository / "src"
    tests_dir = repository / "tests"
    hidden_dir = task_root / "hidden_tests"

    write_text(source_dir / "__init__.py", '"""Manual challenge package."""')
    source_key = "fixed" if fixed else "defective"
    write_text(source_dir / str(task["module"]), str(task[source_key]))
    stem = Path(str(task["module"])).stem
    write_text(tests_dir / f"test_{stem}.py", str(task["visible"]))
    write_text(hidden_dir / f"test_hidden_{stem}.py", str(task["hidden"]))
    return repository


def validate_clean_reference(
    task: dict[str, Any],
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    """Prove the clean reference satisfies visible and hidden suites."""
    with tempfile.TemporaryDirectory(prefix="patchpilot-challenge-clean-") as raw:
        task_root = Path(raw) / str(task["id"])
        repository = write_repository(task_root, task, fixed=True)
        visible = run_pytest(
            repository,
            repository / "tests",
            timeout_seconds=timeout_seconds,
        )
        hidden = run_pytest(
            repository,
            task_root / "hidden_tests",
            timeout_seconds=timeout_seconds,
        )
        if visible.returncode != 0 or hidden.returncode != 0:
            raise RuntimeError(
                f"clean reference failed for {task['id']}\n"
                f"VISIBLE\n{visible.stdout}\n{visible.stderr}\n"
                f"HIDDEN\n{hidden.stdout}\n{hidden.stderr}"
            )
        return {
            "visible_returncode": visible.returncode,
            "hidden_returncode": hidden.returncode,
            "visible_passed": True,
            "hidden_passed": True,
        }


def build_task(
    project_root: Path,
    output_root: Path,
    task: dict[str, Any],
    *,
    rank: int,
    force: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    """Create and validate one defective manual challenge."""
    task_id = str(task["id"])
    task_root = output_root / task_id
    if task_root.exists():
        if not force:
            raise FileExistsError(f"task already exists: {task_root}; use --force")
        shutil.rmtree(task_root)

    repository = write_repository(task_root, task, fixed=False)
    visible_result = run_pytest(
        repository,
        repository / "tests",
        timeout_seconds=timeout_seconds,
    )
    hidden_result = run_pytest(
        repository,
        task_root / "hidden_tests",
        timeout_seconds=timeout_seconds,
    )
    if visible_result.returncode == 0:
        raise RuntimeError(f"defective visible suite unexpectedly passed: {task_id}")
    if hidden_result.returncode == 0:
        raise RuntimeError(f"defective hidden suite unexpectedly passed: {task_id}")

    visible_output = visible_result.stdout + "\n" + visible_result.stderr
    hidden_output = hidden_result.stdout + "\n" + hidden_result.stderr
    visible_failures = failure_count(visible_output)
    hidden_failures = failure_count(hidden_output)
    if visible_failures < 1 or hidden_failures < 1:
        raise RuntimeError(f"could not count failures for {task_id}")

    clean_validation = validate_clean_reference(
        task,
        timeout_seconds=timeout_seconds,
    )
    module = str(task["module"])
    fixed_source = str(task["fixed"])
    defective_source = str(task["defective"])
    diff = source_diff(module, fixed_source, defective_source)
    line_count = changed_line_count(diff)
    if line_count < 3:
        raise RuntimeError(f"manual challenge is not multi-line enough: {task_id}")

    relative_task_root = task_root.relative_to(project_root).as_posix()
    manifest = {
        "task_id": task_id,
        "title": task["title"],
        "goal": task["goal"],
        "repository_root": f"{relative_task_root}/repository",
        "defect_category": task["category"],
        "difficulty": task["difficulty"],
        "allowed_paths": ["src"],
        "forbidden_paths": ["tests"],
        "test_command": ["python", "-m", "pytest", "-q"],
        "expected_initial_failures": visible_failures,
        "hidden_test_root": f"{relative_task_root}/hidden_tests",
        "expected_hidden_tests": count_tests(str(task["hidden"])),
    }
    write_json(task_root / "task.json", manifest)

    provenance = {
        "schema_version": "1.0",
        "origin_type": "manual_challenge",
        "task_id": task_id,
        "selection_rank": rank,
        "difficulty": task["difficulty"],
        "defect_category": task["category"],
        "defect_patterns": task["patterns"],
        "rationale": task["rationale"],
        "source_file": f"src/{module}",
        "clean_source_sha256": sha256_bytes(fixed_source.encode("utf-8")),
        "defective_source_sha256": sha256_bytes(defective_source.encode("utf-8")),
        "repository_tree_sha256": tree_sha256(repository),
        "hidden_tests_tree_sha256": tree_sha256(task_root / "hidden_tests"),
        "defect_diff": diff,
        "changed_line_count": line_count,
        "visible_validation": {
            "returncode": visible_result.returncode,
            "failed": visible_failures,
        },
        "hidden_validation": {
            "returncode": hidden_result.returncode,
            "failed": hidden_failures,
            "collected_tests": count_tests(str(task["hidden"])),
        },
        "clean_reference_validation": clean_validation,
        "generation": {
            "script": "scripts/create_manual_challenges.py",
            "command": (
                "python scripts/create_manual_challenges.py "
                "--force --timeout-seconds 60"
            ),
            "deterministic": True,
        },
    }
    write_json(task_root / "provenance.json", provenance)

    print(
        f"VALIDATED {rank:02d}/{len(TASKS)} {task_id}: "
        f"visible_failures={visible_failures} "
        f"hidden_failures={hidden_failures}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "manifest_path": f"{relative_task_root}/task.json",
        "repository_path": f"{relative_task_root}/repository",
        "provenance_path": f"{relative_task_root}/provenance.json",
        "difficulty": task["difficulty"],
        "defect_category": task["category"],
        "origin_type": "manual_challenge",
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create eight validated PatchPilot manual challenges."
    )
    parser.add_argument(
        "--output-root",
        default="challenge_benchmarks",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create all manual challenge tasks."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [
        build_task(
            project_root,
            output_root,
            task,
            rank=index,
            force=args.force,
            timeout_seconds=args.timeout_seconds,
        )
        for index, task in enumerate(TASKS, start=1)
    ]
    write_json(
        output_root / "manual_challenges.json",
        {
            "schema_version": "1.0",
            "suite_id": "patchpilot-manual-challenges",
            "task_count": len(rows),
            "tasks": rows,
        },
    )
    print(f"MANUAL_CHALLENGES={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
