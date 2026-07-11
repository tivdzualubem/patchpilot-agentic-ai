"""Tests for validated evaluation catalog loading and selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patchpilot.evaluation.catalog import (
    discover_manifest_root,
    load_benchmark_catalog,
    load_primary_research_catalog,
    select_catalog_tasks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_primary_research_catalog_has_exact_composition() -> None:
    catalog = load_primary_research_catalog(ROOT)

    assert catalog.suite_id == "patchpilot-primary-research-benchmark"
    assert catalog.task_count == 53
    assert catalog.declared_task_count == 53
    assert catalog.catalog_sha256 is not None
    assert len(catalog.catalog_sha256) == 64
    assert catalog.composition()["origin_type"] == {
        "manual_challenge": 8,
        "mutmut": 45,
    }
    assert len({task.task_id for task in catalog.tasks}) == 53
    assert all(task.manifest_path.is_file() for task in catalog.tasks)


def test_catalog_selection_preserves_canonical_order() -> None:
    catalog = load_primary_research_catalog(ROOT)
    requested = [
        catalog.tasks[4].task_id,
        catalog.tasks[1].task_id,
    ]

    selected = select_catalog_tasks(
        catalog,
        task_ids=requested,
    )

    assert [task.task_id for task in selected] == [
        catalog.tasks[1].task_id,
        catalog.tasks[4].task_id,
    ]


def test_catalog_selection_rejects_unknown_task() -> None:
    catalog = load_primary_research_catalog(ROOT)

    with pytest.raises(ValueError, match="not in the selected catalog"):
        select_catalog_tasks(
            catalog,
            task_ids=["missing-task"],
        )


def test_catalog_rejects_manifest_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-task.json"
    outside.write_text("{}", encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "suite_id": "unsafe",
                "task_count": 1,
                "tasks": [
                    {
                        "task_id": "unsafe",
                        "manifest_path": "../outside-task.json",
                        "origin_type": "custom",
                        "difficulty": "easy",
                        "defect_category": "logic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the project root"):
        load_benchmark_catalog(tmp_path, catalog_path)


def test_custom_manifest_root_supports_sanity_suite() -> None:
    catalog = discover_manifest_root(ROOT, "benchmarks")

    assert catalog.task_count == 12
    assert catalog.catalog_path is None
    assert catalog.suite_id == "custom:benchmarks"
    assert all(task.origin_type == "custom" for task in catalog.tasks)
