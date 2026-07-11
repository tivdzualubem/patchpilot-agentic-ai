"""Tests for research evaluation runner and paired summarization integration."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from patchpilot.evaluation.catalog import load_primary_research_catalog
from patchpilot.evaluation.conditions import PRIMARY_CONDITIONS

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(path: Path, name: str) -> ModuleType:
    """Import one repository script as a testable module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_evaluation_module() -> ModuleType:
    return load_script_module(
        ROOT / "scripts" / "run_evaluation.py",
        "patchpilot_run_evaluation",
    )


@pytest.fixture
def summarizer_module() -> ModuleType:
    return load_script_module(
        ROOT / "scripts" / "summarize_research_evaluation.py",
        "patchpilot_research_summarizer",
    )


def write_runs_csv(
    path: Path,
    *,
    condition: str,
    task_ids: list[str],
    success_pattern: list[bool],
) -> None:
    """Write a minimal valid runs.csv fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, task_id in enumerate(task_ids):
        succeeded = success_pattern[index]
        rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "status": "succeeded" if succeeded else "failed",
                "succeeded": str(succeeded).lower(),
                "hidden_verified_success": str(succeeded).lower(),
                "patch_attempts": "1",
                "steps": "4",
                "tool_calls": "3",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_runner_defaults_to_primary_research_catalog(
    run_evaluation_module: ModuleType,
) -> None:
    args = argparse.Namespace(
        manifest_root=None,
        catalog="generated_benchmarks/research_suite.json",
    )

    catalog = run_evaluation_module.resolve_catalog(ROOT, args)

    assert catalog.task_count == 53
    assert catalog.suite_id == "patchpilot-primary-research-benchmark"


def test_runner_writes_lf_only_csv(
    run_evaluation_module: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs.csv"
    run_evaluation_module.write_csv(
        output,
        [{"task_id": "alpha", "succeeded": True}],
    )

    raw = output.read_bytes()
    assert b"\r\n" not in raw
    assert raw == b"task_id,succeeded\nalpha,True\n"


def test_research_summarizer_builds_six_pairwise_comparisons(
    summarizer_module: ModuleType,
    tmp_path: Path,
) -> None:
    catalog = load_primary_research_catalog(ROOT)
    selected = list(catalog.tasks[:3])
    task_ids = [task.task_id for task in selected]
    experiment_root = tmp_path / "experiment"

    patterns = {
        "one-shot": [True, False, False],
        "fixed-workflow": [True, True, False],
        "tool-agent-no-reflection": [True, False, True],
        "full-reflective-agent": [True, True, True],
    }
    for condition in (item.value for item in PRIMARY_CONDITIONS):
        write_runs_csv(
            experiment_root / condition / "runs.csv",
            condition=condition,
            task_ids=task_ids,
            success_pattern=patterns[condition],
        )

    condition_rows = {
        condition.value: summarizer_module.load_condition_runs(
            experiment_root,
            condition.value,
        )
        for condition in PRIMARY_CONDITIONS
    }
    paired_tasks = summarizer_module.selected_catalog_tasks(
        catalog,
        condition_rows,
        allow_partial=True,
    )
    comparisons = summarizer_module.mcnemar_rows(
        paired_tasks,
        condition_rows,
    )

    assert len(paired_tasks) == 3
    assert len(comparisons) == 6
    full_vs_one_shot = next(
        row
        for row in comparisons
        if row["first_condition"] == "one-shot"
        and row["second_condition"] == "full-reflective-agent"
    )
    assert full_vs_one_shot["paired_tasks"] == 3
    assert full_vs_one_shot["second_only_success"] == 2


def test_research_summarizer_requires_full_catalog_by_default(
    summarizer_module: ModuleType,
    tmp_path: Path,
) -> None:
    catalog = load_primary_research_catalog(ROOT)
    task_id = catalog.tasks[0].task_id
    experiment_root = tmp_path / "experiment"

    for condition in (item.value for item in PRIMARY_CONDITIONS):
        write_runs_csv(
            experiment_root / condition / "runs.csv",
            condition=condition,
            task_ids=[task_id],
            success_pattern=[True],
        )

    condition_rows = {
        condition.value: summarizer_module.load_condition_runs(
            experiment_root,
            condition.value,
        )
        for condition in PRIMARY_CONDITIONS
    }

    with pytest.raises(ValueError, match="all 53 catalog tasks"):
        summarizer_module.selected_catalog_tasks(
            catalog,
            condition_rows,
            allow_partial=False,
        )
