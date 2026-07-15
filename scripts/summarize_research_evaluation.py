"""Summarize the canonical paired PatchPilot research evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from statistics import fmean

from patchpilot.evaluation.catalog import (
    BenchmarkCatalog,
    CatalogTask,
    load_primary_research_catalog,
)
from patchpilot.evaluation.conditions import PRIMARY_CONDITIONS
from patchpilot.evaluation.statistics import exact_mcnemar_test

BOOLEAN_TRUE = {"1", "true", "yes"}
BOOLEAN_FALSE = {"0", "false", "no"}


def parse_bool(value: object, *, field_name: str) -> bool:
    """Parse a CSV boolean without Python truthiness ambiguity."""
    normalized = str(value).strip().lower()
    if normalized in BOOLEAN_TRUE:
        return True
    if normalized in BOOLEAN_FALSE:
        return False
    raise ValueError(f"Invalid boolean for {field_name}: {value!r}")


def parse_number(value: object, *, field_name: str) -> float:
    """Parse one numeric CSV metric."""
    try:
        return float(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV file as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write deterministic LF-only CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_condition_values() -> tuple[str, ...]:
    """Return canonical primary conditions in experiment order."""
    return tuple(condition.value for condition in PRIMARY_CONDITIONS)


def load_condition_runs(
    experiment_root: Path,
    condition: str,
) -> dict[str, dict[str, str]]:
    """Load one condition and reject duplicate task rows."""
    path = experiment_root / condition / "runs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing runs.csv for condition {condition!r}: {path}")

    rows = read_csv_rows(path)
    required = {
        "task_id",
        "condition",
        "status",
        "succeeded",
        "hidden_verified_success",
        "patch_attempts",
        "steps",
        "tool_calls",
    }
    by_task: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}:{row_number} missing columns: {missing}")
        task_id = row["task_id"]
        if not task_id:
            raise ValueError(f"{path}:{row_number} has an empty task_id.")
        if row["condition"] != condition:
            raise ValueError(
                f"{path}:{row_number} condition mismatch: "
                f"{row['condition']!r} != {condition!r}."
            )
        if task_id in by_task:
            raise ValueError(f"Duplicate task_id in {path}: {task_id}")
        by_task[task_id] = row

    if not by_task:
        raise ValueError(f"No run rows found in {path}.")
    return by_task


def selected_catalog_tasks(
    catalog: BenchmarkCatalog,
    condition_rows: dict[str, dict[str, dict[str, str]]],
    *,
    allow_partial: bool,
) -> tuple[CatalogTask, ...]:
    """Validate paired task coverage and preserve catalog order."""
    task_sets = {condition: set(rows) for condition, rows in condition_rows.items()}
    first_condition = canonical_condition_values()[0]
    reference = task_sets[first_condition]

    for condition, task_ids in task_sets.items():
        if task_ids != reference:
            missing = sorted(reference - task_ids)
            extra = sorted(task_ids - reference)
            raise ValueError(
                f"Condition {condition!r} is not paired with "
                f"{first_condition!r}; missing={missing}, extra={extra}."
            )

    catalog_ids = {task.task_id for task in catalog.tasks}
    unknown = sorted(reference - catalog_ids)
    if unknown:
        raise ValueError(
            "Evaluation contains tasks outside the research catalog: "
            + ", ".join(unknown)
        )

    if not allow_partial and reference != catalog_ids:
        missing = sorted(catalog_ids - reference)
        raise ValueError(
            f"Full research summary requires all 53 catalog tasks; missing={missing}."
        )

    selected = tuple(task for task in catalog.tasks if task.task_id in reference)
    if not selected:
        raise ValueError("No paired catalog tasks were selected.")
    return selected


def mean_metric(
    rows: Iterable[dict[str, str]],
    field_name: str,
) -> float:
    """Return the arithmetic mean of one numeric run field."""
    values = [parse_number(row[field_name], field_name=field_name) for row in rows]
    return fmean(values) if values else 0.0


def summary_row(
    *,
    condition: str,
    rows: list[dict[str, str]],
    stratum: str,
    stratum_value: str,
) -> dict[str, object]:
    """Aggregate one condition within one benchmark stratum."""
    count = len(rows)
    successes = sum(
        parse_bool(row["succeeded"], field_name="succeeded") for row in rows
    )
    hidden_successes = sum(
        parse_bool(
            row["hidden_verified_success"],
            field_name="hidden_verified_success",
        )
        for row in rows
    )
    return {
        "condition": condition,
        "stratum": stratum,
        "stratum_value": stratum_value,
        "runs": count,
        "successes": successes,
        "repair_rate": successes / count if count else 0.0,
        "hidden_verified_successes": hidden_successes,
        "hidden_verified_repair_rate": (hidden_successes / count if count else 0.0),
        "mean_patch_attempts": mean_metric(rows, "patch_attempts"),
        "mean_steps": mean_metric(rows, "steps"),
        "mean_tool_calls": mean_metric(rows, "tool_calls"),
    }


def build_summary_rows(
    tasks: tuple[CatalogTask, ...],
    condition_rows: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    """Build overall and stratified canonical condition summaries."""
    task_by_id = {task.task_id: task for task in tasks}
    output: list[dict[str, object]] = []

    for condition in canonical_condition_values():
        rows_by_task = condition_rows[condition]
        ordered_rows = [rows_by_task[task.task_id] for task in tasks]
        output.append(
            summary_row(
                condition=condition,
                rows=ordered_rows,
                stratum="overall",
                stratum_value="all",
            )
        )

        for field_name in ("origin_type", "difficulty"):
            values = sorted({str(getattr(task, field_name)) for task in tasks})
            for value in values:
                rows = [
                    rows_by_task[task_id]
                    for task_id, task in task_by_id.items()
                    if str(getattr(task, field_name)) == value
                ]
                output.append(
                    summary_row(
                        condition=condition,
                        rows=rows,
                        stratum=field_name,
                        stratum_value=value,
                    )
                )

    return output


def paired_outcome_rows(
    tasks: tuple[CatalogTask, ...],
    condition_rows: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    """Build one wide paired row per catalog task."""
    output: list[dict[str, object]] = []
    for task in tasks:
        row: dict[str, object] = {
            "task_id": task.task_id,
            "origin_type": task.origin_type,
            "difficulty": task.difficulty,
            "defect_category": task.defect_category,
        }
        for condition in canonical_condition_values():
            source = condition_rows[condition][task.task_id]
            prefix = condition.replace("-", "_")
            row[f"{prefix}_succeeded"] = parse_bool(
                source["succeeded"],
                field_name="succeeded",
            )
            row[f"{prefix}_hidden_verified_success"] = parse_bool(
                source["hidden_verified_success"],
                field_name="hidden_verified_success",
            )
            row[f"{prefix}_status"] = source["status"]
            row[f"{prefix}_patch_attempts"] = int(
                parse_number(
                    source["patch_attempts"],
                    field_name="patch_attempts",
                )
            )
            row[f"{prefix}_steps"] = int(
                parse_number(source["steps"], field_name="steps")
            )
            row[f"{prefix}_tool_calls"] = int(
                parse_number(
                    source["tool_calls"],
                    field_name="tool_calls",
                )
            )
        output.append(row)
    return output


def mcnemar_rows(
    tasks: tuple[CatalogTask, ...],
    condition_rows: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    """Compute all six exact pairwise canonical comparisons."""
    output: list[dict[str, object]] = []
    for first, second in combinations(canonical_condition_values(), 2):
        first_success = [
            parse_bool(
                condition_rows[first][task.task_id]["succeeded"],
                field_name="succeeded",
            )
            for task in tasks
        ]
        second_success = [
            parse_bool(
                condition_rows[second][task.task_id]["succeeded"],
                field_name="succeeded",
            )
            for task in tasks
        ]
        result = exact_mcnemar_test(
            full_success=first_success,
            baseline_success=second_success,
        )
        output.append(
            {
                "first_condition": first,
                "second_condition": second,
                "paired_tasks": len(tasks),
                "both_success": result.both_success,
                "first_only_success": result.full_only_success,
                "second_only_success": result.baseline_only_success,
                "both_failure": result.both_failure,
                "discordant": result.discordant,
                "exact_p_value": result.p_value,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    """Parse summary arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize a canonical four-condition PatchPilot "
            "research experiment."
        )
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help=(
            "Experiment directory containing one runs.csv under each "
            "canonical condition directory."
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("generated_benchmarks/research_suite.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Summary output directory. Defaults to <experiment-root>/research_summary."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a paired catalog subset for smoke experiments.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate paired runs and write research summary artifacts."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    experiment_root = args.experiment_root.expanduser().resolve(strict=True)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else experiment_root / "research_summary"
    )

    catalog = load_primary_research_catalog(
        project_root,
        args.catalog,
    )
    condition_rows = {
        condition: load_condition_runs(experiment_root, condition)
        for condition in canonical_condition_values()
    }
    tasks = selected_catalog_tasks(
        catalog,
        condition_rows,
        allow_partial=args.allow_partial,
    )

    paired = paired_outcome_rows(tasks, condition_rows)
    summaries = build_summary_rows(tasks, condition_rows)
    comparisons = mcnemar_rows(tasks, condition_rows)

    write_csv_rows(output_root / "paired_outcomes.csv", paired)
    write_csv_rows(output_root / "condition_summary.csv", summaries)
    write_csv_rows(output_root / "pairwise_mcnemar.csv", comparisons)
    write_json(
        output_root / "research_summary.json",
        {
            "schema_version": "1.0",
            "suite_id": catalog.suite_id,
            "catalog_sha256": catalog.catalog_sha256,
            "paired_task_count": len(tasks),
            "complete_catalog": len(tasks) == catalog.task_count,
            "conditions": list(canonical_condition_values()),
            "composition": {
                field: dict(
                    sorted(
                        {
                            value: sum(
                                1
                                for task in tasks
                                if str(getattr(task, field)) == value
                            )
                            for value in {str(getattr(task, field)) for task in tasks}
                        }.items()
                    )
                )
                for field in (
                    "origin_type",
                    "difficulty",
                    "defect_category",
                )
            },
            "condition_summary": summaries,
            "pairwise_mcnemar": comparisons,
        },
    )

    print(f"RESEARCH_SUITE={catalog.suite_id}")
    print(f"PAIRED_TASKS={len(tasks)}")
    print(f"CANONICAL_CONDITIONS={len(canonical_condition_values())}")
    print(f"PAIRWISE_COMPARISONS={len(comparisons)}")
    print(f"SUMMARY_ROOT={output_root}")


if __name__ == "__main__":
    main()
