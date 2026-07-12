"""Run paired PatchPilot evaluation conditions and write metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patchpilot.benchmark import BenchmarkRunner, load_manifest
from patchpilot.evaluation import (
    PRIMARY_CONDITIONS,
    ConfiguredCondition,
    EvaluationCondition,
    RunMetricRow,
    build_condition,
    collect_run_metrics,
    summarise_runs,
)
from patchpilot.evaluation.catalog import (
    BenchmarkCatalog,
    CatalogTask,
    discover_manifest_root,
    load_primary_research_catalog,
    select_catalog_tasks,
)
from patchpilot.evaluation.conditions import (
    VERIFICATION_ABLATION_CONDITIONS,
    all_condition_values,
)
from patchpilot.models import OllamaChatModel

DEFAULT_RESEARCH_CATALOG = "generated_benchmarks/research_suite.json"


def build_run_id(
    condition: str,
    task_id: str,
    experiment_id: str,
) -> str:
    """Build a trace-safe run id with a stable hash when needed."""
    raw = f"{condition}-{task_id}-{experiment_id}"
    if len(raw) <= 100:
        return raw

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    suffix = f"-{digest}"
    return f"{raw[: 100 - len(suffix)].rstrip('-')}{suffix}"


def safe_experiment_id(value: str) -> str:
    """Normalize an experiment id for paths and trace identifiers."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    if len(normalized) < 3:
        raise ValueError(
            "experiment id must contain at least three letters or numbers."
        )
    return normalized[:80].rstrip("-")


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries as one deterministic LF-only CSV file."""
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


def write_json(
    path: Path,
    payload: object,
) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def git_commit(project_root: Path) -> str:
    """Return the evaluated Git commit, or an explicit unknown marker."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def parse_args() -> argparse.Namespace:
    """Parse reproducible experiment parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical PatchPilot conditions on the validated research "
            "catalog or an explicitly selected custom manifest root."
        )
    )
    parser.add_argument(
        "--condition",
        default="all",
        choices=("all", "verification-ablation", *all_condition_values()),
        help=(
            "Condition to run; 'all' runs the four primary conditions and "
            "'verification-ablation' runs the paired reflective-agent arms."
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:3b",
        help="Ollama model name used identically across selected conditions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ollama generation seed.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per model call.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for each Ollama model call.",
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for each bounded test execution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of tasks from the stable catalog order.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help=(
            "Optional exact catalog task ID. Repeat to select several tasks "
            "while retaining catalog order."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/evaluation",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_RESEARCH_CATALOG,
        help=("Validated research catalog. Used unless --manifest-root is provided."),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Explicit custom one-level task directory. This is intended for "
            "sanity or development runs and overrides --catalog."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional stable experiment id. Defaults to the UTC timestamp.",
    )
    return parser.parse_args()


def selected_conditions(
    value: str,
) -> tuple[EvaluationCondition, ...]:
    """Resolve one condition or the complete paired set."""
    if value == "all":
        return PRIMARY_CONDITIONS
    if value == "verification-ablation":
        return VERIFICATION_ABLATION_CONDITIONS
    return (EvaluationCondition(value),)


def build_model(args: argparse.Namespace) -> OllamaChatModel:
    """Create one identically configured model backend."""
    return OllamaChatModel(
        model=args.model,
        timeout_seconds=args.model_timeout_seconds,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


def resolve_catalog(
    project_root: Path,
    args: argparse.Namespace,
) -> BenchmarkCatalog:
    """Resolve the canonical research catalog or one explicit custom root."""
    if args.manifest_root is not None:
        return discover_manifest_root(project_root, args.manifest_root)
    return load_primary_research_catalog(project_root, args.catalog)


def relative_path(project_root: Path, path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(project_root):
        return resolved.relative_to(project_root).as_posix()
    return str(resolved)


def run_condition(
    *,
    project_root: Path,
    result_root: Path,
    tasks: tuple[CatalogTask, ...],
    benchmark_suite: str,
    configured: ConfiguredCondition,
    experiment_id: str,
    args: argparse.Namespace,
) -> list[RunMetricRow]:
    """Run one condition across the shared ordered task list."""
    condition = configured.spec.condition
    condition_root = result_root / condition.value
    runner = BenchmarkRunner(
        project_root=project_root,
        output_root=condition_root,
    )
    rows: list[RunMetricRow] = []
    base_metadata = configured.spec.trace_metadata()

    print(
        f"CONDITION {condition.value}: policy={type(configured.policy).__name__}",
        flush=True,
    )

    for task in tasks:
        run_id = build_run_id(
            condition.value,
            task.task_id,
            experiment_id,
        )
        print(
            f"RUN {condition.value} {task.task_id} -> {run_id}",
            flush=True,
        )
        metadata = {
            **base_metadata,
            "experiment_id": experiment_id,
            "task_id": task.task_id,
            "benchmark_suite": benchmark_suite,
            "origin_type": task.origin_type,
            "difficulty": task.difficulty,
            "defect_category": task.defect_category,
            "manifest_path": relative_path(
                project_root,
                task.manifest_path,
            ),
        }
        run = runner.run(
            manifest_path=task.manifest_path,
            policy=configured.policy,
            run_id=run_id,
            budget=configured.spec.budget,
            metadata=metadata,
            test_timeout_seconds=args.test_timeout_seconds,
            verification_mode=configured.spec.verification_mode,
        )
        row = collect_run_metrics(
            run_id=run_id,
            condition=condition.value,
            state=run.state,
            runtime_verification_mode=configured.spec.verification_mode.value,
            benchmark_suite=benchmark_suite,
            origin_type=task.origin_type,
            difficulty=task.difficulty,
            defect_category=task.defect_category,
        )
        rows.append(row)
        print(
            f"DONE {condition.value} {task.task_id}: "
            f"status={row.status} success={row.succeeded} "
            f"hidden={row.hidden_suite_status} "
            f"steps={row.steps} patches={row.patch_attempts}",
            flush=True,
        )

    run_dicts = [row.model_dump(mode="json") for row in rows]
    summary_dicts = [row.model_dump(mode="json") for row in summarise_runs(rows)]
    write_csv(condition_root / "runs.csv", run_dicts)
    write_csv(condition_root / "summary.csv", summary_dicts)
    write_json(condition_root / "summary.json", summary_dicts)
    return rows


def experiment_configuration(
    *,
    project_root: Path,
    experiment_id: str,
    catalog: BenchmarkCatalog,
    tasks: tuple[CatalogTask, ...],
    conditions: tuple[EvaluationCondition, ...],
    configured: list[ConfiguredCondition],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the complete machine-readable experiment configuration."""
    catalog_path = (
        relative_path(project_root, catalog.catalog_path)
        if catalog.catalog_path is not None
        else None
    )
    selected_composition: dict[str, dict[str, int]] = {}
    for field in ("origin_type", "difficulty", "defect_category"):
        counts: dict[str, int] = {}
        for task in tasks:
            value = str(getattr(task, field))
            counts[value] = counts.get(value, 0) + 1
        selected_composition[field] = dict(sorted(counts.items()))

    return {
        "experiment_id": experiment_id,
        "git_commit": git_commit(project_root),
        "conditions": [item.spec.condition.value for item in configured],
        "condition_specs": {
            item.spec.condition.value: {
                **item.spec.trace_metadata(),
                "budget": item.spec.budget.model_dump(mode="json"),
                "policy_class": (
                    f"{type(item.policy).__module__}.{type(item.policy).__qualname__}"
                ),
            }
            for item in configured
        },
        "condition_order": [condition.value for condition in conditions],
        "model": {
            "backend": "ollama",
            "name": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.model_timeout_seconds,
        },
        "benchmark": {
            "suite_id": catalog.suite_id,
            "catalog_path": catalog_path,
            "catalog_sha256": catalog.catalog_sha256,
            "catalog_task_count": catalog.task_count,
            "selected_task_count": len(tasks),
            "catalog_composition": catalog.composition(),
            "selected_composition": selected_composition,
            "custom_manifest_root": args.manifest_root,
        },
        "hidden_verification": {
            "phase": "post_run_external",
            "output_exposed_to_agent": False,
            "configured_manifest_count": sum(
                1
                for task in tasks
                if load_manifest(task.manifest_path).hidden_test_root is not None
            ),
        },
        "test_timeout_seconds": args.test_timeout_seconds,
        "manifest_count": len(tasks),
        "manifest_order": [
            relative_path(project_root, task.manifest_path) for task in tasks
        ],
        "task_order": [task.task_id for task in tasks],
    }


def main() -> None:
    """Execute the requested paired evaluation."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    experiment_id = safe_experiment_id(args.experiment_id or timestamp)
    result_root = Path(args.output_root).expanduser().resolve() / experiment_id

    catalog = resolve_catalog(project_root, args)
    tasks = select_catalog_tasks(
        catalog,
        task_ids=args.task_id,
        limit=args.limit,
    )
    conditions = selected_conditions(args.condition)
    model = build_model(args)
    print("MODEL_WARMUP=starting")
    model.warmup()
    print("MODEL_WARMUP=complete")
    configured = [build_condition(condition, model) for condition in conditions]

    write_json(
        result_root / "experiment_config.json",
        experiment_configuration(
            project_root=project_root,
            experiment_id=experiment_id,
            catalog=catalog,
            tasks=tasks,
            conditions=conditions,
            configured=configured,
            args=args,
        ),
    )

    all_rows: list[RunMetricRow] = []
    for item in configured:
        all_rows.extend(
            run_condition(
                project_root=project_root,
                result_root=result_root,
                tasks=tasks,
                benchmark_suite=catalog.suite_id,
                configured=item,
                experiment_id=experiment_id,
                args=args,
            )
        )

    run_dicts = [row.model_dump(mode="json") for row in all_rows]
    summary_rows = summarise_runs(all_rows)
    summary_dicts = [row.model_dump(mode="json") for row in summary_rows]
    write_csv(result_root / "runs.csv", run_dicts)
    write_csv(result_root / "summary.csv", summary_dicts)
    write_json(result_root / "summary.json", summary_dicts)

    print(f"BENCHMARK_SUITE={catalog.suite_id}")
    print(f"CATALOG_TASKS={catalog.task_count}")
    print(f"SELECTED_TASKS={len(tasks)}")
    print(f"RESULT_ROOT={result_root}")
    print(f"RUNS_CSV={result_root / 'runs.csv'}")
    print(f"SUMMARY_CSV={result_root / 'summary.csv'}")
    print(json.dumps(summary_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
