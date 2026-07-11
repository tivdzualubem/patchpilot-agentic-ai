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
    condition_values,
    summarise_runs,
)
from patchpilot.models import OllamaChatModel


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


def benchmark_manifests(
    project_root: Path,
    manifest_root: str,
) -> list[Path]:
    """Return all benchmark manifests in stable order."""
    root = Path(manifest_root)
    if not root.is_absolute():
        root = project_root / root
    return sorted(root.glob("*/task.json"))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries as one deterministic CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
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
        description=("Run one or all four paired PatchPilot evaluation conditions.")
    )
    parser.add_argument(
        "--condition",
        default="all",
        choices=("all", *condition_values()),
        help="Condition to run; 'all' runs the complete paired experiment.",
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
        help="Optional number of benchmark tasks in stable order.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/evaluation",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--manifest-root",
        default="benchmarks",
        help="Directory containing benchmark task folders.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help=("Optional stable experiment id. Defaults to the current UTC timestamp."),
    )
    return parser.parse_args()


def selected_conditions(
    value: str,
) -> tuple[EvaluationCondition, ...]:
    """Resolve one condition or the complete paired set."""
    if value == "all":
        return PRIMARY_CONDITIONS
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


def run_condition(
    *,
    project_root: Path,
    result_root: Path,
    manifests: list[Path],
    configured: ConfiguredCondition,
    experiment_id: str,
    args: argparse.Namespace,
) -> list[RunMetricRow]:
    """Run one condition across the shared ordered manifest list."""
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

    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        run_id = build_run_id(
            condition.value,
            manifest.task_id,
            experiment_id,
        )
        print(
            f"RUN {condition.value} {manifest.task_id} -> {run_id}",
            flush=True,
        )
        metadata = {
            **base_metadata,
            "experiment_id": experiment_id,
            "task_id": manifest.task_id,
        }
        run = runner.run(
            manifest_path=manifest_path,
            policy=configured.policy,
            run_id=run_id,
            budget=configured.spec.budget,
            metadata=metadata,
            test_timeout_seconds=args.test_timeout_seconds,
        )
        row = collect_run_metrics(
            run_id=run_id,
            condition=condition.value,
            state=run.state,
        )
        rows.append(row)
        print(
            f"DONE {condition.value} {manifest.task_id}: "
            f"status={row.status} success={row.succeeded} "
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
    manifests: list[Path],
    conditions: tuple[EvaluationCondition, ...],
    configured: list[ConfiguredCondition],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the complete machine-readable experiment configuration."""
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
        "test_timeout_seconds": args.test_timeout_seconds,
        "manifest_root": args.manifest_root,
        "manifest_count": len(manifests),
        "manifest_order": [
            str(path.relative_to(project_root))
            if path.is_relative_to(project_root)
            else str(path)
            for path in manifests
        ],
    }


def main() -> None:
    """Execute the requested paired evaluation."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    experiment_id = safe_experiment_id(args.experiment_id or timestamp)
    result_root = Path(args.output_root).expanduser().resolve() / experiment_id

    manifests = benchmark_manifests(
        project_root,
        args.manifest_root,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        manifests = manifests[: args.limit]
    if not manifests:
        raise FileNotFoundError("No benchmark manifests were found for evaluation.")

    conditions = selected_conditions(args.condition)
    configured = [
        build_condition(condition, build_model(args)) for condition in conditions
    ]

    write_json(
        result_root / "experiment_config.json",
        experiment_configuration(
            project_root=project_root,
            experiment_id=experiment_id,
            manifests=manifests,
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
                manifests=manifests,
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

    print(f"RESULT_ROOT={result_root}")
    print(f"RUNS_CSV={result_root / 'runs.csv'}")
    print(f"SUMMARY_CSV={result_root / 'summary.csv'}")
    print(json.dumps(summary_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
