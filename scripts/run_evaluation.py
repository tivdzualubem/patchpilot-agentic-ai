"""Run PatchPilot benchmark evaluations and write CSV/JSON metrics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from patchpilot.agent.llm_policy import StructuredLLMPolicy
from patchpilot.benchmark import BenchmarkRunner, load_manifest
from patchpilot.evaluation import collect_run_metrics, summarise_runs
from patchpilot.models.ollama import OllamaChatModel
from patchpilot.schemas import ExecutionBudget


def benchmark_manifests(project_root: Path) -> list[Path]:
    """Return all benchmark manifests in stable order."""
    return sorted((project_root / "benchmarks").glob("*/task.json"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PatchPilot benchmark evaluation."
    )
    parser.add_argument(
        "--condition",
        default="full-agent-live-qwen",
        choices=["full-agent-live-qwen"],
        help="Evaluation condition to run.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:1.5b",
        help="Ollama model name for the live full-agent condition.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of benchmark tasks to run.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/evaluation",
        help="Directory for workspaces, traces, and metric files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    result_root = output_root / timestamp

    manifests = benchmark_manifests(project_root)
    if args.limit is not None:
        manifests = manifests[: args.limit]

    runner = BenchmarkRunner(
        project_root=project_root,
        output_root=result_root,
    )
    policy = StructuredLLMPolicy(
        OllamaChatModel(
            model=args.model,
            timeout_seconds=300,
            temperature=0.0,
            seed=42,
        )
    )
    budget = ExecutionBudget(
        max_steps=10,
        max_tool_calls=10,
        max_patch_attempts=3,
        max_seconds=1800,
    )

    run_rows = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        run_id = f"{args.condition}-{manifest.task_id}-{timestamp}"
        print(f"RUN {manifest.task_id} -> {run_id}", flush=True)
        run = runner.run(
            manifest_path=manifest_path,
            policy=policy,
            run_id=run_id,
            budget=budget,
            metadata={
                "condition": args.condition,
                "model": args.model,
                "task_id": manifest.task_id,
            },
            test_timeout_seconds=60,
        )
        row = collect_run_metrics(
            run_id=run_id,
            condition=args.condition,
            state=run.state,
        )
        run_rows.append(row)
        print(
            f"DONE {manifest.task_id}: "
            f"status={row.status} "
            f"success={row.succeeded} "
            f"steps={row.steps} "
            f"patches={row.patch_attempts}",
            flush=True,
        )

    summary_rows = summarise_runs(run_rows)
    run_dicts = [row.model_dump(mode="json") for row in run_rows]
    summary_dicts = [row.model_dump(mode="json") for row in summary_rows]

    write_csv(result_root / "runs.csv", run_dicts)
    write_csv(result_root / "summary.csv", summary_dicts)
    (result_root / "summary.json").write_text(
        json.dumps(summary_dicts, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"RESULT_ROOT={result_root}")
    print(f"RUNS_CSV={result_root / 'runs.csv'}")
    print(f"SUMMARY_CSV={result_root / 'summary.csv'}")
    print(json.dumps(summary_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
