"""Run PatchPilot on a small external QuixBugs Python smoke benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from patchpilot.agent.llm_policy import StructuredLLMPolicy
from patchpilot.benchmark import BenchmarkRunner, load_manifest
from patchpilot.evaluation import collect_run_metrics, summarise_runs
from patchpilot.models.ollama import OllamaChatModel
from patchpilot.schemas import ExecutionBudget

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "external_benchmarks" / "quixbugs"
SOURCE = EXTERNAL_ROOT / "source"
TASKS_ROOT = EXTERNAL_ROOT / "tasks"

DEFAULT_TASKS = [
    "gcd",
    "get_factors",
    "is_valid_parenthesization",
    "max_sublist_sum",
    "mergesort",
    "quicksort",
    "sieve",
    "to_base",
]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_quixbugs_source() -> None:
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE.exists():
        return

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/jkoppel/QuixBugs.git",
            str(SOURCE),
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_manifests(tasks: list[str]) -> list[Path]:
    TASKS_ROOT.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []

    for name in tasks:
        program = SOURCE / "python_programs" / f"{name}.py"
        testcase = SOURCE / "python_testcases" / f"test_{name}.py"

        if not program.exists():
            raise FileNotFoundError(program)
        if not testcase.exists():
            raise FileNotFoundError(testcase)

        task_dir = TASKS_ROOT / name
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "task_id": f"quixbugs-{name.replace('_', '-')}",
            "title": f"QuixBugs external repair task: {name}",
            "goal": (
                f"Repair the QuixBugs Python program {name}.py so that "
                f"python_testcases/test_{name}.py passes without modifying tests."
            ),
            "repository_root": "external_benchmarks/quixbugs/source",
            "test_command": [
                "python",
                "-m",
                "pytest",
                "-q",
                f"python_testcases/test_{name}.py",
            ],
            "allowed_paths": ["python_programs"],
            "forbidden_paths": [
                "python_testcases",
                "correct_python_programs",
                "json_testcases",
            ],
            "defect_category": "external-public-benchmark",
            "difficulty": "medium",
            "expected_initial_failures": 1,
        }
        manifest_path = task_dir / "task.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifests.append(manifest_path)

    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--clean-source",
        action="store_true",
        help="Remove and reclone QuixBugs before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean_source and SOURCE.exists():
        shutil.rmtree(SOURCE)

    ensure_quixbugs_source()

    selected = DEFAULT_TASKS if args.limit is None else DEFAULT_TASKS[: args.limit]
    manifests = create_manifests(selected)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    result_root = ROOT / "artifacts" / "external" / "quixbugs" / timestamp

    runner = BenchmarkRunner(project_root=ROOT, output_root=result_root)
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

    rows = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        run_id = f"quixbugs-full-agent-{manifest.task_id}-{timestamp}"
        print(f"RUN {manifest.task_id}", flush=True)
        run = runner.run(
            manifest_path=manifest_path,
            policy=policy,
            run_id=run_id,
            budget=budget,
            metadata={
                "condition": "quixbugs-full-agent-live-qwen",
                "model": args.model,
                "task_id": manifest.task_id,
                "benchmark": "QuixBugs",
            },
            test_timeout_seconds=60,
        )
        row = collect_run_metrics(
            run_id=run_id,
            condition="quixbugs-full-agent-live-qwen",
            state=run.state,
        )
        rows.append(row)
        print(
            f"DONE {manifest.task_id}: status={row.status} "
            f"success={row.succeeded} steps={row.steps} patches={row.patch_attempts}",
            flush=True,
        )

    run_dicts = [row.model_dump(mode="json") for row in rows]
    summary_dicts = [row.model_dump(mode="json") for row in summarise_runs(rows)]

    write_csv(result_root / "runs.csv", run_dicts)
    write_csv(result_root / "summary.csv", summary_dicts)
    (result_root / "summary.json").write_text(
        json.dumps(summary_dicts, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"RESULT_ROOT={result_root}")
    print(json.dumps(summary_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
