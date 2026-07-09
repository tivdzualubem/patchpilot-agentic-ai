"""Run PatchPilot benchmark evaluations and write CSV/JSON metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from patchpilot.agent.llm_policy import PolicyResponseError, StructuredLLMPolicy
from patchpilot.benchmark import BenchmarkRunner, load_manifest
from patchpilot.evaluation import collect_run_metrics, summarise_runs
from patchpilot.models.ollama import OllamaChatModel
from patchpilot.schemas import AgentState, ExecutionBudget, ObservationStatus, ToolName

EVALUATION_CONDITIONS = [
    "full-agent-live-qwen",
    "no-retry-live-qwen",
    "one-shot-live-qwen",
]


class OneShotRepairPolicy(StructuredLLMPolicy):
    """Single-patch baseline with no repair retry after failed verification."""

    @staticmethod
    def _finish(status: str, message: str):
        return StructuredLLMPolicy._make_decision(
            summary=f"Finish one-shot baseline with status {status}.",
            plan="Stop after the single allowed repair attempt.",
            tool=ToolName.FINISH,
            arguments={"status": status, "message": message},
            rationale="The one-shot baseline does not perform iterative repair.",
        )

    def decide(self, state: AgentState):
        """Run tests, inspect once, patch once, verify once, then stop."""
        if not state.actions or not state.observations:
            return self._make_decision(
                summary="Run the full test suite before one-shot repair.",
                plan="Reproduce the failing test signal.",
                tool=ToolName.RUN_TESTS,
                rationale="Establish the baseline failure signal first.",
            )

        last_action = state.actions[-1]
        last_observation = state.observations[-1]

        if last_action.tool is ToolName.RUN_TESTS:
            if last_observation.status is ObservationStatus.OK:
                return self._finish(
                    "succeeded",
                    "Full test suite passed after one-shot repair.",
                )

            if state.changed_files:
                return self._finish(
                    "escalated",
                    "One-shot patch did not pass full-suite verification.",
                )

            return self._make_decision(
                summary="Search source code for the failing symbol.",
                plan="Locate the implementation connected to the failing tests.",
                tool=ToolName.SEARCH_CODE,
                arguments={
                    "query": self._failure_query(state),
                    "relative_path": state.task.allowed_paths[0],
                },
                rationale="Find the likely defective source implementation.",
            )

        if last_action.tool is ToolName.SEARCH_CODE:
            if last_observation.status is not ObservationStatus.OK:
                return self._finish(
                    "escalated",
                    "One-shot baseline could not locate relevant source code.",
                )

            try:
                relative_path = self._source_path_from_search(
                    last_observation.output,
                    state.task.allowed_paths,
                )
            except PolicyResponseError:
                return self._finish(
                    "escalated",
                    "One-shot baseline could not extract a source path.",
                )

            return self._make_decision(
                summary="Read the source file found by search.",
                plan="Inspect the candidate defective implementation.",
                tool=ToolName.READ_FILE,
                arguments={"relative_path": relative_path},
                rationale="Read the source before generating a single patch.",
            )

        if last_action.tool is ToolName.READ_FILE:
            if last_observation.status is not ObservationStatus.OK:
                return self._finish(
                    "escalated",
                    "One-shot baseline could not read the source file.",
                )

            if state.usage.patch_attempts >= 1:
                return self._finish(
                    "escalated",
                    "One-shot patch budget was already used.",
                )

            return self._generate_patch_decision(state)

        if last_action.tool is ToolName.APPLY_PATCH:
            if last_observation.status is ObservationStatus.OK:
                return self._make_decision(
                    summary="Verify the single one-shot patch.",
                    plan="Run the full test suite once after patching.",
                    tool=ToolName.RUN_TESTS,
                    rationale="Check whether the one-shot patch repaired the task.",
                )

            return self._finish(
                "escalated",
                "One-shot patch was rejected by the patch boundary.",
            )

        return self._finish(
            "escalated",
            "One-shot baseline reached an unsupported state.",
        )



def build_run_id(condition: str, task_id: str, timestamp: str) -> str:
    """Build a trace-safe run id with a stable hash when it is too long."""
    raw = f"{condition}-{task_id}-{timestamp}"
    if len(raw) <= 100:
        return raw

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    suffix = f"-{digest}-{timestamp}"
    prefix = f"{condition}-{task_id}"
    keep = 100 - len(suffix)
    return f"{prefix[:keep].rstrip('-')}{suffix}"

def budget_for_condition(condition: str) -> ExecutionBudget:
    """Return the execution budget for one evaluation condition."""
    if condition in {"no-retry-live-qwen", "one-shot-live-qwen"}:
        return ExecutionBudget(
            max_steps=6,
            max_tool_calls=6,
            max_patch_attempts=1,
            max_seconds=1800,
        )

    return ExecutionBudget(
        max_steps=10,
        max_tool_calls=10,
        max_patch_attempts=3,
        max_seconds=1800,
    )


def benchmark_manifests(project_root: Path, manifest_root: str) -> list[Path]:
    """Return all benchmark manifests in stable order."""
    root = Path(manifest_root)
    if not root.is_absolute():
        root = project_root / root
    return sorted(root.glob("*/task.json"))


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
        choices=EVALUATION_CONDITIONS,
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
    parser.add_argument(
        "--manifest-root",
        default="benchmarks",
        help="Directory containing benchmark task folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    result_root = output_root / timestamp

    manifests = benchmark_manifests(project_root, args.manifest_root)
    if args.limit is not None:
        manifests = manifests[: args.limit]

    runner = BenchmarkRunner(
        project_root=project_root,
        output_root=result_root,
    )
    model = OllamaChatModel(
        model=args.model,
        timeout_seconds=300,
        temperature=0.0,
        seed=42,
    )
    if args.condition == "one-shot-live-qwen":
        policy = OneShotRepairPolicy(model)
    else:
        policy = StructuredLLMPolicy(model)
    budget = budget_for_condition(args.condition)

    run_rows = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        run_id = build_run_id(args.condition, manifest.task_id, timestamp)
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
