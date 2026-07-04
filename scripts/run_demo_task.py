"""Run one PatchPilot benchmark task for the Streamlit demo."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from patchpilot.agent.llm_policy import StructuredLLMPolicy
from patchpilot.benchmark import BenchmarkRunner, load_manifest
from patchpilot.models.ollama import OllamaChatModel
from patchpilot.schemas import ExecutionBudget


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one PatchPilot demo repair task."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument(
        "--output-root",
        default="artifacts/demo",
    )
    return parser.parse_args()


def main() -> None:
    """Run one selected benchmark through the live PatchPilot agent."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)
    manifest_path = project_root / "benchmarks" / args.task_id / "task.json"
    manifest = load_manifest(manifest_path)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"demo-{manifest.task_id}-{timestamp}"

    runner = BenchmarkRunner(
        project_root=project_root,
        output_root=Path(args.output_root).resolve() / timestamp,
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

    run = runner.run(
        manifest_path=manifest_path,
        policy=policy,
        run_id=run_id,
        budget=budget,
        metadata={
            "condition": "interactive-demo",
            "model": args.model,
            "task_id": manifest.task_id,
        },
        test_timeout_seconds=60,
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": manifest.task_id,
                "status": run.state.status.value,
                "succeeded": run.state.status.value == "succeeded",
                "full_suite_passed": run.state.full_suite_passed,
                "steps": run.state.usage.steps,
                "tool_calls": run.state.usage.tool_calls,
                "patch_attempts": run.state.usage.patch_attempts,
                "changed_files": run.state.changed_files,
                "final_message": run.state.final_message,
                "workspace": str(run.prepared.workspace_root),
                "repository": str(run.prepared.repository_root),
                "trace": str(run.trace_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
