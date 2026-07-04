"""Run Qwen through the complete PatchPilot repair workflow."""

from pathlib import Path

from patchpilot.agent import StructuredLLMPolicy
from patchpilot.benchmark import BenchmarkRunner
from patchpilot.models import OllamaChatModel
from patchpilot.schemas import ExecutionBudget


def main() -> None:
    project_root = Path.cwd()

    runner = BenchmarkRunner(
        project_root=project_root,
        output_root=project_root / "artifacts" / "live-runs",
    )
    policy = StructuredLLMPolicy(
        OllamaChatModel(
            model="qwen2.5-coder:1.5b",
            timeout_seconds=300,
            temperature=0.0,
            seed=42,
        )
    )

    run = runner.run(
        manifest_path=Path(
            "benchmarks/calculator-001/task.json"
        ),
        policy=policy,
        run_id="qwen-calculator-005",
        budget=ExecutionBudget(
            max_steps=10,
            max_tool_calls=10,
            max_patch_attempts=3,
            max_seconds=1800,
        ),
        metadata={
            "model": "qwen2.5-coder:1.5b",
            "mode": "live-agent",
        },
    )

    print(f"STATUS={run.state.status.value}")
    print(f"STEPS={run.state.usage.steps}")
    print(f"PATCH_ATTEMPTS={run.state.usage.patch_attempts}")
    print(f"CHANGED_FILES={run.state.changed_files}")
    print(f"FINAL_MESSAGE={run.state.final_message}")
    print(f"WORKSPACE={run.prepared.workspace_root}")
    print(f"TRACE={run.trace_path}")

    for number, (action, observation) in enumerate(
        zip(run.state.actions, run.state.observations, strict=True),
        start=1,
    ):
        print(
            f"STEP_{number}="
            f"{action.tool.value}|"
            f"{observation.status.value}|"
            f"{observation.summary}"
        )


if __name__ == "__main__":
    main()
