"""End-to-end execution of PatchPilot benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from patchpilot.agent import (
    AgentControlLoop,
    AgentPolicy,
    AgentToolExecutor,
    TraceRecorder,
)
from patchpilot.benchmark.workspace import (
    BenchmarkWorkspace,
    PreparedBenchmark,
)
from patchpilot.schemas import AgentState, ExecutionBudget


@dataclass(frozen=True)
class BenchmarkRun:
    """Artifacts produced by one benchmark execution."""

    run_id: str
    prepared: PreparedBenchmark
    state: AgentState
    trace_path: Path


class BenchmarkRunner:
    """Run a policy against an isolated, validated benchmark."""

    def __init__(
        self,
        project_root: Path,
        output_root: Path,
    ) -> None:
        self.project_root = project_root.expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        self.workspace_manager = BenchmarkWorkspace(
            self.project_root,
            self.output_root / "workspaces",
        )
        self.trace_recorder = TraceRecorder(
            self.output_root / "traces"
        )

    def run(
        self,
        manifest_path: Path,
        policy: AgentPolicy,
        run_id: str,
        budget: ExecutionBudget | None = None,
        metadata: dict[str, str] | None = None,
        test_timeout_seconds: int = 60,
    ) -> BenchmarkRun:
        """Execute one complete bounded benchmark run."""
        resolved_manifest = (
            manifest_path
            if manifest_path.is_absolute()
            else self.project_root / manifest_path
        )

        prepared = self.workspace_manager.prepare(
            resolved_manifest
        )
        state = AgentState(
            task=prepared.task,
            budget=budget or ExecutionBudget(),
        )
        executor = AgentToolExecutor(
            prepared.workspace_root,
            prepared.task,
            test_timeout_seconds=test_timeout_seconds,
        )
        loop = AgentControlLoop(
            policy=policy,
            executor=executor,
            recorder=self.trace_recorder,
        )

        final_state = loop.run(
            state,
            run_id=run_id,
            metadata=metadata,
        )

        return BenchmarkRun(
            run_id=run_id,
            prepared=prepared,
            state=final_state,
            trace_path=(
                self.trace_recorder.output_directory
                / f"{run_id}.json"
            ),
        )

    def cleanup(self, run: BenchmarkRun) -> None:
        """Remove the disposable workspace for a completed run."""
        self.workspace_manager.cleanup(run.prepared)
