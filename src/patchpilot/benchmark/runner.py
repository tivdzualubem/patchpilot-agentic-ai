"""End-to-end execution of PatchPilot benchmark tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from patchpilot.agent import (
    AgentControlLoop,
    AgentPolicy,
    AgentToolExecutor,
    TraceRecorder,
)
from patchpilot.agent.executor import VerificationMode
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
    trace_event_path: Path


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
        self.trace_recorder = TraceRecorder(self.output_root / "traces")

    @staticmethod
    def _qualified_name(value: object) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"

    @classmethod
    def _trace_metadata(
        cls,
        policy: AgentPolicy,
        metadata: dict[str, str] | None,
        test_timeout_seconds: int,
        verification_mode: VerificationMode,
    ) -> dict[str, str]:
        result = dict(metadata or {})
        result.update(
            {
                "policy_class": cls._qualified_name(policy),
                "test_timeout_seconds": str(test_timeout_seconds),
                "runtime_verification_mode": verification_mode.value,
                "trace_schema_version": "2.0",
            }
        )

        model = getattr(policy, "model", None)
        provider = getattr(model, "trace_metadata", None)
        if callable(provider):
            model_metadata = provider()
            if not isinstance(model_metadata, dict):
                raise TypeError("Model trace_metadata() must return a dictionary.")

            for key, value in sorted(model_metadata.items()):
                result[f"model_{key}"] = (
                    value
                    if isinstance(value, str)
                    else json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )

        return result

    def run(
        self,
        manifest_path: Path,
        policy: AgentPolicy,
        run_id: str,
        budget: ExecutionBudget | None = None,
        metadata: dict[str, str] | None = None,
        test_timeout_seconds: int = 60,
        verification_mode: VerificationMode = VerificationMode.STRICT,
    ) -> BenchmarkRun:
        """Execute one complete bounded benchmark run."""
        resolved_manifest = (
            manifest_path
            if manifest_path.is_absolute()
            else self.project_root / manifest_path
        )

        prepared = self.workspace_manager.prepare(resolved_manifest)
        state = AgentState(
            task=prepared.task,
            budget=budget or ExecutionBudget(),
        )
        executor = AgentToolExecutor(
            prepared.workspace_root,
            prepared.task,
            test_timeout_seconds=test_timeout_seconds,
            verification_mode=verification_mode,
        )
        loop = AgentControlLoop(
            policy=policy,
            executor=executor,
            recorder=self.trace_recorder,
        )

        trace_metadata = self._trace_metadata(
            policy,
            metadata,
            test_timeout_seconds,
            verification_mode,
        )
        final_state = loop.run(
            state,
            run_id=run_id,
            metadata=trace_metadata,
        )

        return BenchmarkRun(
            run_id=run_id,
            prepared=prepared,
            state=final_state,
            trace_path=self.trace_recorder.snapshot_path(run_id),
            trace_event_path=self.trace_recorder.event_log_path(run_id),
        )

    def cleanup(self, run: BenchmarkRun) -> None:
        """Remove the disposable workspace for a completed run."""
        self.workspace_manager.cleanup(run.prepared)
