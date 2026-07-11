"""Tests for the fixed-workflow one-shot baseline."""

from __future__ import annotations

from pathlib import Path

from patchpilot.agent import OneShotRepairPolicy
from patchpilot.benchmark import BenchmarkRunner
from patchpilot.evaluation import (
    EvaluationCondition,
    get_condition_spec,
)
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


class RepairLineModel:
    """Return the one corrected source line for the controlled task."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt, response_schema
        self.calls += 1
        return "    return left + right"


def make_state() -> AgentState:
    task = RepairTask(
        task_id="one-shot-policy-001",
        goal="Repair the defective add function and verify all tests.",
        repository_root="benchmarks/example",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )
    return AgentState(task=task)


def append_result(
    state: AgentState,
    tool: ToolName,
    status: ObservationStatus,
    *,
    arguments: dict[str, object] | None = None,
    output: str = "",
) -> None:
    state.actions.append(
        ToolAction(
            tool=tool,
            arguments=arguments or {},
            rationale="Advance the one-shot repair workflow.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=tool,
            status=status,
            summary="Recorded one-shot workflow evidence.",
            output=output,
        )
    )


def test_one_shot_requires_syntax_before_full_tests() -> None:
    state = make_state()
    append_result(
        state,
        ToolName.APPLY_PATCH,
        ObservationStatus.OK,
        arguments={"patch_text": "diff --git ..."},
    )

    decision = OneShotRepairPolicy(RepairLineModel()).decide(state)

    assert decision.action.tool is ToolName.CHECK_SYNTAX


def test_one_shot_runs_full_suite_after_syntax_success() -> None:
    state = make_state()
    state.usage.patch_attempts = 1
    append_result(
        state,
        ToolName.CHECK_SYNTAX,
        ObservationStatus.OK,
    )

    decision = OneShotRepairPolicy(RepairLineModel()).decide(state)

    assert decision.action.tool is ToolName.RUN_TESTS
    assert decision.action.arguments == {}


def test_one_shot_stops_after_transactional_rollback() -> None:
    state = make_state()
    state.usage.patch_attempts = 1
    state.last_rolled_back_attempt_id = 1
    state.last_rolled_back_attempt_files = ["src/calculator.py"]
    append_result(
        state,
        ToolName.RESTORE_FILE,
        ObservationStatus.OK,
        arguments={
            "scope": "failed_attempt",
            "attempt_id": 1,
        },
        output="src/calculator.py",
    )

    decision = OneShotRepairPolicy(RepairLineModel()).decide(state)

    assert decision.action.tool is ToolName.FINISH
    assert decision.action.arguments["status"] == "escalated"
    assert "rolled back" in str(decision.action.arguments["message"]).lower()


def test_one_shot_controlled_repair_runs_exactly_one_patch(
    tmp_path: Path,
) -> None:
    model = RepairLineModel()
    policy = OneShotRepairPolicy(model)
    spec = get_condition_spec(EvaluationCondition.ONE_SHOT)
    runner = BenchmarkRunner(
        Path("."),
        tmp_path / "outputs",
    )

    run = runner.run(
        Path("benchmarks/calculator-001/task.json"),
        policy,
        run_id="one-shot-controlled-001",
        budget=spec.budget,
        metadata=spec.trace_metadata(),
    )

    assert run.state.status.value == "succeeded"
    assert run.state.full_suite_passed is True
    assert run.state.usage.patch_attempts == 1
    assert model.calls == 1
    assert [action.tool for action in run.state.actions] == [
        ToolName.RUN_TESTS,
        ToolName.SEARCH_CODE,
        ToolName.READ_FILE,
        ToolName.APPLY_PATCH,
        ToolName.CHECK_SYNTAX,
        ToolName.RUN_TESTS,
        ToolName.FINISH,
    ]
