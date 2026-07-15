"""Tests for the paired runtime-verification ablation."""

from __future__ import annotations

from pathlib import Path

from patchpilot.agent import AgentDecision, AgentToolExecutor
from patchpilot.agent.executor import VerificationMode
from patchpilot.benchmark import BenchmarkRunner
from patchpilot.evaluation import collect_run_metrics, summarise_runs
from patchpilot.evaluation.conditions import (
    VERIFICATION_ABLATION_CONDITIONS,
    EvaluationCondition,
    build_condition,
    get_condition_spec,
)
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
)


class NoCallModel:
    """Model stub used only to construct condition policies."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt, response_schema
        raise AssertionError("model should not be called")


def make_executor(
    tmp_path: Path,
    mode: VerificationMode,
) -> tuple[AgentToolExecutor, AgentState]:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src" / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_calculator.py").write_text(
        "from src.calculator import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    task = RepairTask(
        task_id="verification-ablation-001",
        goal="Repair the incorrect calculator addition operation.",
        repository_root="repository",
    )
    return (
        AgentToolExecutor(
            tmp_path,
            task,
            verification_mode=mode,
        ),
        AgentState(task=task),
    )


def repair_patch() -> str:
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(left: int, right: int) -> int:\n"
        "-    return left - right\n"
        "+    return left + right\n"
    )


def action(
    tool: ToolName,
    arguments: dict[str, object] | None = None,
) -> ToolAction:
    return ToolAction(
        tool=tool,
        arguments=arguments or {},
        rationale="Exercise the runtime verification ablation.",
    )


def test_strict_mode_blocks_tests_until_syntax_check(tmp_path: Path) -> None:
    executor, state = make_executor(tmp_path, VerificationMode.STRICT)
    executor.execute(
        state,
        action(ToolName.APPLY_PATCH, {"patch_text": repair_patch()}),
    )

    result = executor.execute(state, action(ToolName.RUN_TESTS))

    assert result.status is ObservationStatus.REJECTED
    assert "syntax check is required" in result.summary


def test_disabled_mode_allows_tests_without_syntax_gate(tmp_path: Path) -> None:
    executor, state = make_executor(tmp_path, VerificationMode.DISABLED)
    executor.execute(
        state,
        action(ToolName.APPLY_PATCH, {"patch_text": repair_patch()}),
    )

    result = executor.execute(state, action(ToolName.RUN_TESTS))

    assert result.status is ObservationStatus.OK
    assert state.full_suite_passed is True
    assert state.current_attempt_id is None


def test_disabled_mode_can_record_unverified_success(tmp_path: Path) -> None:
    executor, state = make_executor(tmp_path, VerificationMode.DISABLED)
    executor.execute(
        state,
        action(ToolName.APPLY_PATCH, {"patch_text": repair_patch()}),
    )

    result = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "succeeded",
                "message": "The policy claims the patch is complete.",
            },
        ),
    )

    assert result.status is ObservationStatus.OK
    assert state.status is AgentStatus.SUCCEEDED
    assert state.full_suite_passed is False
    assert state.verified_revision is None
    assert state.current_attempt_id is None


def test_verification_ablation_changes_only_runtime_enforcement() -> None:
    strict_condition, disabled_condition = VERIFICATION_ABLATION_CONDITIONS
    strict = get_condition_spec(strict_condition)
    disabled = get_condition_spec(disabled_condition)

    assert strict_condition is EvaluationCondition.FULL_REFLECTIVE_AGENT
    assert disabled_condition is (
        EvaluationCondition.FULL_REFLECTIVE_AGENT_NO_RUNTIME_VERIFICATION
    )
    assert strict.budget == disabled.budget
    assert strict.model_selects_tools == disabled.model_selects_tools
    assert strict.reflection_enabled == disabled.reflection_enabled
    assert strict.retry_enabled == disabled.retry_enabled
    assert strict.verification_mode is VerificationMode.STRICT
    assert disabled.verification_mode is VerificationMode.DISABLED
    assert type(build_condition(strict_condition, NoCallModel()).policy) is type(
        build_condition(disabled_condition, NoCallModel()).policy
    )


class UnverifiedFinishPolicy:
    """Apply one patch and claim success without running verification."""

    def decide(self, state: AgentState) -> AgentDecision:
        if not state.actions:
            selected = action(
                ToolName.APPLY_PATCH,
                {
                    "patch_text": (
                        "diff --git a/src/calculator.py b/src/calculator.py\n"
                        "--- a/src/calculator.py\n"
                        "+++ b/src/calculator.py\n"
                        "@@ -4,3 +4,3 @@\n"
                        " def add(left: int, right: int) -> int:\n"
                        '     """Return the sum of two integers."""\n'
                        "-    return left - right\n"
                        "+    return left + right\n"
                    )
                },
            )
        else:
            selected = action(
                ToolName.FINISH,
                {
                    "status": "succeeded",
                    "message": "Claim success without runtime verification.",
                },
            )

        return AgentDecision(
            reasoning_summary="Execute the verification-ablation test step.",
            action=selected,
        )


def test_runner_records_disabled_verification_mode(tmp_path: Path) -> None:
    runner = BenchmarkRunner(Path("."), tmp_path / "outputs")
    run = runner.run(
        Path("benchmarks/calculator-001/task.json"),
        UnverifiedFinishPolicy(),
        run_id="verification-disabled-run",
        verification_mode=VerificationMode.DISABLED,
    )

    trace = runner.trace_recorder.load("verification-disabled-run")

    assert run.state.status is AgentStatus.SUCCEEDED
    assert run.state.full_suite_passed is False
    assert trace.metadata["runtime_verification_mode"] == "disabled"


def test_metrics_distinguish_verified_and_unverified_success() -> None:
    task = RepairTask(
        task_id="verification-metrics-001",
        goal="Measure an unverified policy success claim.",
        repository_root="repository",
    )
    state = AgentState(task=task)
    state.status = AgentStatus.SUCCEEDED
    state.repository_revision = 1

    row = collect_run_metrics(
        run_id="verification-metrics-run",
        condition="full-reflective-agent-no-runtime-verification",
        state=state,
        runtime_verification_mode="disabled",
    )
    summary = summarise_runs([row])[0]

    assert row.succeeded is True
    assert row.verified_success is False
    assert row.unverified_success is True
    assert row.runtime_verification_mode == "disabled"
    assert summary.verified_successes == 0
    assert summary.unverified_successes == 1
    assert summary.verified_repair_rate == 0.0
    assert summary.unverified_success_rate == 1.0
