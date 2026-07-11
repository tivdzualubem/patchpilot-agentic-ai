"""Tests for the audited agent tool executor."""

from pathlib import Path

import pytest

from patchpilot.agent import AgentToolExecutor
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    ExecutionBudget,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
)


@pytest.fixture()
def executor_state(
    tmp_path: Path,
) -> tuple[AgentToolExecutor, AgentState]:
    """Create one defective calculator task."""
    repository = tmp_path / "benchmarks" / "calculator"
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
        task_id="executor-calculator-001",
        goal="Repair the incorrect calculator addition operation.",
        repository_root="benchmarks/calculator",
    )

    return (
        AgentToolExecutor(tmp_path, task),
        AgentState(task=task),
    )


def repair_patch() -> str:
    """Return the exact valid calculator repair patch."""
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
    """Construct one valid agent action."""
    return ToolAction(
        tool=tool,
        arguments=arguments or {},
        rationale="This action advances the repair investigation.",
    )


def test_action_is_audited_and_budgeted(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(ToolName.LIST_FILES),
    )

    assert result.status is ObservationStatus.OK
    assert state.status is AgentStatus.RUNNING
    assert state.usage.steps == 1
    assert state.usage.tool_calls == 1
    assert len(state.actions) == 1
    assert len(state.observations) == 1


def test_invalid_arguments_are_rejected(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(
            ToolName.READ_FILE,
            {"unexpected": "value"},
        ),
    )

    assert result.status is ObservationStatus.REJECTED
    assert "Invalid tool arguments" in result.summary


def test_patch_updates_revision_and_invalidates_verification(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state
    state.full_suite_passed = True
    state.verified_revision = 0

    result = executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )

    assert result.status is ObservationStatus.OK
    assert state.repository_revision == 1
    assert state.full_suite_passed is False
    assert state.verified_revision is None
    assert state.changed_files == ["src/calculator.py"]


def test_success_requires_full_suite_verification(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "succeeded",
                "message": "The repair is complete.",
            },
        ),
    )

    assert result.status is ObservationStatus.REJECTED
    assert state.status is AgentStatus.READY


def test_targeted_test_does_not_authorize_success(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state
    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )

    test_result = executor.execute(
        state,
        action(
            ToolName.RUN_TESTS,
            {"target": ("tests/test_calculator.py::test_add")},
        ),
    )
    finish_result = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "succeeded",
                "message": "The targeted test passed.",
            },
        ),
    )

    assert test_result.status is ObservationStatus.OK
    assert finish_result.status is ObservationStatus.REJECTED


def test_verified_current_revision_can_finish_successfully(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )
    tests = executor.execute(
        state,
        action(ToolName.RUN_TESTS),
    )
    finished = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "succeeded",
                "message": "All regression tests pass.",
            },
        ),
    )

    assert tests.status is ObservationStatus.OK
    assert state.verified_revision == 1
    assert finished.status is ObservationStatus.OK
    assert state.status is AgentStatus.SUCCEEDED


def test_patch_limit_still_allows_verification(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, original_state = executor_state
    state = AgentState(
        task=original_state.task,
        budget=ExecutionBudget(max_patch_attempts=1),
    )

    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )

    blocked_patch = executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )
    verification = executor.execute(
        state,
        action(ToolName.RUN_TESTS),
    )

    assert blocked_patch.status is ObservationStatus.REJECTED
    assert verification.status is ObservationStatus.OK
    assert state.full_suite_passed is True


def test_repository_mutation_after_verification_blocks_success(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )
    executor.execute(
        state,
        action(ToolName.RUN_TESTS),
    )
    executor.execute(
        state,
        action(
            ToolName.RESTORE_FILE,
            {"relative_path": "src/calculator.py"},
        ),
    )

    result = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "succeeded",
                "message": "Attempting stale verification.",
            },
        ),
    )

    assert result.status is ObservationStatus.REJECTED
    assert state.full_suite_passed is False


def test_global_budget_blocks_additional_tool_calls(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, original_state = executor_state
    state = AgentState(
        task=original_state.task,
        budget=ExecutionBudget(
            max_steps=1,
            max_tool_calls=1,
        ),
    )

    executor.execute(
        state,
        action(ToolName.LIST_FILES),
    )
    result = executor.execute(
        state,
        action(ToolName.LIST_FILES),
    )

    assert result.status is ObservationStatus.REJECTED
    assert state.status is AgentStatus.BUDGET_EXHAUSTED


def test_failed_finish_records_terminal_status(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "failed",
                "message": "No valid repair was found.",
            },
        ),
    )

    assert result.status is ObservationStatus.OK
    assert state.status is AgentStatus.FAILED
    assert state.final_message == "No valid repair was found."


def test_terminal_state_rejects_later_actions(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    executor.execute(
        state,
        action(
            ToolName.FINISH,
            {
                "status": "escalated",
                "message": "Human review is required.",
            },
        ),
    )
    result = executor.execute(
        state,
        action(ToolName.LIST_FILES),
    )

    assert result.status is ObservationStatus.REJECTED
    assert state.status is AgentStatus.ESCALATED


def test_restore_without_changes_is_rejected(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(ToolName.RESTORE_FILE),
    )

    assert result.status is ObservationStatus.REJECTED
    assert "No changed files" in result.summary
    assert state.repository_revision == 0


def test_repeated_identical_action_is_rejected(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    first = executor.execute(state, action(ToolName.LIST_FILES))
    repeated = executor.execute(state, action(ToolName.LIST_FILES))

    assert first.status is ObservationStatus.OK
    assert repeated.status is ObservationStatus.REJECTED
    assert "repeated identical action" in repeated.summary


def test_different_action_resets_repeated_action_sequence(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    executor.execute(state, action(ToolName.LIST_FILES))
    middle = executor.execute(
        state,
        action(ToolName.READ_FILE, {"relative_path": "src/calculator.py"}),
    )
    repeated_after_reset = executor.execute(state, action(ToolName.LIST_FILES))

    assert middle.status is ObservationStatus.OK
    assert repeated_after_reset.status is ObservationStatus.OK


def test_run_tests_after_successful_patch_is_allowed(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    initial_tests = executor.execute(state, action(ToolName.RUN_TESTS))
    patch = executor.execute(
        state,
        action(ToolName.APPLY_PATCH, {"patch_text": repair_patch()}),
    )
    verification = executor.execute(state, action(ToolName.RUN_TESTS))

    assert initial_tests.status is ObservationStatus.ERROR
    assert patch.status is ObservationStatus.OK
    assert verification.status is ObservationStatus.OK


def test_repeated_action_rejection_is_audited(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    executor.execute(state, action(ToolName.LIST_FILES))
    result = executor.execute(state, action(ToolName.LIST_FILES))

    assert result.status is ObservationStatus.REJECTED
    assert state.actions[-1].tool is ToolName.LIST_FILES
    assert state.observations[-1].status is ObservationStatus.REJECTED
    assert state.observations[-1].summary == result.summary


def invalid_syntax_patch() -> str:
    """Return a patch that applies but creates invalid Python syntax."""
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(left: int, right: int) -> int:\n"
        "-    return left - right\n"
        "+    return (\n"
    )


def test_syntax_check_passes_after_valid_patch(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state
    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": repair_patch()},
        ),
    )

    result = executor.execute(
        state,
        action(ToolName.CHECK_SYNTAX),
    )

    assert result.status is ObservationStatus.OK
    assert result.output == "src/calculator.py"


def test_syntax_check_reports_invalid_changed_source(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state
    executor.execute(
        state,
        action(
            ToolName.APPLY_PATCH,
            {"patch_text": invalid_syntax_patch()},
        ),
    )

    result = executor.execute(
        state,
        action(ToolName.CHECK_SYNTAX),
    )

    assert result.status is ObservationStatus.ERROR
    assert "src/calculator.py:2:" in result.output


def test_syntax_check_rejects_unexpected_arguments(
    executor_state: tuple[AgentToolExecutor, AgentState],
) -> None:
    executor, state = executor_state

    result = executor.execute(
        state,
        action(
            ToolName.CHECK_SYNTAX,
            {"relative_path": "src/calculator.py"},
        ),
    )

    assert result.status is ObservationStatus.REJECTED
    assert "Invalid tool arguments" in result.summary
