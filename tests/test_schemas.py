"""Tests for PatchPilot's validated agent contracts."""

import pytest
from pydantic import ValidationError

from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    BudgetUsage,
    ExecutionBudget,
    RepairTask,
    ToolAction,
    ToolName,
)


def build_task() -> RepairTask:
    """Create one valid controlled repair task."""
    return RepairTask(
        task_id="boundary-condition-001",
        goal="Repair the failing boundary-condition implementation.",
        repository_root="benchmarks/task-001",
    )


def test_repair_task_defaults_are_safe() -> None:
    task = build_task()

    assert task.test_command == [
        "python",
        "-m",
        "pytest",
        "-q",
    ]
    assert task.allowed_paths == ["src"]
    assert task.forbidden_paths == ["tests"]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/etc/passwd",
        "../outside",
        "src/../../outside",
    ],
)
def test_repair_task_rejects_unsafe_paths(
    unsafe_path: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match="must be relative",
    ):
        RepairTask(
            task_id="unsafe-task",
            goal="Attempt a repair using an unsafe repository path.",
            repository_root=unsafe_path,
        )


def test_budget_usage_detects_exhaustion() -> None:
    budget = ExecutionBudget(
        max_steps=3,
        max_tool_calls=5,
        max_patch_attempts=2,
        max_seconds=60,
    )

    assert BudgetUsage(steps=2).exhausted(budget) is False
    assert BudgetUsage(steps=3).exhausted(budget) is True
    assert BudgetUsage(tool_calls=5).exhausted(budget) is True
    assert BudgetUsage(patch_attempts=2).exhausted(budget) is True
    assert BudgetUsage(elapsed_seconds=60).exhausted(budget) is True


def test_tool_action_requires_explanation() -> None:
    with pytest.raises(ValidationError):
        ToolAction(
            tool=ToolName.RUN_TESTS,
            rationale="",
        )


def test_agent_state_stops_at_budget_limit() -> None:
    state = AgentState(
        task=build_task(),
        budget=ExecutionBudget(max_steps=2),
        usage=BudgetUsage(steps=2),
        status=AgentStatus.RUNNING,
    )

    assert state.can_continue is False


def test_terminal_agent_state_cannot_continue() -> None:
    state = AgentState(
        task=build_task(),
        status=AgentStatus.SUCCEEDED,
        final_message="All required tests passed.",
    )

    assert state.can_continue is False
