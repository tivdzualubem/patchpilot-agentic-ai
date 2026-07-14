"""Regression tests for Step 17D action convergence."""

from __future__ import annotations

from patchpilot.agent import LLMToolPolicy
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


def make_state() -> AgentState:
    return AgentState(
        task=RepairTask(
            task_id="step17d",
            goal="Repair the defective function and pass all tests.",
            repository_root="repository",
            allowed_paths=["src"],
            forbidden_paths=["tests"],
        )
    )


def append(
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
            rationale="Advance the bounded repair.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=tool,
            status=status,
            summary=f"{tool.value} completed.",
            output=output,
        )
    )


def test_successful_read_after_failure_forces_patch() -> None:
    state = make_state()
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert -1 == 5",
    )
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:1:def add(left, right):",
    )
    append(
        state,
        ToolName.READ_FILE,
        ObservationStatus.OK,
        arguments={"relative_path": "src/calculator.py"},
        output="def add(left, right):\n    return left - right",
    )

    assert LLMToolPolicy._legal_actions(state) == ["apply_patch"]


def test_search_after_read_cannot_restart_exploration_loop() -> None:
    state = make_state()
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert -1 == 5",
    )
    append(
        state,
        ToolName.READ_FILE,
        ObservationStatus.OK,
        arguments={"relative_path": "src/calculator.py"},
        output="def add(left, right):\n    return left - right",
    )
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:1:def add(left, right):",
    )

    assert LLMToolPolicy._legal_actions(state) == ["apply_patch"]


def test_successful_search_forces_read() -> None:
    state = make_state()
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert -1 == 5",
    )
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:1:def add(left, right):",
    )

    assert LLMToolPolicy._legal_actions(state) == ["read_file"]


def test_new_read_after_failed_patch_forces_revised_patch() -> None:
    state = make_state()
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert -1 == 5",
    )
    append(
        state,
        ToolName.READ_FILE,
        ObservationStatus.OK,
        arguments={"relative_path": "src/calculator.py"},
        output="def add(left, right):\n    return left - right",
    )
    append(
        state,
        ToolName.APPLY_PATCH,
        ObservationStatus.OK,
        arguments={"patch_text": "diff --git a/x b/x"},
    )
    append(
        state,
        ToolName.CHECK_SYNTAX,
        ObservationStatus.OK,
    )
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert 4 == 5",
    )
    append(
        state,
        ToolName.RESTORE_FILE,
        ObservationStatus.OK,
        arguments={"scope": "failed_attempt"},
    )
    append(
        state,
        ToolName.READ_FILE,
        ObservationStatus.OK,
        arguments={"relative_path": "src/calculator.py"},
        output="def add(left, right):\n    return left - right",
    )

    assert LLMToolPolicy._legal_actions(state) == ["apply_patch"]


def test_initial_repository_inspection_still_requires_tests() -> None:
    state = make_state()
    append(
        state,
        ToolName.LIST_FILES,
        ObservationStatus.OK,
        arguments={"relative_path": "src"},
        output="src/calculator.py",
    )

    assert LLMToolPolicy._legal_actions(state) == ["run_tests"]


def test_syntax_failure_rollback_allows_reflective_source_read() -> None:
    state = make_state()
    state.current_hypothesis = "The replacement expression is valid."
    state.repository_revision = 2
    state.last_failed_attempt_id = 1
    state.last_failed_attempt_files = ["src/calculator.py"]
    state.last_failed_verification_tool = ToolName.CHECK_SYNTAX
    state.last_rolled_back_attempt_id = 1
    state.last_rolled_back_attempt_files = ["src/calculator.py"]

    append(
        state,
        ToolName.APPLY_PATCH,
        ObservationStatus.OK,
        arguments={"patch_text": "diff --git a/x b/x"},
    )
    append(
        state,
        ToolName.CHECK_SYNTAX,
        ObservationStatus.ERROR,
        output="src/calculator.py:5:12: invalid syntax",
    )
    append(
        state,
        ToolName.RESTORE_FILE,
        ObservationStatus.OK,
        arguments={"scope": "failed_attempt", "attempt_id": 1},
        output="src/calculator.py",
    )

    assert LLMToolPolicy._legal_actions(state) == ["read_file"]
