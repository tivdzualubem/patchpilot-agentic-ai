"""Tests for repeated-action and no-progress cycle detection."""

from patchpilot.agent.loop_guard import RepeatedActionGuard
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


def make_state() -> AgentState:
    task = RepairTask(
        task_id="loop-guard-001",
        goal="Repair the defective Python implementation.",
        repository_root="benchmarks/example",
    )
    return AgentState(task=task)


def make_action(
    tool: ToolName,
    arguments: dict[str, object] | None = None,
) -> ToolAction:
    return ToolAction(
        tool=tool,
        arguments=arguments or {},
        rationale="Advance the repair process.",
    )


def record_action(
    guard: RepeatedActionGuard,
    state: AgentState,
    action: ToolAction,
) -> None:
    state.actions.append(action)
    guard.record_progress(state)


def test_first_action_is_allowed() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()

    assert guard.blocks(state, make_action(ToolName.RUN_TESTS)) is False


def test_second_identical_action_is_blocked() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()
    record_action(guard, state, make_action(ToolName.RUN_TESTS))

    reason = guard.rejection_reason(
        state,
        make_action(ToolName.RUN_TESTS),
    )

    assert reason is not None
    assert "repeated identical action" in reason


def test_identical_action_is_allowed_after_hypothesis_progress() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()
    action = make_action(ToolName.RUN_TESTS)
    record_action(guard, state, action)
    state.current_hypothesis = "The loop bound is incorrect."

    assert guard.blocks(state, action) is False


def test_different_action_breaks_sequence() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()
    record_action(guard, state, make_action(ToolName.RUN_TESTS))

    assert guard.blocks(state, make_action(ToolName.READ_FILE)) is False


def test_two_action_cycle_is_blocked_without_progress() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    read = make_action(
        ToolName.READ_FILE,
        {"relative_path": "src/example.py"},
    )
    search = make_action(
        ToolName.SEARCH_CODE,
        {"query": "return"},
    )

    record_action(guard, state, read)
    record_action(guard, state, search)
    record_action(guard, state, read)

    reason = guard.rejection_reason(state, search)

    assert reason is not None
    assert "repeated action cycle" in reason


def test_repository_change_breaks_action_cycle() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    read = make_action(
        ToolName.READ_FILE,
        {"relative_path": "src/example.py"},
    )
    search = make_action(
        ToolName.SEARCH_CODE,
        {"query": "return"},
    )

    record_action(guard, state, read)
    record_action(guard, state, search)
    state.repository_revision = 1
    state.changed_files = ["src/example.py"]
    record_action(guard, state, read)

    assert guard.blocks(state, search) is False


def test_hypothesis_revision_breaks_action_cycle() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    read = make_action(
        ToolName.READ_FILE,
        {"relative_path": "src/example.py"},
    )
    search = make_action(
        ToolName.SEARCH_CODE,
        {"query": "return"},
    )

    record_action(guard, state, read)
    record_action(guard, state, search)
    state.current_hypothesis = "The boundary condition is incorrect."
    record_action(guard, state, read)

    assert guard.blocks(state, search) is False


def test_new_test_evidence_breaks_action_cycle() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    read = make_action(
        ToolName.READ_FILE,
        {"relative_path": "src/example.py"},
    )
    search = make_action(
        ToolName.SEARCH_CODE,
        {"query": "return"},
    )

    record_action(guard, state, read)
    record_action(guard, state, search)
    state.observations.append(
        ToolObservation(
            tool=ToolName.RUN_TESTS,
            status=ObservationStatus.ERROR,
            summary="Tests failed with different evidence.",
            output="test_boundary failed",
        )
    )
    record_action(guard, state, read)

    assert guard.blocks(state, search) is False


def test_progress_resets_no_progress_streak() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    state.no_progress_streak = 1
    record_action(guard, state, make_action(ToolName.LIST_FILES))

    state.repository_revision = 1
    record_action(
        guard,
        state,
        make_action(
            ToolName.READ_FILE,
            {"relative_path": "src/example.py"},
        ),
    )

    assert state.no_progress_streak == 0


def test_syntax_verification_resets_no_progress_streak() -> None:
    guard = RepeatedActionGuard()
    state = make_state()
    state.changed_files = ["src/example.py"]
    state.repository_revision = 1
    record_action(
        guard,
        state,
        make_action(
            ToolName.READ_FILE,
            {"relative_path": "src/example.py"},
        ),
    )

    state.no_progress_streak = 1
    state.syntax_verified_revision = 1
    record_action(
        guard,
        state,
        make_action(
            ToolName.SEARCH_CODE,
            {"query": "return"},
        ),
    )

    assert state.no_progress_streak == 0
