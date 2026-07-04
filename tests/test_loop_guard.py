from patchpilot.agent.loop_guard import RepeatedActionGuard
from patchpilot.schemas import (
    AgentState,
    RepairTask,
    ToolAction,
    ToolName,
)


def make_state() -> AgentState:
    task = RepairTask(
        task_id="loop-guard-001",
        goal="Repair the defective Python implementation.",
        repository_root="benchmarks/example",
    )
    return AgentState(task=task)


def make_action(tool: ToolName) -> ToolAction:
    return ToolAction(
        tool=tool,
        rationale="Advance the repair process.",
    )


def test_first_action_is_allowed() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()

    assert guard.blocks(state, make_action(ToolName.RUN_TESTS)) is False


def test_second_identical_action_is_blocked() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()
    state.actions.append(make_action(ToolName.RUN_TESTS))

    assert guard.blocks(state, make_action(ToolName.RUN_TESTS)) is True


def test_different_action_breaks_sequence() -> None:
    guard = RepeatedActionGuard(max_repeats=2)
    state = make_state()
    state.actions.append(make_action(ToolName.RUN_TESTS))

    assert guard.blocks(state, make_action(ToolName.READ_FILE)) is False
