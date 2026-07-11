from pathlib import Path

import pytest

from patchpilot.agent import TraceRecorder
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    FailureCategory,
    RepairTask,
)


def make_state() -> AgentState:
    task = RepairTask(
        task_id="trace-task-001",
        goal="Repair the incorrect calculator implementation.",
        repository_root="benchmarks/calculator",
    )
    return AgentState(task=task)


def test_trace_round_trip(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs")
    state = make_state()
    path = recorder.save(state, "trace-run-001")
    assert path.is_file()
    assert recorder.load("trace-run-001").state == state


def test_terminal_trace_has_completion_time(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs")
    state = make_state()
    state.status = AgentStatus.SUCCEEDED
    recorder.save(state, "completed-run-001")
    assert recorder.load("completed-run-001").completed_at is not None


def test_invalid_run_id_is_rejected(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs")
    with pytest.raises(ValueError):
        recorder.save(make_state(), "../escape")


def test_trace_preserves_failure_accounting(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs")
    state = make_state()
    state.model_calls = 2
    state.decision_parse_failures = 1
    state.failed_attempt_ids = [1]
    state.last_failure_category = FailureCategory.DECISION_PARSE_ERROR

    recorder.save(state, "accounting-run-001")
    loaded = recorder.load("accounting-run-001").state

    assert loaded.model_calls == 2
    assert loaded.decision_parse_failures == 1
    assert loaded.failed_attempt_ids == [1]
    assert loaded.last_failure_category is FailureCategory.DECISION_PARSE_ERROR
