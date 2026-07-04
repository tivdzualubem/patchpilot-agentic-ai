from patchpilot.evaluation import collect_run_metrics, summarise_runs
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


def make_state() -> AgentState:
    state = AgentState(
        task=RepairTask(
            task_id="example-001",
            goal="Repair the example defect.",
            repository_root="repository",
        )
    )
    state.status = AgentStatus.SUCCEEDED
    state.full_suite_passed = True
    state.changed_files = ["src/example.py"]
    state.usage.steps = 6
    state.usage.tool_calls = 5
    state.usage.patch_attempts = 1
    state.usage.elapsed_seconds = 2.5
    state.actions.append(
        ToolAction(
            tool=ToolName.APPLY_PATCH,
            rationale="Try an invalid patch.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.APPLY_PATCH,
            status=ObservationStatus.ERROR,
            summary="Patch failed.",
        )
    )
    return state


def test_collect_run_metrics_counts_patch_errors() -> None:
    row = collect_run_metrics(
        run_id="run-001",
        condition="full-agent",
        state=make_state(),
    )

    assert row.task_id == "example-001"
    assert row.succeeded is True
    assert row.full_suite_passed is True
    assert row.invalid_patch_count == 1
    assert row.changed_files == "src/example.py"


def test_summarise_runs_computes_rates() -> None:
    row = collect_run_metrics(
        run_id="run-001",
        condition="full-agent",
        state=make_state(),
    )

    summary = summarise_runs([row])[0]

    assert summary.condition == "full-agent"
    assert summary.runs == 1
    assert summary.repair_rate == 1.0
    assert summary.full_suite_pass_rate == 1.0
    assert summary.invalid_patch_rate == 1.0
