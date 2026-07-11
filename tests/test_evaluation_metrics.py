"""Tests for PatchPilot run-level and summary metrics."""

from patchpilot.evaluation import (
    RunMetricRow,
    collect_run_metrics,
    summarise_runs,
)
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    FailureCategory,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


def action(
    tool: ToolName,
    *,
    arguments: dict[str, object] | None = None,
) -> ToolAction:
    return ToolAction(
        tool=tool,
        arguments=arguments or {},
        rationale="Record one evaluation trajectory action.",
    )


def observation(
    tool: ToolName,
    status: ObservationStatus,
    summary: str,
) -> ToolObservation:
    return ToolObservation(
        tool=tool,
        status=status,
        summary=summary,
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
    state.reflections = ["The first repair did not address the root cause."]
    state.rejected_hypotheses = ["The comparison operator is incorrect."]
    state.usage.steps = 8
    state.usage.tool_calls = 8
    state.usage.patch_attempts = 2
    state.usage.elapsed_seconds = 2.5
    state.model_calls = 3
    state.decision_parse_failures = 1
    state.patch_rejection_count = 1
    state.patch_application_failure_count = 1
    state.verification_failure_count = 2
    state.failed_attempt_ids = [1, 2]
    state.last_failed_attempt_id = 2
    state.last_rolled_back_attempt_id = 2
    state.last_failure_category = FailureCategory.TEST_VERIFICATION_FAILED
    state.terminal_failure_category = None

    state.actions.extend(
        [
            action(ToolName.APPLY_PATCH),
            action(ToolName.APPLY_PATCH),
            action(ToolName.CHECK_SYNTAX),
            action(
                ToolName.RUN_TESTS,
                arguments={"target": "tests/test_example.py::test_case"},
            ),
            action(ToolName.RUN_TESTS),
            action(ToolName.RESTORE_FILE),
            action(
                ToolName.FINISH,
                arguments={
                    "status": "succeeded",
                    "message": "Attempt premature completion.",
                },
            ),
        ]
    )
    state.observations.extend(
        [
            observation(
                ToolName.APPLY_PATCH,
                ObservationStatus.ERROR,
                "Patch failed.",
            ),
            observation(
                ToolName.APPLY_PATCH,
                ObservationStatus.OK,
                "Patch applied.",
            ),
            observation(
                ToolName.CHECK_SYNTAX,
                ObservationStatus.OK,
                "Syntax passed.",
            ),
            observation(
                ToolName.RUN_TESTS,
                ObservationStatus.ERROR,
                "Targeted test failed.",
            ),
            observation(
                ToolName.RUN_TESTS,
                ObservationStatus.OK,
                "Full suite passed.",
            ),
            observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.OK,
                "File restored.",
            ),
            observation(
                ToolName.FINISH,
                ObservationStatus.REJECTED,
                "Success requires current verification.",
            ),
        ]
    )
    return state


def test_collect_run_metrics_counts_agentic_behaviour() -> None:
    row = collect_run_metrics(
        run_id="run-001",
        condition="full-agent",
        state=make_state(),
    )

    assert isinstance(row, RunMetricRow)
    assert row.task_id == "example-001"
    assert row.succeeded is True
    assert row.full_suite_passed is True
    assert row.policy_failure is False
    assert row.model_calls == 3
    assert row.decision_parse_failures == 1
    assert row.patch_rejection_count == 1
    assert row.patch_application_failure_count == 1
    assert row.verification_failure_count == 2
    assert row.failed_attempt_count == 2
    assert row.failed_attempt_ids == "1;2"
    assert row.last_failed_attempt_id == 2
    assert row.last_rolled_back_attempt_id == 2
    assert row.last_failure_category == FailureCategory.TEST_VERIFICATION_FAILED.value
    assert row.terminal_failure_category is None
    assert row.invalid_patch_count == 1
    assert row.successful_patch_count == 1
    assert row.syntax_check_count == 1
    assert row.syntax_error_count == 0
    assert row.targeted_test_count == 1
    assert row.full_test_count == 1
    assert row.failed_test_count == 1
    assert row.rollback_count == 1
    assert row.finish_rejection_count == 1
    assert row.reflection_count == 1
    assert row.hypothesis_revision_count == 1
    assert row.changed_files == "src/example.py"


def test_collect_run_metrics_detects_safe_policy_failure() -> None:
    state = make_state()
    state.status = AgentStatus.ESCALATED
    state.final_message = (
        "The decision policy failed safely: PolicyResponseError: invalid JSON"
    )

    row = collect_run_metrics(
        run_id="run-002",
        condition="llm-tool-agent",
        state=state,
    )

    assert row.policy_failure is True
    assert row.succeeded is False


def test_summarise_runs_computes_agentic_rates_and_means() -> None:
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
    assert summary.reflection_rate == 1.0
    assert summary.syntax_error_rate == 0.0
    assert summary.policy_failures == 0
    assert summary.decision_parse_failure_runs == 1
    assert summary.patch_rejection_runs == 1
    assert summary.verification_failure_runs == 1
    assert summary.mean_model_calls == 3.0
    assert summary.mean_decision_parse_failures == 1.0
    assert summary.mean_patch_rejections == 1.0
    assert summary.mean_verification_failures == 2.0
    assert summary.mean_failed_attempts == 2.0
    assert summary.mean_successful_patches == 1.0
    assert summary.mean_syntax_checks == 1.0
    assert summary.mean_targeted_tests == 1.0
    assert summary.mean_full_tests == 1.0
    assert summary.mean_failed_tests == 1.0
    assert summary.mean_reflections == 1.0
    assert summary.mean_hypothesis_revisions == 1.0


def test_no_progress_rejections_are_measured() -> None:
    state = make_state()
    state.actions.append(
        action(
            ToolName.SEARCH_CODE,
            arguments={"query": "return"},
        )
    )
    state.observations.append(
        observation(
            ToolName.SEARCH_CODE,
            ObservationStatus.REJECTED,
            (
                "Rejected no-progress: repeated action cycle with unchanged "
                "repository, tests, hypothesis, and changed files."
            ),
        )
    )

    row = collect_run_metrics(
        run_id="run-003",
        condition="full-agent",
        state=state,
    )
    summary = summarise_runs([row])[0]

    assert row.no_progress_rejection_count == 1
    assert summary.no_progress_runs == 1
    assert summary.no_progress_rate == 1.0
    assert summary.mean_no_progress_rejections == 1.0
