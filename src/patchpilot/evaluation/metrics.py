"""Metric extraction for PatchPilot benchmark evaluations."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    ObservationStatus,
    ToolName,
)


class RunMetricRow(BaseModel):
    """Flat, CSV-friendly metrics for one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    status: str = Field(min_length=1)
    succeeded: bool
    full_suite_passed: bool
    policy_failure: bool
    model_calls: int = Field(ge=0)
    decision_parse_failures: int = Field(ge=0)
    patch_rejection_count: int = Field(ge=0)
    patch_application_failure_count: int = Field(ge=0)
    verification_failure_count: int = Field(ge=0)
    failed_attempt_count: int = Field(ge=0)
    failed_attempt_ids: str
    current_attempt_id: int | None = Field(default=None, ge=1)
    last_failed_attempt_id: int | None = Field(default=None, ge=1)
    last_rolled_back_attempt_id: int | None = Field(default=None, ge=1)
    last_failure_category: str | None = None
    terminal_failure_category: str | None = None
    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    patch_attempts: int = Field(ge=0)
    successful_patch_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    changed_file_count: int = Field(ge=0)
    changed_files: str
    invalid_patch_count: int = Field(ge=0)
    rejected_tool_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    rollback_count: int = Field(ge=0)
    syntax_check_count: int = Field(ge=0)
    syntax_error_count: int = Field(ge=0)
    targeted_test_count: int = Field(ge=0)
    full_test_count: int = Field(ge=0)
    failed_test_count: int = Field(ge=0)
    finish_rejection_count: int = Field(ge=0)
    reflection_count: int = Field(ge=0)
    hypothesis_revision_count: int = Field(ge=0)
    no_progress_rejection_count: int = Field(ge=0)
    budget_exhausted: bool
    final_message: str | None = None


class SummaryMetricRow(BaseModel):
    """Aggregate metrics for one evaluation condition."""

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=1)
    runs: int = Field(ge=0)
    successes: int = Field(ge=0)
    full_suite_passes: int = Field(ge=0)
    repair_rate: float = Field(ge=0, le=1)
    full_suite_pass_rate: float = Field(ge=0, le=1)
    invalid_patch_runs: int = Field(ge=0)
    invalid_patch_rate: float = Field(ge=0, le=1)
    reflection_runs: int = Field(ge=0)
    reflection_rate: float = Field(ge=0, le=1)
    syntax_error_runs: int = Field(ge=0)
    syntax_error_rate: float = Field(ge=0, le=1)
    no_progress_runs: int = Field(ge=0)
    no_progress_rate: float = Field(ge=0, le=1)
    budget_exhaustions: int = Field(ge=0)
    policy_failures: int = Field(ge=0)
    decision_parse_failure_runs: int = Field(ge=0)
    patch_rejection_runs: int = Field(ge=0)
    verification_failure_runs: int = Field(ge=0)
    escalations: int = Field(ge=0)
    failures: int = Field(ge=0)
    mean_model_calls: float = Field(ge=0)
    mean_decision_parse_failures: float = Field(ge=0)
    mean_patch_rejections: float = Field(ge=0)
    mean_verification_failures: float = Field(ge=0)
    mean_failed_attempts: float = Field(ge=0)
    mean_steps: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    mean_patch_attempts: float = Field(ge=0)
    mean_successful_patches: float = Field(ge=0)
    mean_syntax_checks: float = Field(ge=0)
    mean_targeted_tests: float = Field(ge=0)
    mean_full_tests: float = Field(ge=0)
    mean_failed_tests: float = Field(ge=0)
    mean_reflections: float = Field(ge=0)
    mean_hypothesis_revisions: float = Field(ge=0)
    mean_no_progress_rejections: float = Field(ge=0)
    mean_elapsed_seconds: float = Field(ge=0)


def collect_run_metrics(
    *,
    run_id: str,
    condition: str,
    state: AgentState,
) -> RunMetricRow:
    """Extract reproducible evaluation metrics from one final state."""
    pairs = list(
        zip(
            state.actions,
            state.observations,
            strict=False,
        )
    )

    invalid_patch_count = sum(
        1
        for observation in state.observations
        if observation.tool == ToolName.APPLY_PATCH
        and observation.status != ObservationStatus.OK
    )
    successful_patch_count = sum(
        1
        for observation in state.observations
        if observation.tool == ToolName.APPLY_PATCH
        and observation.status == ObservationStatus.OK
    )
    rejected_tool_count = sum(
        1
        for observation in state.observations
        if observation.status == ObservationStatus.REJECTED
    )
    timeout_count = sum(
        1
        for observation in state.observations
        if observation.status == ObservationStatus.TIMEOUT
    )
    rollback_count = sum(
        1
        for action, observation in pairs
        if action.tool == ToolName.RESTORE_FILE
        and observation.status == ObservationStatus.OK
    )
    syntax_check_count = sum(
        1 for action in state.actions if action.tool == ToolName.CHECK_SYNTAX
    )
    syntax_error_count = sum(
        1
        for observation in state.observations
        if observation.tool == ToolName.CHECK_SYNTAX
        and observation.status == ObservationStatus.ERROR
    )
    targeted_test_count = sum(
        1
        for action in state.actions
        if action.tool == ToolName.RUN_TESTS
        and isinstance(action.arguments.get("target"), str)
        and bool(str(action.arguments["target"]).strip())
    )
    full_test_count = sum(
        1
        for action in state.actions
        if action.tool == ToolName.RUN_TESTS and action.arguments.get("target") is None
    )
    failed_test_count = sum(
        1
        for observation in state.observations
        if observation.tool == ToolName.RUN_TESTS
        and observation.status == ObservationStatus.ERROR
    )
    finish_rejection_count = sum(
        1
        for action, observation in pairs
        if action.tool == ToolName.FINISH
        and observation.status == ObservationStatus.REJECTED
    )
    no_progress_rejection_count = sum(
        1
        for observation in state.observations
        if observation.status == ObservationStatus.REJECTED
        and observation.summary.startswith("Rejected no-progress:")
    )
    final_message = state.final_message
    policy_failure = bool(
        final_message and final_message.startswith("The decision policy failed safely:")
    )

    return RunMetricRow(
        run_id=run_id,
        task_id=state.task.task_id,
        condition=condition,
        status=state.status.value,
        succeeded=state.status == AgentStatus.SUCCEEDED,
        full_suite_passed=state.full_suite_passed,
        policy_failure=policy_failure,
        model_calls=state.model_calls,
        decision_parse_failures=state.decision_parse_failures,
        patch_rejection_count=state.patch_rejection_count,
        patch_application_failure_count=state.patch_application_failure_count,
        verification_failure_count=state.verification_failure_count,
        failed_attempt_count=len(state.failed_attempt_ids),
        failed_attempt_ids=";".join(
            str(attempt_id) for attempt_id in state.failed_attempt_ids
        ),
        current_attempt_id=state.current_attempt_id,
        last_failed_attempt_id=state.last_failed_attempt_id,
        last_rolled_back_attempt_id=state.last_rolled_back_attempt_id,
        last_failure_category=(
            state.last_failure_category.value
            if state.last_failure_category is not None
            else None
        ),
        terminal_failure_category=(
            state.terminal_failure_category.value
            if state.terminal_failure_category is not None
            else None
        ),
        steps=state.usage.steps,
        tool_calls=state.usage.tool_calls,
        patch_attempts=state.usage.patch_attempts,
        successful_patch_count=successful_patch_count,
        elapsed_seconds=state.usage.elapsed_seconds,
        changed_file_count=len(state.changed_files),
        changed_files=";".join(state.changed_files),
        invalid_patch_count=invalid_patch_count,
        rejected_tool_count=rejected_tool_count,
        timeout_count=timeout_count,
        rollback_count=rollback_count,
        syntax_check_count=syntax_check_count,
        syntax_error_count=syntax_error_count,
        targeted_test_count=targeted_test_count,
        full_test_count=full_test_count,
        failed_test_count=failed_test_count,
        finish_rejection_count=finish_rejection_count,
        reflection_count=len(state.reflections),
        hypothesis_revision_count=len(state.rejected_hypotheses),
        no_progress_rejection_count=no_progress_rejection_count,
        budget_exhausted=state.status == AgentStatus.BUDGET_EXHAUSTED,
        final_message=final_message,
    )


def summarise_runs(rows: Iterable[RunMetricRow]) -> list[SummaryMetricRow]:
    """Aggregate run rows by evaluation condition."""
    grouped: dict[str, list[RunMetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.condition, []).append(row)

    summaries: list[SummaryMetricRow] = []
    for condition, condition_rows in sorted(grouped.items()):
        count = len(condition_rows)
        successes = sum(1 for row in condition_rows if row.succeeded)
        full_passes = sum(1 for row in condition_rows if row.full_suite_passed)
        invalid_patch_runs = sum(
            1 for row in condition_rows if row.invalid_patch_count > 0
        )
        reflection_runs = sum(1 for row in condition_rows if row.reflection_count > 0)
        syntax_error_runs = sum(
            1 for row in condition_rows if row.syntax_error_count > 0
        )
        no_progress_runs = sum(
            1 for row in condition_rows if row.no_progress_rejection_count > 0
        )
        budget_exhaustions = sum(1 for row in condition_rows if row.budget_exhausted)
        policy_failures = sum(1 for row in condition_rows if row.policy_failure)
        decision_parse_failure_runs = sum(
            1 for row in condition_rows if row.decision_parse_failures > 0
        )
        patch_rejection_runs = sum(
            1 for row in condition_rows if row.patch_rejection_count > 0
        )
        verification_failure_runs = sum(
            1 for row in condition_rows if row.verification_failure_count > 0
        )
        escalations = sum(
            1 for row in condition_rows if row.status == AgentStatus.ESCALATED.value
        )
        failures = sum(
            1 for row in condition_rows if row.status == AgentStatus.FAILED.value
        )

        summaries.append(
            SummaryMetricRow(
                condition=condition,
                runs=count,
                successes=successes,
                full_suite_passes=full_passes,
                repair_rate=successes / count,
                full_suite_pass_rate=full_passes / count,
                invalid_patch_runs=invalid_patch_runs,
                invalid_patch_rate=invalid_patch_runs / count,
                reflection_runs=reflection_runs,
                reflection_rate=reflection_runs / count,
                syntax_error_runs=syntax_error_runs,
                syntax_error_rate=syntax_error_runs / count,
                no_progress_runs=no_progress_runs,
                no_progress_rate=no_progress_runs / count,
                budget_exhaustions=budget_exhaustions,
                policy_failures=policy_failures,
                decision_parse_failure_runs=decision_parse_failure_runs,
                patch_rejection_runs=patch_rejection_runs,
                verification_failure_runs=verification_failure_runs,
                escalations=escalations,
                failures=failures,
                mean_model_calls=(
                    sum(row.model_calls for row in condition_rows) / count
                ),
                mean_decision_parse_failures=(
                    sum(row.decision_parse_failures for row in condition_rows) / count
                ),
                mean_patch_rejections=(
                    sum(row.patch_rejection_count for row in condition_rows) / count
                ),
                mean_verification_failures=(
                    sum(row.verification_failure_count for row in condition_rows)
                    / count
                ),
                mean_failed_attempts=(
                    sum(row.failed_attempt_count for row in condition_rows) / count
                ),
                mean_steps=(sum(row.steps for row in condition_rows) / count),
                mean_tool_calls=(sum(row.tool_calls for row in condition_rows) / count),
                mean_patch_attempts=(
                    sum(row.patch_attempts for row in condition_rows) / count
                ),
                mean_successful_patches=(
                    sum(row.successful_patch_count for row in condition_rows) / count
                ),
                mean_syntax_checks=(
                    sum(row.syntax_check_count for row in condition_rows) / count
                ),
                mean_targeted_tests=(
                    sum(row.targeted_test_count for row in condition_rows) / count
                ),
                mean_full_tests=(
                    sum(row.full_test_count for row in condition_rows) / count
                ),
                mean_failed_tests=(
                    sum(row.failed_test_count for row in condition_rows) / count
                ),
                mean_reflections=(
                    sum(row.reflection_count for row in condition_rows) / count
                ),
                mean_hypothesis_revisions=(
                    sum(row.hypothesis_revision_count for row in condition_rows) / count
                ),
                mean_no_progress_rejections=(
                    sum(row.no_progress_rejection_count for row in condition_rows)
                    / count
                ),
                mean_elapsed_seconds=(
                    sum(row.elapsed_seconds for row in condition_rows) / count
                ),
            )
        )

    return summaries
