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
    budget_exhaustions: int = Field(ge=0)
    policy_failures: int = Field(ge=0)
    escalations: int = Field(ge=0)
    failures: int = Field(ge=0)
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
        budget_exhaustions = sum(1 for row in condition_rows if row.budget_exhausted)
        policy_failures = sum(1 for row in condition_rows if row.policy_failure)
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
                budget_exhaustions=budget_exhaustions,
                policy_failures=policy_failures,
                escalations=escalations,
                failures=failures,
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
                mean_elapsed_seconds=(
                    sum(row.elapsed_seconds for row in condition_rows) / count
                ),
            )
        )

    return summaries
