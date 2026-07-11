"""Audited action execution for the PatchPilot agent."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patchpilot.agent.loop_guard import RepeatedActionGuard
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
from patchpilot.tools import (
    PatchManager,
    RepositorySandbox,
    SyntaxChecker,
    TestRunner,
)


class _Arguments(BaseModel):
    """Strict base model for tool arguments."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class _ListFilesArguments(_Arguments):
    relative_path: str = "."


class _ReadFileArguments(_Arguments):
    relative_path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class _SearchCodeArguments(_Arguments):
    query: str = Field(min_length=1, max_length=200)
    relative_path: str = "."
    max_hits: int = Field(default=50, ge=1, le=100)


class _RunTestsArguments(_Arguments):
    target: str | None = None


class _CheckSyntaxArguments(_Arguments):
    pass


class _ApplyPatchArguments(_Arguments):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    patch_text: str = Field(min_length=1)


class _ViewDiffArguments(_Arguments):
    pass


class _RestoreFileArguments(_Arguments):
    relative_path: str | None = None


class _FinishArguments(_Arguments):
    status: Literal["succeeded", "failed", "escalated"]
    message: str = Field(min_length=3, max_length=2000)


class VerificationMode(StrEnum):
    """Runtime enforcement mode used by verification ablations."""

    STRICT = "strict"
    DISABLED = "disabled"


_TERMINAL_STATES = {
    AgentStatus.SUCCEEDED,
    AgentStatus.FAILED,
    AgentStatus.ESCALATED,
    AgentStatus.BUDGET_EXHAUSTED,
}


class AgentToolExecutor:
    """Validate, execute, record, and account for agent actions."""

    def __init__(
        self,
        workspace_root: Path,
        task: RepairTask,
        test_timeout_seconds: int = 60,
        repeated_action_guard: RepeatedActionGuard | None = None,
        verification_mode: VerificationMode = VerificationMode.STRICT,
    ) -> None:
        self.sandbox = RepositorySandbox(workspace_root, task)
        self.test_runner = TestRunner(
            self.sandbox,
            task,
            timeout_seconds=test_timeout_seconds,
        )
        self.patch_manager = PatchManager(self.sandbox, task)
        self.syntax_checker = SyntaxChecker(self.sandbox)
        self.repeated_action_guard = repeated_action_guard or RepeatedActionGuard()
        self.verification_mode = VerificationMode(verification_mode)
        self._started_at = perf_counter()

    @staticmethod
    def _observation(
        tool: ToolName,
        status: ObservationStatus,
        summary: str,
        output: str = "",
    ) -> ToolObservation:
        """Create an executor-level observation."""
        return ToolObservation(
            tool=tool,
            status=status,
            summary=summary,
            output=output,
        )

    def _record(
        self,
        state: AgentState,
        action: ToolAction,
        observation: ToolObservation,
    ) -> ToolObservation:
        """Append an action-observation pair to the trajectory."""
        state.actions.append(action)
        state.observations.append(observation)
        self.repeated_action_guard.record_progress(state)
        state.usage.elapsed_seconds = perf_counter() - self._started_at
        return observation

    @staticmethod
    def _invalidate_verification(state: AgentState) -> None:
        """Invalidate test evidence after a repository mutation."""
        state.full_suite_passed = False
        state.verified_revision = None

    @staticmethod
    def _update_syntax_gate(state: AgentState) -> None:
        """Require fresh syntax evidence for the current changed Python files."""
        if any(Path(path).suffix == ".py" for path in state.changed_files):
            state.syntax_verified_revision = None
        else:
            state.syntax_verified_revision = state.repository_revision

    @staticmethod
    def _clear_current_attempt(state: AgentState) -> None:
        """Clear active-attempt state after acceptance or rollback."""
        state.current_attempt_id = None
        state.current_attempt_files = []

    def _sync_current_attempt(self, state: AgentState) -> None:
        """Synchronize state with the patch manager's active attempt."""
        state.current_attempt_id = self.patch_manager.current_attempt_id
        state.current_attempt_files = list(self.patch_manager.current_attempt_files)

    @staticmethod
    def _mark_failed_attempt(
        state: AgentState,
        verification_tool: ToolName,
    ) -> None:
        """Require rollback for the active attempt after failed verification."""
        if state.current_attempt_id is None:
            return

        state.rollback_required = True
        state.last_failed_attempt_id = state.current_attempt_id
        state.last_failed_attempt_files = list(state.current_attempt_files)
        state.last_failed_verification_tool = verification_tool
        if state.current_attempt_id not in state.failed_attempt_ids:
            state.failed_attempt_ids.append(state.current_attempt_id)

    def _reject(
        self,
        state: AgentState,
        action: ToolAction,
        summary: str,
    ) -> ToolObservation:
        """Record a rejected action."""
        return self._record(
            state,
            action,
            self._observation(
                action.tool,
                ObservationStatus.REJECTED,
                summary,
            ),
        )

    def _execute_finish(
        self,
        state: AgentState,
        action: ToolAction,
    ) -> ToolObservation:
        """Apply test-gated terminal-state policy."""
        try:
            arguments = _FinishArguments.model_validate(action.arguments)
        except ValidationError as exc:
            return self._reject(
                state,
                action,
                f"Invalid finish arguments: {exc}",
            )

        if arguments.status == "succeeded":
            if state.rollback_required:
                return self._reject(
                    state,
                    action,
                    "Success is blocked until the failed patch attempt is "
                    "transactionally rolled back.",
                )

            if self.verification_mode is VerificationMode.STRICT:
                if state.syntax_check_required:
                    return self._reject(
                        state,
                        action,
                        "Success requires a passing syntax check for the current "
                        "repository revision.",
                    )

                if state.current_attempt_id is not None:
                    return self._reject(
                        state,
                        action,
                        "Success requires full-suite acceptance of the current "
                        "patch attempt.",
                    )

                verified = (
                    state.full_suite_passed
                    and state.verified_revision == state.repository_revision
                )

                if not verified:
                    return self._reject(
                        state,
                        action,
                        "Success requires a passing full test suite "
                        "for the current repository revision.",
                    )
            elif state.current_attempt_id is not None:
                self.patch_manager.accept_attempt(state.current_attempt_id)
                self._clear_current_attempt(state)

            state.status = AgentStatus.SUCCEEDED
            state.terminal_failure_category = None

        elif arguments.status == "failed":
            state.status = AgentStatus.FAILED
            state.last_failure_category = FailureCategory.USER_FAILED
            state.terminal_failure_category = FailureCategory.USER_FAILED

        else:
            state.status = AgentStatus.ESCALATED
            state.last_failure_category = FailureCategory.USER_ESCALATED
            state.terminal_failure_category = FailureCategory.USER_ESCALATED

        state.final_message = arguments.message

        return self._record(
            state,
            action,
            self._observation(
                ToolName.FINISH,
                ObservationStatus.OK,
                f"Run finished with status {state.status.value}.",
                arguments.message,
            ),
        )

    def _dispatch(
        self,
        action: ToolAction,
    ) -> tuple[ToolObservation, str | None]:
        """Validate arguments and dispatch one non-terminal tool."""
        try:
            if action.tool is ToolName.LIST_FILES:
                list_args = _ListFilesArguments.model_validate(action.arguments)
                return (
                    self.sandbox.list_files(list_args.relative_path),
                    None,
                )

            if action.tool is ToolName.READ_FILE:
                read_args = _ReadFileArguments.model_validate(action.arguments)
                return (
                    self.sandbox.read_file(
                        read_args.relative_path,
                        read_args.start_line,
                        read_args.end_line,
                    ),
                    None,
                )

            if action.tool is ToolName.SEARCH_CODE:
                search_args = _SearchCodeArguments.model_validate(action.arguments)
                return (
                    self.sandbox.search_code(
                        search_args.query,
                        search_args.relative_path,
                        search_args.max_hits,
                    ),
                    None,
                )

            if action.tool is ToolName.RUN_TESTS:
                test_args = _RunTestsArguments.model_validate(action.arguments)
                return (
                    self.test_runner.run_tests(test_args.target),
                    test_args.target,
                )

            if action.tool is ToolName.CHECK_SYNTAX:
                _CheckSyntaxArguments.model_validate(action.arguments)
                return (
                    self.syntax_checker.check_files(self.patch_manager.changed_files),
                    None,
                )

            if action.tool is ToolName.APPLY_PATCH:
                patch_args = _ApplyPatchArguments.model_validate(action.arguments)
                return (
                    self.patch_manager.apply_patch(patch_args.patch_text),
                    None,
                )

            if action.tool is ToolName.VIEW_DIFF:
                _ViewDiffArguments.model_validate(action.arguments)
                return self.patch_manager.view_diff(), None

            if action.tool is ToolName.RESTORE_FILE:
                restore_args = _RestoreFileArguments.model_validate(action.arguments)

                if restore_args.relative_path is None:
                    return (
                        self.patch_manager.restore_all(),
                        None,
                    )

                return (
                    self.patch_manager.restore_file(restore_args.relative_path),
                    None,
                )

            return (
                self._observation(
                    action.tool,
                    ObservationStatus.REJECTED,
                    "Unsupported tool action.",
                ),
                None,
            )

        except ValidationError as exc:
            return (
                self._observation(
                    action.tool,
                    ObservationStatus.REJECTED,
                    f"Invalid tool arguments: {exc}",
                ),
                None,
            )

    def rollback_failed_attempt(
        self,
        state: AgentState,
    ) -> ToolObservation:
        """Rollback the active failed attempt outside normal action budgets."""
        attempt_id = state.current_attempt_id
        attempt_files = list(state.current_attempt_files)
        action = ToolAction(
            tool=ToolName.RESTORE_FILE,
            arguments={
                "scope": "failed_attempt",
                "attempt_id": attempt_id,
            },
            rationale=(
                "Runtime-enforced transactional rollback after failed verification."
            ),
        )

        if not state.rollback_required or attempt_id is None:
            return self._reject(
                state,
                action,
                "No failed patch attempt is pending transactional rollback.",
            )

        observation = self.patch_manager.restore_attempt()

        if observation.status is ObservationStatus.OK:
            state.repository_revision += 1
            state.changed_files = list(self.patch_manager.changed_files)
            self._invalidate_verification(state)
            self._update_syntax_gate(state)
            state.last_rolled_back_attempt_id = attempt_id
            state.last_rolled_back_attempt_files = attempt_files
            state.rollback_required = False
            self._sync_current_attempt(state)
        else:
            state.status = AgentStatus.ESCALATED
            state.last_failure_category = FailureCategory.ROLLBACK_FAILED
            state.terminal_failure_category = FailureCategory.ROLLBACK_FAILED
            state.final_message = (
                "Transactional rollback of the failed patch attempt failed."
            )

        return self._record(
            state,
            action,
            observation,
        )

    def execute(
        self,
        state: AgentState,
        action: ToolAction,
    ) -> ToolObservation:
        """Execute one policy-checked action and update agent state."""
        if state.status in _TERMINAL_STATES:
            return self._reject(
                state,
                action,
                "No actions are permitted after terminal status.",
            )

        if action.tool is ToolName.FINISH:
            return self._execute_finish(state, action)

        if state.rollback_required:
            return self._reject(
                state,
                action,
                "Runtime transactional rollback must complete before another "
                "policy-selected action.",
            )

        state.usage.elapsed_seconds = perf_counter() - self._started_at

        if state.usage.exhausted(state.budget):
            state.status = AgentStatus.BUDGET_EXHAUSTED
            state.last_failure_category = FailureCategory.BUDGET_EXHAUSTED
            state.terminal_failure_category = FailureCategory.BUDGET_EXHAUSTED
            state.final_message = "A global execution budget was exhausted."
            return self._reject(
                state,
                action,
                state.final_message,
            )

        if action.tool is ToolName.APPLY_PATCH and state.usage.patch_limit_reached(
            state.budget
        ):
            return self._reject(
                state,
                action,
                "The patch-attempt budget has been exhausted.",
            )

        if state.status is AgentStatus.READY:
            state.status = AgentStatus.RUNNING

        state.usage.steps += 1
        state.usage.tool_calls += 1

        if action.tool is ToolName.APPLY_PATCH:
            state.usage.patch_attempts += 1

        if self.verification_mode is VerificationMode.STRICT:
            if (
                action.tool is ToolName.APPLY_PATCH
                and state.current_attempt_id is not None
                and not state.syntax_check_required
            ):
                return self._reject(
                    state,
                    action,
                    "The current patch attempt must pass the full test suite or "
                    "be rolled back before another patch attempt.",
                )

            if state.syntax_check_required and action.tool in {
                ToolName.APPLY_PATCH,
                ToolName.RUN_TESTS,
            }:
                return self._reject(
                    state,
                    action,
                    "A passing syntax check is required for the current repository "
                    "revision before tests or another patch.",
                )
        elif (
            action.tool is ToolName.APPLY_PATCH and state.current_attempt_id is not None
        ):
            self.patch_manager.accept_attempt(state.current_attempt_id)
            self._clear_current_attempt(state)

        no_progress_reason = self.repeated_action_guard.rejection_reason(
            state,
            action,
        )
        if no_progress_reason is not None:
            state.no_progress_streak += 1
            if (
                state.no_progress_streak
                >= self.repeated_action_guard.max_no_progress_events
            ):
                state.status = AgentStatus.ESCALATED
                state.last_failure_category = FailureCategory.NO_PROGRESS
                state.terminal_failure_category = FailureCategory.NO_PROGRESS
                state.final_message = (
                    "Run escalated after repeated no-progress action patterns."
                )
                no_progress_reason = f"{no_progress_reason} {state.final_message}"

            return self._reject(
                state,
                action,
                no_progress_reason,
            )

        if action.tool is ToolName.RESTORE_FILE and not state.changed_files:
            return self._reject(
                state,
                action,
                "No changed files are available to restore.",
            )

        observation, test_target = self._dispatch(action)

        if action.tool is ToolName.APPLY_PATCH:
            if observation.status is ObservationStatus.REJECTED:
                state.patch_rejection_count += 1
                state.last_failure_category = FailureCategory.PATCH_REJECTED
            elif observation.status in {
                ObservationStatus.ERROR,
                ObservationStatus.TIMEOUT,
            }:
                state.patch_application_failure_count += 1
                state.last_failure_category = FailureCategory.PATCH_APPLICATION_ERROR

        if action.tool is ToolName.CHECK_SYNTAX and observation.status in {
            ObservationStatus.ERROR,
            ObservationStatus.TIMEOUT,
        }:
            state.verification_failure_count += 1
            state.last_failure_category = FailureCategory.SYNTAX_VERIFICATION_FAILED

        if action.tool is ToolName.RUN_TESTS and observation.status in {
            ObservationStatus.ERROR,
            ObservationStatus.TIMEOUT,
        }:
            state.verification_failure_count += 1
            state.last_failure_category = FailureCategory.TEST_VERIFICATION_FAILED

        if (
            action.tool is ToolName.APPLY_PATCH
            and observation.status is ObservationStatus.OK
        ) or (
            action.tool is ToolName.RESTORE_FILE
            and observation.status is ObservationStatus.OK
        ):
            state.repository_revision += 1
            state.changed_files = list(self.patch_manager.changed_files)
            self._sync_current_attempt(state)
            self._invalidate_verification(state)
            self._update_syntax_gate(state)

        elif action.tool is ToolName.CHECK_SYNTAX:
            if observation.status is ObservationStatus.OK:
                state.syntax_verified_revision = state.repository_revision
            else:
                state.syntax_verified_revision = None
                if observation.status in {
                    ObservationStatus.ERROR,
                    ObservationStatus.TIMEOUT,
                }:
                    self._mark_failed_attempt(
                        state,
                        ToolName.CHECK_SYNTAX,
                    )

        elif action.tool is ToolName.RUN_TESTS:
            if test_target is None and observation.status is ObservationStatus.OK:
                state.full_suite_passed = True
                state.verified_revision = state.repository_revision
                if state.current_attempt_id is not None:
                    self.patch_manager.accept_attempt(state.current_attempt_id)
                    self._clear_current_attempt(state)
            elif observation.status in {
                ObservationStatus.ERROR,
                ObservationStatus.TIMEOUT,
            }:
                self._invalidate_verification(state)
                self._mark_failed_attempt(
                    state,
                    ToolName.RUN_TESTS,
                )

        return self._record(
            state,
            action,
            observation,
        )
