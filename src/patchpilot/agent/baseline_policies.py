"""Reproducible fixed-workflow and one-shot evaluation baselines."""

from __future__ import annotations

from patchpilot.agent.llm_policy import (
    PolicyResponseError,
    StructuredLLMPolicy,
)
from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    ToolName,
)


class FixedWorkflowPolicy(StructuredLLMPolicy):
    """Fixed staged repair baseline with bounded retries.

    The model generates patch content, while the policy chooses the
    deterministic test-search-read-patch-verify workflow.
    """


class OneShotRepairPolicy(FixedWorkflowPolicy):
    """Fixed-workflow baseline limited to one applied patch attempt."""

    @staticmethod
    def _finish(
        status: str,
        message: str,
    ) -> AgentDecision:
        return StructuredLLMPolicy._make_decision(
            summary=f"Finish one-shot baseline with status {status}.",
            plan="Stop after the single allowed repair attempt.",
            tool=ToolName.FINISH,
            arguments={
                "status": status,
                "message": message,
            },
            rationale=(
                "The one-shot baseline does not perform another repair attempt."
            ),
        )

    def decide(self, state: AgentState) -> AgentDecision:
        """Run one complete inspect-patch-syntax-test sequence."""
        if state.rollback_required:
            raise PolicyResponseError(
                "Runtime transactional rollback must complete before the "
                "one-shot policy can continue."
            )

        if not state.actions or not state.observations:
            return self._make_decision(
                summary="Run the full test suite before one-shot repair.",
                plan="Reproduce the failing test signal.",
                tool=ToolName.RUN_TESTS,
                rationale="Establish the baseline failure signal first.",
            )

        last_action = state.actions[-1]
        last_observation = state.observations[-1]

        if last_action.tool is ToolName.RUN_TESTS:
            if last_observation.status is ObservationStatus.OK:
                return self._finish(
                    "succeeded",
                    "Full test suite passed after one-shot repair.",
                )

            if state.usage.patch_attempts >= 1:
                return self._finish(
                    "escalated",
                    "The one-shot patch did not pass full-suite verification.",
                )

            return self._make_decision(
                summary="Search source code for the failing symbol.",
                plan="Locate the implementation connected to the failing tests.",
                tool=ToolName.SEARCH_CODE,
                arguments={
                    "query": self._failure_query(state),
                    "relative_path": state.task.allowed_paths[0],
                },
                rationale="Find the likely defective source implementation.",
            )

        if last_action.tool is ToolName.SEARCH_CODE:
            if last_observation.status is not ObservationStatus.OK:
                return self._finish(
                    "escalated",
                    "The one-shot baseline could not locate relevant source code.",
                )

            try:
                relative_path = self._source_path_from_search(
                    last_observation.output,
                    state.task.allowed_paths,
                )
            except PolicyResponseError:
                return self._finish(
                    "escalated",
                    "The one-shot baseline could not extract a source path.",
                )

            return self._make_decision(
                summary="Read the source file found by search.",
                plan="Inspect the candidate defective implementation.",
                tool=ToolName.READ_FILE,
                arguments={"relative_path": relative_path},
                rationale="Read the source before generating one patch.",
            )

        if last_action.tool is ToolName.READ_FILE:
            if last_observation.status is not ObservationStatus.OK:
                return self._finish(
                    "escalated",
                    "The one-shot baseline could not read the source file.",
                )

            if state.usage.patch_attempts >= 1:
                return self._finish(
                    "escalated",
                    "The one-shot patch budget was already used.",
                )

            return self._generate_patch_decision(state)

        if last_action.tool is ToolName.APPLY_PATCH:
            if last_observation.status is ObservationStatus.OK:
                return self._make_decision(
                    summary="Check syntax for the one-shot patch.",
                    plan="Validate changed Python files before test execution.",
                    tool=ToolName.CHECK_SYNTAX,
                    rationale="Reject a syntactically invalid one-shot patch.",
                )

            return self._finish(
                "escalated",
                "The one-shot patch was rejected by the patch boundary.",
            )

        if last_action.tool is ToolName.CHECK_SYNTAX:
            if last_observation.status is ObservationStatus.OK:
                return self._make_decision(
                    summary="Run full-suite verification once.",
                    plan="Verify the syntax-checked one-shot patch.",
                    tool=ToolName.RUN_TESTS,
                    rationale="Measure whether the single patch repaired the task.",
                )

            return self._finish(
                "escalated",
                "The one-shot patch failed syntax validation.",
            )

        if last_action.tool is ToolName.RESTORE_FILE:
            return self._finish(
                "escalated",
                "The failed one-shot patch was rolled back transactionally.",
            )

        return self._finish(
            "escalated",
            "The one-shot baseline reached an unsupported state.",
        )
