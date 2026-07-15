"""Bounded Plan-Act-Observe-Reflect-Verify control loop."""

from __future__ import annotations

from patchpilot.agent.executor import AgentToolExecutor
from patchpilot.agent.llm_policy import PolicyResponseError
from patchpilot.agent.policy import AgentPolicy
from patchpilot.agent.tracing import TraceRecorder
from patchpilot.schemas import AgentState, AgentStatus, FailureCategory
from patchpilot.schemas.models import DecisionRecord


class AgentControlLoop:
    """Coordinate policy decisions, tools, budgets, and traces."""

    def __init__(
        self,
        policy: AgentPolicy,
        executor: AgentToolExecutor,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.recorder = recorder

    @staticmethod
    def _qualified_name(value: object) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"

    def _checkpoint(
        self,
        state: AgentState,
        run_id: str | None,
        metadata: dict[str, str] | None,
        checkpoint_kind: str,
    ) -> None:
        if self.recorder is None:
            return

        if run_id is None:
            raise ValueError("run_id is required when trace recording is enabled.")

        self.recorder.save(
            state,
            run_id,
            metadata,
            checkpoint_kind=checkpoint_kind,
        )

    def run(
        self,
        state: AgentState,
        run_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentState:
        """Run until verified completion, escalation, or budget exhaustion."""
        self._checkpoint(
            state,
            run_id,
            metadata,
            "initial",
        )

        while state.can_continue:
            model_call_start = state.model_calls + 1
            try:
                decision = self.policy.decide(state)
            except Exception as exc:
                state.status = AgentStatus.ESCALATED
                category = (
                    FailureCategory.DECISION_PARSE_ERROR
                    if isinstance(exc, PolicyResponseError)
                    else FailureCategory.MODEL_ERROR
                )
                state.last_failure_category = category
                state.terminal_failure_category = category
                detail = str(exc).strip()
                raw_response = getattr(exc, "raw_response", None)
                if raw_response:
                    detail = f"{detail} Raw response: {raw_response[:500]}"
                state.final_message = (
                    f"The decision policy failed safely: {type(exc).__name__}: {detail}"
                )
                self._checkpoint(
                    state,
                    run_id,
                    metadata,
                    "policy_failure",
                )
                return state

            model_call_end = state.model_calls
            has_model_calls = model_call_end >= model_call_start
            state.decision_records.append(
                DecisionRecord(
                    decision_index=len(state.decision_records) + 1,
                    policy=self._qualified_name(self.policy),
                    model_call_start=(model_call_start if has_model_calls else None),
                    model_call_end=(model_call_end if has_model_calls else None),
                    reasoning_summary=decision.reasoning_summary,
                    plan=list(decision.plan),
                    hypothesis=decision.hypothesis,
                    reflection=decision.reflection,
                    action=decision.action,
                )
            )

            if decision.plan:
                state.plan = list(decision.plan)

            previous_hypothesis = state.current_hypothesis

            if decision.reflection is not None:
                state.reflections.append(decision.reflection)
                if state.last_failed_attempt_id is not None:
                    state.last_reflected_attempt_id = state.last_failed_attempt_id

            if decision.hypothesis is not None:
                if (
                    previous_hypothesis is not None
                    and decision.hypothesis != previous_hypothesis
                ):
                    state.rejected_hypotheses.append(previous_hypothesis)
                state.current_hypothesis = decision.hypothesis

            self._checkpoint(
                state,
                run_id,
                metadata,
                "post_decision",
            )
            self.executor.execute(state, decision.action)
            self._checkpoint(
                state,
                run_id,
                metadata,
                "post_action",
            )

            if state.rollback_required:
                self.executor.rollback_failed_attempt(state)
                self._checkpoint(
                    state,
                    run_id,
                    metadata,
                    "post_rollback",
                )

            if state.status in {
                AgentStatus.SUCCEEDED,
                AgentStatus.FAILED,
                AgentStatus.ESCALATED,
                AgentStatus.BUDGET_EXHAUSTED,
            }:
                return state

        if state.status not in {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.ESCALATED,
        }:
            state.status = AgentStatus.BUDGET_EXHAUSTED
            state.last_failure_category = FailureCategory.BUDGET_EXHAUSTED
            state.terminal_failure_category = FailureCategory.BUDGET_EXHAUSTED
            state.final_message = "The configured execution budget was exhausted."
            self._checkpoint(
                state,
                run_id,
                metadata,
                "budget_exhausted",
            )

        return state
