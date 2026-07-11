"""Bounded Plan-Act-Observe-Reflect-Verify control loop."""

from __future__ import annotations

from patchpilot.agent.executor import AgentToolExecutor
from patchpilot.agent.policy import AgentPolicy
from patchpilot.agent.tracing import TraceRecorder
from patchpilot.schemas import AgentState, AgentStatus


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

    def _checkpoint(
        self,
        state: AgentState,
        run_id: str | None,
        metadata: dict[str, str] | None,
    ) -> None:
        if self.recorder is None:
            return

        if run_id is None:
            raise ValueError("run_id is required when trace recording is enabled.")

        self.recorder.save(state, run_id, metadata)

    def run(
        self,
        state: AgentState,
        run_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentState:
        """Run until verified completion, escalation, or budget exhaustion."""
        self._checkpoint(state, run_id, metadata)

        while state.can_continue:
            try:
                decision = self.policy.decide(state)
            except Exception as exc:
                state.status = AgentStatus.ESCALATED
                detail = str(exc).strip()
                raw_response = getattr(exc, "raw_response", None)
                if raw_response:
                    detail = f"{detail} Raw response: {raw_response[:500]}"
                state.final_message = (
                    f"The decision policy failed safely: {type(exc).__name__}: {detail}"
                )
                self._checkpoint(state, run_id, metadata)
                return state

            if decision.plan:
                state.plan = list(decision.plan)

            previous_hypothesis = state.current_hypothesis

            if decision.reflection is not None:
                state.reflections.append(decision.reflection)

            if decision.hypothesis is not None:
                if (
                    previous_hypothesis is not None
                    and decision.hypothesis != previous_hypothesis
                ):
                    state.rejected_hypotheses.append(previous_hypothesis)
                state.current_hypothesis = decision.hypothesis

            self.executor.execute(state, decision.action)
            self._checkpoint(state, run_id, metadata)

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
            state.final_message = "The configured execution budget was exhausted."
            self._checkpoint(state, run_id, metadata)

        return state
