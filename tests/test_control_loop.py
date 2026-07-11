from patchpilot.agent import AgentControlLoop, AgentDecision
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
    task = RepairTask(
        task_id="control-loop-001",
        goal="Repair the defective Python implementation.",
        repository_root="benchmarks/control-loop",
    )
    return AgentState(task=task)


class SuccessPolicy:
    def decide(self, state: AgentState) -> AgentDecision:
        return AgentDecision(
            reasoning_summary="The task is complete.",
            plan=["Verify completion."],
            action=ToolAction(
                tool=ToolName.FINISH,
                arguments={
                    "status": "succeeded",
                    "message": "Repair verified.",
                },
                rationale="Finish after verification.",
            ),
        )


class SuccessExecutor:
    def execute(
        self,
        state: AgentState,
        action: ToolAction,
    ) -> None:
        state.actions.append(action)
        state.status = AgentStatus.SUCCEEDED
        state.final_message = "Repair verified."


class FailingPolicy:
    def decide(self, state: AgentState) -> AgentDecision:
        raise RuntimeError("model unavailable")


def test_control_loop_reaches_success() -> None:
    state = make_state()
    loop = AgentControlLoop(
        policy=SuccessPolicy(),
        executor=SuccessExecutor(),
    )

    result = loop.run(state)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.plan == ["Verify completion."]
    assert len(result.actions) == 1


def test_policy_failure_escalates_safely() -> None:
    state = make_state()
    loop = AgentControlLoop(
        policy=FailingPolicy(),
        executor=SuccessExecutor(),
    )

    result = loop.run(state)

    assert result.status is AgentStatus.ESCALATED
    assert "RuntimeError" in str(result.final_message)


class ReflectingPolicy:
    def decide(self, state: AgentState) -> AgentDecision:
        return AgentDecision(
            reasoning_summary="Revise the failed repair hypothesis.",
            plan=["Continue using the revised failure explanation."],
            hypothesis="The loop bound is incorrect.",
            reflection=(
                "The comparison-only hypothesis did not explain "
                "the remaining boundary failure."
            ),
            action=ToolAction(
                tool=ToolName.FINISH,
                arguments={
                    "status": "succeeded",
                    "message": "Reflection state recorded.",
                },
                rationale="Stop after recording the revised hypothesis.",
            ),
        )


def test_control_loop_tracks_reflection_and_rejected_hypothesis() -> None:
    state = make_state()
    state.current_hypothesis = "The comparison operator is incorrect."
    state.last_failed_attempt_id = 7

    loop = AgentControlLoop(
        policy=ReflectingPolicy(),
        executor=SuccessExecutor(),
    )
    result = loop.run(state)

    assert result.current_hypothesis == "The loop bound is incorrect."
    assert result.rejected_hypotheses == ["The comparison operator is incorrect."]
    assert result.reflections == [
        (
            "The comparison-only hypothesis did not explain "
            "the remaining boundary failure."
        )
    ]
    assert result.plan == ["Continue using the revised failure explanation."]
    assert result.last_reflected_attempt_id == 7


class RollbackThenStopPolicy:
    def decide(self, state: AgentState) -> AgentDecision:
        if not state.actions:
            return AgentDecision(
                reasoning_summary="Verify the current patch attempt.",
                plan=["Run verification."],
                action=ToolAction(
                    tool=ToolName.RUN_TESTS,
                    rationale="Exercise the patched repository.",
                ),
            )

        assert state.rollback_required is False
        assert state.last_rolled_back_attempt_id == 1
        return AgentDecision(
            reasoning_summary="Stop after the failed attempt is restored.",
            plan=["Record safe failure."],
            action=ToolAction(
                tool=ToolName.FINISH,
                arguments={
                    "status": "failed",
                    "message": "The attempted repair did not verify.",
                },
                rationale="Stop after safe rollback.",
            ),
        )


class TransactionalRollbackExecutor:
    def execute(
        self,
        state: AgentState,
        action: ToolAction,
    ) -> ToolObservation:
        state.actions.append(action)

        if action.tool is ToolName.RUN_TESTS:
            observation = ToolObservation(
                tool=ToolName.RUN_TESTS,
                status=ObservationStatus.ERROR,
                summary="Tests failed.",
            )
            state.current_attempt_id = 1
            state.current_attempt_files = ["src/example.py"]
            state.rollback_required = True
            state.last_failed_attempt_id = 1
            state.last_failed_attempt_files = ["src/example.py"]
        else:
            observation = ToolObservation(
                tool=ToolName.FINISH,
                status=ObservationStatus.OK,
                summary="Run finished.",
            )
            state.status = AgentStatus.FAILED
            state.final_message = "The attempted repair did not verify."

        state.observations.append(observation)
        return observation

    def rollback_failed_attempt(
        self,
        state: AgentState,
    ) -> ToolObservation:
        action = ToolAction(
            tool=ToolName.RESTORE_FILE,
            arguments={"scope": "failed_attempt", "attempt_id": 1},
            rationale="Runtime-enforced transactional rollback.",
        )
        observation = ToolObservation(
            tool=ToolName.RESTORE_FILE,
            status=ObservationStatus.OK,
            summary="Rolled back patch attempt 1 across 1 file(s).",
            output="src/example.py",
        )
        state.actions.append(action)
        state.observations.append(observation)
        state.rollback_required = False
        state.current_attempt_id = None
        state.current_attempt_files = []
        state.last_rolled_back_attempt_id = 1
        state.last_rolled_back_attempt_files = ["src/example.py"]
        return observation


def test_control_loop_rolls_back_failed_attempt_before_next_decision() -> None:
    state = make_state()
    loop = AgentControlLoop(
        policy=RollbackThenStopPolicy(),
        executor=TransactionalRollbackExecutor(),
    )

    result = loop.run(state)

    assert [action.tool for action in result.actions] == [
        ToolName.RUN_TESTS,
        ToolName.RESTORE_FILE,
        ToolName.FINISH,
    ]
    assert result.status is AgentStatus.FAILED
    assert result.rollback_required is False
    assert result.last_rolled_back_attempt_id == 1
