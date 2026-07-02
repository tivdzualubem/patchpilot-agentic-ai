
from patchpilot.agent import AgentControlLoop, AgentDecision
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    RepairTask,
    ToolAction,
    ToolName,
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
