import pytest

from patchpilot.agent import (
    PolicyResponseError,
    StructuredLLMPolicy,
)
from patchpilot.schemas import AgentState, RepairTask, ToolName


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        assert "PatchPilot" in system_prompt
        assert "Current validated state" in user_prompt
        return self.response


def make_state() -> AgentState:
    task = RepairTask(
        task_id="llm-policy-001",
        goal="Repair the defective Python implementation.",
        repository_root="benchmarks/example",
    )
    return AgentState(task=task)


def valid_response() -> str:
    return """
{
  "reasoning_summary": "Inspect the repository first.",
  "plan": ["List repository files."],
  "hypothesis": null,
  "reflection": null,
  "action": {
    "tool": "list_files",
    "arguments": {"relative_path": "."},
    "rationale": "Inspect the repository structure."
  }
}
"""


def test_valid_json_creates_decision() -> None:
    policy = StructuredLLMPolicy(FakeModel(valid_response()))

    decision = policy.decide(make_state())

    assert decision.action.tool is ToolName.LIST_FILES
    assert decision.plan == ["List repository files."]


def test_json_code_fence_is_supported() -> None:
    response = f"```json\n{valid_response()}\n```"
    policy = StructuredLLMPolicy(FakeModel(response))

    decision = policy.decide(make_state())

    assert decision.action.tool is ToolName.LIST_FILES


def test_invalid_response_fails_safely() -> None:
    policy = StructuredLLMPolicy(
        FakeModel('{"reasoning_summary": "missing action"}')
    )

    with pytest.raises(PolicyResponseError):
        policy.decide(make_state())
