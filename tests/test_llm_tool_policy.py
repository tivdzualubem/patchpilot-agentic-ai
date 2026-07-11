"""Tests for genuine model-directed tool selection."""

from __future__ import annotations

import json

import pytest

from patchpilot.agent import LLMToolPolicy, PolicyResponseError
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []
        self.response_schemas: list[dict[str, object] | None] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        if not self.responses:
            raise AssertionError("No scripted model response remains.")

        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        self.response_schemas.append(response_schema)
        return self.responses.pop(0)


def make_task() -> RepairTask:
    return RepairTask(
        task_id="tool-agent-001",
        goal="Repair the defective add function and verify all tests.",
        repository_root="benchmarks/example",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )


def decision_json(
    *,
    tool: str = "run_tests",
    arguments: dict[str, object] | None = None,
    plan: list[str] | None = None,
    reflection: str | None = None,
    extra: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "reasoning_summary": "Gather executable evidence before modifying source.",
        "plan": plan if plan is not None else ["Reproduce the reported failure."],
        "hypothesis": "The add implementation may use the wrong operator.",
        "reflection": reflection,
        "action": {
            "tool": tool,
            "arguments": arguments or {},
            "rationale": "Choose the next bounded repair action.",
        },
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


def failed_state() -> AgentState:
    state = AgentState(task=make_task())
    state.actions.append(
        ToolAction(
            tool=ToolName.RUN_TESTS,
            arguments={},
            rationale="Reproduce the failure.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RUN_TESTS,
            status=ObservationStatus.ERROR,
            summary="Tests failed.",
            output="E assert -1 == 5\nE where -1 = add(2, 3)",
        )
    )
    return state


def test_model_selects_first_tool_and_receives_decision_schema() -> None:
    model = ScriptedModel(
        [
            decision_json(
                tool="list_files",
                arguments={"relative_path": "src"},
            )
        ]
    )
    policy = LLMToolPolicy(model)

    decision = policy.decide(AgentState(task=make_task()))

    assert model.calls == 1
    assert decision.action.tool is ToolName.LIST_FILES
    assert decision.action.arguments == {"relative_path": "src"}
    assert model.response_schemas[0] is not None
    assert "properties" in model.response_schemas[0]
    assert "list_files" in model.system_prompts[0]
    assert "tool-agent-001" in model.user_prompts[0]
    assert '"allowed_paths": [' in model.user_prompts[0]


def test_recent_failure_evidence_is_included_in_prompt() -> None:
    model = ScriptedModel(
        [
            decision_json(
                tool="search_code",
                arguments={
                    "query": "add",
                    "relative_path": "src",
                },
            )
        ]
    )

    decision = LLMToolPolicy(model).decide(failed_state())

    assert decision.action.tool is ToolName.SEARCH_CODE
    assert "E where -1 = add(2, 3)" in model.user_prompts[0]
    assert '"status": "error"' in model.user_prompts[0]


def test_markdown_fenced_json_is_accepted() -> None:
    response = "```json\n" + decision_json() + "\n```"
    decision = LLMToolPolicy(ScriptedModel([response])).decide(
        AgentState(task=make_task())
    )

    assert decision.action.tool is ToolName.RUN_TESTS


def test_malformed_json_is_retried_once() -> None:
    model = ScriptedModel(
        [
            "not valid json",
            decision_json(
                tool="read_file",
                arguments={"relative_path": "src/calculator.py"},
            ),
        ]
    )
    policy = LLMToolPolicy(model)

    decision = policy.decide(failed_state())

    assert model.calls == 2
    assert decision.action.tool is ToolName.READ_FILE
    assert "CORRECTION REQUIRED" in model.user_prompts[1]
    assert "not valid json" in model.user_prompts[1]


def test_invalid_tool_after_retry_preserves_last_raw_response() -> None:
    invalid = decision_json(tool="delete_repository")
    model = ScriptedModel([invalid, invalid])

    with pytest.raises(PolicyResponseError) as captured:
        LLMToolPolicy(model).decide(failed_state())

    assert model.calls == 2
    assert captured.value.raw_response == invalid
    assert "valid tool decision" in str(captured.value)


def test_no_reflection_policy_rejects_reflection() -> None:
    raw = decision_json(
        reflection="The first repair hypothesis was incomplete.",
    )

    with pytest.raises(
        PolicyResponseError,
        match="reflection to be null",
    ):
        LLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(failed_state())


def test_empty_plan_is_rejected() -> None:
    raw = decision_json(plan=[])

    with pytest.raises(
        PolicyResponseError,
        match="at least one plan step",
    ):
        LLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(AgentState(task=make_task()))


def test_extra_top_level_field_is_rejected_by_schema() -> None:
    raw = decision_json(extra={"unexpected": "not allowed"})

    with pytest.raises(
        PolicyResponseError,
        match="schema validation",
    ):
        LLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(AgentState(task=make_task()))


@pytest.mark.parametrize("attempts", [0, 4])
def test_invalid_parse_attempt_limit_is_rejected(attempts: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        LLMToolPolicy(
            ScriptedModel([decision_json()]),
            max_parse_attempts=attempts,
        )
