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
                tool="run_tests",
                arguments={},
            )
        ]
    )
    policy = LLMToolPolicy(model)

    decision = policy.decide(AgentState(task=make_task()))

    assert model.calls == 1
    assert decision.action.tool is ToolName.RUN_TESTS
    assert decision.action.arguments == {}
    assert model.response_schemas[0] is not None
    assert "properties" in model.response_schemas[0]
    assert "list_files" in model.system_prompts[0]
    assert "check_syntax" in model.system_prompts[0]
    assert "After every successful patch" in model.system_prompts[0]
    assert "tool-agent-001" in model.user_prompts[0]
    assert '"allowed_paths": [' in model.user_prompts[0]
    assert '"syntax_check_required": false' in model.user_prompts[0]
    assert '"current_attempt_id": null' in model.user_prompts[0]
    assert '"rollback_required": false' in model.user_prompts[0]


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


def test_pending_syntax_gate_is_exposed_to_model() -> None:
    state = AgentState(task=make_task())
    state.changed_files = ["src/calculator.py"]
    state.repository_revision = 1
    model = ScriptedModel(
        [
            decision_json(
                tool="check_syntax",
                arguments={},
            )
        ]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert decision.action.tool is ToolName.CHECK_SYNTAX
    assert '"syntax_check_required": true' in model.user_prompts[0]
    assert '"syntax_verified_revision": null' in model.user_prompts[0]


def test_transactional_attempt_state_is_exposed_to_model() -> None:
    state = AgentState(task=make_task())
    state.current_attempt_id = 3
    state.current_attempt_files = ["src/calculator.py"]
    state.last_failed_attempt_id = 2
    state.last_failed_attempt_files = ["src/helpers.py"]
    state.last_rolled_back_attempt_id = 2
    state.last_rolled_back_attempt_files = ["src/helpers.py"]
    model = ScriptedModel(
        [
            decision_json(
                tool="run_tests",
                arguments={},
            )
        ]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert decision.action.tool is ToolName.RUN_TESTS
    prompt = model.user_prompts[0]
    assert '"current_attempt_id": 3' in prompt
    assert '"last_failed_attempt_id": 2' in prompt
    assert '"last_rolled_back_attempt_id": 2' in prompt


def test_model_decision_is_rejected_while_runtime_rollback_is_pending() -> None:
    state = AgentState(task=make_task())
    state.current_attempt_id = 1
    state.current_attempt_files = ["src/calculator.py"]
    state.rollback_required = True
    model = ScriptedModel(
        [
            decision_json(
                tool="read_file",
                arguments={"relative_path": "src/calculator.py"},
            )
        ]
    )

    with pytest.raises(
        PolicyResponseError,
        match="transactional rollback",
    ):
        LLMToolPolicy(
            model,
            max_parse_attempts=1,
        ).decide(state)

    assert model.calls == 1


def test_model_call_and_parse_retry_accounting() -> None:
    state = failed_state()
    model = ScriptedModel(
        [
            "not valid json",
            decision_json(
                tool="read_file",
                arguments={"relative_path": "src/calculator.py"},
            ),
        ]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert state.model_calls == 2
    assert state.decision_parse_failures == 1


def test_exhausted_parse_retries_are_all_counted() -> None:
    state = failed_state()
    invalid = decision_json(tool="delete_repository")

    with pytest.raises(PolicyResponseError):
        LLMToolPolicy(
            ScriptedModel([invalid, invalid]),
        ).decide(state)

    assert state.model_calls == 2
    assert state.decision_parse_failures == 2


def test_tool_policy_model_call_trace_contains_schema_and_response() -> None:
    state = failed_state()
    raw = decision_json(
        tool="read_file",
        arguments={"relative_path": "src/calculator.py"},
    )

    decision = LLMToolPolicy(ScriptedModel([raw])).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert len(state.model_call_records) == 1
    record = state.model_call_records[0]
    assert record.policy == "LLMToolPolicy"
    assert record.purpose == "tool_decision"
    assert record.attempt == 1
    assert record.response_schema is not None
    assert "properties" in record.response_schema
    assert record.raw_response == raw
    assert record.generation_succeeded is True
    assert record.parse_succeeded is True
    assert "CURRENT REPAIR STATE" in record.user_prompt


def test_tool_policy_retry_trace_marks_invalid_then_valid() -> None:
    state = failed_state()
    valid = decision_json(
        tool="read_file",
        arguments={"relative_path": "src/calculator.py"},
    )

    LLMToolPolicy(ScriptedModel(["not valid json", valid])).decide(state)

    assert len(state.model_call_records) == 2
    first, second = state.model_call_records
    assert first.attempt == 1
    assert first.parse_succeeded is False
    assert first.error_type == "PolicyResponseError"
    assert second.attempt == 2
    assert second.parse_succeeded is True
    assert "CORRECTION REQUIRED" in second.user_prompt
