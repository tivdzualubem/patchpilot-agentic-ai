"""Final regression tests for agent tool-argument and transition plumbing."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from patchpilot.agent import LLMToolPolicy, ReflectiveLLMToolPolicy
from patchpilot.agent.executor import (
    tool_argument_schema,
    validate_tool_arguments,
)
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
        self.schemas: list[dict[str, object] | None] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt
        if not self.responses:
            raise AssertionError("No scripted response remains.")
        self.calls += 1
        self.schemas.append(response_schema)
        return self.responses.pop(0)


class NoCallModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt, response_schema
        raise AssertionError("A mandatory transition called the model.")


def task() -> RepairTask:
    return RepairTask(
        task_id="final-agent-runtime",
        goal="Repair the defective add function and pass all tests.",
        repository_root="repository",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )


def append(
    state: AgentState,
    tool: ToolName,
    status: ObservationStatus,
    *,
    arguments: dict[str, object] | None = None,
    output: str = "",
) -> None:
    state.actions.append(
        ToolAction(
            tool=tool,
            arguments=arguments or {},
            rationale="Advance the bounded repair.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=tool,
            status=status,
            summary=f"{tool.value} completed.",
            output=output,
        )
    )


def failed_state() -> AgentState:
    state = AgentState(task=task())
    append(
        state,
        ToolName.RUN_TESTS,
        ObservationStatus.ERROR,
        output="E assert -1 == 5\nE where -1 = add(2, 3)",
    )
    return state


def decision_json(
    tool: str,
    arguments: dict[str, object],
    *,
    reflection: str | None = None,
    hypothesis: str | None = None,
) -> str:
    return json.dumps(
        {
            "reasoning_summary": "Choose one evidence-grounded action.",
            "plan": ["Advance the bounded repair."],
            "hypothesis": hypothesis,
            "reflection": reflection,
            "action": {
                "tool": tool,
                "arguments": arguments,
                "rationale": "Use repository evidence.",
            },
        }
    )


def test_executor_schema_is_single_source_of_truth() -> None:
    schema = tool_argument_schema(ToolName.READ_FILE)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["relative_path"]
    assert set(schema["properties"]) == {
        "relative_path",
        "start_line",
        "end_line",
    }

    with pytest.raises(ValidationError):
        validate_tool_arguments(
            ToolName.READ_FILE,
            {"path": "src/calculator.py"},
        )


def test_policy_schema_exposes_exact_single_tool_arguments() -> None:
    state = failed_state()
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:4:def add(left, right):",
    )

    schema = LLMToolPolicy._decision_response_schema(state)
    action = schema["properties"]["action"]
    arguments = action["properties"]["arguments"]

    assert action["properties"]["tool"]["enum"] == ["read_file"]
    assert arguments["additionalProperties"] is False
    assert arguments["required"] == ["relative_path"]


def test_wrong_search_arguments_are_canonicalized() -> None:
    model = ScriptedModel(
        [
            decision_json(
                "search_code",
                {
                    "path": "tests",
                    "query": "fix read_file arguments",
                },
            )
        ]
    )

    decision = LLMToolPolicy(model).decide(failed_state())

    assert decision.action.tool is ToolName.SEARCH_CODE
    assert decision.action.arguments == {
        "query": "add",
        "relative_path": "src",
    }


def test_successful_search_forces_valid_read_without_model_call() -> None:
    state = failed_state()
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:4:def add(left, right):",
    )

    decision = LLMToolPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {"relative_path": "src/calculator.py"}


def test_successful_read_uses_dedicated_patch_generation() -> None:
    state = failed_state()
    append(
        state,
        ToolName.SEARCH_CODE,
        ObservationStatus.OK,
        arguments={"query": "add", "relative_path": "src"},
        output="src/calculator.py:4:def add(left, right):",
    )
    append(
        state,
        ToolName.READ_FILE,
        ObservationStatus.OK,
        arguments={"relative_path": "src/calculator.py"},
        output=(
            "4: def add(left: int, right: int) -> int:\n5:     return left - right"
        ),
    )
    model = ScriptedModel(
        [("def add(left: int, right: int) -> int:\n    return left + right")]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert model.calls == 1
    assert decision.action.tool is ToolName.APPLY_PATCH
    assert "return left + right" in str(decision.action.arguments["patch_text"])


def test_reflective_rollback_keeps_reflection_and_normalizes_path() -> None:
    state = failed_state()
    state.current_hypothesis = "The operator alone is wrong."
    state.last_failed_attempt_id = 1
    state.last_failed_attempt_files = ["src/calculator.py"]
    state.last_failed_verification_tool = ToolName.RUN_TESTS
    state.last_rolled_back_attempt_id = 1
    state.last_rolled_back_attempt_files = ["src/calculator.py"]

    append(
        state,
        ToolName.RESTORE_FILE,
        ObservationStatus.OK,
        arguments={"scope": "failed_attempt", "attempt_id": 1},
        output="src/calculator.py",
    )

    model = ScriptedModel(
        [
            decision_json(
                "read_file",
                {"path": "invented.py"},
                reflection=(
                    "The prior hypothesis was incomplete because the "
                    "verification evidence still failed."
                ),
                hypothesis=(
                    "The implementation must be revised while preserving "
                    "the public API."
                ),
            )
        ]
    )

    decision = ReflectiveLLMToolPolicy(model).decide(state)

    assert model.calls == 1
    assert decision.reflection is not None
    assert decision.hypothesis != state.current_hypothesis
    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {
        "relative_path": "src/calculator.py",
    }
