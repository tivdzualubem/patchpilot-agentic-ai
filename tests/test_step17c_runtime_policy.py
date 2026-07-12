"""Regression tests for Step 17C bounded model runtime and legal actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from patchpilot.agent import LLMToolPolicy, PolicyResponseError
from patchpilot.models import OllamaChatModel
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


class ScriptedModel:
    """Return scripted decisions and retain correction prompts."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.user_prompts: list[str] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, response_schema
        self.user_prompts.append(user_prompt)
        return self.responses.pop(0)


def make_task() -> RepairTask:
    return RepairTask(
        task_id="step17c",
        goal="Repair the defective add function and pass all tests.",
        repository_root="repository",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )


def decision_json(
    tool: str,
    *,
    arguments: dict[str, object] | None = None,
) -> str:
    return json.dumps(
        {
            "reasoning_summary": (
                "Use the current evidence to select the next action."
            ),
            "plan": ["Advance the bounded repair."],
            "hypothesis": None,
            "reflection": None,
            "action": {
                "tool": tool,
                "arguments": arguments or {},
                "rationale": "Choose one currently legal action.",
            },
        }
    )


def failed_test_state() -> AgentState:
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


def read_after_search_state() -> AgentState:
    state = failed_test_state()
    state.actions.extend(
        [
            ToolAction(
                tool=ToolName.SEARCH_CODE,
                arguments={"query": "add", "relative_path": "src"},
                rationale="Locate the source.",
            ),
            ToolAction(
                tool=ToolName.READ_FILE,
                arguments={"relative_path": "src/calculator.py"},
                rationale="Inspect the source.",
            ),
        ]
    )
    state.observations.extend(
        [
            ToolObservation(
                tool=ToolName.SEARCH_CODE,
                status=ObservationStatus.OK,
                summary="Found one source match.",
                output="src/calculator.py:1:def add(left, right):",
            ),
            ToolObservation(
                tool=ToolName.READ_FILE,
                status=ObservationStatus.OK,
                summary="Read source.",
                output=("1: def add(left, right):\n2:     return left - right"),
            ),
        ]
    )
    return state


def schema_tools(schema: dict[str, object]) -> list[str]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    action = properties["action"]
    assert isinstance(action, dict)
    action_properties = action["properties"]
    assert isinstance(action_properties, dict)
    tool = action_properties["tool"]
    assert isinstance(tool, dict)
    values = tool["enum"]
    assert isinstance(values, list)
    return [str(value) for value in values]


def test_dynamic_schema_restricts_initial_action_to_tests() -> None:
    state = AgentState(task=make_task())
    schema = LLMToolPolicy._decision_response_schema(state)

    assert schema_tools(schema) == ["run_tests"]


def test_dynamic_schema_converges_from_search_and_read_to_patch() -> None:
    state = read_after_search_state()
    schema = LLMToolPolicy._decision_response_schema(state)

    assert schema_tools(schema) == ["apply_patch"]


def test_illegal_repeated_action_receives_legal_choices() -> None:
    state = failed_test_state()
    model = ScriptedModel(
        [
            decision_json("run_tests"),
            decision_json(
                "search_code",
                arguments={"query": "add", "relative_path": "src"},
            ),
        ]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert decision.action.tool is ToolName.SEARCH_CODE
    assert "exactly repeats the latest action" in model.user_prompts[1]
    assert "LEGAL ACTIONS NOW" in model.user_prompts[1]


def test_compact_prompt_bounds_large_observation() -> None:
    state = failed_test_state()
    state.observations[-1].output = "x" * 20_000

    prompt = LLMToolPolicy._state_prompt(state)

    assert "...[observation output truncated]" in prompt
    assert len(prompt) < 3500


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return self._body


def test_ollama_runtime_caps_and_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request: object, timeout: int) -> _FakeResponse:
        del timeout
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"message": {"content": "OK"}})

    monkeypatch.setattr(
        "patchpilot.models.ollama.urlopen",
        fake_urlopen,
    )

    model = OllamaChatModel(max_tokens=512)
    model.warmup()
    model.generate("system", "user")
    model.generate(
        "system",
        "user",
        response_schema={
            "type": "object",
            "properties": {},
        },
    )

    warmup, unstructured, structured = payloads
    assert warmup["keep_alive"] == "30m"
    assert warmup["options"]["num_predict"] == 1
    assert warmup["options"]["num_ctx"] == 4096
    assert unstructured["options"]["num_predict"] == 384
    assert structured["options"]["num_predict"] == 256
    assert structured["format"] == {
        "type": "object",
        "properties": {},
    }


def test_evaluation_warms_and_reuses_one_model() -> None:
    source = Path("scripts/run_evaluation.py").read_text(encoding="utf-8")

    assert "model = build_model(args)" in source
    assert "model.warmup()" in source
    assert "build_condition(condition, model)" in source


def test_schema_validation_rejects_non_legal_tool() -> None:
    state = read_after_search_state()
    raw = decision_json("restore_file")

    with pytest.raises(PolicyResponseError, match="not currently legal"):
        LLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(state)
