"""Tests for explicit reflective tool decisions."""

from __future__ import annotations

import json

import pytest

from patchpilot.agent import (
    PolicyResponseError,
    ReflectiveLLMToolPolicy,
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
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del response_schema
        if not self.responses:
            raise AssertionError("No scripted model response remains.")

        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.responses.pop(0)


def make_task() -> RepairTask:
    return RepairTask(
        task_id="reflective-agent-001",
        goal="Repair the defective add function and verify all tests.",
        repository_root="benchmarks/example",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )


def decision_json(
    *,
    reflection: str | None,
    hypothesis: str | None,
    tool: str = "restore_file",
) -> str:
    return json.dumps(
        {
            "reasoning_summary": (
                "Use the failed verification evidence to revise the repair."
            ),
            "plan": ["Rollback the failed patch and inspect the clean source."],
            "hypothesis": hypothesis,
            "reflection": reflection,
            "action": {
                "tool": tool,
                "arguments": {"relative_path": "src/calculator.py"},
                "rationale": "Continue from a clean, evidence-grounded state.",
            },
        }
    )


def failed_verification_state() -> AgentState:
    state = AgentState(task=make_task())
    state.current_hypothesis = "The comparison operator is incorrect."
    state.changed_files = ["src/calculator.py"]
    state.repository_revision = 1

    state.actions.extend(
        [
            ToolAction(
                tool=ToolName.APPLY_PATCH,
                arguments={"patch_text": "diff --git ..."},
                rationale="Apply the first hypothesis.",
            ),
            ToolAction(
                tool=ToolName.RUN_TESTS,
                arguments={},
                rationale="Verify the patched repository.",
            ),
        ]
    )
    state.observations.extend(
        [
            ToolObservation(
                tool=ToolName.APPLY_PATCH,
                status=ObservationStatus.OK,
                summary="Patch applied.",
            ),
            ToolObservation(
                tool=ToolName.RUN_TESTS,
                status=ObservationStatus.ERROR,
                summary="Tests still fail.",
                output="Two boundary tests still fail.",
            ),
        ]
    )
    return state


def failed_syntax_state() -> AgentState:
    state = AgentState(task=make_task())
    state.current_hypothesis = "The replacement expression is valid."
    state.changed_files = ["src/calculator.py"]
    state.repository_revision = 1
    state.actions.extend(
        [
            ToolAction(
                tool=ToolName.APPLY_PATCH,
                arguments={"patch_text": "diff --git ..."},
                rationale="Apply the first hypothesis.",
            ),
            ToolAction(
                tool=ToolName.CHECK_SYNTAX,
                arguments={},
                rationale="Validate changed Python syntax.",
            ),
        ]
    )
    state.observations.extend(
        [
            ToolObservation(
                tool=ToolName.APPLY_PATCH,
                status=ObservationStatus.OK,
                summary="Patch applied.",
            ),
            ToolObservation(
                tool=ToolName.CHECK_SYNTAX,
                status=ObservationStatus.ERROR,
                summary="Python syntax check failed.",
                output="src/calculator.py:5:12: invalid syntax",
            ),
        ]
    )
    return state


def initial_state() -> AgentState:
    return AgentState(task=make_task())


def test_failed_verification_requires_and_accepts_reflection() -> None:
    model = ScriptedModel(
        [
            decision_json(
                reflection=(
                    "The comparison-only hypothesis did not explain "
                    "the remaining boundary failures."
                ),
                hypothesis="The loop bound is off by one.",
            )
        ]
    )

    decision = ReflectiveLLMToolPolicy(model).decide(failed_verification_state())

    assert decision.reflection is not None
    assert decision.hypothesis == "The loop bound is off by one."
    assert decision.action.tool is ToolName.RESTORE_FILE
    assert "REFLECTION REQUIRED" in model.user_prompts[0]
    assert "failed post-patch verification" in model.system_prompts[0]


def test_missing_reflection_after_failed_verification_is_rejected() -> None:
    raw = decision_json(
        reflection=None,
        hypothesis="The loop bound is off by one.",
    )

    with pytest.raises(
        PolicyResponseError,
        match="meaningful reflection",
    ):
        ReflectiveLLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(failed_verification_state())


def test_missing_revised_hypothesis_is_rejected() -> None:
    raw = decision_json(
        reflection="The previous explanation did not fit the new evidence.",
        hypothesis=None,
    )

    with pytest.raises(
        PolicyResponseError,
        match="revised non-empty hypothesis",
    ):
        ReflectiveLLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(failed_verification_state())


def test_unchanged_hypothesis_is_rejected() -> None:
    raw = decision_json(
        reflection="The previous explanation should be reconsidered.",
        hypothesis="The comparison operator is incorrect.",
    )

    with pytest.raises(
        PolicyResponseError,
        match="must differ",
    ):
        ReflectiveLLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(failed_verification_state())


def test_reflection_is_rejected_when_not_required() -> None:
    raw = decision_json(
        reflection="Reflect even though no patch has failed yet.",
        hypothesis="Inspect the source before forming a repair.",
        tool="read_file",
    )

    with pytest.raises(
        PolicyResponseError,
        match="only allowed",
    ):
        ReflectiveLLMToolPolicy(
            ScriptedModel([raw]),
            max_parse_attempts=1,
        ).decide(initial_state())


def test_non_reflective_initial_decision_is_accepted() -> None:
    raw = json.dumps(
        {
            "reasoning_summary": "Reproduce the failure first.",
            "plan": ["Run the full test suite."],
            "hypothesis": None,
            "reflection": None,
            "action": {
                "tool": "run_tests",
                "arguments": {},
                "rationale": "Establish executable failure evidence.",
            },
        }
    )

    decision = ReflectiveLLMToolPolicy(ScriptedModel([raw])).decide(initial_state())

    assert decision.action.tool is ToolName.RUN_TESTS
    assert decision.reflection is None


def test_invalid_reflection_is_retried_with_correction_prompt() -> None:
    model = ScriptedModel(
        [
            decision_json(
                reflection=None,
                hypothesis="The loop bound is off by one.",
            ),
            decision_json(
                reflection=(
                    "The first hypothesis failed to explain the boundary-test evidence."
                ),
                hypothesis="The loop excludes the final item.",
            ),
        ]
    )

    decision = ReflectiveLLMToolPolicy(model).decide(failed_verification_state())

    assert model.calls == 2
    assert decision.hypothesis == "The loop excludes the final item."
    assert "CORRECTION REQUIRED" in model.user_prompts[1]


def test_failed_syntax_check_requires_reflection() -> None:
    model = ScriptedModel(
        [
            decision_json(
                reflection=(
                    "The proposed replacement introduced an incomplete expression "
                    "and therefore could not be parsed."
                ),
                hypothesis="The operator must be corrected without changing grouping.",
            )
        ]
    )

    decision = ReflectiveLLMToolPolicy(model).decide(failed_syntax_state())

    assert decision.reflection is not None
    assert decision.hypothesis == (
        "The operator must be corrected without changing grouping."
    )
    assert "REFLECTION REQUIRED" in model.user_prompts[0]
