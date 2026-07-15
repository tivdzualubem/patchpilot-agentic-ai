"""Regression tests for Step 17B policy semantics and prompt efficiency."""

from __future__ import annotations

import json

from patchpilot.agent import (
    AgentDecision,
    LLMToolPolicy,
    ReflectiveLLMToolPolicy,
    StructuredLLMPolicy,
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
    """Return scripted responses and retain prompts for assertions."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.user_prompts: list[str] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, response_schema
        self.calls += 1
        self.user_prompts.append(user_prompt)
        return self.responses.pop(0)


def make_task() -> RepairTask:
    return RepairTask(
        task_id="step17b-policy",
        goal="Repair the defective add function.",
        repository_root="benchmarks/example",
        allowed_paths=["src"],
        forbidden_paths=["tests"],
    )


def decision_json(
    tool: str,
    *,
    arguments: dict[str, object] | None = None,
    reflection: str | None = None,
    hypothesis: str | None = None,
) -> str:
    return json.dumps(
        {
            "reasoning_summary": "Choose the next evidence-grounded action.",
            "plan": ["Advance the bounded repair using current evidence."],
            "hypothesis": hypothesis,
            "reflection": reflection,
            "action": {
                "tool": tool,
                "arguments": arguments or {},
                "rationale": "Use the current repository evidence.",
            },
        }
    )


def test_tool_schema_requires_plan_without_breaking_programmatic_policy() -> None:
    schema = LLMToolPolicy._decision_response_schema()

    assert "plan" in schema["required"]
    assert schema["properties"]["plan"]["minItems"] == 1

    decision = AgentDecision(
        reasoning_summary="Programmatic policies remain backward compatible.",
        action=ToolAction(
            tool=ToolName.RUN_TESTS,
            arguments={},
            rationale="Establish executable evidence.",
        ),
    )
    assert decision.plan == []


def test_initial_semantic_guard_retries_to_run_tests() -> None:
    model = ScriptedModel(
        [
            decision_json("check_syntax"),
            decision_json("run_tests"),
        ]
    )

    decision = LLMToolPolicy(model).decide(AgentState(task=make_task()))

    assert model.calls == 2
    assert decision.action.tool is ToolName.RUN_TESTS
    assert "first action must be run_tests" in model.user_prompts[1]


def test_reflective_policy_uses_same_initial_semantic_guard() -> None:
    model = ScriptedModel(
        [
            decision_json("list_files"),
            decision_json("run_tests"),
        ]
    )

    decision = ReflectiveLLMToolPolicy(model).decide(AgentState(task=make_task()))

    assert model.calls == 2
    assert decision.action.tool is ToolName.RUN_TESTS


def test_repeated_identical_action_is_corrected_before_execution() -> None:
    state = AgentState(task=make_task())
    state.actions.append(
        ToolAction(
            tool=ToolName.LIST_FILES,
            arguments={"relative_path": "src"},
            rationale="Inspect repository files.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.LIST_FILES,
            status=ObservationStatus.OK,
            summary="Listed files.",
            output="src/calculator.py",
        )
    )
    model = ScriptedModel(
        [
            decision_json(
                "list_files",
                arguments={"relative_path": "src"},
            ),
            decision_json("run_tests"),
        ]
    )

    decision = LLMToolPolicy(model).decide(state)

    assert model.calls == 2
    assert decision.action.tool is ToolName.RUN_TESTS
    assert "exactly repeats the latest action" in model.user_prompts[1]


def test_trajectory_output_is_bounded() -> None:
    state = AgentState(task=make_task())
    state.actions.append(
        ToolAction(
            tool=ToolName.READ_FILE,
            arguments={"relative_path": "src/calculator.py"},
            rationale="Inspect source.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.READ_FILE,
            status=ObservationStatus.OK,
            summary="Read source.",
            output="x" * 5000,
        )
    )

    prompt = LLMToolPolicy._state_prompt(state)

    assert "...[observation output truncated]" in prompt
    assert len(prompt) < 5000


def interval_source() -> str:
    return "\n".join(
        [
            (
                "6: def merge_intervals("
                "intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]:"
            ),
            '7:     """Merge overlapping or touching closed intervals."""',
            ("8:     normalized = [(start, end) for start, end in intervals]"),
            "9:     if not normalized:",
            "10:         return []",
            "11: ",
            ("12:     normalized.sort(key=lambda item: (item[1], item[0]))"),
            ("13:     merged: list[tuple[int, int]] = [normalized[0]]"),
            "14: ",
            "15:     for start, end in normalized[1:]:",
            ("16:         previous_start, previous_end = merged[-1]"),
            "17:         if start < previous_end:",
            ("18:             merged[-1] = (start, max(previous_end, end))"),
            "19:         else:",
            "20:             merged.append((start, end))",
            "21:     return merged",
        ]
    )


def partial_interval_function() -> str:
    return "\n".join(
        [
            (
                "def merge_intervals("
                "intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]:"
            ),
            '    """Merge overlapping or touching closed intervals."""',
            ("    normalized = [(start, end) for start, end in intervals]"),
            "    if not normalized:",
            "        return []",
            "",
            ("    normalized.sort(key=lambda item: (item[1], item[0]))"),
            ("    merged: list[tuple[int, int]] = [normalized[0]]"),
            "",
            "    for start, end in normalized[1:]:",
            "        previous_start, previous_end = merged[-1]",
            "        if start <= previous_end:",
            (
                "            merged[-1] = "
                "(min(previous_start, start), max(previous_end, end))"
            ),
            "        else:",
            "            merged.append((start, end))",
            "    return merged",
        ]
    )


def complete_interval_function() -> str:
    return (
        partial_interval_function()
        .replace(
            "normalized = [(start, end) for start, end in intervals]",
            (
                "normalized = [(min(start, end), max(start, end)) "
                "for start, end in intervals]"
            ),
        )
        .replace(
            "key=lambda item: (item[1], item[0])",
            "key=lambda item: (item[0], item[1])",
        )
    )


def test_failed_patch_is_not_repeated_by_fixed_policy() -> None:
    state = AgentState(
        task=RepairTask(
            task_id="interval-retry",
            goal="Repair merge_intervals.",
            repository_root="repository",
            allowed_paths=["src"],
            forbidden_paths=["tests"],
        )
    )
    prior_patch = StructuredLLMPolicy._extract_diff(
        partial_interval_function(),
        "src/intervals.py",
        interval_source(),
        "merge_intervals",
    )

    state.actions.extend(
        [
            ToolAction(
                tool=ToolName.APPLY_PATCH,
                arguments={"patch_text": prior_patch},
                rationale="Apply partial repair.",
            ),
            ToolAction(
                tool=ToolName.RUN_TESTS,
                arguments={},
                rationale="Verify repair.",
            ),
            ToolAction(
                tool=ToolName.RESTORE_FILE,
                arguments={"scope": "failed_attempt"},
                rationale="Rollback failed repair.",
            ),
            ToolAction(
                tool=ToolName.READ_FILE,
                arguments={"relative_path": "src/intervals.py"},
                rationale="Inspect restored source.",
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
                summary="Tests failed.",
                output=("E assert merge_intervals([(8, 5), (6, 9)]) == [(5, 9)]"),
            ),
            ToolObservation(
                tool=ToolName.RESTORE_FILE,
                status=ObservationStatus.OK,
                summary="Rolled back.",
            ),
            ToolObservation(
                tool=ToolName.READ_FILE,
                status=ObservationStatus.OK,
                summary="Read source.",
                output=interval_source(),
            ),
        ]
    )

    model = ScriptedModel(
        [
            partial_interval_function(),
            complete_interval_function(),
        ]
    )
    decision = StructuredLLMPolicy(model).decide(state)

    assert model.calls == 2
    assert decision.action.tool is ToolName.APPLY_PATCH
    assert "previous failed patches" in model.user_prompts[0].lower()
    assert "exactly repeats a previous failed patch" in model.user_prompts[1]
    patch_text = decision.action.arguments["patch_text"]
    assert "min(start, end)" in str(patch_text)
    assert "item[0], item[1]" in str(patch_text)
