"""Regression tests for model-output robustness."""

from __future__ import annotations

import json
from pathlib import Path

from patchpilot.agent import (
    LLMToolPolicy,
    OneShotRepairPolicy,
    StructuredLLMPolicy,
)
from patchpilot.benchmark import BenchmarkRunner
from patchpilot.evaluation import EvaluationCondition, get_condition_spec
from patchpilot.schemas import AgentState, RepairTask, ToolName


class CapturingModel:
    """Return one response and capture the supplied schema."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.response_schema: dict[str, object] | None = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt
        self.response_schema = response_schema
        return self.response


class IntervalFunctionModel:
    """Return a complete corrected interval function."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt, response_schema
        self.calls += 1
        return corrected_interval_function()


def corrected_interval_function() -> str:
    """Return a multi-line repair within the patch boundary."""
    return "\n".join(
        [
            (
                "def merge_intervals("
                "intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]:"
            ),
            '    """Merge overlapping or touching closed intervals."""',
            (
                "    normalized = [(min(start, end), max(start, end)) "
                "for start, end in intervals]"
            ),
            "    if not normalized:",
            "        return []",
            "",
            ("    normalized.sort(key=lambda item: (item[0], item[1]))"),
            ("    merged: list[tuple[int, int]] = [normalized[0]]"),
            "",
            "    for start, end in normalized[1:]:",
            "        previous_start, previous_end = merged[-1]",
            "        if start <= previous_end:",
            ("            merged[-1] = (previous_start, max(previous_end, end))"),
            "        else:",
            "            merged.append((start, end))",
            "    return merged",
        ]
    )


def interval_read_output() -> str:
    """Return the defective source in read_file observation format."""
    return "\n".join(
        [
            '1: """Interval normalization helpers."""',
            "2: ",
            "3: from __future__ import annotations",
            "4: ",
            "5: ",
            (
                "6: def merge_intervals("
                "intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]:"
            ),
            ('7:     """Merge overlapping or touching closed intervals."""'),
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


def valid_decision_json() -> str:
    """Return one valid tool decision."""
    return json.dumps(
        {
            "reasoning_summary": "Run tests before modifying source.",
            "plan": ["Reproduce the failure."],
            "hypothesis": None,
            "reflection": None,
            "action": {
                "tool": "run_tests",
                "arguments": {},
                "rationale": "Establish executable evidence.",
            },
        }
    )


def make_task() -> RepairTask:
    """Create one controlled repair task."""
    return RepairTask(
        task_id="policy-robustness",
        goal="Repair the defective implementation.",
        repository_root="repository",
    )


def test_tool_schema_requires_non_empty_plan() -> None:
    model = CapturingModel(valid_decision_json())
    decision = LLMToolPolicy(model).decide(AgentState(task=make_task()))

    assert decision.action.tool is ToolName.RUN_TESTS
    schema = model.response_schema
    assert schema is not None
    required = schema["required"]
    properties = schema["properties"]
    assert isinstance(required, list)
    assert isinstance(properties, dict)
    assert "plan" in required
    plan_schema = properties["plan"]
    assert isinstance(plan_schema, dict)
    assert plan_schema["minItems"] == 1


def test_complete_function_builds_bounded_multiline_diff() -> None:
    patch_text = StructuredLLMPolicy._extract_diff(
        corrected_interval_function(),
        "src/intervals.py",
        interval_read_output(),
        "merge_intervals",
    )

    assert "-    normalized = [(start, end)" in patch_text
    assert "+    normalized = [(min(start, end)" in patch_text
    assert "-        if start < previous_end:" in patch_text
    assert "+        if start <= previous_end:" in patch_text
    assert "-            merged[-1] = (start," in patch_text
    assert "+            merged[-1] = (previous_start," in patch_text


def test_one_shot_repairs_multiline_manual_challenge(
    tmp_path: Path,
) -> None:
    model = IntervalFunctionModel()
    policy = OneShotRepairPolicy(model)
    spec = get_condition_spec(EvaluationCondition.ONE_SHOT)
    runner = BenchmarkRunner(Path("."), tmp_path / "outputs")

    run = runner.run(
        Path("challenge_benchmarks/challenge-interval-merge/task.json"),
        policy,
        run_id="one-shot-interval-challenge",
        budget=spec.budget,
        metadata=spec.trace_metadata(),
    )

    assert run.state.status.value == "succeeded"
    assert run.state.full_suite_passed is True
    assert run.state.usage.patch_attempts == 1
    assert model.calls == 1
