"""Tests for canonical paired evaluation conditions."""

from __future__ import annotations

import pytest

from patchpilot.agent import (
    FixedWorkflowPolicy,
    LLMToolPolicy,
    OneShotRepairPolicy,
    ReflectiveLLMToolPolicy,
)
from patchpilot.evaluation import (
    PRIMARY_CONDITIONS,
    EvaluationCondition,
    build_condition,
    condition_values,
    get_condition_spec,
    parse_condition,
)


class NoCallModel:
    """Model stub used only to construct policies."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        del system_prompt, user_prompt, response_schema
        raise AssertionError("model should not be called")


def test_primary_condition_order_is_explicit_and_stable() -> None:
    assert condition_values() == (
        "one-shot",
        "fixed-workflow",
        "tool-agent-no-reflection",
        "full-reflective-agent",
    )
    assert tuple(condition.value for condition in PRIMARY_CONDITIONS) == (
        condition_values()
    )


@pytest.mark.parametrize(
    ("condition", "policy_type"),
    [
        (
            EvaluationCondition.ONE_SHOT,
            OneShotRepairPolicy,
        ),
        (
            EvaluationCondition.FIXED_WORKFLOW,
            FixedWorkflowPolicy,
        ),
        (
            EvaluationCondition.TOOL_AGENT_NO_REFLECTION,
            LLMToolPolicy,
        ),
        (
            EvaluationCondition.FULL_REFLECTIVE_AGENT,
            ReflectiveLLMToolPolicy,
        ),
    ],
)
def test_condition_factory_builds_exact_policy(
    condition: EvaluationCondition,
    policy_type: type[object],
) -> None:
    configured = build_condition(
        condition,
        NoCallModel(),
    )

    assert type(configured.policy) is policy_type
    assert configured.spec.condition is condition


def test_reflection_ablation_holds_non_reflection_factors_constant() -> None:
    no_reflection = get_condition_spec(EvaluationCondition.TOOL_AGENT_NO_REFLECTION)
    reflective = get_condition_spec(EvaluationCondition.FULL_REFLECTIVE_AGENT)

    assert no_reflection.budget == reflective.budget
    assert no_reflection.model_selects_tools is True
    assert reflective.model_selects_tools is True
    assert no_reflection.retry_enabled is True
    assert reflective.retry_enabled is True
    assert no_reflection.reflection_enabled is False
    assert reflective.reflection_enabled is True


def test_one_shot_is_the_only_single_patch_condition() -> None:
    one_shot = get_condition_spec(EvaluationCondition.ONE_SHOT)
    fixed = get_condition_spec(EvaluationCondition.FIXED_WORKFLOW)

    assert one_shot.retry_enabled is False
    assert one_shot.budget.max_patch_attempts == 1
    assert fixed.retry_enabled is True
    assert fixed.budget.max_patch_attempts > 1


def test_obsolete_condition_labels_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown evaluation condition",
    ):
        parse_condition("no-retry-live-qwen")


def test_condition_trace_metadata_is_complete() -> None:
    spec = get_condition_spec(EvaluationCondition.FULL_REFLECTIVE_AGENT)

    metadata = spec.trace_metadata()

    assert metadata["condition"] == "full-reflective-agent"
    assert metadata["model_selects_tools"] == "true"
    assert metadata["reflection_enabled"] == "true"
    assert metadata["retry_enabled"] == "true"
    assert metadata["budget_max_patch_attempts"] == str(spec.budget.max_patch_attempts)
