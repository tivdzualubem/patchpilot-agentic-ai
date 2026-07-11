"""Canonical experimental conditions for PatchPilot evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from patchpilot.agent.baseline_policies import (
    FixedWorkflowPolicy,
    OneShotRepairPolicy,
)
from patchpilot.agent.llm_policy import TextGenerationModel
from patchpilot.agent.llm_tool_policy import LLMToolPolicy
from patchpilot.agent.policy import AgentPolicy
from patchpilot.agent.reflective_policy import ReflectiveLLMToolPolicy
from patchpilot.schemas import ExecutionBudget


class EvaluationCondition(StrEnum):
    """The four primary paired experimental conditions."""

    ONE_SHOT = "one-shot"
    FIXED_WORKFLOW = "fixed-workflow"
    TOOL_AGENT_NO_REFLECTION = "tool-agent-no-reflection"
    FULL_REFLECTIVE_AGENT = "full-reflective-agent"


@dataclass(frozen=True)
class ConditionSpec:
    """Reproducible configuration for one evaluation condition."""

    condition: EvaluationCondition
    label: str
    description: str
    model_selects_tools: bool
    reflection_enabled: bool
    retry_enabled: bool
    budget: ExecutionBudget

    def trace_metadata(self) -> dict[str, str]:
        """Return stable condition metadata for every run trace."""
        return {
            "condition": self.condition.value,
            "condition_label": self.label,
            "condition_description": self.description,
            "model_selects_tools": str(self.model_selects_tools).lower(),
            "reflection_enabled": str(self.reflection_enabled).lower(),
            "retry_enabled": str(self.retry_enabled).lower(),
            "budget_max_steps": str(self.budget.max_steps),
            "budget_max_tool_calls": str(self.budget.max_tool_calls),
            "budget_max_patch_attempts": str(self.budget.max_patch_attempts),
            "budget_max_seconds": str(self.budget.max_seconds),
        }


@dataclass(frozen=True)
class ConfiguredCondition:
    """A condition specification bound to its concrete policy."""

    spec: ConditionSpec
    policy: AgentPolicy


_ONE_SHOT_BUDGET = ExecutionBudget(
    max_steps=8,
    max_tool_calls=8,
    max_patch_attempts=1,
    max_seconds=1800,
)
_ITERATIVE_BUDGET = ExecutionBudget(
    max_steps=20,
    max_tool_calls=30,
    max_patch_attempts=5,
    max_seconds=1800,
)

PRIMARY_CONDITIONS: tuple[EvaluationCondition, ...] = (
    EvaluationCondition.ONE_SHOT,
    EvaluationCondition.FIXED_WORKFLOW,
    EvaluationCondition.TOOL_AGENT_NO_REFLECTION,
    EvaluationCondition.FULL_REFLECTIVE_AGENT,
)

CONDITION_SPECS: dict[EvaluationCondition, ConditionSpec] = {
    EvaluationCondition.ONE_SHOT: ConditionSpec(
        condition=EvaluationCondition.ONE_SHOT,
        label="One-shot",
        description=(
            "Fixed workflow with one model-generated patch and no second "
            "repair attempt."
        ),
        model_selects_tools=False,
        reflection_enabled=False,
        retry_enabled=False,
        budget=_ONE_SHOT_BUDGET,
    ),
    EvaluationCondition.FIXED_WORKFLOW: ConditionSpec(
        condition=EvaluationCondition.FIXED_WORKFLOW,
        label="Fixed workflow",
        description=(
            "Deterministic staged tool sequence with model-generated patches "
            "and bounded retries."
        ),
        model_selects_tools=False,
        reflection_enabled=False,
        retry_enabled=True,
        budget=_ITERATIVE_BUDGET,
    ),
    EvaluationCondition.TOOL_AGENT_NO_REFLECTION: ConditionSpec(
        condition=EvaluationCondition.TOOL_AGENT_NO_REFLECTION,
        label="Tool agent without reflection",
        description=(
            "The model selects bounded tools and may retry, but reflection is disabled."
        ),
        model_selects_tools=True,
        reflection_enabled=False,
        retry_enabled=True,
        budget=_ITERATIVE_BUDGET,
    ),
    EvaluationCondition.FULL_REFLECTIVE_AGENT: ConditionSpec(
        condition=EvaluationCondition.FULL_REFLECTIVE_AGENT,
        label="Full reflective agent",
        description=(
            "The model selects bounded tools and must critique failed patch "
            "hypotheses before retrying."
        ),
        model_selects_tools=True,
        reflection_enabled=True,
        retry_enabled=True,
        budget=_ITERATIVE_BUDGET,
    ),
}


def condition_values() -> tuple[str, ...]:
    """Return primary condition names in deterministic experiment order."""
    return tuple(condition.value for condition in PRIMARY_CONDITIONS)


def parse_condition(
    value: str | EvaluationCondition,
) -> EvaluationCondition:
    """Normalize one condition value and reject obsolete labels."""
    if isinstance(value, EvaluationCondition):
        return value

    try:
        return EvaluationCondition(value)
    except ValueError as exc:
        allowed = ", ".join(condition_values())
        raise ValueError(
            f"Unknown evaluation condition {value!r}; choose one of: {allowed}."
        ) from exc


def get_condition_spec(
    value: str | EvaluationCondition,
) -> ConditionSpec:
    """Return the immutable specification for one condition."""
    return CONDITION_SPECS[parse_condition(value)]


def build_condition(
    value: str | EvaluationCondition,
    model: TextGenerationModel,
) -> ConfiguredCondition:
    """Construct the exact policy and budget for one condition."""
    condition = parse_condition(value)
    spec = CONDITION_SPECS[condition]

    if condition is EvaluationCondition.ONE_SHOT:
        policy: AgentPolicy = OneShotRepairPolicy(model)
    elif condition is EvaluationCondition.FIXED_WORKFLOW:
        policy = FixedWorkflowPolicy(model)
    elif condition is EvaluationCondition.TOOL_AGENT_NO_REFLECTION:
        policy = LLMToolPolicy(model)
    else:
        policy = ReflectiveLLMToolPolicy(model)

    return ConfiguredCondition(
        spec=spec,
        policy=policy,
    )
