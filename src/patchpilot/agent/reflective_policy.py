"""Reflective model-directed tool-selection policy."""

from __future__ import annotations

import json

from patchpilot.agent.llm_policy import PolicyResponseError
from patchpilot.agent.llm_tool_policy import LLMToolPolicy
from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    ToolName,
)


class ReflectiveLLMToolPolicy(LLMToolPolicy):
    """Require explicit reflection after failed post-patch verification."""

    @staticmethod
    def _reflection_required(state: AgentState) -> bool:
        if not state.actions or not state.observations:
            return False

        last_action = state.actions[-1]
        last_observation = state.observations[-1]

        return (
            bool(state.changed_files)
            and last_action.tool is ToolName.RUN_TESTS
            and last_observation.status is not ObservationStatus.OK
        )

    @classmethod
    def _system_prompt(cls) -> str:
        tool_contract = json.dumps(
            cls._TOOL_CONTRACT,
            indent=2,
            sort_keys=True,
        )
        return (
            "You are the reflective tool-selection policy for a bounded Python "
            "repair agent. Choose exactly one next action from the restricted "
            "tool contract. The runtime validates and executes every action. "
            "Never edit tests or forbidden paths. Prefer evidence-gathering "
            "before patching. Use a minimal unified diff for apply_patch. Do "
            "not report success until a full test run has passed for the current "
            "revision. Normally set reflection to null. After failed post-patch "
            "verification, provide a concise critique of the previous hypothesis "
            "and a revised hypothesis before continuing. Return only one JSON "
            "object matching the provided schema. Do not use markdown.\n\n"
            f"RESTRICTED TOOL CONTRACT:\n{tool_contract}"
        )

    @classmethod
    def _reflection_instruction(cls, state: AgentState) -> str:
        if cls._reflection_required(state):
            return (
                "REFLECTION REQUIRED: the latest post-patch test run failed. "
                "Provide a non-empty reflection that critiques the previous "
                "hypothesis and provide a revised, non-empty hypothesis before "
                "choosing the next action."
            )

        return (
            "REFLECTION NOT REQUIRED: set the reflection field to null. "
            "Keep or update the current hypothesis only when evidence supports it."
        )

    def _validate_decision(
        self,
        state: AgentState,
        decision: AgentDecision,
        raw_response: str,
    ) -> AgentDecision:
        required = self._reflection_required(state)

        if not required:
            if decision.reflection is not None:
                raise PolicyResponseError(
                    "Reflection is only allowed after failed post-patch verification.",
                    raw_response=raw_response,
                )
            return decision

        reflection = decision.reflection
        if reflection is None or len(reflection.strip()) < 10:
            raise PolicyResponseError(
                "Failed post-patch verification requires a meaningful reflection.",
                raw_response=raw_response,
            )

        hypothesis = decision.hypothesis
        if hypothesis is None or len(hypothesis.strip()) < 3:
            raise PolicyResponseError(
                "Reflection requires a revised non-empty hypothesis.",
                raw_response=raw_response,
            )

        previous = state.current_hypothesis
        if previous is not None and hypothesis.strip() == previous.strip():
            raise PolicyResponseError(
                "The revised hypothesis must differ from the rejected hypothesis.",
                raw_response=raw_response,
            )

        return decision
