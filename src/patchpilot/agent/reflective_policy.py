"""Reflective model-directed tool-selection policy."""

from __future__ import annotations

from patchpilot.agent.llm_policy import PolicyResponseError
from patchpilot.agent.llm_tool_policy import LLMToolPolicy
from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import AgentState, ToolName


class ReflectiveLLMToolPolicy(LLMToolPolicy):
    """Require explicit reflection after failed post-patch verification."""

    @staticmethod
    def _reflection_required(state: AgentState) -> bool:
        return state.reflection_required

    @classmethod
    def _system_prompt(cls) -> str:
        return (
            "You are a bounded reflective Python repair tool-selection "
            "policy. The JSON schema restricts action.tool to currently "
            "legal choices. Available tools include list_files, read_file, "
            "search_code, run_tests, apply_patch, check_syntax, view_diff, "
            "restore_file, and finish. Never invent evidence, edit tests, "
            "or use forbidden paths. After every successful patch choose "
            "check_syntax before tests. The runtime transactionally rolls "
            "back failed verified attempts. Normally set reflection to null. "
            "When reflection is required, critique the failed hypothesis and "
            "provide a revised hypothesis. Never claim success until the "
            "current revision passes the full suite. Return JSON only."
        )

    @classmethod
    def _reflection_instruction(cls, state: AgentState) -> str:
        if cls._reflection_required(state):
            return (
                "REFLECTION REQUIRED: the latest failed patch attempt has "
                "already been rolled back transactionally. Provide a non-empty "
                "reflection that critiques the previous hypothesis and provide "
                "a revised, non-empty hypothesis before choosing the next action."
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

        if not required and decision.reflection is not None:
            raise PolicyResponseError(
                "Reflection is only allowed after failed post-patch verification.",
                raw_response=raw_response,
            )

        decision = self._validate_common_decision(
            state,
            decision,
            raw_response,
        )

        if not required:
            return decision

        if decision.action.tool is ToolName.RESTORE_FILE:
            raise PolicyResponseError(
                "The failed patch attempt has already been rolled back by the "
                "runtime; choose the next evidence-gathering action.",
                raw_response=raw_response,
            )

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
