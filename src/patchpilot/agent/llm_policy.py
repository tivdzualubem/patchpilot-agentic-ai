"""Structured LLM policy for PatchPilot decisions."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import AgentState


class TextGenerationModel(Protocol):
    """Minimal interface for a text-generation backend."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return one model-generated response."""
        ...


class PolicyResponseError(ValueError):
    """Raised when a model response cannot form a valid decision."""


class StructuredLLMPolicy:
    """Convert validated agent state into a structured next action."""

    def __init__(self, model: TextGenerationModel) -> None:
        self.model = model

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are PatchPilot, a bounded Python repair agent. "
            "Choose exactly one permitted tool action. "
            "Return JSON only, matching the AgentDecision schema. "
            "Do not claim success unless full-suite verification exists. "
            "Do not modify tests or files outside allowed paths."
        )

    @staticmethod
    def _user_prompt(state: AgentState) -> str:
        payload = state.model_dump(mode="json")
        return (
            "Choose the next Plan-Act-Observe-Reflect-Verify step.\n"
            "Current validated state:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _strip_code_fence(response: str) -> str:
        text = response.strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()

        if len(lines) < 3 or lines[-1].strip() != "```":
            raise PolicyResponseError(
                "The model returned an incomplete JSON code fence."
            )

        return "\n".join(lines[1:-1]).strip()

    def decide(self, state: AgentState) -> AgentDecision:
        """Generate and validate one structured agent decision."""
        response = self.model.generate(
            self._system_prompt(),
            self._user_prompt(state),
        )
        payload = self._strip_code_fence(response)

        try:
            return AgentDecision.model_validate_json(payload)
        except ValidationError as exc:
            raise PolicyResponseError(
                "The model returned an invalid AgentDecision."
            ) from exc
