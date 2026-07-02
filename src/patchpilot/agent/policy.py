"""Decision contracts for PatchPilot agent policies."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from patchpilot.schemas import AgentState, ToolAction


class AgentDecision(BaseModel):
    """One structured Plan-Act-Reflect decision."""

    model_config = ConfigDict(extra="forbid")

    reasoning_summary: str = Field(min_length=3, max_length=2000)
    plan: list[str] = Field(default_factory=list, max_length=20)
    hypothesis: str | None = Field(default=None, max_length=2000)
    reflection: str | None = Field(default=None, max_length=2000)
    action: ToolAction


class AgentPolicy(Protocol):
    """Interface implemented by any PatchPilot decision policy."""

    def decide(self, state: AgentState) -> AgentDecision:
        """Choose the next bounded action from the current state."""
        ...
