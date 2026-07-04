"""Detection of repeated no-progress agent actions."""

from __future__ import annotations

import json

from patchpilot.schemas import AgentState, ToolAction


class RepeatedActionGuard:
    """Detect consecutive identical tool calls."""

    def __init__(self, max_repeats: int = 2) -> None:
        if max_repeats < 2:
            raise ValueError("max_repeats must be at least 2.")

        self.max_repeats = max_repeats

    @staticmethod
    def _signature(action: ToolAction) -> str:
        return json.dumps(
            {
                "tool": action.tool.value,
                "arguments": action.arguments,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def blocks(
        self,
        state: AgentState,
        proposed_action: ToolAction,
    ) -> bool:
        """Return whether the proposed action repeats too often."""
        proposed = self._signature(proposed_action)
        trailing_matches = 0

        for previous in reversed(state.actions):
            if self._signature(previous) != proposed:
                break

            trailing_matches += 1

        return trailing_matches + 1 >= self.max_repeats
