"""Detection of repeated no-progress agent action patterns."""

from __future__ import annotations

import hashlib
import json

from patchpilot.schemas import AgentState, ToolAction, ToolName
from patchpilot.schemas.models import ProgressSnapshot


class RepeatedActionGuard:
    """Detect repeated actions and bounded cycles without state progress."""

    def __init__(
        self,
        max_repeats: int = 2,
        max_cycle_length: int = 4,
        cycle_repeats: int = 2,
        max_no_progress_events: int = 2,
    ) -> None:
        if max_repeats < 2:
            raise ValueError("max_repeats must be at least 2.")

        if not 2 <= max_cycle_length <= 10:
            raise ValueError("max_cycle_length must be between 2 and 10.")

        if not 2 <= cycle_repeats <= 5:
            raise ValueError("cycle_repeats must be between 2 and 5.")

        if not 1 <= max_no_progress_events <= 10:
            raise ValueError("max_no_progress_events must be between 1 and 10.")

        self.max_repeats = max_repeats
        self.max_cycle_length = max_cycle_length
        self.cycle_repeats = cycle_repeats
        self.max_no_progress_events = max_no_progress_events

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

    @staticmethod
    def _latest_test_evidence_hash(state: AgentState) -> str | None:
        for observation in reversed(state.observations):
            if observation.tool is not ToolName.RUN_TESTS:
                continue

            evidence = json.dumps(
                {
                    "status": observation.status.value,
                    "summary": observation.summary,
                    "output": observation.output,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(evidence.encode("utf-8")).hexdigest()

        return None

    @classmethod
    def progress_snapshot(cls, state: AgentState) -> ProgressSnapshot:
        """Build the evidence used to determine whether progress occurred."""
        return ProgressSnapshot(
            repository_revision=state.repository_revision,
            latest_test_evidence_hash=cls._latest_test_evidence_hash(state),
            current_hypothesis=state.current_hypothesis,
            changed_files=tuple(sorted(state.changed_files)),
            full_suite_passed=state.full_suite_passed,
            verified_revision=state.verified_revision,
        )

    @classmethod
    def record_progress(cls, state: AgentState) -> None:
        """Append a durable progress snapshot after one recorded action."""
        snapshot = cls.progress_snapshot(state)

        if state.progress_snapshots and snapshot != state.progress_snapshots[-1]:
            state.no_progress_streak = 0

        state.progress_snapshots.append(snapshot)

    def _repeats_identically(
        self,
        state: AgentState,
        proposed_action: ToolAction,
    ) -> bool:
        proposed = self._signature(proposed_action)
        trailing_matches = 0

        for previous in reversed(state.actions):
            if self._signature(previous) != proposed:
                break
            trailing_matches += 1

        if trailing_matches + 1 < self.max_repeats:
            return False

        required_snapshots = min(
            trailing_matches,
            self.max_repeats - 1,
        )
        if len(state.progress_snapshots) < required_snapshots:
            return False

        current = self.progress_snapshot(state)
        recent = state.progress_snapshots[-required_snapshots:]
        return all(snapshot == current for snapshot in recent)

    def _completes_no_progress_cycle(
        self,
        state: AgentState,
        proposed_action: ToolAction,
    ) -> bool:
        signatures = [
            *(self._signature(action) for action in state.actions),
            self._signature(proposed_action),
        ]
        maximum = min(
            self.max_cycle_length,
            len(signatures) // self.cycle_repeats,
        )

        for cycle_length in range(2, maximum + 1):
            window_size = cycle_length * self.cycle_repeats
            tail = signatures[-window_size:]
            unit = tail[:cycle_length]

            if tail != unit * self.cycle_repeats:
                continue

            required_snapshots = window_size - 1
            if len(state.progress_snapshots) < required_snapshots:
                continue

            current = self.progress_snapshot(state)
            recent = state.progress_snapshots[-required_snapshots:]

            if all(snapshot == current for snapshot in recent):
                return True

        return False

    def rejection_reason(
        self,
        state: AgentState,
        proposed_action: ToolAction,
    ) -> str | None:
        """Explain why a proposed action represents no progress."""
        if self._repeats_identically(state, proposed_action):
            return (
                "Rejected no-progress: repeated identical action "
                "with unchanged repair evidence."
            )

        if self._completes_no_progress_cycle(state, proposed_action):
            return (
                "Rejected no-progress: repeated action cycle with unchanged "
                "repository, tests, hypothesis, and changed files."
            )

        return None

    def blocks(
        self,
        state: AgentState,
        proposed_action: ToolAction,
    ) -> bool:
        """Return whether the proposed action repeats without progress."""
        return self.rejection_reason(state, proposed_action) is not None
