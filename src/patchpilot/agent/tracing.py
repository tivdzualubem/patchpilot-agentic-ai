"""Durable, versioned execution traces for PatchPilot."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from patchpilot.schemas import AgentState, AgentStatus

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,99}$")

_TERMINAL_STATUSES = frozenset(
    {
        AgentStatus.SUCCEEDED,
        AgentStatus.FAILED,
        AgentStatus.ESCALATED,
        AgentStatus.BUDGET_EXHAUSTED,
    }
)


class RunTrace(BaseModel):
    """Validated snapshot of one PatchPilot run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]{2,99}$",
    )
    recorded_at: datetime
    completed_at: datetime | None = None
    state: AgentState
    metadata: dict[str, str] = Field(default_factory=dict)


class TraceRecorder:
    """Atomically persist and reload agent execution traces."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory.expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "run_id must use lowercase letters, numbers, "
                "and hyphens only, with length 3-100."
            )
        return run_id

    def build_trace(
        self,
        state: AgentState,
        run_id: str,
        metadata: dict[str, str] | None = None,
    ) -> RunTrace:
        """Build a deep, validated snapshot of the current run."""
        now = datetime.now(UTC)

        return RunTrace(
            run_id=self._validate_run_id(run_id),
            recorded_at=now,
            completed_at=(
                now
                if state.status in _TERMINAL_STATUSES
                else None
            ),
            state=state.model_copy(deep=True),
            metadata=dict(metadata or {}),
        )

    def save(
        self,
        state: AgentState,
        run_id: str,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        """Save a trace using atomic file replacement."""
        trace = self.build_trace(state, run_id, metadata)
        target = self.output_directory / f"{trace.run_id}.json"
        temporary = self.output_directory / (
            f".{trace.run_id}.{uuid4().hex}.tmp"
        )

        try:
            with temporary.open(
                "x",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(trace.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            temporary.chmod(0o600)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

        return target

    def load(self, run_id: str) -> RunTrace:
        """Load and validate a previously saved trace."""
        validated_run_id = self._validate_run_id(run_id)
        path = self.output_directory / f"{validated_run_id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Trace does not exist: {validated_run_id}"
            )

        return RunTrace.model_validate_json(
            path.read_text(encoding="utf-8")
        )
