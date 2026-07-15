"""Durable, append-only execution traces for PatchPilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from patchpilot.schemas import AgentState, AgentStatus

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,99}$")
_CHECKPOINT_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,49}$")

_TERMINAL_STATUSES = frozenset(
    {
        AgentStatus.SUCCEEDED,
        AgentStatus.FAILED,
        AgentStatus.ESCALATED,
        AgentStatus.BUDGET_EXHAUSTED,
    }
)


class RunTrace(BaseModel):
    """Validated, hash-chained checkpoint for one PatchPilot run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    run_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]{2,99}$",
    )
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_kind: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-z][a-z0-9_]{2,49}$",
    )
    recorded_at: datetime
    completed_at: datetime | None = None
    previous_checkpoint_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: AgentState
    metadata: dict[str, str] = Field(default_factory=dict)


class TraceRecorder:
    """Append immutable checkpoints and maintain one latest-state snapshot."""

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

    @staticmethod
    def _validate_checkpoint_kind(checkpoint_kind: str) -> str:
        if _CHECKPOINT_KIND_PATTERN.fullmatch(checkpoint_kind) is None:
            raise ValueError(
                "checkpoint_kind must use lowercase letters, numbers, "
                "and underscores only, with length 3-50."
            )
        return checkpoint_kind

    @staticmethod
    def _canonical_json(payload: object) -> str:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def _state_digest(cls, state: AgentState) -> str:
        payload = state.model_dump(mode="json")
        return hashlib.sha256(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def _checkpoint_digest(
        cls,
        *,
        run_id: str,
        checkpoint_sequence: int,
        checkpoint_kind: str,
        recorded_at: datetime,
        completed_at: datetime | None,
        previous_checkpoint_digest: str | None,
        state_digest: str,
        metadata: dict[str, str],
    ) -> str:
        payload = {
            "schema_version": "2.0",
            "run_id": run_id,
            "checkpoint_sequence": checkpoint_sequence,
            "checkpoint_kind": checkpoint_kind,
            "recorded_at": recorded_at.isoformat(),
            "completed_at": (
                completed_at.isoformat() if completed_at is not None else None
            ),
            "previous_checkpoint_digest": previous_checkpoint_digest,
            "state_digest": state_digest,
            "metadata": metadata,
        }
        return hashlib.sha256(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    def event_log_path(self, run_id: str) -> Path:
        """Return the append-only JSONL path for one run."""
        validated_run_id = self._validate_run_id(run_id)
        return self.output_directory / f"{validated_run_id}.events.jsonl"

    def snapshot_path(self, run_id: str) -> Path:
        """Return the latest validated snapshot path for one run."""
        validated_run_id = self._validate_run_id(run_id)
        return self.output_directory / f"{validated_run_id}.json"

    def build_trace(
        self,
        state: AgentState,
        run_id: str,
        metadata: dict[str, str] | None = None,
        *,
        checkpoint_sequence: int = 1,
        checkpoint_kind: str = "snapshot",
        previous_checkpoint_digest: str | None = None,
    ) -> RunTrace:
        """Build one deep, validated, hash-chained checkpoint."""
        now = datetime.now(UTC)
        validated_run_id = self._validate_run_id(run_id)
        validated_kind = self._validate_checkpoint_kind(checkpoint_kind)
        copied_metadata = dict(metadata or {})
        completed_at = now if state.status in _TERMINAL_STATUSES else None
        state_digest = self._state_digest(state)
        checkpoint_digest = self._checkpoint_digest(
            run_id=validated_run_id,
            checkpoint_sequence=checkpoint_sequence,
            checkpoint_kind=validated_kind,
            recorded_at=now,
            completed_at=completed_at,
            previous_checkpoint_digest=previous_checkpoint_digest,
            state_digest=state_digest,
            metadata=copied_metadata,
        )

        return RunTrace(
            run_id=validated_run_id,
            checkpoint_sequence=checkpoint_sequence,
            checkpoint_kind=validated_kind,
            recorded_at=now,
            completed_at=completed_at,
            previous_checkpoint_digest=previous_checkpoint_digest,
            state_digest=state_digest,
            checkpoint_digest=checkpoint_digest,
            state=state.model_copy(deep=True),
            metadata=copied_metadata,
        )

    def _append_event(self, trace: RunTrace) -> Path:
        path = self.event_log_path(trace.run_id)
        payload = trace.model_dump_json() + "\n"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )

        try:
            with os.fdopen(
                descriptor,
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            with suppress(OSError):
                os.close(descriptor)
            raise

        path.chmod(0o600)
        return path

    def _write_snapshot(self, trace: RunTrace) -> Path:
        target = self.snapshot_path(trace.run_id)
        temporary = self.output_directory / (f".{trace.run_id}.{uuid4().hex}.tmp")

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
            target.chmod(0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

        return target

    def save(
        self,
        state: AgentState,
        run_id: str,
        metadata: dict[str, str] | None = None,
        *,
        checkpoint_kind: str = "snapshot",
    ) -> Path:
        """Append one immutable checkpoint and refresh the latest snapshot."""
        checkpoints = self.load_checkpoints(
            run_id,
            missing_ok=True,
        )
        previous = checkpoints[-1] if checkpoints else None
        trace = self.build_trace(
            state,
            run_id,
            metadata,
            checkpoint_sequence=(
                previous.checkpoint_sequence + 1 if previous is not None else 1
            ),
            checkpoint_kind=checkpoint_kind,
            previous_checkpoint_digest=(
                previous.checkpoint_digest if previous is not None else None
            ),
        )
        self._append_event(trace)
        return self._write_snapshot(trace)

    def _validate_chain(
        self,
        checkpoints: list[RunTrace],
        run_id: str,
    ) -> None:
        previous_digest: str | None = None

        for expected_sequence, trace in enumerate(checkpoints, start=1):
            if trace.run_id != run_id:
                raise ValueError("Trace event log contains a different run_id.")

            if trace.checkpoint_sequence != expected_sequence:
                raise ValueError("Trace checkpoint sequence is not contiguous.")

            if trace.previous_checkpoint_digest != previous_digest:
                raise ValueError("Trace checkpoint digest chain is broken.")

            expected_state_digest = self._state_digest(trace.state)
            if trace.state_digest != expected_state_digest:
                raise ValueError("Trace state digest validation failed.")

            expected_checkpoint_digest = self._checkpoint_digest(
                run_id=trace.run_id,
                checkpoint_sequence=trace.checkpoint_sequence,
                checkpoint_kind=trace.checkpoint_kind,
                recorded_at=trace.recorded_at,
                completed_at=trace.completed_at,
                previous_checkpoint_digest=(trace.previous_checkpoint_digest),
                state_digest=trace.state_digest,
                metadata=trace.metadata,
            )
            if trace.checkpoint_digest != expected_checkpoint_digest:
                raise ValueError("Trace checkpoint digest validation failed.")

            previous_digest = trace.checkpoint_digest

    def load_checkpoints(
        self,
        run_id: str,
        *,
        missing_ok: bool = False,
    ) -> list[RunTrace]:
        """Load and verify every append-only checkpoint for one run."""
        validated_run_id = self._validate_run_id(run_id)
        path = self.event_log_path(validated_run_id)

        if not path.is_file():
            if missing_ok:
                return []
            raise FileNotFoundError(
                f"Trace event log does not exist: {validated_run_id}"
            )

        checkpoints: list[RunTrace] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue

            try:
                checkpoints.append(RunTrace.model_validate_json(raw_line))
            except Exception as exc:
                raise ValueError(f"Invalid trace event at line {line_number}.") from exc

        if not checkpoints:
            raise ValueError("Trace event log is empty.")

        self._validate_chain(checkpoints, validated_run_id)
        return checkpoints

    def load(self, run_id: str) -> RunTrace:
        """Load the latest verified checkpoint for one run."""
        checkpoints = self.load_checkpoints(run_id)
        return checkpoints[-1]
