"""Validated domain models for PatchPilot agent execution."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class AgentStatus(StrEnum):
    """Lifecycle states for one repair attempt."""

    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ESCALATED = "escalated"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ToolName(StrEnum):
    """Restricted tools available to the repair agent."""

    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    RUN_TESTS = "run_tests"
    APPLY_PATCH = "apply_patch"
    VIEW_DIFF = "view_diff"
    RESTORE_FILE = "restore_file"
    FINISH = "finish"


class ObservationStatus(StrEnum):
    """Outcome of a validated tool execution."""

    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ExecutionBudget(BaseModel):
    """Hard limits for one bounded agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=20, ge=1, le=100)
    max_tool_calls: int = Field(default=30, ge=1, le=200)
    max_patch_attempts: int = Field(default=5, ge=1, le=25)
    max_seconds: int = Field(default=600, ge=1, le=3600)


class BudgetUsage(BaseModel):
    """Resources consumed by an agent run."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    patch_attempts: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)

    def exhausted(self, budget: ExecutionBudget) -> bool:
        """Return whether any configured limit has been reached."""
        return (
            self.steps >= budget.max_steps
            or self.tool_calls >= budget.max_tool_calls
            or self.patch_attempts >= budget.max_patch_attempts
            or self.elapsed_seconds >= budget.max_seconds
        )


class RepairTask(BaseModel):
    """One controlled Python repair benchmark task."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    goal: str = Field(min_length=10, max_length=1000)
    repository_root: str = Field(min_length=1)
    test_command: list[str] = Field(
        default_factory=lambda: [
            "python",
            "-m",
            "pytest",
            "-q",
        ],
        min_length=1,
    )
    allowed_paths: list[str] = Field(
        default_factory=lambda: ["src"]
    )
    forbidden_paths: list[str] = Field(
        default_factory=lambda: ["tests"]
    )

    @field_validator(
        "repository_root",
        "allowed_paths",
        "forbidden_paths",
        mode="before",
    )
    @classmethod
    def validate_safe_paths(cls, value: Any) -> Any:
        """Reject absolute paths and parent-directory traversal."""
        values = value if isinstance(value, list) else [value]

        for raw_path in values:
            if not isinstance(raw_path, str):
                continue

            path = PurePosixPath(raw_path)

            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "Paths must be relative and must not contain '..'."
                )

        return value


class ToolAction(BaseModel):
    """One model-requested action before policy validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=3, max_length=2000)


class ToolObservation(BaseModel):
    """One real result returned by the tool boundary."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    tool: ToolName
    status: ObservationStatus
    summary: str = Field(min_length=1, max_length=2000)
    output: str = Field(default="")
    duration_seconds: float = Field(default=0.0, ge=0)


class AgentState(BaseModel):
    """Structured short-term memory for a PatchPilot run."""

    model_config = ConfigDict(extra="forbid")

    task: RepairTask
    budget: ExecutionBudget = Field(
        default_factory=ExecutionBudget
    )
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    status: AgentStatus = AgentStatus.READY

    plan: list[str] = Field(default_factory=list)
    current_hypothesis: str | None = None
    rejected_hypotheses: list[str] = Field(default_factory=list)

    actions: list[ToolAction] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)

    final_message: str | None = None

    @property
    def can_continue(self) -> bool:
        """Return whether the run may take another step."""
        terminal = {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.ESCALATED,
            AgentStatus.BUDGET_EXHAUSTED,
        }

        return (
            self.status not in terminal
            and not self.usage.exhausted(self.budget)
        )
