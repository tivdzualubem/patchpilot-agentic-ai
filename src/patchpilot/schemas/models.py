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
    CHECK_SYNTAX = "check_syntax"
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


class FailureCategory(StrEnum):
    """Normalized failure categories for evaluation and diagnosis."""

    MODEL_ERROR = "model_error"
    DECISION_PARSE_ERROR = "decision_parse_error"
    PATCH_REJECTED = "patch_rejected"
    PATCH_APPLICATION_ERROR = "patch_application_error"
    SYNTAX_VERIFICATION_FAILED = "syntax_verification_failed"
    TEST_VERIFICATION_FAILED = "test_verification_failed"
    ROLLBACK_FAILED = "rollback_failed"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    USER_FAILED = "user_failed"
    USER_ESCALATED = "user_escalated"


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
        """Return whether a global execution limit has been reached."""
        return (
            self.steps >= budget.max_steps
            or self.tool_calls >= budget.max_tool_calls
            or self.elapsed_seconds >= budget.max_seconds
        )

    def patch_limit_reached(
        self,
        budget: ExecutionBudget,
    ) -> bool:
        """Return whether no additional patch may be attempted."""
        return self.patch_attempts >= budget.max_patch_attempts


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
    allowed_paths: list[str] = Field(default_factory=lambda: ["src"])
    forbidden_paths: list[str] = Field(default_factory=lambda: ["tests"])

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
                raise ValueError("Paths must be relative and must not contain '..'.")

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


class ModelCallRecord(BaseModel):
    """One complete model invocation record for reproducible traces."""

    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=1)
    policy: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=100)
    attempt: int = Field(ge=1)
    backend: str = Field(min_length=1, max_length=300)
    model_name: str | None = Field(default=None, max_length=300)
    generation_config: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any] | None = None
    raw_response: str | None = None
    generation_succeeded: bool
    parse_succeeded: bool | None = None
    duration_seconds: float = Field(default=0.0, ge=0)
    error_type: str | None = Field(default=None, max_length=300)
    error_message: str | None = Field(default=None, max_length=2000)


class DecisionRecord(BaseModel):
    """One policy decision linked to the model calls that produced it."""

    model_config = ConfigDict(extra="forbid")

    decision_index: int = Field(ge=1)
    policy: str = Field(min_length=1, max_length=300)
    model_call_start: int | None = Field(default=None, ge=1)
    model_call_end: int | None = Field(default=None, ge=1)
    reasoning_summary: str = Field(min_length=1, max_length=4000)
    plan: list[str] = Field(default_factory=list)
    hypothesis: str | None = None
    reflection: str | None = None
    action: ToolAction


class ProgressSnapshot(BaseModel):
    """One durable no-progress detection checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_revision: int = Field(ge=0)
    syntax_verified_revision: int | None = Field(default=None, ge=0)
    current_attempt_id: int | None = Field(default=None, ge=1)
    rollback_required: bool = False
    last_failed_attempt_id: int | None = Field(default=None, ge=1)
    last_reflected_attempt_id: int | None = Field(default=None, ge=1)
    latest_test_evidence_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    current_hypothesis: str | None = None
    changed_files: tuple[str, ...] = Field(default_factory=tuple)
    full_suite_passed: bool = False
    verified_revision: int | None = Field(default=None, ge=0)


class AgentState(BaseModel):
    """Structured short-term memory for a PatchPilot run."""

    model_config = ConfigDict(extra="forbid")

    task: RepairTask
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    status: AgentStatus = AgentStatus.READY

    plan: list[str] = Field(default_factory=list)
    current_hypothesis: str | None = None
    rejected_hypotheses: list[str] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)

    actions: list[ToolAction] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    progress_snapshots: list[ProgressSnapshot] = Field(default_factory=list)
    no_progress_streak: int = Field(default=0, ge=0)

    model_calls: int = Field(default=0, ge=0)
    model_call_records: list[ModelCallRecord] = Field(default_factory=list)
    decision_records: list[DecisionRecord] = Field(default_factory=list)
    decision_parse_failures: int = Field(default=0, ge=0)
    patch_rejection_count: int = Field(default=0, ge=0)
    patch_application_failure_count: int = Field(default=0, ge=0)
    verification_failure_count: int = Field(default=0, ge=0)
    failed_attempt_ids: list[int] = Field(default_factory=list)
    last_failure_category: FailureCategory | None = None
    terminal_failure_category: FailureCategory | None = None

    current_attempt_id: int | None = Field(default=None, ge=1)
    current_attempt_files: list[str] = Field(default_factory=list)
    rollback_required: bool = False
    last_failed_attempt_id: int | None = Field(default=None, ge=1)
    last_failed_attempt_files: list[str] = Field(default_factory=list)
    last_failed_verification_tool: ToolName | None = None
    last_rolled_back_attempt_id: int | None = Field(default=None, ge=1)
    last_rolled_back_attempt_files: list[str] = Field(default_factory=list)
    last_reflected_attempt_id: int | None = Field(default=None, ge=1)

    repository_revision: int = Field(default=0, ge=0)
    syntax_verified_revision: int | None = Field(default=None, ge=0)
    verified_revision: int | None = Field(default=None, ge=0)
    full_suite_passed: bool = False

    final_message: str | None = None

    @property
    def syntax_check_required(self) -> bool:
        """Return whether changed Python files need current syntax evidence."""
        has_changed_python = any(
            PurePosixPath(path).suffix == ".py" for path in self.changed_files
        )
        return (
            has_changed_python
            and self.syntax_verified_revision != self.repository_revision
        )

    @property
    def reflection_required(self) -> bool:
        """Return whether the latest failed attempt still needs reflection."""
        return (
            self.last_failed_attempt_id is not None
            and self.last_reflected_attempt_id != self.last_failed_attempt_id
        )

    @property
    def can_continue(self) -> bool:
        """Return whether the run may take another step."""
        terminal = {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.ESCALATED,
            AgentStatus.BUDGET_EXHAUSTED,
        }

        return self.status not in terminal and not self.usage.exhausted(self.budget)
