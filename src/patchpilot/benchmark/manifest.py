"""Validated benchmark manifests for PatchPilot."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from patchpilot.schemas import RepairTask


class BenchmarkManifest(BaseModel):
    """Metadata describing one controlled repair benchmark."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    title: str = Field(min_length=3, max_length=200)
    goal: str = Field(min_length=10, max_length=1000)
    repository_root: str = Field(min_length=1)
    defect_category: str = Field(min_length=3, max_length=100)
    difficulty: Literal["easy", "medium", "hard"]
    allowed_paths: list[str] = Field(min_length=1)
    forbidden_paths: list[str] = Field(default_factory=list)
    test_command: list[str] = Field(min_length=3)
    expected_initial_failures: int = Field(ge=1)
    hidden_test_root: str | None = Field(default=None, min_length=1)
    expected_hidden_tests: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_hidden_test_contract(self) -> Self:
        """Require a safe, complete hidden-verification declaration."""
        configured = self.hidden_test_root is not None
        if configured != (self.expected_hidden_tests is not None):
            raise ValueError(
                "hidden_test_root and expected_hidden_tests must be "
                "configured together."
            )

        if self.hidden_test_root is None:
            return self

        hidden = PurePosixPath(self.hidden_test_root)
        repository = PurePosixPath(self.repository_root)
        if hidden.is_absolute() or ".." in hidden.parts:
            raise ValueError(
                "hidden_test_root must be relative and cannot contain '..'."
            )
        if hidden == repository or hidden.is_relative_to(repository):
            raise ValueError(
                "Hidden tests must remain outside the agent-visible repository."
            )

        return self

    def to_repair_task(self) -> RepairTask:
        """Convert benchmark metadata into an executable repair task."""
        return RepairTask(
            task_id=self.task_id,
            goal=self.goal,
            repository_root=self.repository_root,
            test_command=self.test_command,
            allowed_paths=self.allowed_paths,
            forbidden_paths=self.forbidden_paths,
        )


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load and validate one benchmark manifest from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))
