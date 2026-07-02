"""Validated benchmark manifests for PatchPilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

    return BenchmarkManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
