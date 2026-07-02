"""Isolated workspaces for reproducible benchmark execution."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from patchpilot.benchmark.manifest import load_manifest
from patchpilot.schemas import RepairTask


class BenchmarkWorkspaceError(ValueError):
    """Raised when a benchmark cannot be prepared safely."""


@dataclass(frozen=True)
class PreparedBenchmark:
    """One disposable benchmark execution environment."""

    workspace_root: Path
    repository_root: Path
    task: RepairTask


class BenchmarkWorkspace:
    """Copy validated benchmarks into isolated temporary workspaces."""

    def __init__(
        self,
        project_root: Path,
        output_root: Path,
    ) -> None:
        self.project_root = project_root.expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def prepare(self, manifest_path: Path) -> PreparedBenchmark:
        """Create one isolated copy of a benchmark repository."""
        manifest = load_manifest(manifest_path)

        source = (
            self.project_root / manifest.repository_root
        ).resolve(strict=True)

        if not source.is_relative_to(self.project_root):
            raise BenchmarkWorkspaceError(
                "Benchmark repository escapes the project root."
            )

        if not source.is_dir():
            raise BenchmarkWorkspaceError(
                "Benchmark repository is not a directory."
            )

        if any(path.is_symlink() for path in source.rglob("*")):
            raise BenchmarkWorkspaceError(
                "Benchmark repositories cannot contain symbolic links."
            )

        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"{manifest.task_id}-",
                dir=self.output_root,
            )
        )
        repository = workspace / "repository"

        shutil.copytree(source, repository)

        task = manifest.to_repair_task().model_copy(
            update={"repository_root": "repository"}
        )

        return PreparedBenchmark(
            workspace_root=workspace,
            repository_root=repository,
            task=task,
        )

    @staticmethod
    def cleanup(prepared: PreparedBenchmark) -> None:
        """Delete one disposable benchmark workspace."""
        shutil.rmtree(prepared.workspace_root, ignore_errors=False)
