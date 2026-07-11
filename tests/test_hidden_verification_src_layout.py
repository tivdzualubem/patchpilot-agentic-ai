"""Regression test for hidden verification of src-layout projects."""

from __future__ import annotations

import json
from pathlib import Path

from patchpilot.benchmark.hidden_verification import (
    HiddenTestRunner,
    HiddenVerificationStatus,
)
from patchpilot.benchmark.manifest import load_manifest


def test_hidden_runner_imports_src_layout_package(tmp_path: Path) -> None:
    repository = tmp_path / "benchmarks" / "example" / "repository"
    hidden = tmp_path / "benchmarks" / "example" / "hidden_tests"
    package = repository / "src" / "example_pkg"
    package.mkdir(parents=True)
    hidden.mkdir(parents=True)

    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    (hidden / "test_hidden_core.py").write_text(
        "from example_pkg.core import add\n\n"
        "def test_hidden_add() -> None:\n"
        "    assert add(40, 2) == 42\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "benchmarks" / "example" / "task.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": "src-layout-001",
                "title": "Src layout hidden verification",
                "goal": "Verify hidden tests for a src-layout package.",
                "repository_root": "benchmarks/example/repository",
                "defect_category": "hidden_verification",
                "difficulty": "easy",
                "allowed_paths": ["src"],
                "forbidden_paths": ["tests"],
                "test_command": ["python", "-m", "pytest", "-q"],
                "expected_initial_failures": 1,
                "hidden_test_root": "benchmarks/example/hidden_tests",
                "expected_hidden_tests": 1,
            }
        ),
        encoding="utf-8",
    )

    runner = HiddenTestRunner(
        project_root=tmp_path,
        output_root=tmp_path / "judge",
    )
    result = runner.run(
        manifest=load_manifest(manifest_path),
        repaired_repository=repository,
        run_id="src-layout-hidden-001",
    )

    assert result.status is HiddenVerificationStatus.PASSED
    assert result.passed is True
    assert result.test_count == 1
