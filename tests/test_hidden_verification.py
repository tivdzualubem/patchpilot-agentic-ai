"""Tests for post-run hidden verification isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patchpilot.benchmark.hidden_verification import (
    HiddenTestRunner,
    HiddenVerificationError,
    HiddenVerificationStatus,
)
from patchpilot.benchmark.manifest import load_manifest


def write_manifest(
    project_root: Path,
    *,
    expected_hidden_tests: int = 1,
) -> Path:
    manifest_path = project_root / "benchmarks" / "example" / "task.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": "hidden-example-001",
                "title": "Hidden verification example",
                "goal": "Repair the function and pass private judging tests.",
                "repository_root": "benchmarks/example/repository",
                "defect_category": "hidden_verification",
                "difficulty": "easy",
                "allowed_paths": ["src"],
                "forbidden_paths": ["tests"],
                "test_command": ["python", "-m", "pytest", "-q"],
                "expected_initial_failures": 1,
                "hidden_test_root": "benchmarks/example/hidden_tests",
                "expected_hidden_tests": expected_hidden_tests,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def make_project(tmp_path: Path, implementation: str) -> tuple[Path, Path]:
    repository = tmp_path / "benchmarks" / "example" / "repository"
    hidden_tests = tmp_path / "benchmarks" / "example" / "hidden_tests"
    (repository / "src").mkdir(parents=True)
    hidden_tests.mkdir(parents=True)

    (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "src" / "calculator.py").write_text(
        implementation,
        encoding="utf-8",
    )
    (hidden_tests / "test_hidden_calculator.py").write_text(
        "from src.calculator import add\n\n"
        "def test_hidden_addition() -> None:\n"
        "    assert add(100, -40) == 60\n",
        encoding="utf-8",
    )
    return repository, write_manifest(tmp_path)


def test_hidden_suite_passes_without_exposing_output(tmp_path: Path) -> None:
    repository, manifest_path = make_project(
        tmp_path,
        "def add(left: int, right: int) -> int:\n    return left + right\n",
    )
    runner = HiddenTestRunner(
        project_root=tmp_path,
        output_root=tmp_path / "judge-runs",
    )

    result = runner.run(
        manifest=load_manifest(manifest_path),
        repaired_repository=repository,
        run_id="hidden-pass-001",
    )

    assert result.status is HiddenVerificationStatus.PASSED
    assert result.passed is True
    assert result.test_count == 1
    assert result.return_code == 0
    assert result.output_sha256 is not None
    assert not hasattr(result, "output")
    assert list((tmp_path / "judge-runs").iterdir()) == []


def test_hidden_suite_catches_visible_only_repair(tmp_path: Path) -> None:
    repository, manifest_path = make_project(
        tmp_path,
        "def add(left: int, right: int) -> int:\n"
        "    if (left, right) == (2, 3):\n"
        "        return 5\n"
        "    return left - right\n",
    )
    runner = HiddenTestRunner(
        project_root=tmp_path,
        output_root=tmp_path / "judge-runs",
    )

    result = runner.run(
        manifest=load_manifest(manifest_path),
        repaired_repository=repository,
        run_id="hidden-fail-001",
    )

    assert result.status is HiddenVerificationStatus.FAILED
    assert result.passed is False
    assert result.test_count == 1
    assert result.return_code != 0
    assert result.output_sha256 is not None


def test_hidden_test_symlink_is_rejected(tmp_path: Path) -> None:
    repository, manifest_path = make_project(
        tmp_path,
        "def add(left: int, right: int) -> int:\n    return left + right\n",
    )
    hidden_root = tmp_path / "benchmarks" / "example" / "hidden_tests"
    outside = tmp_path / "outside.py"
    outside.write_text("def test_outside() -> None:\n    pass\n", encoding="utf-8")
    (hidden_root / "linked.py").symlink_to(outside)
    runner = HiddenTestRunner(
        project_root=tmp_path,
        output_root=tmp_path / "judge-runs",
    )

    with pytest.raises(HiddenVerificationError, match="symbolic links"):
        runner.run(
            manifest=load_manifest(manifest_path),
            repaired_repository=repository,
            run_id="hidden-symlink-001",
        )
