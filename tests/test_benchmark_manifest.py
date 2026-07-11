from pathlib import Path

import pytest

from patchpilot.benchmark import load_manifest


def test_loads_calculator_manifest() -> None:
    path = Path("benchmarks/calculator-001/task.json")
    manifest = load_manifest(path)
    task = manifest.to_repair_task()

    assert manifest.task_id == "calculator-001"
    assert manifest.expected_initial_failures == 2
    assert manifest.hidden_test_root == ("benchmarks/calculator-001/hidden_tests")
    assert manifest.expected_hidden_tests == 2
    assert task.allowed_paths == ["src"]
    assert task.forbidden_paths == ["tests"]
    assert task.repository_root.endswith("/repository")


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")


def test_hidden_tests_must_be_outside_agent_repository(tmp_path: Path) -> None:
    manifest_path = tmp_path / "task.json"
    manifest_path.write_text(
        """
        {
          "task_id": "unsafe-hidden-001",
          "title": "Unsafe hidden tests",
          "goal": "Reject hidden tests placed inside the agent repository.",
          "repository_root": "benchmarks/example/repository",
          "defect_category": "test_isolation",
          "difficulty": "easy",
          "allowed_paths": ["src"],
          "forbidden_paths": ["tests"],
          "test_command": ["python", "-m", "pytest", "-q"],
          "expected_initial_failures": 1,
          "hidden_test_root": "benchmarks/example/repository/hidden_tests",
          "expected_hidden_tests": 1
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the agent-visible repository"):
        load_manifest(manifest_path)
