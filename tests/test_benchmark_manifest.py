from pathlib import Path

import pytest

from patchpilot.benchmark import load_manifest


def test_loads_calculator_manifest() -> None:
    path = Path("benchmarks/calculator-001/task.json")
    manifest = load_manifest(path)
    task = manifest.to_repair_task()

    assert manifest.task_id == "calculator-001"
    assert manifest.expected_initial_failures == 2
    assert task.allowed_paths == ["src"]
    assert task.forbidden_paths == ["tests"]
    assert task.repository_root.endswith("/repository")


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")
