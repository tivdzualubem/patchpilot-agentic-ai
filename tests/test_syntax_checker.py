"""Tests for bounded Python syntax validation."""

from pathlib import Path

from patchpilot.schemas import (
    ObservationStatus,
    RepairTask,
    ToolName,
)
from patchpilot.tools import RepositorySandbox, SyntaxChecker


def make_checker(tmp_path: Path) -> tuple[Path, SyntaxChecker]:
    repository = tmp_path / "benchmarks" / "syntax"
    (repository / "src").mkdir(parents=True)

    task = RepairTask(
        task_id="syntax-check-001",
        goal="Check changed Python files before running tests.",
        repository_root="benchmarks/syntax",
    )
    sandbox = RepositorySandbox(tmp_path, task)
    return repository, SyntaxChecker(sandbox)


def test_valid_changed_python_file_passes(tmp_path: Path) -> None:
    repository, checker = make_checker(tmp_path)
    (repository / "src" / "valid.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )

    result = checker.check_files(("src/valid.py",))

    assert result.tool is ToolName.CHECK_SYNTAX
    assert result.status is ObservationStatus.OK
    assert "1 Python file" in result.summary
    assert result.output == "src/valid.py"


def test_invalid_changed_python_file_reports_location(
    tmp_path: Path,
) -> None:
    repository, checker = make_checker(tmp_path)
    (repository / "src" / "broken.py").write_text(
        "def broken(:\n    return 1\n",
        encoding="utf-8",
    )

    result = checker.check_files(("src/broken.py",))

    assert result.status is ObservationStatus.ERROR
    assert result.summary == "Python syntax check failed."
    assert "src/broken.py:1:" in result.output


def test_no_changed_python_files_is_rejected(tmp_path: Path) -> None:
    _, checker = make_checker(tmp_path)

    result = checker.check_files(())

    assert result.status is ObservationStatus.REJECTED
    assert "No changed Python files" in result.summary


def test_escape_path_is_rejected(tmp_path: Path) -> None:
    _, checker = make_checker(tmp_path)

    result = checker.check_files(("../outside.py",))

    assert result.status is ObservationStatus.REJECTED
    assert "cannot contain '..'" in result.summary
