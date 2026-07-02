"""Tests for sandboxed repository inspection tools."""

from pathlib import Path

import pytest

from patchpilot.schemas import ObservationStatus, RepairTask
from patchpilot.tools import (
    RepositoryAccessError,
    RepositorySandbox,
)


@pytest.fixture()
def sandbox(tmp_path: Path) -> RepositorySandbox:
    """Create a small controlled repair repository."""
    repository = tmp_path / "benchmarks" / "task-001"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / ".git").mkdir()

    (repository / "src" / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_calculator.py").write_text(
        "from src.calculator import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (repository / ".git" / "config").write_text(
        "secret",
        encoding="utf-8",
    )
    (repository / ".env").write_text(
        "API_TOKEN=secret",
        encoding="utf-8",
    )
    (repository / "binary.bin").write_bytes(b"\x00\x01\x02")

    task = RepairTask(
        task_id="calculator-001",
        goal="Repair the incorrect calculator addition implementation.",
        repository_root="benchmarks/task-001",
    )

    return RepositorySandbox(tmp_path, task)


def test_list_files_hides_sensitive_paths(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.list_files()

    assert result.status is ObservationStatus.OK
    assert "src/calculator.py" in result.output
    assert "tests/test_calculator.py" in result.output
    assert ".git/config" not in result.output
    assert ".env" not in result.output


def test_read_file_returns_numbered_lines(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.read_file(
        "src/calculator.py",
        start_line=1,
        end_line=2,
    )

    assert result.status is ObservationStatus.OK
    assert "1: def add" in result.output
    assert "2:     return left - right" in result.output


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.txt",
        "/etc/passwd",
        ".git/config",
        ".env",
    ],
)
def test_read_file_rejects_unsafe_paths(
    sandbox: RepositorySandbox,
    unsafe_path: str,
) -> None:
    result = sandbox.read_file(unsafe_path)

    assert result.status is ObservationStatus.REJECTED


def test_read_file_rejects_binary_content(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.read_file("binary.bin")

    assert result.status is ObservationStatus.REJECTED
    assert "Binary" in result.summary


def test_search_code_returns_file_and_line(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.search_code("left - right")

    assert result.status is ObservationStatus.OK
    assert (
        "src/calculator.py:2: return left - right"
        in result.output
    )


def test_search_code_is_case_insensitive(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.search_code("ASSERT ADD")

    assert result.status is ObservationStatus.OK
    assert "tests/test_calculator.py:4:" in result.output


def test_symlink_escape_is_rejected(
    sandbox: RepositorySandbox,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    link = sandbox.repository_root / "src" / "outside-link.txt"
    link.symlink_to(outside)

    result = sandbox.read_file("src/outside-link.txt")

    assert result.status is ObservationStatus.REJECTED
    assert "escapes" in result.summary


def test_repository_root_must_remain_in_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "linked-repository").symlink_to(outside)

    task = RepairTask(
        task_id="escape-001",
        goal="Verify that repository sandbox escape is rejected.",
        repository_root="linked-repository",
    )

    with pytest.raises(
        RepositoryAccessError,
        match="inside the workspace",
    ):
        RepositorySandbox(workspace, task)


def test_read_range_is_bounded(
    sandbox: RepositorySandbox,
) -> None:
    result = sandbox.read_file(
        "src/calculator.py",
        start_line=1,
        end_line=500,
    )

    assert result.status is ObservationStatus.REJECTED
    assert "maximum" in result.summary
