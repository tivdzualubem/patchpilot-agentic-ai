"""Tests for bounded executable repair verification."""

from pathlib import Path

import pytest

from patchpilot.schemas import ObservationStatus, RepairTask
from patchpilot.tools import (
    RepositorySandbox,
    TestExecutionError,
    TestRunner,
)


@pytest.fixture()
def repair_repository(
    tmp_path: Path,
) -> tuple[Path, RepairTask, RepositorySandbox]:
    """Create a repository containing one deliberate defect."""
    repository = tmp_path / "benchmarks" / "calculator"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()

    implementation = repository / "src" / "calculator.py"
    implementation.write_text(
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

    task = RepairTask(
        task_id="calculator-addition-001",
        goal="Repair the incorrect calculator addition operation.",
        repository_root="benchmarks/calculator",
    )
    sandbox = RepositorySandbox(tmp_path, task)

    return implementation, task, sandbox


def test_failing_suite_returns_error(
    repair_repository: tuple[
        Path,
        RepairTask,
        RepositorySandbox,
    ],
) -> None:
    _, task, sandbox = repair_repository
    result = TestRunner(sandbox, task).run_tests()

    assert result.status is ObservationStatus.ERROR
    assert "exit code 1" in result.summary
    assert "1 failed" in result.output


def test_repaired_suite_returns_success(
    repair_repository: tuple[
        Path,
        RepairTask,
        RepositorySandbox,
    ],
) -> None:
    implementation, task, sandbox = repair_repository
    implementation.write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n",
        encoding="utf-8",
    )

    result = TestRunner(sandbox, task).run_tests()

    assert result.status is ObservationStatus.OK
    assert "exit code 0" in result.summary
    assert "1 passed" in result.output


def test_targeted_test_execution(
    repair_repository: tuple[
        Path,
        RepairTask,
        RepositorySandbox,
    ],
) -> None:
    _, task, sandbox = repair_repository
    result = TestRunner(sandbox, task).run_tests(
        "tests/test_calculator.py::test_add"
    )

    assert result.status is ObservationStatus.ERROR
    assert "test_add" in result.output


@pytest.mark.parametrize(
    "command",
    [
        ["bash", "-c", "pytest"],
        ["python", "script.py"],
        ["python", "-m", "pytest", "--rootdir=/tmp"],
        ["python", "-m", "pytest", "-c", "/tmp/config"],
    ],
)
def test_unsafe_commands_are_rejected(
    repair_repository: tuple[
        Path,
        RepairTask,
        RepositorySandbox,
    ],
    command: list[str],
) -> None:
    _, _, sandbox = repair_repository
    task = RepairTask(
        task_id="unsafe-command-001",
        goal="Verify that unsafe test commands are rejected.",
        repository_root="benchmarks/calculator",
        test_command=command,
    )

    with pytest.raises(TestExecutionError):
        TestRunner(sandbox, task)


@pytest.mark.parametrize(
    "target",
    [
        "../outside.py",
        "/etc/passwd",
        "-k expression",
        "tests/missing.py",
    ],
)
def test_unsafe_targets_are_rejected(
    repair_repository: tuple[
        Path,
        RepairTask,
        RepositorySandbox,
    ],
    target: str,
) -> None:
    _, task, sandbox = repair_repository
    result = TestRunner(sandbox, task).run_tests(target)

    assert result.status is ObservationStatus.REJECTED


def test_timeout_is_reported(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "benchmarks" / "slow"
    (repository / "tests").mkdir(parents=True)

    (repository / "tests" / "test_slow.py").write_text(
        "import time\n\n"
        "def test_slow() -> None:\n"
        "    time.sleep(3)\n",
        encoding="utf-8",
    )

    task = RepairTask(
        task_id="slow-test-001",
        goal="Verify that long-running tests are terminated safely.",
        repository_root="benchmarks/slow",
    )
    sandbox = RepositorySandbox(tmp_path, task)

    result = TestRunner(
        sandbox,
        task,
        timeout_seconds=1,
    ).run_tests()

    assert result.status is ObservationStatus.TIMEOUT
    assert "1-second timeout" in result.summary


def test_environment_does_not_expose_parent_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "benchmarks" / "environment"
    (repository / "tests").mkdir(parents=True)

    (repository / "tests" / "test_environment.py").write_text(
        "import os\n\n"
        "def test_secret_is_absent() -> None:\n"
        "    assert os.getenv('PATCHPILOT_PRIVATE_TOKEN') is None\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "PATCHPILOT_PRIVATE_TOKEN",
        "must-not-leak",
    )

    task = RepairTask(
        task_id="environment-isolation-001",
        goal="Verify that inherited secrets are removed from tests.",
        repository_root="benchmarks/environment",
    )
    sandbox = RepositorySandbox(tmp_path, task)

    result = TestRunner(sandbox, task).run_tests()

    assert result.status is ObservationStatus.OK
