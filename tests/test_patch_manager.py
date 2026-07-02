"""Tests for safe patch application and rollback."""

from pathlib import Path

import pytest

from patchpilot.schemas import ObservationStatus, RepairTask
from patchpilot.tools import PatchManager, RepositorySandbox


@pytest.fixture()
def patch_environment(
    tmp_path: Path,
) -> tuple[Path, PatchManager]:
    """Create a small repository and patch manager."""
    repository = tmp_path / "benchmarks" / "calculator"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()

    source = repository / "src" / "calculator.py"
    source.write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n",
        encoding="utf-8",
    )

    (repository / "tests" / "test_calculator.py").write_text(
        "def test_add() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )

    task = RepairTask(
        task_id="calculator-patch-001",
        goal="Repair the calculator addition implementation.",
        repository_root="benchmarks/calculator",
    )

    sandbox = RepositorySandbox(tmp_path, task)

    return source, PatchManager(sandbox, task)


def valid_patch() -> str:
    """Return a valid one-line calculator repair."""
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(left: int, right: int) -> int:\n"
        "-    return left - right\n"
        "+    return left + right\n"
    )


def test_valid_patch_is_applied(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment

    result = manager.apply_patch(valid_patch())

    assert result.status is ObservationStatus.OK
    assert manager.changed_files == ("src/calculator.py",)
    assert "return left + right" in source.read_text()


def test_view_diff_records_actual_change(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    _, manager = patch_environment
    manager.apply_patch(valid_patch())

    result = manager.view_diff()

    assert result.status is ObservationStatus.OK
    assert "-    return left - right" in result.output
    assert "+    return left + right" in result.output


def test_restore_file_reverts_patch(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    manager.apply_patch(valid_patch())

    result = manager.restore_file("src/calculator.py")

    assert result.status is ObservationStatus.OK
    assert "return left - right" in source.read_text()
    assert manager.changed_files == ()


def test_invalid_patch_does_not_mutate_file(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    original = source.read_text()

    malformed = valid_patch().replace(
        "return left - right",
        "return unknown text",
    )
    result = manager.apply_patch(malformed)

    assert result.status is ObservationStatus.ERROR
    assert source.read_text() == original


def test_test_file_modification_is_rejected(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    _, manager = patch_environment

    patch = (
        "diff --git a/tests/test_calculator.py "
        "b/tests/test_calculator.py\n"
        "--- a/tests/test_calculator.py\n"
        "+++ b/tests/test_calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def test_add() -> None:\n"
        "-    assert True\n"
        "+    assert False\n"
    )

    result = manager.apply_patch(patch)

    assert result.status is ObservationStatus.REJECTED
    assert "outside allowed paths" in result.summary


@pytest.mark.parametrize(
    "patch",
    [
        (
            "diff --git a/../outside.py b/../outside.py\n"
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
        (
            "diff --git a/src/new.py b/src/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/src/new.py\n"
            "@@ -0,0 +1 @@\n"
            "+value = 1\n"
        ),
        (
            "diff --git a/src/calculator.py b/src/renamed.py\n"
            "--- a/src/calculator.py\n"
            "+++ b/src/renamed.py\n"
        ),
    ],
)
def test_unsafe_patch_shapes_are_rejected(
    patch_environment: tuple[Path, PatchManager],
    patch: str,
) -> None:
    _, manager = patch_environment

    result = manager.apply_patch(patch)

    assert result.status is ObservationStatus.REJECTED


def test_restore_all_reverts_multiple_patch_attempts(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    manager.apply_patch(valid_patch())

    second_patch = valid_patch().replace(
        "left - right",
        "left + right",
    ).replace(
        "left + right\n",
        "left * right\n",
        1,
    )
    manager.apply_patch(second_patch)

    result = manager.restore_all()

    assert result.status is ObservationStatus.OK
    assert "return left - right" in source.read_text()
    assert manager.changed_files == ()
