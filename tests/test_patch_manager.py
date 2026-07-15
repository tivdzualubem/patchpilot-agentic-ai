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
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )

    (repository / "tests" / "test_calculator.py").write_text(
        "def test_add() -> None:\n    assert True\n",
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

    second_patch = (
        valid_patch()
        .replace(
            "left - right",
            "left + right",
        )
        .replace(
            "left + right\n",
            "left * right\n",
            1,
        )
    )
    manager.apply_patch(second_patch)

    result = manager.restore_all()

    assert result.status is ObservationStatus.OK
    assert "return left - right" in source.read_text()
    assert manager.changed_files == ()


def multiline_patch() -> str:
    """Return a valid bounded multi-line calculator repair."""
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def add(left: int, right: int) -> int:\n"
        '+    """Return the arithmetic sum."""\n'
        "-    return left - right\n"
        "+    return left + right\n"
    )


def test_bounded_multiline_patch_is_applied(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment

    result = manager.apply_patch(multiline_patch())

    assert result.status is ObservationStatus.OK
    content = source.read_text(encoding="utf-8")
    assert '"""Return the arithmetic sum."""' in content
    assert "return left + right" in content


def test_patch_exceeding_changed_line_limit_is_rejected(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    removed = "".join(f"-old_{index}\n" for index in range(11))
    added = "".join(f"+new_{index}\n" for index in range(11))
    patch = (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,11 +1,11 @@\n"
        f"{removed}{added}"
    )
    original = source.read_text(encoding="utf-8")

    result = manager.apply_patch(patch)

    assert result.status is ObservationStatus.REJECTED
    assert "configured limit" in result.summary
    assert "22 > 20" in result.summary
    assert source.read_text(encoding="utf-8") == original


def test_patch_modifying_more_than_two_files_is_rejected(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    source_root = source.parent

    for name in ("one.py", "two.py", "three.py"):
        (source_root / name).write_text("value = 1\n", encoding="utf-8")

    sections = []
    for name in ("one.py", "two.py", "three.py"):
        sections.append(
            f"diff --git a/src/{name} b/src/{name}\n"
            f"--- a/src/{name}\n"
            f"+++ b/src/{name}\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )

    result = manager.apply_patch("".join(sections))

    assert result.status is ObservationStatus.REJECTED
    assert "more files than the configured limit" in result.summary


def two_file_patch() -> str:
    """Return one bounded patch attempt that changes two source files."""
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(left: int, right: int) -> int:\n"
        "-    return left + right\n"
        "+    return left * right\n"
        "diff --git a/src/helpers.py b/src/helpers.py\n"
        "--- a/src/helpers.py\n"
        "+++ b/src/helpers.py\n"
        "@@ -1 +1 @@\n"
        "-SCALE = 1\n"
        "+SCALE = 2\n"
    )


def test_latest_attempt_rollback_restores_all_attempt_files(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    helper = source.parent / "helpers.py"
    helper.write_text("SCALE = 1\n", encoding="utf-8")

    first = manager.apply_patch(valid_patch())
    second = manager.apply_patch(two_file_patch())

    assert first.status is ObservationStatus.OK
    assert second.status is ObservationStatus.OK
    assert manager.current_attempt_id == 2
    assert manager.current_attempt_files == (
        "src/calculator.py",
        "src/helpers.py",
    )

    rolled_back = manager.restore_attempt()

    assert rolled_back.status is ObservationStatus.OK
    assert "patch attempt 2" in rolled_back.summary
    assert rolled_back.output.splitlines() == [
        "src/calculator.py",
        "src/helpers.py",
    ]
    assert "return left + right" in source.read_text(encoding="utf-8")
    assert helper.read_text(encoding="utf-8") == "SCALE = 1\n"
    assert manager.current_attempt_id == 1
    assert manager.current_attempt_files == ("src/calculator.py",)
    assert manager.changed_files == ("src/calculator.py",)


def test_attempt_rollback_preserves_earlier_successful_attempt(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    helper = source.parent / "helpers.py"
    helper.write_text("SCALE = 1\n", encoding="utf-8")

    manager.apply_patch(valid_patch())
    manager.apply_patch(two_file_patch())
    manager.restore_attempt()

    rolled_back_first = manager.restore_attempt()

    assert rolled_back_first.status is ObservationStatus.OK
    assert "patch attempt 1" in rolled_back_first.summary
    assert "return left - right" in source.read_text(encoding="utf-8")
    assert helper.read_text(encoding="utf-8") == "SCALE = 1\n"
    assert manager.current_attempt_id is None
    assert manager.current_attempt_files == ()
    assert manager.changed_files == ()


def test_attempt_rollback_is_atomic_on_write_failure(
    patch_environment: tuple[Path, PatchManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manager = patch_environment
    helper = source.parent / "helpers.py"
    helper.write_text("SCALE = 1\n", encoding="utf-8")

    manager.apply_patch(valid_patch())
    manager.apply_patch(two_file_patch())
    patched_source = source.read_bytes()
    patched_helper = helper.read_bytes()

    original_write_bytes = Path.write_bytes
    failed = False

    def fail_second_restore(path: Path, data: bytes) -> int:
        nonlocal failed
        if path == helper and data == b"SCALE = 1\n" and not failed:
            failed = True
            raise OSError("simulated rollback write failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_second_restore)

    result = manager.restore_attempt()

    assert result.status is ObservationStatus.ERROR
    assert "partial rollback" in result.summary
    assert source.read_bytes() == patched_source
    assert helper.read_bytes() == patched_helper
    assert manager.current_attempt_id == 2
    assert manager.current_attempt_files == (
        "src/calculator.py",
        "src/helpers.py",
    )


def test_restore_attempt_without_active_attempt_reports_error(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    _, manager = patch_environment

    result = manager.restore_attempt()

    assert result.status is ObservationStatus.ERROR
    assert "No active patch attempt" in result.summary


def test_accept_attempt_preserves_changes_and_closes_transaction(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    source, manager = patch_environment
    manager.apply_patch(valid_patch())

    manager.accept_attempt(1)

    assert "return left + right" in source.read_text(encoding="utf-8")
    assert manager.current_attempt_id is None
    assert manager.current_attempt_files == ()
    assert manager.changed_files == ("src/calculator.py",)


def test_accept_attempt_rejects_non_latest_identifier(
    patch_environment: tuple[Path, PatchManager],
) -> None:
    _, manager = patch_environment
    manager.apply_patch(valid_patch())

    with pytest.raises(
        ValueError,
        match="latest active",
    ):
        manager.accept_attempt(2)
