from pathlib import Path

from patchpilot.benchmark.workspace import BenchmarkWorkspace


def make_workspace(tmp_path: Path) -> BenchmarkWorkspace:
    return BenchmarkWorkspace(
        project_root=Path("."),
        output_root=tmp_path / "runs",
    )


def test_prepare_creates_isolated_copy(tmp_path: Path) -> None:
    manager = make_workspace(tmp_path)
    prepared = manager.prepare(
        Path("benchmarks/calculator-001/task.json")
    )

    assert prepared.repository_root.is_dir()
    assert prepared.task.repository_root == "repository"
    assert (
        prepared.repository_root / "src" / "calculator.py"
    ).is_file()


def test_changes_do_not_modify_original(tmp_path: Path) -> None:
    manager = make_workspace(tmp_path)
    prepared = manager.prepare(
        Path("benchmarks/calculator-001/task.json")
    )

    copied = prepared.repository_root / "src" / "calculator.py"
    original = Path(
        "benchmarks/calculator-001/repository/src/calculator.py"
    )

    original_content = original.read_text(encoding="utf-8")
    copied.write_text("# modified copy\n", encoding="utf-8")

    assert original.read_text(encoding="utf-8") == original_content


def test_cleanup_removes_workspace(tmp_path: Path) -> None:
    manager = make_workspace(tmp_path)
    prepared = manager.prepare(
        Path("benchmarks/calculator-001/task.json")
    )

    workspace_root = prepared.workspace_root
    manager.cleanup(prepared)

    assert not workspace_root.exists()
