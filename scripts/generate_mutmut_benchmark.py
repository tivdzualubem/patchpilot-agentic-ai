"""Generate PatchPilot benchmark tasks from killed mutmut mutants."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

COPY_IGNORE_PATTERNS = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".mutmut-cache",
    "mutants",
    "artifacts",
    "generated_benchmarks",
    "htmlcov",
    "*.pyc",
)


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result."""

    args: Sequence[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class MutantResult:
    """One parsed mutmut result row."""

    name: str
    status: str


@dataclass(frozen=True)
class GeneratedTask:
    """Metadata for one generated PatchPilot benchmark task."""

    task_id: str
    mutant_name: str
    mutant_status: str
    manifest_path: Path
    repository_path: Path
    expected_initial_failures: int


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    check: bool = False,
) -> CommandResult:
    """Run a command and return captured output."""
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        command = shlex.join(args)
        raise SystemExit(
            f"Command timed out after {timeout_seconds}s in {cwd}: {command}"
        ) from exc

    result = CommandResult(
        args=args,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        raise SystemExit(format_command_failure(result))
    return result


def format_command_failure(result: CommandResult) -> str:
    """Format a subprocess failure for terminal output."""
    command = shlex.join(result.args)
    return (
        f"Command failed with exit code {result.returncode}\n"
        f"cwd: {result.cwd}\n"
        f"command: {command}\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )


def copy_source_tree(source_root: Path, destination: Path) -> None:
    """Copy a clean source tree while excluding caches and generated files."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source_root,
        destination,
        ignore=shutil.ignore_patterns(*COPY_IGNORE_PATTERNS),
    )



def copy_mutmut_work_tree(source_root: Path, destination: Path) -> None:
    """Copy mutmut work tree while preserving mutant state needed by apply."""
    if destination.exists():
        shutil.rmtree(destination)
    ignore_patterns = tuple(
        pattern for pattern in COPY_IGNORE_PATTERNS
        if pattern not in {".mutmut-cache", "mutants"}
    )
    shutil.copytree(
        source_root,
        destination,
        ignore=shutil.ignore_patterns(*ignore_patterns),
    )

def has_toml_table(text: str, table_name: str) -> bool:
    """Return whether a TOML table exists in raw TOML text."""
    pattern = rf"(?m)^\[{re.escape(table_name)}\]\s*$"
    return re.search(pattern, text) is not None


def toml_string_list(values: Sequence[str]) -> str:
    """Render a simple TOML string list."""
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def ensure_mutmut_config(
    repository_root: Path,
    *,
    source_paths: Sequence[str],
    test_paths: Sequence[str],
    test_command: str,
    pytest_pythonpath: Sequence[str],
) -> None:
    """Add minimal mutmut config to the working copy if it is missing."""
    pyproject_path = repository_root / "pyproject.toml"
    text = ""
    if pyproject_path.exists():
        text = pyproject_path.read_text(encoding="utf-8")

    blocks: list[str] = []
    if pytest_pythonpath and not has_toml_table(text, "tool.pytest.ini_options"):
        blocks.append(
            "[tool.pytest.ini_options]\n"
            f"pythonpath = {toml_string_list(pytest_pythonpath)}"
        )

    if not has_toml_table(text, "tool.mutmut"):
        blocks.append(
            "[tool.mutmut]\n"
            f"source_paths = {toml_string_list(source_paths)}\n"
            f"runner = {json.dumps(test_command)}\n"
            f"pytest_add_cli_args_test_selection = {toml_string_list(test_paths)}"
        )

    if not blocks:
        return

    updated = text.rstrip()
    if updated:
        updated += "\n\n"
    updated += "\n\n".join(blocks)
    updated += "\n"
    pyproject_path.write_text(updated, encoding="utf-8")


def parse_mutmut_results(output: str) -> list[MutantResult]:
    """Parse `mutmut results --all true` output."""
    pattern = re.compile(r"^\s*(?P<name>[^:\s][^:]*):\s*(?P<status>[A-Za-z_-]+)")
    rows: list[MutantResult] = []
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        rows.append(
            MutantResult(
                name=match.group("name").strip(),
                status=match.group("status").strip().lower(),
            )
        )
    return rows


def slugify(value: str) -> str:
    """Convert a mutant name into a PatchPilot-safe task-id fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "mutant"



def mutant_group_key(mutant_name: str) -> str:
    """Return a stable function-level grouping key for a mutmut mutant."""
    match = re.search(r"\.x_(?P<function>.+?)__mutmut_\d+$", mutant_name)
    if match is not None:
        return match.group("function")
    return mutant_name.rsplit("__mutmut_", 1)[0]


def select_diverse_mutants(
    mutants: Sequence[MutantResult],
    max_tasks: int,
) -> list[MutantResult]:
    """Select mutants round-robin across mutated functions."""
    groups: dict[str, list[MutantResult]] = {}
    for mutant in mutants:
        groups.setdefault(mutant_group_key(mutant.name), []).append(mutant)

    selected: list[MutantResult] = []
    while len(selected) < max_tasks and groups:
        for key in sorted(list(groups)):
            bucket = groups[key]
            if not bucket:
                del groups[key]
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= max_tasks:
                break
    return selected

def build_task_id(prefix: str, mutant_name: str) -> str:
    """Build a schema-compatible task id no longer than 100 chars."""
    slug = slugify(mutant_name)
    max_slug_length = 99 - len(prefix)
    return f"{prefix}-{slug[:max_slug_length]}".strip("-")


def estimate_initial_failures(output: str) -> int:
    """Estimate initial failing-test count from pytest output."""
    patterns = (
        r"(?P<count>\d+)\s+failed",
        r"(?P<count>\d+)\s+errors?",
        r"(?P<count>\d+)\s+error",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is not None:
            return max(1, int(match.group("count")))
    return 1


def repository_root_for_manifest(repository_path: Path, project_root: Path) -> str:
    """Return a manifest repository root string."""
    try:
        return repository_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(repository_path.resolve())


def remove_runtime_artifacts(repository_root: Path) -> None:
    """Remove files that should never be part of exported repair tasks."""
    for name in (
        ".mutmut-cache",
        "mutants",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
    ):
        path = repository_root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    for pycache in repository_root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)


def write_json(path: Path, payload: object) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write generated-task metadata as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_mutant_task(
    *,
    mutant: MutantResult,
    work_root: Path,
    output_root: Path,
    project_root: Path,
    task_prefix: str,
    source_paths: Sequence[str],
    forbidden_paths: Sequence[str],
    test_command_args: Sequence[str],
    test_timeout_seconds: int,
    difficulty: str,
    force: bool,
) -> GeneratedTask | None:
    """Apply one killed mutant and export it as a PatchPilot task."""
    task_id = build_task_id(task_prefix, mutant.name)
    task_root = output_root / task_id
    repository_path = task_root / "repository"

    if task_root.exists():
        if not force:
            raise SystemExit(
                f"Task already exists: {task_root}. Re-run with --force."
            )
        shutil.rmtree(task_root)

    copy_mutmut_work_tree(work_root, repository_path)

    apply_result = run_command(
        ["mutmut", "apply", mutant.name],
        cwd=repository_path,
        timeout_seconds=120,
        check=False,
    )
    if apply_result.returncode != 0:
        print(f"SKIP {mutant.name}: mutmut apply failed")
        print(apply_result.stderr)
        shutil.rmtree(task_root)
        return None

    remove_runtime_artifacts(repository_path)

    test_result = run_command(
        test_command_args,
        cwd=repository_path,
        timeout_seconds=test_timeout_seconds,
        check=False,
    )
    combined_test_output = test_result.stdout + "\n" + test_result.stderr
    if test_result.returncode == 0:
        print(f"SKIP {mutant.name}: exported mutant did not fail tests")
        shutil.rmtree(task_root)
        return None

    expected_failures = estimate_initial_failures(combined_test_output)
    remove_runtime_artifacts(repository_path)
    manifest = {
        "task_id": task_id,
        "title": f"Mutmut-generated killed mutant: {mutant.name}",
        "goal": (
            "Repair the mutmut-generated Python defect so that all "
            "regression tests pass. Do not edit tests."
        ),
        "repository_root": repository_root_for_manifest(
            repository_path,
            project_root,
        ),
        "defect_category": "mutmut_killed_mutant",
        "difficulty": difficulty,
        "allowed_paths": list(source_paths),
        "forbidden_paths": list(forbidden_paths),
        "test_command": list(test_command_args),
        "expected_initial_failures": expected_failures,
    }
    write_json(task_root / "task.json", manifest)
    write_json(
        task_root / "mutmut_metadata.json",
        {
            "mutant_name": mutant.name,
            "mutant_status": mutant.status,
            "expected_initial_failures": expected_failures,
        },
    )

    return GeneratedTask(
        task_id=task_id,
        mutant_name=mutant.name,
        mutant_status=mutant.status,
        manifest_path=task_root / "task.json",
        repository_path=repository_path,
        expected_initial_failures=expected_failures,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PatchPilot tasks from killed mutmut mutants."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--test-path", action="append", default=None)
    parser.add_argument("--test-command", default="python -m pytest -q")
    parser.add_argument("--pytest-pythonpath", action="append", default=[])
    parser.add_argument("--output-root", default="generated_benchmarks/mutmut")
    parser.add_argument("--work-root", default="artifacts/mutmut_generation")
    parser.add_argument("--task-prefix", default="mutmut")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
    )
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--mutmut-timeout-seconds", type=int, default=1800)
    parser.add_argument("--test-timeout-seconds", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate mutmut benchmark tasks."""
    args = parse_args()
    project_root = Path.cwd().resolve()
    source_root = Path(args.source_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    work_base = Path(args.work_root).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    generation_root = work_base / timestamp
    work_root = generation_root / "work"

    test_paths = args.test_path or ["tests"]
    test_command_args = shlex.split(args.test_command)
    if len(test_command_args) < 3:
        raise SystemExit("--test-command must contain at least 3 arguments")

    generation_root.mkdir(parents=True, exist_ok=True)
    copy_source_tree(source_root, work_root)
    ensure_mutmut_config(
        work_root,
        source_paths=args.source_path,
        test_paths=test_paths,
        test_command=args.test_command,
        pytest_pythonpath=args.pytest_pythonpath,
    )

    print(f"WORK_ROOT={work_root}", flush=True)
    print("RUN clean tests", flush=True)
    run_command(
        test_command_args,
        cwd=work_root,
        timeout_seconds=args.test_timeout_seconds,
        check=True,
    )

    print("RUN mutmut", flush=True)
    run_command(
        ["mutmut", "run"],
        cwd=work_root,
        timeout_seconds=args.mutmut_timeout_seconds,
        check=True,
    )

    print("READ mutmut results", flush=True)
    results = run_command(
        ["mutmut", "results", "--all", "true"],
        cwd=work_root,
        timeout_seconds=120,
        check=True,
    )
    mutants = parse_mutmut_results(results.stdout)
    killed = [mutant for mutant in mutants if mutant.status == "killed"]
    selected = select_diverse_mutants(killed, args.max_tasks)

    if not selected:
        raise SystemExit("No killed mutants found.")

    forbidden_paths = sorted(
        {*test_paths, ".mutmut-cache", "mutants", "__pycache__"}
    )
    generated: list[GeneratedTask] = []
    for mutant in selected:
        print(f"EXPORT {mutant.name}", flush=True)
        task = export_mutant_task(
            mutant=mutant,
            work_root=work_root,
            output_root=output_root,
            project_root=project_root,
            task_prefix=args.task_prefix,
            source_paths=args.source_path,
            forbidden_paths=forbidden_paths,
            test_command_args=test_command_args,
            test_timeout_seconds=args.test_timeout_seconds,
            difficulty=args.difficulty,
            force=args.force,
        )
        if task is not None:
            generated.append(task)
            print(f"DONE {task.task_id}", flush=True)

    rows = [
        {
            "task_id": task.task_id,
            "mutant_name": task.mutant_name,
            "mutant_status": task.mutant_status,
            "manifest_path": repository_root_for_manifest(
                task.manifest_path,
                project_root,
            ),
            "repository_path": repository_root_for_manifest(
                task.repository_path,
                project_root,
            ),
            "expected_initial_failures": task.expected_initial_failures,
        }
        for task in generated
    ]
    write_csv(output_root / "mutmut_tasks.csv", rows)
    write_json(
        output_root / "generation_summary.json",
        {
            "source_dataset": source_root.name,
            "output_root": repository_root_for_manifest(output_root, project_root),
            "mutants_total": len(mutants),
            "killed_mutants": len(killed),
            "generated_tasks": len(generated),
            "max_tasks": args.max_tasks,
        },
    )

    print(f"OUTPUT_ROOT={output_root}")
    print(f"GENERATED_TASKS={len(generated)}")
    print(f"KILLED_MUTANTS={len(killed)}")


if __name__ == "__main__":
    main()
