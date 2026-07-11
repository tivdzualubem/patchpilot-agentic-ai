"""Generate validated PatchPilot benchmarks from killed Mutmut mutants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from patchpilot.benchmark.provenance import (
    MutationOperatorFamily,
    MutmutProvenance,
    classify_mutation,
    mutation_diff_sha256,
)

COPY_IGNORE_DIRECTORIES = {
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
}
_MAX_SHOW_BYTES = 1_000_000


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
    """One parsed Mutmut result row."""

    name: str
    status: str


@dataclass(frozen=True)
class MutantDescriptor:
    """One killed mutant enriched with its exact mutation diff."""

    name: str
    status: str
    function: str
    source_file: str
    source_line: int | None
    operator_family: MutationOperatorFamily
    mutation_diff: str


@dataclass(frozen=True)
class GeneratedTask:
    """Metadata for one generated PatchPilot benchmark task."""

    task_id: str
    mutant_name: str
    mutant_status: str
    operator_family: str
    mutated_function: str
    manifest_path: Path
    repository_path: Path
    provenance_path: Path
    visible_initial_failures: int
    hidden_initial_failures: int


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command without a shell and capture its complete output."""
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
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


def _is_ignored(relative: Path) -> bool:
    return (
        any(part in COPY_IGNORE_DIRECTORIES for part in relative.parts)
        or relative.suffix == ".pyc"
    )


def validate_source_tree(source_root: Path) -> None:
    """Reject symlinks and empty source trees before generation."""
    if not source_root.is_dir():
        raise SystemExit(f"Source root is not a directory: {source_root}")
    if any(path.is_symlink() for path in source_root.rglob("*")):
        raise SystemExit("Seed projects cannot contain symbolic links.")
    if not any(
        path.is_file()
        for path in source_root.rglob("*")
        if not _is_ignored(path.relative_to(source_root))
    ):
        raise SystemExit("Seed project contains no usable files.")


def source_tree_sha256(source_root: Path) -> str:
    """Hash all included seed files and paths deterministically."""
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if _is_ignored(relative):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_source_tree(source_root: Path, destination: Path) -> None:
    """Copy a clean source tree while excluding runtime artifacts."""
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(directory: str, names: list[str]) -> set[str]:
        root = Path(directory)
        ignored: set[str] = set()
        for name in names:
            candidate = root / name
            try:
                relative = candidate.relative_to(source_root)
            except ValueError:
                continue
            if _is_ignored(relative):
                ignored.add(name)
        return ignored

    shutil.copytree(source_root, destination, ignore=ignore)


def copy_mutmut_work_tree(work_root: Path, destination: Path) -> None:
    """Copy the Mutmut work tree while retaining state needed by apply."""
    if destination.exists():
        shutil.rmtree(destination)

    ignored = COPY_IGNORE_DIRECTORIES - {".mutmut-cache", "mutants"}

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    shutil.copytree(work_root, destination, ignore=ignore)


def has_toml_table(text: str, table_name: str) -> bool:
    """Return whether a TOML table exists in raw TOML text."""
    pattern = rf"(?m)^\[{re.escape(table_name)}\]\s*$"
    return re.search(pattern, text) is not None


def toml_string_list(values: Sequence[str]) -> str:
    """Render a deterministic TOML string list."""
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def ensure_mutmut_config(
    repository_root: Path,
    *,
    source_paths: Sequence[str],
    test_paths: Sequence[str],
    test_command: str,
    pytest_pythonpath: Sequence[str],
) -> None:
    """Add minimal Mutmut and pytest configuration when absent."""
    pyproject_path = repository_root / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8") if pyproject_path.exists() else ""

    blocks: list[str] = []
    if pytest_pythonpath and not has_toml_table(
        text,
        "tool.pytest.ini_options",
    ):
        blocks.append(
            "[tool.pytest.ini_options]\n"
            f"pythonpath = {toml_string_list(pytest_pythonpath)}"
        )

    if not has_toml_table(text, "tool.mutmut"):
        blocks.append(
            "[tool.mutmut]\n"
            f"source_paths = {toml_string_list(source_paths)}\n"
            f"runner = {json.dumps(test_command)}\n"
            f"pytest_add_cli_args_test_selection = "
            f"{toml_string_list(test_paths)}"
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
    """Parse ``mutmut results --all true`` output."""
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


def mutant_group_key(mutant_name: str) -> str:
    """Return a stable function-level grouping key."""
    match = re.search(r"\.x_(?P<function>.+?)__mutmut_\d+$", mutant_name)
    if match is not None:
        return match.group("function")
    return mutant_name.rsplit("__mutmut_", 1)[0]


def parse_mutmut_show(
    mutant: MutantResult,
    output: str,
) -> MutantDescriptor:
    """Parse ``mutmut show`` into a validated descriptor."""
    encoded = output.encode("utf-8")
    if not output.strip() or len(encoded) > _MAX_SHOW_BYTES:
        raise ValueError(f"Invalid Mutmut show output for {mutant.name}.")

    lines = output.splitlines()
    diff_start = next(
        (index for index, line in enumerate(lines) if line.startswith("--- ")),
        None,
    )
    if diff_start is None:
        raise ValueError(f"Mutmut show output has no diff for {mutant.name}.")
    mutation_diff = "\n".join(lines[diff_start:]).strip() + "\n"

    source_line_text = lines[diff_start][4:].strip()
    source_file = source_line_text.removeprefix("a/")
    hunk = re.search(r"^@@\s+-(?P<line>\d+)", mutation_diff, re.MULTILINE)
    source_line = int(hunk.group("line")) if hunk is not None else None

    return MutantDescriptor(
        name=mutant.name,
        status=mutant.status,
        function=mutant_group_key(mutant.name),
        source_file=source_file,
        source_line=source_line,
        operator_family=classify_mutation(mutation_diff),
        mutation_diff=mutation_diff,
    )


def describe_killed_mutants(
    killed: Sequence[MutantResult],
    *,
    work_root: Path,
) -> list[MutantDescriptor]:
    """Read the exact diff for every killed mutant."""
    descriptors: list[MutantDescriptor] = []
    for mutant in sorted(killed, key=lambda item: item.name):
        result = run_command(
            ["mutmut", "show", mutant.name],
            cwd=work_root,
            timeout_seconds=120,
            check=True,
        )
        descriptors.append(parse_mutmut_show(mutant, result.stdout))
    return descriptors


def select_diverse_mutants(
    descriptors: Sequence[MutantDescriptor],
    max_tasks: int,
    max_per_function: int,
) -> list[MutantDescriptor]:
    """Select deterministically across functions and operator families."""
    if max_tasks < 1:
        raise ValueError("max_tasks must be at least 1.")
    if max_per_function < 1:
        raise ValueError("max_per_function must be at least 1.")

    groups: dict[str, list[MutantDescriptor]] = {}
    seen_diffs: set[str] = set()
    for descriptor in sorted(
        descriptors,
        key=lambda item: (
            item.function,
            item.operator_family.value,
            item.name,
        ),
    ):
        digest = mutation_diff_sha256(descriptor.mutation_diff)
        if digest in seen_diffs:
            continue
        seen_diffs.add(digest)
        bucket = groups.setdefault(descriptor.function, [])
        if len(bucket) < max_per_function:
            bucket.append(descriptor)

    selected: list[MutantDescriptor] = []
    while len(selected) < max_tasks and groups:
        progressed = False
        for key in sorted(list(groups)):
            bucket = groups[key]
            if not bucket:
                del groups[key]
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= max_tasks:
                break
        if not progressed:
            break
    return selected


def slugify(value: str) -> str:
    """Convert text into a PatchPilot-safe task-id fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "mutant"


def build_task_id(prefix: str, mutant_name: str) -> str:
    """Build a schema-compatible task id no longer than 100 chars."""
    slug = slugify(mutant_name)
    max_slug_length = 99 - len(prefix)
    return f"{prefix}-{slug[:max_slug_length]}".strip("-")


def estimate_initial_failures(output: str) -> int:
    """Extract a failing-test count from pytest output."""
    patterns = (
        r"(?P<count>\d+)\s+failed",
        r"(?P<count>\d+)\s+errors?",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is not None:
            return max(1, int(match.group("count")))
    return 1


def pytest_test_count(output: str) -> int:
    """Return the total pytest outcome count in one summary."""
    pattern = re.compile(
        r"(?P<count>\d+)\s+"
        r"(?:passed|failed|errors?|skipped|xfailed|xpassed)"
    )
    return sum(int(match.group("count")) for match in pattern.finditer(output))


def repository_root_for_manifest(path: Path, project_root: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def remove_runtime_artifacts(repository_root: Path) -> None:
    """Remove files that must not appear in exported repair tasks."""
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


def hidden_environment(repository_root: Path) -> dict[str, str]:
    """Build a minimal environment for hidden tests."""
    python_paths = [repository_root]
    source_root = repository_root / "src"
    if source_root.is_dir():
        python_paths.append(source_root)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def run_hidden_tests(
    *,
    repository_root: Path,
    hidden_test_root: Path,
    timeout_seconds: int,
) -> CommandResult:
    """Run hidden tests outside the agent-visible repository."""
    return run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(hidden_test_root.resolve(strict=True)),
        ],
        cwd=repository_root,
        timeout_seconds=timeout_seconds,
        env=hidden_environment(repository_root),
    )


def difficulty_for(
    family: MutationOperatorFamily,
    requested: str,
) -> str:
    """Resolve explicit or deterministic automatic difficulty."""
    if requested != "auto":
        return requested
    if family in {
        MutationOperatorFamily.ARITHMETIC,
        MutationOperatorFamily.BOUNDARY,
        MutationOperatorFamily.COMPARISON,
        MutationOperatorFamily.CONSTANT,
    }:
        return "easy"
    if family in {
        MutationOperatorFamily.BOOLEAN,
        MutationOperatorFamily.COLLECTION,
        MutationOperatorFamily.RETURN_VALUE,
    }:
        return "medium"
    return "hard"


def git_commit(project_root: Path) -> str:
    """Return the generator Git commit."""
    result = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        timeout_seconds=30,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def mutmut_version(project_root: Path) -> str:
    """Return the installed Mutmut version string."""
    result = run_command(
        ["mutmut", "--version"],
        cwd=project_root,
        timeout_seconds=30,
        check=True,
    )
    text = (result.stdout or result.stderr).strip()
    match = re.search(r"version\s+(?P<version>\S+)", text)
    return match.group("version") if match is not None else text


def write_json(path: Path, payload: object) -> None:
    """Write stable human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write generated-task metadata as deterministic CSV."""
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
    descriptor: MutantDescriptor,
    selection_rank: int,
    work_root: Path,
    output_root: Path,
    project_root: Path,
    source_project: str,
    source_hash: str,
    generator_commit_value: str,
    mutmut_version_value: str,
    generation_command: list[str],
    selected_from_total: int,
    selected_from_killed: int,
    task_prefix: str,
    source_paths: Sequence[str],
    forbidden_paths: Sequence[str],
    test_command_args: Sequence[str],
    hidden_source_root: Path,
    hidden_relative_path: Path,
    clean_hidden_test_count: int,
    test_timeout_seconds: int,
    requested_difficulty: str,
    force: bool,
) -> GeneratedTask | None:
    """Apply, validate, and export one killed mutant."""
    task_id = build_task_id(task_prefix, descriptor.name)
    task_root = output_root / task_id
    repository_path = task_root / "repository"
    hidden_export_root = task_root / "hidden_tests"

    if task_root.exists():
        if not force:
            raise SystemExit(f"Task already exists: {task_root}. Re-run with --force.")
        shutil.rmtree(task_root)

    copy_mutmut_work_tree(work_root, repository_path)
    exposed_hidden = repository_path / hidden_relative_path
    if exposed_hidden.is_dir():
        shutil.rmtree(exposed_hidden)
    elif exposed_hidden.exists():
        exposed_hidden.unlink()

    apply_result = run_command(
        ["mutmut", "apply", descriptor.name],
        cwd=repository_path,
        timeout_seconds=120,
    )
    if apply_result.returncode != 0:
        print(f"SKIP {descriptor.name}: mutmut apply failed")
        shutil.rmtree(task_root)
        return None

    remove_runtime_artifacts(repository_path)
    shutil.copytree(hidden_source_root, hidden_export_root)

    visible_result = run_command(
        test_command_args,
        cwd=repository_path,
        timeout_seconds=test_timeout_seconds,
    )
    visible_output = visible_result.stdout + "\n" + visible_result.stderr
    if visible_result.returncode == 0:
        print(f"SKIP {descriptor.name}: mutant passed visible tests")
        shutil.rmtree(task_root)
        return None

    hidden_result = run_hidden_tests(
        repository_root=repository_path,
        hidden_test_root=hidden_export_root,
        timeout_seconds=test_timeout_seconds,
    )
    hidden_output = hidden_result.stdout + "\n" + hidden_result.stderr
    if hidden_result.returncode == 0:
        print(f"SKIP {descriptor.name}: mutant passed hidden tests")
        shutil.rmtree(task_root)
        return None

    hidden_count = pytest_test_count(hidden_output)
    if hidden_count != clean_hidden_test_count:
        print(
            f"SKIP {descriptor.name}: hidden test count changed "
            f"from {clean_hidden_test_count} to {hidden_count}"
        )
        shutil.rmtree(task_root)
        return None

    visible_failures = estimate_initial_failures(visible_output)
    hidden_failures = estimate_initial_failures(hidden_output)
    difficulty = difficulty_for(
        descriptor.operator_family,
        requested_difficulty,
    )
    remove_runtime_artifacts(repository_path)

    hidden_manifest_root = repository_root_for_manifest(
        hidden_export_root,
        project_root,
    )
    manifest = {
        "task_id": task_id,
        "title": f"Mutmut-generated killed mutant: {descriptor.name}",
        "goal": (
            "Repair the Mutmut-generated Python defect so that all "
            "regression tests pass. Do not edit tests."
        ),
        "repository_root": repository_root_for_manifest(
            repository_path,
            project_root,
        ),
        "defect_category": (f"mutmut_{descriptor.operator_family.value}"),
        "difficulty": difficulty,
        "allowed_paths": list(source_paths),
        "forbidden_paths": list(forbidden_paths),
        "test_command": list(test_command_args),
        "expected_initial_failures": visible_failures,
        "hidden_test_root": hidden_manifest_root,
        "expected_hidden_tests": clean_hidden_test_count,
    }
    write_json(task_root / "task.json", manifest)

    provenance = MutmutProvenance(
        source_project=source_project,
        source_root_sha256=source_hash,
        generator_commit=generator_commit_value,
        mutmut_version=mutmut_version_value,
        generation_command=generation_command,
        selection_rank=selection_rank,
        selected_from_total=selected_from_total,
        selected_from_killed=selected_from_killed,
        mutant_name=descriptor.name,
        mutant_status="killed",
        mutated_function=descriptor.function,
        source_file=descriptor.source_file,
        source_line=descriptor.source_line,
        operator_family=descriptor.operator_family,
        mutation_diff=descriptor.mutation_diff,
        mutation_diff_sha256=mutation_diff_sha256(descriptor.mutation_diff),
        test_command=list(test_command_args),
        visible_tests_pass_on_clean=True,
        hidden_tests_pass_on_clean=True,
        visible_initial_failures=visible_failures,
        hidden_initial_failures=hidden_failures,
        hidden_test_count=clean_hidden_test_count,
        difficulty=difficulty,
    )
    provenance_path = task_root / "provenance.json"
    write_json(provenance_path, provenance)
    write_json(
        task_root / "mutmut_metadata.json",
        {
            "mutant_name": descriptor.name,
            "mutant_status": descriptor.status,
            "operator_family": descriptor.operator_family.value,
            "mutated_function": descriptor.function,
            "visible_initial_failures": visible_failures,
            "hidden_initial_failures": hidden_failures,
            "provenance_file": "provenance.json",
        },
    )

    return GeneratedTask(
        task_id=task_id,
        mutant_name=descriptor.name,
        mutant_status=descriptor.status,
        operator_family=descriptor.operator_family.value,
        mutated_function=descriptor.function,
        manifest_path=task_root / "task.json",
        repository_path=repository_path,
        provenance_path=provenance_path,
        visible_initial_failures=visible_failures,
        hidden_initial_failures=hidden_failures,
    )


def parse_args() -> argparse.Namespace:
    """Parse deterministic generation parameters."""
    parser = argparse.ArgumentParser(
        description=("Generate validated PatchPilot tasks from killed Mutmut mutants.")
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--test-path", action="append", default=None)
    parser.add_argument("--hidden-test-root", required=True)
    parser.add_argument("--test-command", default="python -m pytest -q")
    parser.add_argument("--pytest-pythonpath", action="append", default=[])
    parser.add_argument(
        "--output-root",
        default="generated_benchmarks/mutmut",
    )
    parser.add_argument(
        "--work-root",
        default="artifacts/mutmut_generation",
    )
    parser.add_argument("--task-prefix", default="mutmut")
    parser.add_argument(
        "--difficulty",
        choices=["auto", "easy", "medium", "hard"],
        default="auto",
    )
    parser.add_argument("--max-tasks", type=int, default=15)
    parser.add_argument("--max-per-function", type=int, default=2)
    parser.add_argument(
        "--mutmut-timeout-seconds",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate one complete project slice of the research benchmark."""
    args = parse_args()
    project_root = Path.cwd().resolve(strict=True)
    source_root = Path(args.source_root).resolve(strict=True)
    validate_source_tree(source_root)

    hidden_source_root = Path(args.hidden_test_root).resolve(strict=True)
    if not hidden_source_root.is_dir():
        raise SystemExit("Hidden test root must be a directory.")
    if not hidden_source_root.is_relative_to(source_root):
        raise SystemExit(
            "Hidden test root must be inside the seed project for provenance."
        )
    if any(path.is_symlink() for path in hidden_source_root.rglob("*")):
        raise SystemExit("Hidden test roots cannot contain symbolic links.")

    hidden_relative_path = hidden_source_root.relative_to(source_root)
    output_root = Path(args.output_root).resolve()
    work_base = Path(args.work_root).resolve()
    project_id = args.project_id or slugify(source_root.name)
    generation_root = work_base / project_id
    work_root = generation_root / "work"

    test_paths = args.test_path or ["tests"]
    test_command_args = shlex.split(args.test_command)
    if len(test_command_args) < 3:
        raise SystemExit("--test-command must contain at least 3 arguments")
    if args.max_tasks < 1:
        raise SystemExit("--max-tasks must be at least 1")
    if args.max_per_function < 1:
        raise SystemExit("--max-per-function must be at least 1")

    generation_root.mkdir(parents=True, exist_ok=True)
    copy_source_tree(source_root, work_root)
    ensure_mutmut_config(
        work_root,
        source_paths=args.source_path,
        test_paths=test_paths,
        test_command=args.test_command,
        pytest_pythonpath=args.pytest_pythonpath,
    )
    work_hidden_root = work_root / hidden_relative_path

    print(f"WORK_ROOT={work_root}", flush=True)
    print("RUN clean visible tests", flush=True)
    run_command(
        test_command_args,
        cwd=work_root,
        timeout_seconds=args.test_timeout_seconds,
        check=True,
    )

    print("RUN clean hidden tests", flush=True)
    clean_hidden = run_hidden_tests(
        repository_root=work_root,
        hidden_test_root=work_hidden_root,
        timeout_seconds=args.test_timeout_seconds,
    )
    if clean_hidden.returncode != 0:
        raise SystemExit(format_command_failure(clean_hidden))
    clean_hidden_count = pytest_test_count(
        clean_hidden.stdout + "\n" + clean_hidden.stderr
    )
    if clean_hidden_count < 1:
        raise SystemExit("Hidden suite did not report any tests.")

    print("RUN Mutmut", flush=True)
    run_command(
        ["mutmut", "run"],
        cwd=work_root,
        timeout_seconds=args.mutmut_timeout_seconds,
        check=True,
    )

    print("READ Mutmut results", flush=True)
    results = run_command(
        ["mutmut", "results", "--all", "true"],
        cwd=work_root,
        timeout_seconds=120,
        check=True,
    )
    mutants = parse_mutmut_results(results.stdout)
    killed = [mutant for mutant in mutants if mutant.status == "killed"]
    if not killed:
        raise SystemExit("No killed mutants found.")

    print("DESCRIBE killed mutants", flush=True)
    descriptors = describe_killed_mutants(killed, work_root=work_root)
    selected = select_diverse_mutants(
        descriptors,
        args.max_tasks,
        args.max_per_function,
    )
    if len(selected) < args.max_tasks:
        raise SystemExit(
            f"Only {len(selected)} diverse killed mutants were available; "
            f"{args.max_tasks} were requested."
        )

    forbidden_paths = sorted(
        {
            *test_paths,
            hidden_relative_path.as_posix(),
            ".mutmut-cache",
            "mutants",
            "__pycache__",
        }
    )
    source_hash = source_tree_sha256(source_root)
    version = mutmut_version(project_root)
    generator_commit_value = git_commit(project_root)
    generation_command = [sys.executable, *sys.argv]
    generated: list[GeneratedTask] = []

    for rank, descriptor in enumerate(selected, start=1):
        print(f"EXPORT {rank}/{len(selected)} {descriptor.name}", flush=True)
        task = export_mutant_task(
            descriptor=descriptor,
            selection_rank=rank,
            work_root=work_root,
            output_root=output_root,
            project_root=project_root,
            source_project=project_id,
            source_hash=source_hash,
            generator_commit_value=generator_commit_value,
            mutmut_version_value=version,
            generation_command=generation_command,
            selected_from_total=len(mutants),
            selected_from_killed=len(killed),
            task_prefix=args.task_prefix,
            source_paths=args.source_path,
            forbidden_paths=forbidden_paths,
            test_command_args=test_command_args,
            hidden_source_root=hidden_source_root,
            hidden_relative_path=hidden_relative_path,
            clean_hidden_test_count=clean_hidden_count,
            test_timeout_seconds=args.test_timeout_seconds,
            requested_difficulty=args.difficulty,
            force=args.force,
        )
        if task is not None:
            generated.append(task)
            print(f"DONE {task.task_id}", flush=True)

    if len(generated) != args.max_tasks:
        raise SystemExit(
            f"Generated {len(generated)} validated tasks; "
            f"{args.max_tasks} were required."
        )

    rows = [
        {
            "task_id": task.task_id,
            "mutant_name": task.mutant_name,
            "mutant_status": task.mutant_status,
            "operator_family": task.operator_family,
            "mutated_function": task.mutated_function,
            "manifest_path": repository_root_for_manifest(
                task.manifest_path,
                project_root,
            ),
            "repository_path": repository_root_for_manifest(
                task.repository_path,
                project_root,
            ),
            "provenance_path": repository_root_for_manifest(
                task.provenance_path,
                project_root,
            ),
            "visible_initial_failures": task.visible_initial_failures,
            "hidden_initial_failures": task.hidden_initial_failures,
        }
        for task in generated
    ]
    write_csv(output_root / "mutmut_tasks.csv", rows)

    operator_counts = Counter(task.operator_family for task in generated)
    write_json(
        output_root / "generation_summary.json",
        {
            "schema_version": "2.0",
            "source_project": project_id,
            "source_root": repository_root_for_manifest(
                source_root,
                project_root,
            ),
            "source_root_sha256": source_hash,
            "generator_commit": generator_commit_value,
            "mutmut_version": version,
            "generation_command": generation_command,
            "output_root": repository_root_for_manifest(
                output_root,
                project_root,
            ),
            "mutants_total": len(mutants),
            "killed_mutants": len(killed),
            "described_killed_mutants": len(descriptors),
            "selected_mutants": len(selected),
            "generated_tasks": len(generated),
            "max_tasks": args.max_tasks,
            "max_per_function": args.max_per_function,
            "hidden_test_count": clean_hidden_count,
            "operator_family_counts": dict(sorted(operator_counts.items())),
        },
    )

    print(f"OUTPUT_ROOT={output_root}")
    print(f"GENERATED_TASKS={len(generated)}")
    print(f"KILLED_MUTANTS={len(killed)}")
    print(f"HIDDEN_TESTS={clean_hidden_count}")


if __name__ == "__main__":
    main()
