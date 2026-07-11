"""Independently validate the 45-task Mutmut research benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from patchpilot.benchmark.manifest import load_manifest
from patchpilot.benchmark.provenance import load_mutmut_provenance

PROJECT_COUNTS = {
    "mutmut_algorithms": 15,
    "mutmut_collections": 15,
    "mutmut_textdata": 15,
}
IGNORED_DIRECTORIES = {
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
OUTCOME_PATTERN = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?:passed|failed|errors?|skipped|xfailed|xpassed)"
)


def ignored(relative: Path) -> bool:
    """Return whether a path is excluded from deterministic tree hashes."""
    return (
        any(part in IGNORED_DIRECTORIES for part in relative.parts)
        or relative.suffix == ".pyc"
    )


def tree_sha256(root: Path) -> str:
    """Hash included paths and file contents deterministically."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ignored(relative):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(path: Path, root: Path, label: str) -> Path:
    """Resolve a required path and ensure it remains inside the project."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise SystemExit(f"{label} escapes project root: {resolved}")
    return resolved


def test_environment(repository: Path) -> dict[str, str]:
    """Build a deterministic environment for visible and hidden tests."""
    python_paths = [repository]
    source_root = repository / "src"
    if source_root.is_dir():
        python_paths.insert(0, source_root)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def run_visible(
    repository: Path,
    command: list[str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run the agent-visible regression suite without creating caches."""
    if len(command) < 3 or command[1:3] != ["-m", "pytest"]:
        raise SystemExit(f"Unsupported test command: {command!r}")
    final = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *command[3:],
    ]
    return subprocess.run(
        final,
        cwd=repository,
        env=test_environment(repository),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def run_hidden(
    repository: Path,
    hidden_root: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run the hidden judge suite outside the visible repository."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(hidden_root),
        ],
        cwd=repository,
        env=test_environment(repository),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def outcome_count(output: str) -> int:
    """Count pytest outcomes from its terminal summary."""
    return sum(int(match.group("count")) for match in OUTCOME_PATTERN.finditer(output))


def failure_count(output: str) -> int:
    """Estimate the initial failure count using generator semantics."""
    for pattern in (r"(\d+)\s+failed", r"(\d+)\s+errors?"):
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is not None:
            return max(1, int(match.group(1)))
    return 1


def validate_summary(root: Path, project_id: str) -> dict[str, Any]:
    """Validate one generation summary."""
    path = root / "generated_benchmarks" / project_id / "generation_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_project": project_id,
        "mutmut_version": "3.6.0",
        "generated_tasks": 15,
        "max_tasks": 15,
        "max_per_function": 3,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"Invalid {project_id} summary field {key}: {payload.get(key)!r}"
            )
    if int(payload.get("attempted_candidates", 0)) < 15:
        raise SystemExit(f"{project_id} attempted fewer than 15 candidates.")
    return payload


def main() -> None:
    """Validate every generated task and write the suite catalog."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-catalog", action="store_true")
    parser.add_argument("--test-timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    root = Path.cwd().resolve(strict=True)
    generated_root = root / "generated_benchmarks"
    seed_registry = json.loads(
        (root / "benchmark_seeds" / "projects.json").read_text(encoding="utf-8")
    )
    seed_by_id = {str(item["project_id"]): item for item in seed_registry}
    if set(seed_by_id) != set(PROJECT_COUNTS):
        raise SystemExit("Seed registry does not match the research projects.")

    sanity_manifests = sorted((root / "benchmarks").glob("*/task.json"))
    if len(sanity_manifests) != 12:
        raise SystemExit(
            f"Expected 12 separate sanity tasks, found {len(sanity_manifests)}."
        )

    expected_commit = os.environ.get(
        "PATCHPILOT_GENERATOR_COMMIT",
        "",
    ).strip()

    project_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    task_ids: set[str] = set()
    catalog_rows: list[dict[str, object]] = []

    for project_id, expected_count in PROJECT_COUNTS.items():
        validate_summary(root, project_id)
        manifests = sorted((generated_root / project_id).glob("*/task.json"))
        if len(manifests) != expected_count:
            raise SystemExit(
                f"{project_id}: expected {expected_count} manifests, "
                f"found {len(manifests)}."
            )

        seed_root = safe_relative(
            root / str(seed_by_id[project_id]["source_root"]),
            root,
            f"{project_id} seed root",
        )
        expected_seed_hash = tree_sha256(seed_root)

        for manifest_path in manifests:
            manifest = load_manifest(manifest_path)
            if manifest.task_id in task_ids:
                raise SystemExit(f"Duplicate task id: {manifest.task_id}")
            task_ids.add(manifest.task_id)

            task_root = manifest_path.parent.resolve(strict=True)
            repository = safe_relative(
                root / manifest.repository_root,
                root,
                f"{manifest.task_id} repository",
            )
            if manifest.hidden_test_root is None:
                raise SystemExit(f"Hidden verification missing: {manifest.task_id}")
            hidden_root = safe_relative(
                root / manifest.hidden_test_root,
                root,
                f"{manifest.task_id} hidden root",
            )
            expected_hidden = manifest.expected_hidden_tests
            if expected_hidden is None:
                raise SystemExit(f"Hidden test count missing: {manifest.task_id}")

            if repository != task_root / "repository":
                raise SystemExit(f"Repository path mismatch: {manifest.task_id}")
            if hidden_root != task_root / "hidden_tests":
                raise SystemExit(f"Hidden path mismatch: {manifest.task_id}")
            if hidden_root == repository or hidden_root.is_relative_to(repository):
                raise SystemExit(f"Hidden tests exposed to agent: {manifest.task_id}")
            if (repository / "hidden_tests").exists():
                raise SystemExit(
                    f"Visible repository contains hidden tests: {manifest.task_id}"
                )
            if any(path.is_symlink() for path in task_root.rglob("*")):
                raise SystemExit(f"Symlink found in task: {manifest.task_id}")

            provenance_path = task_root / "provenance.json"
            provenance = load_mutmut_provenance(provenance_path)
            if provenance.source_project != project_id:
                raise SystemExit(f"Source project mismatch: {manifest.task_id}")
            if provenance.mutmut_version != "3.6.0":
                raise SystemExit(f"Wrong Mutmut version: {manifest.task_id}")
            if expected_commit and provenance.generator_commit != expected_commit:
                raise SystemExit(f"Generator commit mismatch: {manifest.task_id}")
            if provenance.mutant_status != "killed":
                raise SystemExit(f"Non-killed mutant: {manifest.task_id}")
            if provenance.source_root_sha256 != expected_seed_hash:
                raise SystemExit(f"Seed hash mismatch: {manifest.task_id}")
            repository_hash = tree_sha256(repository)
            if provenance.exported_repository_sha256 != repository_hash:
                raise SystemExit(f"Repository hash mismatch: {manifest.task_id}")
            hidden_hash = tree_sha256(hidden_root)
            if provenance.hidden_tests_sha256 != hidden_hash:
                raise SystemExit(f"Hidden-suite hash mismatch: {manifest.task_id}")
            if list(manifest.allowed_paths) != provenance.source_paths:
                raise SystemExit(f"Allowed source paths mismatch: {manifest.task_id}")
            if manifest.test_command != provenance.test_command:
                raise SystemExit(f"Test command mismatch: {manifest.task_id}")
            if provenance.max_tasks != 15:
                raise SystemExit(f"Invalid max_tasks provenance: {manifest.task_id}")
            if provenance.max_per_function != 3:
                raise SystemExit(
                    f"Invalid max_per_function provenance: {manifest.task_id}"
                )
            if not provenance.generation_command:
                raise SystemExit(f"Generation command missing: {manifest.task_id}")
            source_file = Path(provenance.source_file)
            if source_file.is_absolute() or ".." in source_file.parts:
                raise SystemExit(f"Unsafe source location: {manifest.task_id}")
            if not provenance.mutation_diff.strip():
                raise SystemExit(f"Empty mutation diff: {manifest.task_id}")

            visible = run_visible(
                repository,
                list(manifest.test_command),
                args.test_timeout_seconds,
            )
            visible_output = visible.stdout + "\n" + visible.stderr
            if visible.returncode == 0:
                raise SystemExit(
                    f"Visible suite unexpectedly passes: {manifest.task_id}"
                )
            observed_visible_failures = failure_count(visible_output)
            if observed_visible_failures != manifest.expected_initial_failures:
                raise SystemExit(
                    f"Visible failure mismatch for {manifest.task_id}: "
                    f"expected {manifest.expected_initial_failures}, "
                    f"found {observed_visible_failures}."
                )
            if observed_visible_failures != provenance.visible_initial_failures:
                raise SystemExit(f"Visible provenance mismatch: {manifest.task_id}")

            hidden = run_hidden(
                repository,
                hidden_root,
                args.test_timeout_seconds,
            )
            hidden_output = hidden.stdout + "\n" + hidden.stderr
            if hidden.returncode == 0:
                raise SystemExit(
                    f"Hidden suite unexpectedly passes: {manifest.task_id}"
                )
            observed_hidden_count = outcome_count(hidden_output)
            if observed_hidden_count != expected_hidden:
                raise SystemExit(
                    f"Hidden count mismatch for {manifest.task_id}: "
                    f"expected {expected_hidden}, "
                    f"found {observed_hidden_count}."
                )
            observed_hidden_failures = failure_count(hidden_output)
            if observed_hidden_failures != provenance.hidden_initial_failures:
                raise SystemExit(f"Hidden provenance mismatch: {manifest.task_id}")

            project_counts[project_id] += 1
            operator_counts[provenance.operator_family.value] += 1
            difficulty_counts[manifest.difficulty] += 1
            catalog_rows.append(
                {
                    "task_id": manifest.task_id,
                    "project_id": project_id,
                    "manifest_path": manifest_path.relative_to(root).as_posix(),
                    "repository_path": repository.relative_to(root).as_posix(),
                    "hidden_test_path": hidden_root.relative_to(root).as_posix(),
                    "provenance_path": provenance_path.relative_to(root).as_posix(),
                    "mutant_name": provenance.mutant_name,
                    "mutated_function": provenance.mutated_function,
                    "operator_family": provenance.operator_family.value,
                    "difficulty": manifest.difficulty,
                    "source_root_sha256": provenance.source_root_sha256,
                    "repository_sha256": repository_hash,
                    "hidden_tests_sha256": hidden_hash,
                    "manifest_sha256": file_sha256(manifest_path),
                    "provenance_sha256": file_sha256(provenance_path),
                }
            )
            print(
                f"VALID {len(catalog_rows):02d}/45 {manifest.task_id}",
                flush=True,
            )

    if dict(project_counts) != PROJECT_COUNTS:
        raise SystemExit(f"Unexpected project composition: {dict(project_counts)}")
    if len(task_ids) != 45:
        raise SystemExit(f"Expected 45 unique task ids, found {len(task_ids)}.")

    catalog = {
        "schema_version": "1.0",
        "suite_id": "patchpilot-mutmut-research-v1",
        "benchmark_kind": "mutmut",
        "task_count": 45,
        "mutmut_version": "3.6.0",
        "project_breakdown": dict(sorted(project_counts.items())),
        "operator_breakdown": dict(sorted(operator_counts.items())),
        "difficulty_breakdown": dict(sorted(difficulty_counts.items())),
        "sanity_tasks_excluded": len(sanity_manifests),
        "tasks": sorted(
            catalog_rows,
            key=lambda item: str(item["task_id"]),
        ),
    }
    if args.write_catalog:
        catalog_path = generated_root / "mutmut_research_suite.json"
        catalog_path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"CATALOG={catalog_path}", flush=True)

    print("MUTMUT_RESEARCH_VALIDATED=45", flush=True)
    print("SANITY_TASKS_EXCLUDED=12", flush=True)


if __name__ == "__main__":
    main()
