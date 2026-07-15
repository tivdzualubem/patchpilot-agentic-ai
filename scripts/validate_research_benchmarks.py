"""Validate the complete 53-task PatchPilot research benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from patchpilot.benchmark import load_manifest
from patchpilot.benchmark.provenance import load_mutmut_provenance

MUTMUT_ROOTS = (
    Path("generated_benchmarks/mutmut_algorithms"),
    Path("generated_benchmarks/mutmut_collections"),
    Path("generated_benchmarks/mutmut_textdata"),
)
CHALLENGE_ROOT = Path("challenge_benchmarks")
CATALOG_PATH = Path("generated_benchmarks/research_suite.json")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pytest_environment() -> dict[str, str]:
    """Return an isolated pytest environment."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def run_pytest(
    repository: Path,
    target: Path,
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run one benchmark test suite."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(target),
        ],
        cwd=repository,
        env=pytest_environment(),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def failure_count(output: str) -> int:
    """Extract pytest's failing-test count."""
    match = re.search(r"(\d+) failed", output)
    return int(match.group(1)) if match is not None else 0


def ensure_safe_tree(root: Path) -> None:
    """Reject symlinks and runtime artifacts."""
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden: {path}")
        if path.name in {"__pycache__", ".pytest_cache"}:
            raise ValueError(f"runtime artifact is forbidden: {path}")


def validate_mutmut(project_root: Path) -> list[dict[str, object]]:
    """Validate the 45 generated killed-mutant task records."""
    rows: list[dict[str, object]] = []
    project_counts: dict[str, int] = {}

    for root in MUTMUT_ROOTS:
        manifests = sorted((project_root / root).glob("*/task.json"))
        project_counts[root.name] = len(manifests)
        for path in manifests:
            manifest = load_manifest(path)
            task_root = path.parent
            provenance_path = task_root / "provenance.json"
            metadata_path = task_root / "mutmut_metadata.json"
            if not provenance_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"incomplete Mutmut task: {task_root}")
            provenance = load_mutmut_provenance(provenance_path)
            metadata = read_json(metadata_path)
            if provenance.benchmark_kind != "mutmut":
                raise ValueError(f"invalid Mutmut benchmark kind: {provenance_path}")
            if provenance.source_project != root.name:
                raise ValueError(f"Mutmut source project mismatch: {provenance_path}")
            if provenance.mutant_status != "killed":
                raise ValueError(
                    f"Mutmut task is not a killed mutant: {provenance_path}"
                )
            if provenance.difficulty != manifest.difficulty:
                raise ValueError(f"Mutmut difficulty mismatch: {provenance_path}")
            if provenance.test_command != manifest.test_command:
                raise ValueError(f"Mutmut test command mismatch: {provenance_path}")
            if metadata.get("mutant_status") != "killed":
                raise ValueError(f"invalid Mutmut metadata status: {metadata_path}")
            if metadata.get("mutant_name") != provenance.mutant_name:
                raise ValueError(f"Mutmut metadata name mismatch: {metadata_path}")
            if metadata.get("operator_family") != provenance.operator_family.value:
                raise ValueError(f"Mutmut operator mismatch: {metadata_path}")
            if manifest.hidden_test_root is None:
                raise ValueError(f"hidden verification missing: {path}")
            rows.append(
                {
                    "task_id": manifest.task_id,
                    "manifest_path": path.relative_to(project_root).as_posix(),
                    "origin_type": "mutmut",
                    "difficulty": manifest.difficulty,
                    "defect_category": manifest.defect_category,
                }
            )

    expected = {
        "mutmut_algorithms": 15,
        "mutmut_collections": 15,
        "mutmut_textdata": 15,
    }
    if project_counts != expected:
        raise ValueError(f"expected 15+15+15 Mutmut tasks, found {project_counts}")
    return rows


def validate_challenges(
    project_root: Path,
    *,
    timeout_seconds: int,
    runtime: bool,
) -> list[dict[str, object]]:
    """Validate all eight manual challenge tasks."""
    manifests = sorted((project_root / CHALLENGE_ROOT).glob("*/task.json"))
    if len(manifests) != 8:
        raise ValueError(f"expected 8 manual challenges, found {len(manifests)}")

    rows: list[dict[str, object]] = []
    for index, path in enumerate(manifests, start=1):
        manifest = load_manifest(path)
        task_root = path.parent
        repository = project_root / manifest.repository_root
        hidden_root = project_root / str(manifest.hidden_test_root)
        provenance_path = task_root / "provenance.json"

        if repository.parent != task_root:
            raise ValueError(f"repository escaped task root: {path}")
        if hidden_root.parent != task_root:
            raise ValueError(f"hidden tests escaped task root: {path}")
        if hidden_root.is_relative_to(repository):
            raise ValueError(f"hidden tests leaked into repository: {path}")
        if not repository.is_dir() or not hidden_root.is_dir():
            raise FileNotFoundError(f"incomplete challenge task: {path}")

        ensure_safe_tree(task_root)
        provenance = read_json(provenance_path)
        required = {
            "origin_type",
            "task_id",
            "defect_patterns",
            "rationale",
            "clean_source_sha256",
            "defective_source_sha256",
            "repository_tree_sha256",
            "hidden_tests_tree_sha256",
            "defect_diff",
            "changed_line_count",
            "visible_validation",
            "hidden_validation",
            "clean_reference_validation",
        }
        missing = sorted(required - provenance.keys())
        if missing:
            raise ValueError(
                f"missing challenge provenance fields in {provenance_path}: {missing}"
            )
        if provenance["origin_type"] != "manual_challenge":
            raise ValueError(f"wrong origin type: {provenance_path}")
        if provenance["task_id"] != manifest.task_id:
            raise ValueError(f"task id mismatch: {provenance_path}")
        if int(provenance["changed_line_count"]) < 3:
            raise ValueError(f"challenge is not multi-line: {provenance_path}")
        clean_validation = provenance["clean_reference_validation"]
        if not isinstance(clean_validation, dict):
            raise ValueError(f"invalid clean validation: {provenance_path}")
        if not clean_validation.get("visible_passed"):
            raise ValueError(f"clean visible suite was not proven: {path}")
        if not clean_validation.get("hidden_passed"):
            raise ValueError(f"clean hidden suite was not proven: {path}")

        if runtime:
            visible = run_pytest(
                repository,
                repository / "tests",
                timeout_seconds=timeout_seconds,
            )
            hidden = run_pytest(
                repository,
                hidden_root,
                timeout_seconds=timeout_seconds,
            )
            visible_failures = failure_count(visible.stdout + "\n" + visible.stderr)
            hidden_failures = failure_count(hidden.stdout + "\n" + hidden.stderr)
            if visible.returncode == 0:
                raise ValueError(f"defective visible suite passed: {path}")
            if hidden.returncode == 0:
                raise ValueError(f"defective hidden suite passed: {path}")
            if visible_failures != manifest.expected_initial_failures:
                raise ValueError(
                    f"visible failure count mismatch for {manifest.task_id}: "
                    f"{visible_failures} != "
                    f"{manifest.expected_initial_failures}"
                )
            if hidden_failures < 1:
                raise ValueError(f"hidden failure count missing for {manifest.task_id}")

        rows.append(
            {
                "task_id": manifest.task_id,
                "manifest_path": path.relative_to(project_root).as_posix(),
                "origin_type": "manual_challenge",
                "difficulty": manifest.difficulty,
                "defect_category": manifest.defect_category,
            }
        )
        print(
            f"VALID CHALLENGE {index:02d}/08 {manifest.task_id}",
            flush=True,
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate the complete PatchPilot research benchmark."
    )
    parser.add_argument("--write-catalog", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--test-timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    """Validate and optionally write the 53-task suite catalog."""
    args = parse_args()
    project_root = Path(".").resolve(strict=True)

    mutmut_rows = validate_mutmut(project_root)
    challenge_rows = validate_challenges(
        project_root,
        timeout_seconds=args.test_timeout_seconds,
        runtime=not args.skip_runtime,
    )
    all_rows = sorted(
        [*mutmut_rows, *challenge_rows],
        key=lambda item: str(item["task_id"]),
    )
    task_ids = [str(row["task_id"]) for row in all_rows]

    if len(mutmut_rows) != 45:
        raise ValueError(f"expected 45 Mutmut tasks, found {len(mutmut_rows)}")
    if len(challenge_rows) != 8:
        raise ValueError(f"expected 8 manual challenges, found {len(challenge_rows)}")
    if len(all_rows) != 53 or len(set(task_ids)) != 53:
        raise ValueError("research suite must contain 53 unique task IDs")

    sanity_manifests = sorted((project_root / "benchmarks").glob("*/task.json"))
    if len(sanity_manifests) != 12:
        raise ValueError(f"sanity benchmark count changed: {len(sanity_manifests)}")

    catalog = {
        "schema_version": "1.0",
        "suite_id": "patchpilot-primary-research-benchmark",
        "task_count": 53,
        "mutmut_task_count": 45,
        "manual_challenge_task_count": 8,
        "sanity_task_count_excluded": 12,
        "manifest_paths": [row["manifest_path"] for row in all_rows],
        "tasks": all_rows,
    }
    if args.write_catalog:
        write_json(project_root / CATALOG_PATH, catalog)

    print("RESEARCH_BENCHMARK_VALIDATED=53", flush=True)
    print("MUTMUT_TASKS=45", flush=True)
    print("MANUAL_CHALLENGES=8", flush=True)
    print("SANITY_TASKS_EXCLUDED=12", flush=True)
    print(f"CATALOG={project_root / CATALOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
